"""tnoq (middle-out TNO growth with quimb arbitrary-geometry compression) with a selectable array backend
(backend.py: np128 | np64 | mlx | mlxcpu), per-phase timers, and a wall-clock cap.  Logic identical to tnoq.py."""
import time
import numpy as np
import quimb.tensor as qtn
from quimb.tensor.tensor_arbgeom_compress import tensor_network_ag_compress
import backend as B
from tnoq import CZ, _nid, unitG, split_gate

TIMERS = {'gate': 0.0, 'compress': 0.0, 'sync': 0.0}


def _sync(tn):
    if B.NAME.startswith('mlx'):
        t = time.perf_counter()
        B._mx.eval([x.data for x in tn])
        TIMERS['sync'] += time.perf_counter() - t


class QTNO:
    def __init__(self, n):
        self.n = n
        ts = [qtn.Tensor(B.asb(np.eye(2)), inds=(f"k{q}", f"b{q}"), tags={f"I{q}"}) for q in range(n)]
        self.tn = qtn.TensorNetwork(ts)
        self.site_tags = [f"I{q}" for q in range(n)]

    def gate(self, G, a, b, side):
        t0 = time.perf_counter()
        A, Bm = split_gate(G)
        A, Bm = B.asb(A), B.asb(Bm)
        r = _nid("r")
        ta, tb = _nid("t"), _nid("t")
        if side == 'after':
            self.tn.reindex_({f"k{a}": ta, f"k{b}": tb})
            self.tn |= qtn.Tensor(A, inds=(f"k{a}", ta, r), tags={f"I{a}"})
            self.tn |= qtn.Tensor(Bm, inds=(r, f"k{b}", tb), tags={f"I{b}"})
        else:
            self.tn.reindex_({f"b{a}": ta, f"b{b}": tb})
            self.tn |= qtn.Tensor(A, inds=(ta, f"b{a}", r), tags={f"I{a}"})
            self.tn |= qtn.Tensor(Bm, inds=(r, tb, f"b{b}"), tags={f"I{b}"})
        TIMERS['gate'] += time.perf_counter() - t0

    def compress(self, max_bond=None, cutoff=1e-3, method='local-late'):
        t0 = time.perf_counter()
        self.tn = tensor_network_ag_compress(self.tn, max_bond=max_bond, cutoff=cutoff, method=method,
                                             site_tags=self.site_tags, canonize=True, equalize_norms=True)
        self.tn.squeeze_()
        _sync(self.tn)
        TIMERS['compress'] += time.perf_counter() - t0

    def stats(self):
        mb = max((self.tn.ind_size(ix) for ix in self.tn.inner_inds()), default=1)
        return dict(elems=sum(t.size for t in self.tn), max_bond=mb, ntensors=self.tn.num_tensors,
                    nbonds=len(self.tn.inner_inds()))


def grow(n, units, L, c, cutoff=1e-3, max_bond=None, chunk=1, max_elems=5e6, log=True, method='local-late',
         stop_layers=None, snapshots=None, tmax=None):
    """as tnoq.grow; snapshots are stored as numpy complex128 networks (for the exact final contraction)."""
    D = max(L) + 1
    bylayer = [[] for _ in range(D)]
    for k in range(len(units)):
        bylayer[L[k]].append(k)
    W = QTNO(n)
    lo = hi = int(np.ceil(c)); t0 = time.time(); hist = []
    lim_lo, lim_hi = (0, D) if stop_layers is None else stop_layers
    while lo > lim_lo or hi < lim_hi:
        ts = time.perf_counter()
        for _ in range(chunk):
            if hi < lim_hi:
                for k in bylayer[hi]:
                    W.gate(unitG(units[k]), units[k][0], units[k][1], 'after')
                hi += 1
            if lo > lim_lo:
                for k in reversed(bylayer[lo - 1]):
                    W.gate(unitG(units[k]), units[k][0], units[k][1], 'before')
                lo -= 1
        W.compress(max_bond=max_bond, cutoff=cutoff, method=method)
        st = W.stats(); st.update(lo=lo, hi=hi, t=round(time.time() - t0, 2), step=round(time.perf_counter() - ts, 3))
        hist.append(st)
        if snapshots is not None:
            snapshots.append((lo, hi, B.to_numpy_tn(W.tn)))
        if log:
            print("  ", st, flush=True)
        if st['elems'] > max_elems or (tmax and time.time() - t0 > tmax):
            break
    return W, lo, hi, hist
