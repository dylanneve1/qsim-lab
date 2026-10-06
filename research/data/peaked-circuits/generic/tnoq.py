"""Middle-out operator cancellation on fixed wires with quimb arbitrary-geometry compression.

W (window operator) is a quimb TensorNetwork: per wire q one site (tag I{q}), upper index k{q} (output),
lower index b{q} (input).  Gates are split into two site tensors (operator Schmidt decomposition, rank 2
for CZ-class gates) so every tensor belongs to exactly one site; after each chunk of layers all tensors of
a site are fused and every bond is compressed (tensor_network_ag_compress, canonised local compression).
The idea follows Kremer-Dupuis' TNO iterative cancellation (2604.21908 / tracker #47, #105); this is an
independent implementation on our parsed CZ units, with the mirror centre found from the data.
"""
import time
import numpy as np
import quimb.tensor as qtn
from quimb.tensor.tensor_arbgeom_compress import tensor_network_ag_compress

CZ = np.diag([1, 1, 1, -1]).astype(complex)
_uid = [0]


def _nid(p):
    _uid[0] += 1
    return f"{p}{_uid[0]}"


def unitG(u):
    a, b, Pa, Pb = u
    return CZ @ np.kron(Pa, Pb)


def split_gate(G):
    """G (4x4, (a_out b_out),(a_in b_in)) -> A[a_out, a_in, r], B[r, b_out, b_in]."""
    Gt = G.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)     # (a_out a_in),(b_out b_in)
    U, s, Vh = np.linalg.svd(Gt)
    r = int(np.sum(s > 1e-12 * s[0]))
    A = (U[:, :r] * np.sqrt(s[:r])).reshape(2, 2, r)
    B = (np.sqrt(s[:r])[:, None] * Vh[:r]).reshape(r, 2, 2)
    return A, B


class QTNO:
    def __init__(self, n):
        self.n = n
        ts = [qtn.Tensor(np.eye(2, dtype=complex), inds=(f"k{q}", f"b{q}"), tags={f"I{q}"}) for q in range(n)]
        self.tn = qtn.TensorNetwork(ts)
        self.site_tags = [f"I{q}" for q in range(n)]

    def gate(self, G, a, b, side):
        A, B = split_gate(G)
        r = _nid("r")
        if side == 'after':
            ta, tb = _nid("t"), _nid("t")
            self.tn.reindex_({f"k{a}": ta, f"k{b}": tb})
            self.tn |= qtn.Tensor(A, inds=(f"k{a}", ta, r), tags={f"I{a}"})
            self.tn |= qtn.Tensor(B, inds=(r, f"k{b}", tb), tags={f"I{b}"})
        else:
            ta, tb = _nid("t"), _nid("t")
            self.tn.reindex_({f"b{a}": ta, f"b{b}": tb})
            self.tn |= qtn.Tensor(A, inds=(ta, f"b{a}", r), tags={f"I{a}"})
            self.tn |= qtn.Tensor(B, inds=(r, tb, f"b{b}"), tags={f"I{b}"})

    def compress(self, max_bond=None, cutoff=1e-3, method='local-late'):
        self.tn = tensor_network_ag_compress(self.tn, max_bond=max_bond, cutoff=cutoff, method=method,
                                             site_tags=self.site_tags, canonize=True, equalize_norms=True)
        self.tn.squeeze_()

    def stats(self):
        bonds = {}
        for t in self.tn:
            pass
        mb = max((self.tn.ind_size(ix) for ix in self.tn.inner_inds()), default=1)
        return dict(elems=sum(t.size for t in self.tn), max_bond=mb, ntensors=self.tn.num_tensors,
                    nbonds=len(self.tn.inner_inds()))


def grow(n, units, L, c, cutoff=1e-3, max_bond=None, chunk=1, max_elems=5e6, log=True, method='local-late',
         stop_layers=None, snapshots=None):
    """layer-synchronous growth from ASAP-layer cut c; returns (W, lo, hi, history)."""
    D = max(L) + 1
    bylayer = [[] for _ in range(D)]
    for k in range(len(units)):
        bylayer[L[k]].append(k)
    W = QTNO(n)
    lo = hi = int(np.ceil(c)); t0 = time.time(); hist = []
    lim_lo, lim_hi = (0, D) if stop_layers is None else stop_layers
    while lo > lim_lo or hi < lim_hi:
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
        st = W.stats(); st.update(lo=lo, hi=hi, t=round(time.time() - t0, 1))
        hist.append(st)
        if snapshots is not None:
            snapshots.append((lo, hi, W.tn.copy()))
        if log:
            print("  ", st, flush=True)
        if st['elems'] > max_elems:
            break
    return W, lo, hi, hist
