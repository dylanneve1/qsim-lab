"""Lean numpy re-implementation of tnoq's middle-out growth (drop-in for tnoq.grow / QTNO).

Same algorithm as tnoq + quimb tensor_network_ag_compress(method='local-late', canonize=True, cutoff,
equalize_norms=True):  every 2-qubit unit is split (operator Schmidt) and absorbed into the two site tensors;
after each layer pair all multi-bonds are fused and every bond is compressed with the 'virtual-tree' local gauge
(tree span r=3 around the bond, reduced factors R absorbed from the leaves inwards, oblique projectors from the
truncated SVD of R_l R_r^T with quimb's default 'rsum2' cutoff), then dim-1 bonds are dropped and tensor norms
are equalised.  Differences from quimb: plain ndarrays + integer bond ids (no Tensor/TN objects, no autoray, no
cotengra path search), gates are contracted into the site tensor immediately, bonds are compressed in creation
order, and the tree span uses the same scoring (connectivity = sum log2 bond dims, then ndim) but simplified
tie-breaking.  Numerically equivalent up to these ordering/gauge choices (validated on peaks / p).

Options: dtype (complex128 | complex64), batch_qr (stack same-shape leaf QRs of a tree span into one
np.linalg.qr call)."""
import math, time
import numpy as np
from tnoq import unitG, split_gate, _nid

DTYPE = np.complex128
BATCH_QR = False
R_TREE = 3
QR_IMPL = 'numpy'      # 'numpy' (np.linalg.qr mode='r') | 'lapack' (direct ?geqrf call: same LAPACK routine, no wrapper overhead)
_GEQRF = {}


def _qr_r(M):
    if QR_IMPL == 'numpy':
        return np.linalg.qr(M, mode='r')
    f = _GEQRF.get(M.dtype.char)
    if f is None:
        from scipy.linalg.lapack import get_lapack_funcs
        f = _GEQRF[M.dtype.char] = get_lapack_funcs(('geqrf',), (M,))[0]
    qr, tau, work, info = f(M)
    k = min(M.shape)
    return np.triu(qr[:k])


class Site:
    __slots__ = ('A', 'legs')          # legs: list of ('k',q) | ('b',q) | int bond id

    def __init__(self, A, legs):
        self.A, self.legs = A, legs


class FastTNO:
    def __init__(self, n, dtype=None):
        self.n = n
        self.dtype = dtype or DTYPE
        self.S = [Site(np.eye(2, dtype=self.dtype), [('k', q), ('b', q)]) for q in range(n)]
        self.bonds = {}                  # bond id -> (s1, s2)
        self.nbr = [dict() for _ in range(n)]   # site -> {other site: bond id}
        self.nb = 0

    # ------------------------------------------------------------------ helpers
    def _newbond(self, s, t):
        self.nb += 1
        b = self.nb
        self.bonds[b] = (s, t)
        self.nbr[s][t] = b; self.nbr[t][s] = b
        return b

    def _apply_leg(self, s, leg, G):
        """site s: leg <- G[new, old] . leg (quimb Tensor.gate semantics); leg keeps its name."""
        st = self.S[s]
        ax = st.legs.index(leg)
        st.A = np.moveaxis(np.tensordot(G, st.A, axes=([1], [ax])), 0, ax)

    def gate(self, G, a, b, side):
        A, B = split_gate(np.asarray(G))         # A[a_out, a_in, r], B[r, b_out, b_in]
        A = A.astype(self.dtype); B = B.astype(self.dtype)
        r = A.shape[2]
        kind = 'k' if side == 'after' else 'b'
        for s, F, rax in ((a, A, 2), (b, B, 0)):
            st = self.S[s]
            ax = st.legs.index((kind, s))
            if side == 'after':     # new[out, (r)] = F[out, old, r] . T[old]
                Fm = np.moveaxis(F, rax, 2) if rax != 2 else F           # (out, in, r)
                C = np.tensordot(Fm, st.A, axes=([1], [ax]))              # (out, r, rest...)
            else:                   # new[in, (r)] = T[old] F[old, in, r]
                Fm = np.moveaxis(F, rax, 2) if rax != 2 else F           # (out=old, in=new, r)
                C = np.tensordot(Fm, st.A, axes=([0], [ax]))              # (new in, r, rest...)
                # leg order (new, r, rest)
            rest = [l for i, l in enumerate(st.legs) if i != ax]
            st.A = C
            st.legs = [(kind, s), ('r', None)] + rest
        # bond between a and b (fuse with an existing one)
        if b in self.nbr[a]:
            bid = self.nbr[a][b]
            for s in (a, b):
                st = self.S[s]
                i_new, i_old = st.legs.index(('r', None)), st.legs.index(bid)
                order = [i for i in range(len(st.legs)) if i not in (i_new, i_old)] + [i_old, i_new]
                sh = st.A.shape
                st.A = np.transpose(st.A, order).reshape([sh[i] for i in order[:-2]] + [sh[i_old] * sh[i_new]])
                st.legs = [st.legs[i] for i in order[:-2]] + [bid]
        else:
            bid = self._newbond(a, b)
            for s in (a, b):
                st = self.S[s]
                st.legs[st.legs.index(('r', None))] = bid

    def size(self):
        return sum(st.A.size for st in self.S)

    def bond_dim(self, b):
        s, _ = self.bonds[b]
        st = self.S[s]
        return st.A.shape[st.legs.index(b)]

    # ------------------------------------------------------------------ compression
    def _tree_span(self, region, r):
        """quimb.tensor.networking.get_tree_span (weight_bonds, ndim 'max', distance 'min'), inwards order."""
        region = set(region)
        dist = {s: 0 for s in region}; merges = {}; conn = {}; cands = []
        def check(s, t):
            if t in region:
                return
            if t not in dist:
                merges[t] = s; dist[t] = dist[s] + 1
                if dist[t] <= r:
                    cands.append(t)
            conn[t] = conn.get(t, 0.0) + math.log2(self.bond_dim(self.nbr[s][t]))
        for s in sorted(region):
            for t in self.nbr[s]:
                check(s, t)
        seq = []
        while cands:
            cands.sort(key=lambda t: (conn[t], self.S[t].A.ndim, -dist[t]))
            s = cands.pop()
            region.add(s)
            seq.append((s, merges[s], dist[s]))
            for t in self.nbr[s]:
                check(s, t)
        seq.reverse()
        return seq

    def _reduced_factor(self, A, legs, ix, Gs):
        """absorb gauges Gs[bond] into the legs of A (except ix), then R of the QR with ix as the column leg."""
        for i, l in enumerate(legs):
            if l != ix and l in Gs:
                G = Gs.pop(l)
                A = np.moveaxis(np.tensordot(G, A, axes=([1], [i])), 0, i)
        ax = legs.index(ix)
        M = np.moveaxis(A, ax, -1)
        M = M.reshape(-1, M.shape[-1])
        return M

    def _compress_bond(self, b, cutoff, max_bond):
        s1, s2 = self.bonds[b]
        tree = self._tree_span((s1, s2), R_TREE)
        Gs = {}
        # (optional) batch the QRs of all leaves (outer tensors with no incoming gauges) by shape
        pending = {}
        if BATCH_QR:
            children = {}
            for o, i_, _ in tree:
                children.setdefault(i_, []).append(o)
            leaves = [(o, i_) for o, i_, _ in tree if o not in children]
            byshape = {}
            for o, i_ in leaves:
                bo = self.nbr[o][i_]
                M = self._reduced_factor(self.S[o].A, self.S[o].legs, bo, {})
                byshape.setdefault(M.shape, []).append((o, bo, M))
            for shp, items in byshape.items():
                if len(items) == 1:
                    o, bo, M = items[0]
                    pending[o] = _qr_r(M)
                else:
                    Rs = np.linalg.qr(np.stack([M for _, _, M in items]), mode='r')
                    for (o, bo, M), R in zip(items, Rs):
                        pending[o] = R
        for o, i_, _ in tree:
            bo = self.nbr[o][i_]
            if o in pending:
                R = pending.pop(o)
            else:
                M = self._reduced_factor(self.S[o].A, self.S[o].legs, bo, Gs)
                R = _qr_r(M)
            Gs[bo] = R / np.linalg.norm(R)
        Rs = []
        for s in (s1, s2):
            M = self._reduced_factor(self.S[s].A, self.S[s].legs, b, Gs)
            Rs.append(_qr_r(M))
        Rl, Rr = Rs
        U, sv, VH = np.linalg.svd(Rl @ Rr.T, full_matrices=False)
        # quimb 'rsum2': drop the tail whose summed s^2 <= cutoff * sum s^2
        s2_ = sv.astype(np.float64) ** 2
        tail = np.cumsum(s2_[::-1])[::-1]          # tail[i] = sum_{j>=i} s_j^2
        keep = int(np.sum(tail > cutoff * s2_.sum())) if cutoff > 0 else sv.size
        keep = max(keep, 1)
        if max_bond:
            keep = min(keep, max_bond)
        U, sv, VH = U[:, :keep], sv[:keep], VH[:keep]
        sq = np.sqrt(sv)
        Pl = (Rr.T @ VH.conj().T) / sq[None, :]          # (d, chi)
        Pr = (U.conj().T / sq[:, None]) @ Rl             # (chi, d)
        self._apply_leg(s1, b, Pl.T)
        self._apply_leg(s2, b, Pr)

    def compress(self, cutoff=1e-3, max_bond=None):
        for b in sorted(self.bonds):
            if b in self.bonds:
                self._compress_bond(b, cutoff, max_bond)
        # squeeze dim-1 bonds
        for b in [b for b in self.bonds if self.bond_dim(b) == 1]:
            s1, s2 = self.bonds.pop(b)
            del self.nbr[s1][s2]; del self.nbr[s2][s1]
            for s in (s1, s2):
                st = self.S[s]; i = st.legs.index(b)
                st.A = st.A.reshape([d for j, d in enumerate(st.A.shape) if j != i])
                st.legs = [l for j, l in enumerate(st.legs) if j != i]
        # equalize norms (quimb: strip each tensor to norm 1, then distribute the exponent evenly)
        nrms = [np.linalg.norm(st.A) for st in self.S]
        logm = float(np.mean(np.log(nrms)))
        for st, nr in zip(self.S, nrms):
            st.A = st.A * (math.exp(logm) / nr)

    # ------------------------------------------------------------------ interop
    def to_quimb(self):
        import quimb.tensor as qtn
        def name(l):
            if isinstance(l, tuple):
                return f"{l[0]}{l[1]}"
            return f"_fb{l}"
        ts = [qtn.Tensor(np.asarray(st.A, dtype=np.complex128), inds=[name(l) for l in st.legs], tags={f"I{q}"})
              for q, st in enumerate(self.S)]
        return qtn.TensorNetwork(ts)

    @property
    def tn(self):
        return self.to_quimb()

    def stats(self):
        dims = [self.bond_dim(b) for b in self.bonds]
        return dict(elems=self.size(), max_bond=max(dims, default=1), ntensors=self.n, nbonds=len(self.bonds))


class _W:
    """holder exposing .tn (a quimb TN) like tnoq.QTNO, for solve_generic."""
    pass


TIMERS = {'gate': 0.0, 'compress': 0.0}


def grow(n, units, L, c, cutoff=1e-3, max_bond=None, chunk=1, max_elems=5e6, log=True, method='local-late',
         stop_layers=None, snapshots=None, tmax=None):
    """drop-in for tnoq.grow (returns (W, lo, hi, hist); W.tn is a quimb TensorNetwork)."""
    D = max(L) + 1
    bylayer = [[] for _ in range(D)]
    for k in range(len(units)):
        bylayer[L[k]].append(k)
    W = FastTNO(n)
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
        tg = time.perf_counter(); TIMERS['gate'] += tg - ts
        W.compress(cutoff=cutoff, max_bond=max_bond)
        TIMERS['compress'] += time.perf_counter() - tg
        st = W.stats(); st.update(lo=lo, hi=hi, t=round(time.time() - t0, 2), step=round(time.perf_counter() - ts, 3))
        hist.append(st)
        if snapshots is not None:
            snapshots.append((lo, hi, W.to_quimb()))
        if log:
            print("  ", st, flush=True)
        if st['elems'] > max_elems or (tmax and time.time() - t0 > tmax):
            break
    Wq = _W(); Wq.tn = W.to_quimb(); Wq.fast = W
    return Wq, lo, hi, hist
