"""Own midpoint-MPO grower with leg-permutation tracking and greedy local unswapping (idea from Kremer-Dupuis /
Galda; independent implementation).

W is an MPO on n sites in mixed-canonical form; site tensor A[s] has shape (Dl, up, dn, Dr).
Logical output wire w lives on site pu[w] (upper legs), logical input wire w on site pd[w] (lower legs).
W represents  C_right . C_left  where C_left = units absorbed on the input side, C_right = units absorbed on the output side.
- absorb on the 'up' side (later units):   W <- G W   (G on the upper legs of the sites holding its wires)
- absorb on the 'dn' side (earlier units): W <- W G   (G on the lower legs)
Non-adjacent wires are brought together by SWAPs on the same legs; a SWAP on legs only relabels (pu/pd are updated),
so the represented operator is unchanged. Unswapping = trying SWAPs on up / dn / both legs of a bond and keeping
the choice with the smallest bond after truncation.
Final state: C|0> = all lower legs contracted with |0>; bit of logical wire w = site pu[w]."""
import numpy as np, scipy.linalg as sla, time

SWAP = np.eye(4)[[0, 2, 1, 3]].astype(complex)

class MPOU:
    def __init__(self, n, cutoff=1e-3, max_bond=512):
        self.n = n; self.cutoff = cutoff; self.max_bond = max_bond
        self.A = [np.eye(2, dtype=complex).reshape(1, 2, 2, 1) for _ in range(n)]
        self.c = 0                      # orthogonality centre
        self.pu = list(range(n)); self.pd = list(range(n))   # logical wire -> site
        self.su = list(range(n)); self.sd = list(range(n))   # site -> logical wire
        self.lognorm = 0.0
        self.trunc = 0.0                # accumulated discarded weight (relative)

    # ---------- canonical form ----------
    def _left_qr(self, s):
        T = self.A[s]; Dl, u, d, Dr = T.shape
        Q, R = np.linalg.qr(T.reshape(Dl * u * d, Dr))
        self.A[s] = Q.reshape(Dl, u, d, -1)
        self.A[s + 1] = np.tensordot(R, self.A[s + 1], axes=(1, 0))
    def _right_qr(self, s):
        T = self.A[s]; Dl, u, d, Dr = T.shape
        Q, R = np.linalg.qr(T.reshape(Dl, u * d * Dr).T)
        self.A[s] = Q.T.reshape(-1, u, d, Dr)
        self.A[s - 1] = np.tensordot(self.A[s - 1], R.T, axes=(3, 0))
    def move(self, s):
        while self.c < s: self._left_qr(self.c); self.c += 1
        while self.c > s: self._right_qr(self.c); self.c -= 1
        nrm = np.linalg.norm(self.A[self.c])
        self.A[self.c] /= nrm; self.lognorm += np.log(nrm)

    # ---------- two-site update ----------
    def _theta(self, s):
        return np.tensordot(self.A[s], self.A[s + 1], axes=(3, 0))   # (Dl,u1,d1,u2,d2,Dr)
    def _split(self, th, s, dry=False, max_bond=None):
        Dl, u1, d1, u2, d2, Dr = th.shape
        M = th.reshape(Dl * u1 * d1, u2 * d2 * Dr)
        if dry:
            S = sla.svdvals(M, check_finite=False)
            keep = int(np.sum(S > self.cutoff * S[0]))
            return max(1, min(keep, max_bond or self.max_bond))
        try:
            U, S, Vh = sla.svd(M, full_matrices=False, lapack_driver='gesdd', check_finite=False)
        except Exception:
            U, S, Vh = sla.svd(M, full_matrices=False, lapack_driver='gesvd', check_finite=False)
        keep = int(np.sum(S > self.cutoff * S[0]))
        keep = max(1, min(keep, max_bond or self.max_bond))
        if dry: return keep
        disc = np.sum(S[keep:] ** 2) / np.sum(S ** 2)
        self.trunc += disc
        S = S[:keep]; U = U[:, :keep]; Vh = Vh[:keep]
        # centre goes to the right site (s+1) by default
        self.A[s] = U.reshape(Dl, u1, d1, keep)
        self.A[s + 1] = (S[:, None] * Vh).reshape(keep, u2, d2, Dr)
        self.c = s + 1
        return keep
    @staticmethod
    def _apply(th, G, side):
        G4 = G.reshape(2, 2, 2, 2)
        if side == 'up':     # new_u = sum_m G[u, m] th[m]
            return np.einsum('abcd,lcxdyr->laxbyr', G4, th, optimize=True)
        else:                # new_d = sum_m th[m] G[m, d]
            return np.einsum('cdab,lxcydr->lxaybr', G4, th, optimize=True)
    def gate2(self, s, G, side):
        """G acts on (site s, site s+1) legs of `side`, G index order (site s, site s+1)."""
        self.move(s)
        th = self._apply(self._theta(s), G, side)
        return self._split(th, s)
    def swap_legs(self, s, side):
        k = self.gate2(s, SWAP, side)
        if side == 'up':
            a, b = self.su[s], self.su[s + 1]; self.su[s], self.su[s + 1] = b, a; self.pu[a], self.pu[b] = s + 1, s
        else:
            a, b = self.sd[s], self.sd[s + 1]; self.sd[s], self.sd[s + 1] = b, a; self.pd[a], self.pd[b] = s + 1, s
        return k
    def swap_both(self, s):
        self.move(s)
        th = self._apply(self._apply(self._theta(s), SWAP, 'up'), SWAP, 'dn')
        k = self._split(th, s)
        a, b = self.su[s], self.su[s + 1]; self.su[s], self.su[s + 1] = b, a; self.pu[a], self.pu[b] = s + 1, s
        a, b = self.sd[s], self.sd[s + 1]; self.sd[s], self.sd[s + 1] = b, a; self.pd[a], self.pd[b] = s + 1, s
        return k
    route_both = False
    def gate1(self, w, M, side):
        s = (self.pu if side == 'up' else self.pd)[w]
        if side == 'up': self.A[s] = np.einsum('ab,lbdr->ladr', M, self.A[s])
        else: self.A[s] = np.einsum('lubr,bd->ludr', self.A[s], M)

    # ---------- absorb a 2q unit on logical wires (a,b) ----------
    def absorb(self, a, b, G, side):
        """G acts on kron(a, b)."""
        p = self.pu if side == 'up' else self.pd
        while abs(p[a] - p[b]) > 1:            # move the two wires towards each other
            i, j = p[a], p[b]
            sw = self.swap_both if self.route_both else (lambda t: self.swap_legs(t, side))
            if i < j: sw(j - 1)
            else: sw(i - 1)
        i, j = p[a], p[b]
        if i > j: G = SWAP @ G @ SWAP; i = j
        return self.gate2(i, G, side)

    # ---------- greedy unswapping ----------
    def bonds(self):
        return [self.A[s].shape[3] for s in range(self.n - 1)]
    def unswap_sweep(self, sides=('up', 'dn', 'both'), hot=None):
        gained = 0
        order = list(range(self.n - 1)) + list(range(self.n - 2, -1, -1))
        if hot is not None:
            b = self.bonds(); thr = hot * max(b)
            hs = {s2 for s, d in enumerate(b) if d >= thr for s2 in range(s - 2, s + 3) if 0 <= s2 < self.n - 1}
            order = [s for s in order if s in hs]
        for s in order:
            if self.A[s].shape[3] <= 1: continue
            self.move(s)
            th = self._theta(s); cur = self.A[s].shape[3]
            best = (cur, None)
            for sd in sides:
                t2 = th
                if sd in ('up', 'both'): t2 = self._apply(t2, SWAP, 'up')
                if sd in ('dn', 'both'): t2 = self._apply(t2, SWAP, 'dn')
                k = self._split(t2, s, dry=True, max_bond=10 ** 9)
                if k < best[0]: best = (k, sd)
            if best[1] is not None:
                sd = best[1]
                if sd in ('up', 'both'): self.swap_legs(s, 'up')
                if sd in ('dn', 'both'): self.swap_legs(s, 'dn')
                gained += cur - best[0]
        return gained

    def stats(self):
        b = self.bonds(); return dict(max_bond=max(b), elems=int(sum(t.size for t in self.A)), trunc=round(self.trunc, 6))

    # ---------- final state ----------
    def state(self):
        """MPS (list of (Dl,2,Dr)) of W|0> on sites, plus site->logical map su."""
        z = np.array([1, 0], dtype=complex)
        return [np.einsum('ludr,d->lur', T, z) for T in self.A], list(self.su)

def mps_marginals(M):
    """exact single-site P(bit=0) for an MPS (list of (Dl,2,Dr))."""
    n = len(M)
    L = [np.ones((1, 1), dtype=complex)]
    for T in M:
        L.append(np.einsum('ab,aur,bus->rs', L[-1], T, T.conj(), optimize=True))
    R = [np.ones((1, 1), dtype=complex)]
    for T in reversed(M):
        R.append(np.einsum('rs,aur,bus->ab', R[-1], T, T.conj(), optimize=True))
    R = R[::-1]
    nrm = float(np.real(L[-1][0, 0]))
    p0 = []
    for s, T in enumerate(M):
        rho = np.einsum('ab,aur,bvs,rs->uv', L[s], T, T.conj(), R[s + 1], optimize=True)
        p0.append(float(np.real(rho[0, 0] / np.trace(rho))))
    return np.array(p0), nrm
def mps_amp(M, bits):
    v = np.ones((1,), dtype=complex)
    for T, b in zip(M, bits): v = v @ T[:, int(b), :]
    return v[0]
