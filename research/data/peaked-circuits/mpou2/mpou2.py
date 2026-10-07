"""mpou2: own midpoint MPO grower with leg relabelling, QR-reduced two-site splits, hot-bond greedy unswap and
matching unswap.  (Midpoint MPO + unswapping idea: Kremer-Dupuis arXiv 2604.21908; QR-reduced core SVD idea:
our /tmp/mlx-port/tno_fast.py; everything here is an independent implementation.)

W is an MPO on n sites, site tensor A[s] of shape (Dl, up, dn, Dr), mixed-canonical around centre c.
Site s carries the OUTPUT leg of logical wire su[s] and the INPUT leg of logical wire sd[s]; pu / pd are the inverses.
W represents  C_right . C_left  (C_left absorbed on the input ('dn') side, C_right on the output ('up') side).
Exact relabelling moves (represented operator unchanged, only the tensor network changes):
  'both' swap at bond s : exchange sites s, s+1 completely (layout change)
  'up'   swap at bond s : exchange the output legs of s, s+1 (pairing change)
  'dn'   swap at bond s : exchange the input legs
Every two-site update is done on a QR-reduced core: A[s] = Qa Ra over the legs it keeps, A[s+1] = Rb Qb, SVD of
Ra.G.Rb only (same singular values as the full two-site matrix)."""
import numpy as np, scipy.linalg as sla

SWAP = np.eye(4)[[0, 2, 1, 3]].astype(complex)
_PERM_KEEP = {  # site tensor axes (Dl,u,d,Dr); left site keeps / acts, right site acts / keeps
    'up': ((0, 2), (1, 3), (0, 1), (2, 3)),     # left keep (Dl,d1) act (u1,Dm); right act (Dm,u2) keep (d2,Dr)
    'dn': ((0, 1), (2, 3), (0, 2), (1, 3)),
    'both': ((0,), (1, 2, 3), (0, 1, 2), (3,)),
}


def _qr_left(M):
    """M (dk, da) -> Q (dk, k), R (k, da) with k = min(dk, da); no-op Q=None if dk <= da."""
    dk, da = M.shape
    if dk <= da:
        return None, M
    Q, R = np.linalg.qr(M)
    return Q, R


def _svd(M):
    try:
        return sla.svd(M, full_matrices=False, lapack_driver='gesdd', check_finite=False)
    except Exception:
        return sla.svd(M, full_matrices=False, lapack_driver='gesvd', check_finite=False)


def _svals(M):
    try:
        return sla.svdvals(M, check_finite=False)
    except Exception:
        return np.linalg.svd(M + 1e-15 * np.random.randn(*M.shape), compute_uv=False)


class MPO2:
    def __init__(self, n, eps=1e-6, mode='sum2', max_bond=1024):
        self.n = n; self.eps = eps; self.mode = mode; self.max_bond = max_bond
        self.A = [np.eye(2, dtype=complex).reshape(1, 2, 2, 1) for _ in range(n)]
        self.c = 0
        self.su = list(range(n)); self.sd = list(range(n))
        self.pu = list(range(n)); self.pd = list(range(n))
        self.lognorm = 0.0
        self.trunc = 0.0        # accumulated discarded weight (relative, sum over splits)
        self.nsplit = 0
        self.nroute = 0

    # ------------------------------------------------------------------ state copy (arrays are never modified in place)
    def snapshot(self):
        return (list(self.A), self.c, list(self.su), list(self.sd), list(self.pu), list(self.pd), self.lognorm, self.trunc)

    def restore(self, snap):
        A, self.c, su, sd, pu, pd, self.lognorm, self.trunc = snap
        self.A = list(A); self.su = list(su); self.sd = list(sd); self.pu = list(pu); self.pd = list(pd)

    # ------------------------------------------------------------------ canonical form
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
        while self.c < s:
            self._left_qr(self.c); self.c += 1
        while self.c > s:
            self._right_qr(self.c); self.c -= 1
        nrm = np.linalg.norm(self.A[self.c])
        self.A[self.c] = self.A[self.c] / nrm; self.lognorm += np.log(nrm)

    # ------------------------------------------------------------------ truncation rule
    def _keep(self, S, max_bond=None):
        mb = max_bond or self.max_bond
        if S[0] == 0:
            return 1
        if self.mode == 'rel':
            k = int(np.sum(S > self.eps * S[0]))
        else:   # 'sum2': discarded weight fraction <= eps
            w = S ** 2; tail = np.cumsum(w[::-1])[::-1]       # tail[k] = sum_{i>=k} w_i
            tot = tail[0]
            k = int(np.sum(tail > self.eps * tot))             # smallest k with tail[k] <= eps*tot
        return max(1, min(k, mb))

    # ------------------------------------------------------------------ reduced core of bond s
    def _core(self, s, kind):
        """returns (core (ka, *actL_without_bond, *actR_without_bond, kb), QA info, QB info)."""
        lk, la, ra, rk = _PERM_KEEP[kind]
        A = self.A[s]; B = self.A[s + 1]
        sa = A.shape; sb = B.shape
        MA = A.transpose(lk + la).reshape(int(np.prod([sa[i] for i in lk])), -1)
        QA, RA = _qr_left(MA)
        MB = B.transpose(rk + ra).reshape(int(np.prod([sb[i] for i in rk])), -1)
        QB, RB = _qr_left(MB)
        ka = RA.shape[0]; kb = RB.shape[0]
        dm = sa[3]
        # RA: (ka, act legs of A ... , Dm) ; RB: (kb, Dm, act legs of B ...)
        RA = RA.reshape((ka,) + tuple(sa[i] for i in la))
        RB = RB.reshape((kb,) + tuple(sb[i] for i in ra))
        # contract over Dm (last of RA, first act of RB)
        core = np.tensordot(RA, RB, axes=(RA.ndim - 1, 1))     # (ka, actA.., kb, actB..)
        nA = RA.ndim - 2
        # move kb to the end
        core = np.moveaxis(core, 1 + nA, -1)                    # (ka, actA.., actB.., kb)
        return core, (QA, lk, la, sa), (QB, rk, ra, sb)

    @staticmethod
    def _apply_core(core, kind, G=None, side=None):
        """core (ka, xA.., xB.., kb). kind up/dn: (ka,p1,p2,kb); both: (ka,u1,d1,u2,d2,kb)."""
        if kind == 'both':
            # exchange the two sites: new left site = old right legs
            return core.transpose(0, 3, 4, 1, 2, 5)
        if G is None:
            return core
        G4 = G.reshape(2, 2, 2, 2)
        if side == 'up':   # new_out = G . old_out
            return np.einsum('xyuv,auvb->axyb', G4, core, optimize=True)
        else:              # new_in = old_in . G
            return np.einsum('auvb,uvxy->axyb', core, G4, optimize=True)

    def _rebuild(self, s, kind, U, SVh, infoA, infoB, adL, adR):
        QA, lk, la, sa = infoA; QB, rk, ra, sb = infoB
        k = SVh.shape[0]
        # left site
        if QA is not None:
            NA = QA @ U.reshape(QA.shape[1], -1)                # (dkeep, act'*k)
        else:
            NA = U.reshape(int(np.prod([sa[i] for i in lk])), -1)
        NA = NA.reshape(tuple(sa[i] for i in lk) + tuple(adL) + (k,))
        # current axis order: lk + act + (k); target (Dl,u,d,Dr)
        order = list(lk) + list(la)          # positions in original order; la's last is Dr(=3)
        inv = np.argsort(order)
        self.A[s] = NA.transpose(inv)
        # right site: SVh (k, actB.., kb)
        kb = SVh.shape[-1]
        M = SVh.reshape(-1, kb)
        if QB is not None:
            NB = M @ QB.T                                      # (k*act, dkeep)
        else:
            NB = M
        NB = NB.reshape((k,) + tuple(adR) + tuple(sb[i] for i in rk))
        order = list(ra) + list(rk)
        inv = np.argsort(order)
        self.A[s + 1] = NB.transpose(inv)

    def two(self, s, kind, G=None, side=None, max_bond=None):
        """apply G (kind up/dn) or a full site exchange (kind both) at bond s, split with truncation."""
        if self.c != s and self.c != s + 1:
            self.move(s)
        elif self.c == s + 1:
            pass  # centre inside the pair: still canonical outside
        core, iA, iB = self._core(s, kind)
        core = self._apply_core(core, kind, G, side)
        ka, kb = core.shape[0], core.shape[-1]
        na = 2 if kind == 'both' else 1
        adL = core.shape[1:1 + na]; adR = core.shape[1 + na:-1]
        M = core.reshape(ka * int(np.prod(adL)), -1)
        U, S, Vh = _svd(M)
        k = self._keep(S, max_bond)
        w = S ** 2; tot = w.sum()
        disc = w[k:].sum() / tot if tot > 0 else 0.0
        self.trunc += disc; self.nsplit += 1
        U = U[:, :k]; SVh = (S[:k, None] * Vh[:k])
        nrm = np.linalg.norm(S[:k])
        SVh = SVh / nrm; self.lognorm += np.log(nrm)
        SVh = SVh.reshape((k,) + tuple(adR) + (kb,))
        self._rebuild(s, kind, U.reshape(ka, -1), SVh, iA, iB, adL, adR)
        self.c = s + 1
        return k

    def probe(self, s, kind, G=None, side=None):
        """dry run: (rank at cutoff, purity) of the bond s after the move, without changing W."""
        if self.c != s and self.c != s + 1:
            self.move(s)
        core, _, _ = self._core(s, kind)
        core = self._apply_core(core, kind, G, side)
        ka = core.shape[0]
        na = 2 if kind == 'both' else 1
        S = _svals(core.reshape(ka * int(np.prod(core.shape[1:1 + na])), -1))
        p = S ** 2; p = p / p.sum()
        return self._keep(S, 10 ** 9), float(np.sum(p ** 2))

    # ------------------------------------------------------------------ relabelling moves
    def _relabel(self, s, kind):
        if kind in ('up', 'both'):
            a, b = self.su[s], self.su[s + 1]; self.su[s], self.su[s + 1] = b, a; self.pu[a], self.pu[b] = s + 1, s
        if kind in ('dn', 'both'):
            a, b = self.sd[s], self.sd[s + 1]; self.sd[s], self.sd[s + 1] = b, a; self.pd[a], self.pd[b] = s + 1, s

    def swap(self, s, kind):
        if kind == 'both':
            k = self.two(s, 'both')
        else:
            k = self.two(s, kind, SWAP, kind)
        self._relabel(s, kind)
        return k

    def bonds(self):
        return [self.A[s].shape[3] for s in range(self.n - 1)]

    def elems(self):
        return int(sum(t.size for t in self.A))

    def stats(self):
        b = self.bonds()
        return dict(max_bond=max(b), elems=self.elems(), trunc=float(f'{self.trunc:.3g}'))

    # ------------------------------------------------------------------ gate absorption with routing (site exchanges)
    route_kind = 'both'

    def _route(self, i, j, future=None, lam=1.0, side=None):
        """bring sites i<j adjacent with 'both' exchanges.  Site i moves right to k, site j moves left to k+1;
        k minimises a carried-entanglement proxy (moving i right carries its left bond across every passed site)
        plus lam * (lookahead routing distance of the upcoming gates `future` = [(side, a, b, weight)])."""
        b = [1] + self.bonds() + [1]          # b[t] = bond left of site t, b[t+1] = bond right of site t
        ci, cj = b[i], b[j + 1]
        bbar = max(1.0, float(np.mean(b[i + 1:j + 1])))
        fut = []
        if future:
            for side, x, y, w in future:
                p = self.pu if side == 'up' else self.pd
                fut.append((p[x], p[y], w))
        best = None
        for k in range(i, j):
            cost = (sum(ci * b[t + 1] for t in range(i + 1, k + 1)) + sum(cj * b[t] for t in range(k + 1, j))) / bbar
            if fut:
                def newpos(s):
                    if s == i: return k
                    if s == j: return k + 1
                    if i < s <= k: return s - 1
                    if k + 1 <= s < j: return s + 1
                    return s
                cost += lam * sum(w * max(0, abs(newpos(x) - newpos(y)) - 1) for x, y, w in fut)
            if best is None or cost < best[0] - 1e-12:
                best = (cost, k)
        k = best[1]
        self.nroute += (k - i) + (j - 1 - k)
        kind = 'both' if self.route_kind == 'both' else side
        for t in range(i, k):
            self.swap(t, kind)
        for t in range(j - 1, k, -1):
            self.swap(t, kind)

    def route_cost(self, i, j):
        if i > j:
            i, j = j, i
        if j - i <= 1:
            return 0
        b = [1] + self.bonds() + [1]
        ci, cj = b[i], b[j + 1]
        return min(sum(ci * b[t + 1] for t in range(i + 1, k + 1)) + sum(cj * b[t] for t in range(k + 1, j))
                   for k in range(i, j)) + (j - i)

    def absorb(self, a, b, G, side, future=None, lam=1.0):
        """G (4x4) acts on kron(wire a, wire b); side 'up' = later gate (W <- G W), 'dn' = earlier (W <- W G)."""
        p = self.pu if side == 'up' else self.pd
        i, j = p[a], p[b]
        if abs(i - j) > 1:
            self._route(min(i, j), max(i, j), future, lam, side)
            i, j = p[a], p[b]
        if i > j:
            G = SWAP @ G @ SWAP; i = j
        return self.two(i, side, G, side)

    def gate1(self, w, M, side):
        s = (self.pu if side == 'up' else self.pd)[w]
        if side == 'up':
            self.A[s] = np.einsum('ab,lbdr->ladr', M, self.A[s])
        else:
            self.A[s] = np.einsum('lubr,bd->ludr', self.A[s], M)

    # ------------------------------------------------------------------ greedy unswap
    def unswap_bond(self, s, kinds=('up', 'dn', 'both'), tie=True):
        """probe the relabelling moves at bond s; commit the best if it lowers (rank, -purity). returns gain."""
        if self.A[s].shape[3] <= 1:
            return 0
        k0, p0 = self.probe(s, 'up', None, 'up')        # identity option (G=None)
        best = (k0, -p0, None)
        for kd in kinds:
            if kd == 'both':
                r = self.probe(s, 'both')
            else:
                r = self.probe(s, kd, SWAP, kd)
            key = (r[0], -r[1], kd)
            if key[0] < best[0] or (tie and key[0] == best[0] and key[1] < best[1] - 1e-6):
                best = key
        if best[2] is None:
            return 0
        self.swap(s, best[2])
        return max(0, k0 - best[0]) + 1e-3

    kinds = ('up', 'dn', 'both')

    def unswap_sweep(self, thr=1, kinds=None, tie=True, window=2):
        """probe bonds with bond >= thr (plus `window` neighbours); left->right then right->left."""
        b = self.bonds()
        hot = sorted({t for s, d in enumerate(b) if d >= thr for t in range(s - window, s + window + 1) if 0 <= t < self.n - 1})
        if not hot:
            return 0
        g = 0
        kinds = kinds or self.kinds
        for s in hot + hot[::-1][1:]:
            g += self.unswap_bond(s, kinds, tie)
        return g

    # ------------------------------------------------------------------ final state
    def state(self):
        if self.A[0].shape[2] == 1:
            return [T[:, :, 0, :] for T in self.A]
        z = np.array([1, 0], dtype=complex)
        return [np.einsum('ludr,d->lur', T, z) for T in self.A]

    @staticmethod
    def zero_state(n, eps, mode, max_bond):
        S = MPO2(n, eps=eps, mode=mode, max_bond=max_bond)
        S.A = [np.array([1, 0], dtype=complex).reshape(1, 2, 1, 1) for _ in range(n)]
        S.kinds = ('both',)
        return S

    def apply_to_state(self, phi):
        """W <- W|phi> (phi: MPO2 in state form, dn legs of dimension 1).  phi's sites are first permuted so that
        its wire order equals W's input-leg order, then the input legs are contracted and the result compressed."""
        for pos in range(self.n):                      # bubble phi's sites into W's input order
            want = self.sd[pos]; cur = phi.pu[want]
            while cur > pos:
                phi.swap(cur - 1, 'both'); cur -= 1
        phi.move(0)
        new = []
        for s in range(self.n):
            Wt = self.A[s]; Pt = phi.A[s]
            T = np.einsum('ludr,mdeq->lmueqr', Wt, Pt[:, :, :, :])   # e = 1
            Dl, El = Wt.shape[0], Pt.shape[0]; Dr, Er = Wt.shape[3], Pt.shape[3]
            T = T.transpose(0, 1, 2, 3, 5, 4).reshape(Dl * El, Wt.shape[1], 1, Dr * Er)
            new.append(T)
        self.A = new
        self.lognorm += phi.lognorm
        self.trunc += phi.trunc
        self.c = 0
        self.move(self.n - 1); self.move(0)
        for s in range(self.n - 1):
            self.two(s, 'up', None, 'up')
        self.sd = list(self.su); self.pd = list(self.pu)
        self.kinds = ('both',)

    def to_state(self):
        """contract all input legs with |0> (keeps a dn leg of dimension 1): W -> W|0> as an MPS."""
        self.A = [T[:, :, 0:1, :] for T in self.A]
        self.move(self.n - 1); self.move(0)
        for s in range(self.n - 1):
            self.two(s, 'up', None, 'up')
        self.kinds = ('both',)


def mps_marginals(M):
    """exact single-site P(bit=0) for an MPS (list of (Dl,2,Dr)); returns (p0 per site, norm^2)."""
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
    for T, b in zip(M, bits):
        v = v @ T[:, int(b), :]
    return v[0]
