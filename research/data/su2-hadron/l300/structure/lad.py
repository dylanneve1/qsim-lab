"""Ladder (rung-site, d=4) gate stream + forward MPS TEBD + Heisenberg MPO TEBD.

gate_steps(circ, lam): decodes the QASM (via gauss.blocks: SWAP-unwound, fused, Gaussian-checked)
into per-Trotter-step lists of rung-site gates ('1', s, U4) / ('2', r, U16 on rungs r,r+1),
fused along the gate DAG.  The rung (inter-chain) phase g is scaled by lam (lam=1: the circuit).

Rung basis index = 2*n_i + n_o.  Two-rung index = 4*s_r + s_{r+1}.
"""
import numpy as np, os, time
import scipy.linalg as sla
import gauss

D = os.environ.get('SU2_CIRCUITS', '/tmp/su2-254/circuits') + '/'
N = 60
I4 = np.eye(4, dtype=complex)

def gate_steps(circ, lam=1.0):
    occ, m, blist = gauss.blocks(D + f'x_100_{circ}.qasm')
    st = [[0, 0] for _ in range(N)]
    for w in range(120):
        s, l = m[w]; st[s][l] = occ[w]
    init = [2 * st[s][0] + st[s][1] for s in range(N)]
    steps = []; cur = []; last = {}; gmax = 0.0
    def add(sites, U):
        # DAG fusion
        if len(sites) == 1:
            s = sites[0]; j = last.get(s)
            if j is not None:
                ss, V = cur[j]
                if len(ss) == 1: cur[j] = (ss, U @ V)
                else:
                    Ue = np.kron(U, I4) if s == ss[0] else np.kron(I4, U)
                    cur[j] = (ss, Ue @ V)
                return
            cur.append(((s,), U)); last[s] = len(cur) - 1; return
        r = sites[0]
        jr, jr1 = last.get(r), last.get(r + 1)
        if jr is not None and jr == jr1 and cur[jr][0] == (r, r + 1):
            cur[jr] = (cur[jr][0], U @ cur[jr][1]); return
        V = np.eye(16, dtype=complex)
        for s, j in ((r, jr), (r + 1, jr1)):
            if j is not None and len(cur[j][0]) == 1 and cur[j][1] is not None:
                u = cur[j][1]; V = (np.kron(u, I4) if s == r else np.kron(I4, u)) @ V
                cur[j] = (cur[j][0], None)  # dead
        cur.append(((r, r + 1), U @ V)); last[r] = last[r + 1] = len(cur) - 1
    for b in blist:
        if 'step' in b:
            steps.append([(('2' if len(ss) == 2 else '1'), ss[0], U) for ss, U in cur if U is not None])
            cur = []; last = {}
            continue
        w = b['w']; U = b['U'].copy(); sl = [m[x] for x in w]
        if len(w) == 1:
            s, l = sl[0]; add((s,), gauss.embed(U, [l], 2))
        elif sl[0][1] != sl[1][1]:
            (s1, l1), (s2, l2) = sl
            ph = np.angle(np.diag(U)); g = ph[3] - ph[2] - ph[1] + ph[0]; g = (g + np.pi) % (2 * np.pi) - np.pi
            gmax = max(gmax, abs(g))
            U[3, 3] *= np.exp(1j * (lam - 1) * g)
            add((s1,), gauss.embed(U, [l1, l2], 2))
        else:
            (s1, l1), (s2, l2) = sl; r0 = min(s1, s2)
            add((r0, r0 + 1), gauss.embed(U, [2 * (s1 - r0) + l1, 2 * (s2 - r0) + l2], 4))
    return init, steps

def svd(th):
    try: return np.linalg.svd(th, full_matrices=False)
    except np.linalg.LinAlgError: return sla.svd(th, full_matrices=False, lapack_driver='gesvd')

Zi = np.array([1, 1, -1, -1.]); Zo = np.array([1, -1, 1, -1.])

class MPS:
    """Forward state TEBD."""
    def __init__(self, init, chi, cut=1e-14):
        self.A = []
        for s in range(N):
            t = np.zeros((1, 4, 1), complex); t[0, init[s], 0] = 1; self.A.append(t)
        self.c = 0; self.chi = chi; self.cut = cut
        self.disc = 0.0; self.logF = 0.0; self.maxchi = 1; self.maxdw = 0.0
    def move(self, to):
        A = self.A
        while self.c < to:
            c = self.c; Dl, d, Dr = A[c].shape; Q, R = np.linalg.qr(A[c].reshape(Dl * d, Dr))
            A[c] = Q.reshape(Dl, d, -1); A[c + 1] = np.tensordot(R, A[c + 1], 1); self.c += 1
        while self.c > to:
            c = self.c; Dl, d, Dr = A[c].shape; Q, R = np.linalg.qr(A[c].reshape(Dl, d * Dr).T)
            A[c] = Q.T.reshape(-1, d, Dr); A[c - 1] = np.tensordot(A[c - 1], R.T, 1); self.c -= 1
    def apply(self, g):
        A = self.A
        if g[0] == '1':
            _, s, U = g; A[s] = np.einsum('jk,akb->ajb', U, A[s]); return
        _, r, U = g
        if self.c < r: self.move(r)
        elif self.c > r + 1: self.move(r + 1)
        th = np.einsum('ajb,bkc->ajkc', A[r], A[r + 1]); Dl, _, _, Dr = th.shape
        th = np.einsum('xy,ayc->axc', U, th.reshape(Dl, 16, Dr)).reshape(Dl * 4, 4 * Dr)
        u, sv, vh = svd(th); w = sv ** 2; tot = w.sum()
        keep = min(self.chi, max(1, int(np.sum(w / tot > self.cut))))
        dw = w[keep:].sum() / tot; self.disc += dw; self.logF += np.log1p(-dw); self.maxdw = max(self.maxdw, dw)
        sv = sv[:keep] / np.sqrt(w[:keep].sum())
        A[r] = u[:, :keep].reshape(Dl, 4, keep); A[r + 1] = (sv[:, None] * vh[:keep]).reshape(keep, 4, Dr); self.c = r + 1
        self.maxchi = max(self.maxchi, keep)
    def occupations(self):
        self.move(0); out = []
        for s in range(N):
            self.move(s); p = np.einsum('ajb,ajb->j', self.A[s], self.A[s].conj()).real; p = p / p.sum()
            out.append(((1 - p @ Zi) / 2, (1 - p @ Zo) / 2))
        return np.array(out)
    def entropies(self):
        self.move(0); S = []
        for s in range(N - 1):
            self.move(s); Dl, d, Dr = self.A[s].shape
            sv = np.linalg.svd(self.A[s].reshape(Dl * d, Dr), compute_uv=False); p = sv ** 2; p = p[p > 1e-16]; p /= p.sum()
            S.append(float(-(p * np.log(p)).sum()))
        return S

class MPO:
    """Heisenberg-picture operator, vectorised as an MPS with site tensors W[a, p(out), q(in), b]."""
    def __init__(self, site_ops, chi, cut=1e-12):
        # site_ops: dict s -> 4x4 operator; others identity
        self.W = [(site_ops.get(s, I4)).astype(complex).reshape(1, 4, 4, 1) for s in range(N)]
        self.c = min(site_ops); self.chi = chi; self.cut = cut
        self.lo = min(site_ops); self.hi = max(site_ops)
        self.disc = 0.0; self.maxchi = 1; self.norm2 = self.vnorm2()
    def vnorm2(self):
        # log-free: product of identity norms handled by canonical center
        E = np.ones((1, 1))
        for W in self.W:
            E = np.einsum('ab,apqc,bpqd->cd', E, W, W.conj())
        return float(E.real.squeeze())
    def move(self, to):
        W = self.W
        while self.c < to:
            c = self.c; Dl, _, _, Dr = W[c].shape; Q, R = np.linalg.qr(W[c].reshape(Dl * 16, Dr))
            W[c] = Q.reshape(Dl, 4, 4, -1); W[c + 1] = np.tensordot(R, W[c + 1], 1); self.c += 1
        while self.c > to:
            c = self.c; Dl, _, _, Dr = W[c].shape; Q, R = np.linalg.qr(W[c].reshape(Dl, 16 * Dr).T)
            W[c] = Q.T.reshape(-1, 4, 4, Dr); W[c - 1] = np.tensordot(W[c - 1], R.T, 1); self.c -= 1
    def conj_gate(self, g):
        """O <- G^dag O G"""
        W = self.W
        if g[0] == '1':
            _, s, U = g
            if s < self.lo or s > self.hi: return
            W[s] = np.einsum('xp,apqb,qy->axyb', U.conj().T, W[s], U); return
        _, r, U = g
        if r + 1 < self.lo or r > self.hi: return
        self.lo = min(self.lo, r); self.hi = max(self.hi, r + 1)
        if self.c < r: self.move(r)
        elif self.c > r + 1: self.move(r + 1)
        th = np.einsum('apqb,bxyc->apxqyc', W[r], W[r + 1]); Dl = th.shape[0]; Dr = th.shape[-1]
        th = th.reshape(Dl, 16, 16, Dr)
        th = np.einsum('xp,apqc,qy->axyc', U.conj().T, th, U)        # (Dl, out16, in16, Dr)
        th = th.reshape(Dl, 4, 4, 4, 4, Dr).transpose(0, 1, 3, 2, 4, 5).reshape(Dl * 16, 16 * Dr)
        u, sv, vh = svd(th); w = sv ** 2; tot = w.sum()
        keep = min(self.chi, max(1, int(np.sum(w / tot > self.cut))))
        dw = w[keep:].sum(); self.disc += dw
        sv = sv[:keep]
        W[r] = u[:, :keep].reshape(Dl, 4, 4, keep); W[r + 1] = (sv[:, None] * vh[:keep]).reshape(keep, 4, 4, Dr); self.c = r + 1
        self.maxchi = max(self.maxchi, keep)
    def expect(self, init):
        v = np.ones(1, complex)
        for s, W in enumerate(self.W):
            v = v @ W[:, init[s], init[s], :]
        return v.squeeze()
    def bonds(self):
        return [W.shape[3] for W in self.W[:-1]]

def heisenberg(steps, k, site_ops, chi, cut=1e-12):
    """Evolve O back from step k to 0 through steps[0..k-1]."""
    O = MPO(site_ops, chi, cut)
    for st in reversed(steps[:k]):
        for g in reversed(st):
            O.conj_gate(g)
    return O
