"""Exact second-order perturbation theory in the inter-chain interaction g.

The circuit is U = prod_k [F_k V_k] with F_k free-fermion (Gaussian, number conserving,
chain-diagonal) gates and V_k = exp(i lam g_k n_a(r_k) n_b(r_k)) the on-site rung phases
(a = chain 0, b = chain 1).  In the interaction picture U = F_tot W, W = V~_K ... V~_1 with
V~_k = 1 + c_k P_k,  P_k = A_k B_k,  A_k = n_a(r_k, t_k), B_k = n_b(r_k, t_k)  (Heisenberg,
free evolution), c_k = exp(i lam g_k) - 1 (exact, since P_k^2 = P_k).

<O(T)> = <W^dag O~ W> expanded to O(lam^2):
  a1 = -2 sum_k g_k Im<O~ P_k>
  a2 = sum_k (-g_k^2/2) 2Re<O~ P_k>
       + sum_{k,j} g_k g_j <P_k O~ P_j>
       - 2 Re sum_{k>j} g_k g_j <O~ P_k P_j>
For O on chain 0 every correlator factorises into chain-0 x chain-1 free-fermion
correlators, each evaluated exactly with Wick's theorem (rank-1 vertex operators), so
a1, a2 are exact numbers (no sampling, no truncation).  O = staggered occupation (one-body).
"""
import numpy as np, os, json, sys
import gauss

def collect(circ, lo=0, hi=60, nsteps=20, path=None):
    """Return occupations D[l] (len L), list of vertices (step, r, g, phi0, phi1) in circuit
    order, and the free single-particle propagators w_T[l] at each step marker."""
    if path is None:
        path = os.environ.get('SU2_CIRCUITS', 'circuits') + f'/x_100_{circ}.qasm'
    occ, m, out = gauss.blocks(path)
    L = hi - lo
    D = [np.zeros(L), np.zeros(L)]
    for w in range(120):
        s, l = m[w]
        if lo <= s < hi: D[l][s - lo] = occ[w]
    u = [np.eye(L, dtype=complex), np.eye(L, dtype=complex)]
    verts = []; wT = {}; step = 0
    for b in out:
        if 'step' in b:
            step = b['step']; wT[step] = [u[0].copy(), u[1].copy()]
            if step == nsteps: break
            continue
        w = b['w']; U = b['U']; sl = [m[x] for x in w]
        if any(not (lo <= s < hi) for s, _ in sl): continue
        sl = [(s - lo, l) for s, l in sl]
        if len(w) == 1:
            s, l = sl[0]; u[l][s, :] *= U[1, 1] / U[0, 0]
        elif sl[0][1] == sl[1][1]:
            (s1, l1), (s2, l2) = sl
            V = U[1:3, 1:3] / U[0, 0]; Vsp = np.array([[V[1, 1], V[1, 0]], [V[0, 1], V[0, 0]]])
            idx = [s1, s2]; u[l1][idx, :] = Vsp @ u[l1][idx, :]
        else:
            (s1, l1), (s2, l2) = sl
            ph = np.angle(np.diag(U)); a0 = ph[2] - ph[0]; a1 = ph[1] - ph[0]
            g = ph[3] - ph[2] - ph[1] + ph[0]; g = (g + np.pi) % (2 * np.pi) - np.pi
            u[l1][s1, :] *= np.exp(1j * a0); u[l2][s2, :] *= np.exp(1j * a1)
            assert s1 == s2
            # vertex after the free phases (they commute with the vertex anyway)
            phis = {l1: u[l1][s1, :].copy(), l2: u[l2][s2, :].copy()}
            verts.append((step + 1, s1, g, phis[0], phis[1]))
    return D, verts, wT

class Chain:
    """Wick machinery for one chain: occupations D, vertex rows Phi (K x L), observable M.
    Convention: rho(t)[r,r'] = <c^dag_r'(t) c_r(t)>, c_r(t)=sum_s w[r,s] c_s; vertex operator
    n(k) = c^dag M_k c with M_k[s,s'] = conj(phi_k[s]) phi_k[s']."""
    def __init__(self, D, Phi, M):
        self.D = D; self.Db = 1 - D; self.Phi = Phi; self.M = M
        P = Phi; Pc = Phi.conj()
        self.Gp = (P * D) @ Pc.T          # [k,j] = phi_k^T D conj(phi_j)
        self.Gm = (P * self.Db) @ Pc.T
        self.n = self.Gp.diagonal().real.copy()   # <A_k>
        dg = np.diag(D); dbg = np.diag(self.Db)
        def sand(L1, X, L2):  # [k,j] = phi_k^T L1 X L2 conj(phi_j)
            return (P @ (L1 @ X @ L2)) @ Pc.T
        self.mDbD = sand(dbg, M, dg); self.mDDb = sand(dg, M, dbg)
        self.mDbDb = sand(dbg, M, dbg); self.mDD = sand(dg, M, dg)
        self.Om = float(np.real(np.trace(M * 1.0 @ dg)))  # <O>
    # connected 2-point traces Tr(D X Db Y)
    def c_kj(self):      # X=A_k, Y=A_j   -> Gm[k,j] Gp[j,k]
        return self.Gm * self.Gp.T
    def c_kM(self):      # X=A_k, Y=M     -> phi_k^T Db M D conj(phi_k)
        return self.mDbD.diagonal()
    def c_Mk(self):      # X=M, Y=A_k     -> phi_k^T D M Db conj(phi_k)
        return self.mDDb.diagonal()
    def two(self):       # <A_k A_j>
        return np.outer(self.n, self.n) + self.c_kj()
    def three(self, order):
        """<X Y Z> as K x K matrix [k,j] for order in {'kMj','Mkj','jkM'}.
        <XYZ> = <X><Y><Z> + <X><YZ>c + <Y><XZ>c + <Z><XY>c + Tr(D X Db Y Db Z) - Tr(D X Db Z D Y)."""
        n = self.n; O = self.Om; Gp = self.Gp; Gm = self.Gm
        ckj = Gm * Gp.T           # Tr(D A_k Db A_j)
        cjk = ckj.T               # Tr(D A_j Db A_k)
        ckM = self.c_kM(); cMk = self.c_Mk()
        if order == 'kMj':    # X=A_k, Y=M, Z=A_j
            disc = (O * np.outer(n, n) + n[:, None] * cMk[None, :] + O * ckj + n[None, :] * ckM[:, None])
            # Tr(D A_k Db M Db A_j) = (phi_k^T Db M Db conj phi_j)(phi_j^T D conj phi_k)
            # Tr(D A_k Db A_j D M)  = (phi_k^T Db conj phi_j)(phi_j^T D M D conj phi_k)
            conn = self.mDbDb * Gp.T - Gm * self.mDD.T
        elif order == 'Mkj':  # X=M, Y=A_k, Z=A_j
            disc = (O * np.outer(n, n) + O * ckj + n[:, None] * cMk[None, :] + n[None, :] * cMk[:, None])
            # Tr(D M Db A_k Db A_j) = (phi_k^T Db conj phi_j)(phi_j^T D M Db conj phi_k)
            # Tr(D M Db A_j D A_k)  = (phi_j^T D conj phi_k)(phi_k^T D M Db conj phi_j)
            conn = Gm * self.mDDb.T - Gp.T * self.mDDb
        elif order == 'jkM':  # X=A_j, Y=A_k, Z=M   ([k,j] indexing)
            disc = (O * np.outer(n, n) + n[None, :] * ckM[:, None] + n[:, None] * ckM[None, :] + O * cjk)
            # Tr(D A_j Db A_k Db M) = (phi_j^T Db conj phi_k)(phi_k^T Db M D conj phi_j)
            # Tr(D A_j Db M D A_k)  = (phi_j^T Db M D conj phi_k)(phi_k^T D conj phi_j)
            conn = Gm.T * self.mDbD - self.mDbD.T * Gp
        return disc + conn

def pt_coeffs(D, verts, wT, eps, T):
    """a0, a1, a2 for O = sum_r eps[r] (n_a(r)+n_b(r)) at step T (vertices with step<=T)."""
    V = [v for v in verts if v[0] <= T]
    K = len(V); g = np.array([v[2] for v in V])
    Phi = [np.array([v[3] for v in V]), np.array([v[4] for v in V])]
    a0 = a1 = a2 = 0.0
    E = np.diag(eps).astype(complex)
    for lo_ in (0, 1):   # chain carrying O
        lb = 1 - lo_
        w = wT[T][lo_]; M = w.conj().T @ E @ w
        CA = Chain(D[lo_], Phi[lo_], M)
        CB = Chain(D[lb], Phi[lb], np.zeros_like(M))
        a0 += CA.Om
        nb = CB.n; BB = CB.two()                      # <B_k B_j>
        OA = CA.Om * CA.n + CA.c_Mk()                 # <O A_k>
        OP = OA * nb                                   # <O P_k>
        a1 += -2 * np.sum(g * OP.imag)
        T1 = np.sum(-g**2 / 2 * 2 * OP.real)
        T2 = np.einsum('k,j,kj,kj->', g, g, CA.three('kMj'), BB)
        OPP = CA.three('Mkj') * BB                     # <O P_k P_j>
        low = np.tril(np.ones((K, K), bool), -1)       # k>j
        T3 = -2 * np.real(np.sum((np.outer(g, g) * OPP)[low]))
        a2 += T1 + T2.real + T3
    return a0, a1, a2

if __name__ == '__main__':
    eps = np.array([(-1) ** r for r in range(60)], float)
    res = {}
    for circ in ('SCV', 'meson'):
        D, verts, wT = collect(circ)
        res[circ] = [pt_coeffs(D, verts, wT, eps, T) for T in range(1, 21)]
        Qc = pt_coeffs(D, verts, wT, np.ones(60), 20)
        print(circ, 'Q check (a0,a1,a2) at step 20:', Qc, flush=True)
    for T in range(20):
        s = res['SCV'][T]; mm = res['meson'][T]
        print(f"step {T+1:2d} SCV a0={s[0]:.6f} a1={s[1]:+.6f} a2={s[2]:+.6f} | n_f a0={mm[0]-s[0]:.6f} a1={mm[1]-s[1]:+.6f} a2={mm[2]-s[2]:+.6f}")
    json.dump(res, open('pt2.json', 'w'))
