"""Reusable exact second-order perturbation theory for two free-fermion blocks coupled by
density-density phases.  Circuit-independent: feed it a gate list.

Model
-----
Two blocks of fermionic modes, A (nA modes) and B (nB modes), with NO hopping between them.
Initial state: a product of occupation-number (Slater) states, occA / occB in {0,1}.
Gate list `ops` (applied in order):
  ('u',   blk, idx, V)   number-conserving Gaussian gate on modes idx (list) of block blk in
                         {'A','B'}; V is the len(idx) x len(idx) single-particle unitary,
                         acting as c_idx -> V c_idx in the Heisenberg row convention used
                         by gauss.py (rho -> V rho V^dag).  A one-mode phase is V=[[e^{ia}]].
  ('int', iA, iB, g)     interaction vertex exp(i lam g n_A[iA] n_B[iB]).
  ('mark', label)        measurement point: observables are evaluated here (vertices after a
                         mark don't affect it).
Observables: dict name -> (MA, MB); O = sum_r eps... i.e. any one-body operator
  O = sum_{st} MA[s,t] c^dag_{A,s}(T) c_{A,t}(T) + (same for B), with M given in the
  final-time (physical) mode basis, e.g. MA = diag(eps) for a weighted density sum.

Output: for every mark and observable, (a0, a1, a2) with
  <O>(lam) = a0 + a1 lam + a2 lam^2 + O(lam^3)    (exact coefficients, no truncation).

Method: interaction picture, U = F_tot prod_k (1 + c_k P_k), P_k = n_A(k) n_B(k) (Heisenberg,
free), c_k = e^{i lam g_k} - 1 (exact because P_k^2 = P_k).  Every correlator factorises into
block-A x block-B free-fermion correlators of rank-1 bilinears and one-body O, evaluated with
Wick's theorem as K x K matrices (K = number of vertices).  Cost O(K^2 n + K n^2) per mark.
Validated against exact statevector finite differences in lam (validate_pt2.py).
"""
import numpy as np

class _Block:
    def __init__(self, D, Phi, M):
        self.D = np.asarray(D, float); Db = 1 - self.D; P = Phi; Pc = Phi.conj()
        self.Gp = (P * self.D) @ Pc.T                  # phi_k^T D conj(phi_j)
        self.Gm = (P * Db) @ Pc.T                      # phi_k^T Db conj(phi_j)
        self.n = self.Gp.diagonal().real.copy()        # <A_k>
        dg, dbg = np.diag(self.D), np.diag(Db)
        sand = lambda L1, L2: (P @ (L1 @ M @ L2)) @ Pc.T
        self.mDDb, self.mDD, self.mDbDb, self.mDbD = sand(dg, dbg), sand(dg, dg), sand(dbg, dbg), sand(dbg, dg)
        self.Om = float(np.real(np.trace(M @ dg)))
    def two(self):        # <A_k A_j>
        return np.outer(self.n, self.n) + self.Gm * self.Gp.T
    def OA(self):         # <O A_k>
        return self.Om * self.n + self.mDDb.diagonal()
    def AOA(self):        # <A_k O A_j>
        n, O, Gp, Gm = self.n, self.Om, self.Gp, self.Gm
        ckj = Gm * Gp.T; ckM = self.mDbD.diagonal(); cMk = self.mDDb.diagonal()
        return (O * np.outer(n, n) + n[:, None] * cMk[None, :] + O * ckj + n[None, :] * ckM[:, None]
                + self.mDbDb * Gp.T - Gm * self.mDD.T)
    def OAA(self):        # <O A_k A_j>
        n, O, Gp, Gm = self.n, self.Om, self.Gp, self.Gm
        ckj = Gm * Gp.T; cMk = self.mDDb.diagonal()
        return (O * np.outer(n, n) + O * ckj + n[:, None] * cMk[None, :] + n[None, :] * cMk[:, None]
                + Gm * self.mDDb.T - Gp.T * self.mDDb)

def _coeffs(DA, DB, PhiA, PhiB, g, MA, MB):
    K = len(g); a0 = a1 = a2 = 0.0
    low = np.tril(np.ones((K, K), bool), -1)            # k > j  (k later)
    for (Do, Phio, M, Dx, Phix) in ((DA, PhiA, MA, DB, PhiB), (DB, PhiB, MB, DA, PhiA)):
        if M is None or not np.any(M): continue
        Co = _Block(Do, Phio, M); a0 += Co.Om
        if K == 0: continue
        Cx = _Block(Dx, Phix, np.zeros((len(Dx), len(Dx))))
        BB = Cx.two(); OP = Co.OA() * Cx.n
        a1 += -2 * np.sum(g * OP.imag)
        a2 += (np.sum(-g**2 * OP.real) + np.einsum('k,j,kj,kj->', g, g, Co.AOA(), BB).real
               - 2 * np.real(np.sum((np.outer(g, g) * Co.OAA() * BB)[low])))
    return a0, a1, a2

def pt2(occA, occB, ops, observables):
    """Returns {mark_label: {obs_name: (a0, a1, a2)}}."""
    nA, nB = len(occA), len(occB)
    u = {'A': np.eye(nA, dtype=complex), 'B': np.eye(nB, dtype=complex)}
    vA, vB, gs = [], [], []; out = {}
    for op in ops:
        if op[0] == 'u':
            _, blk, idx, V = op; idx = list(idx); u[blk][idx, :] = np.asarray(V) @ u[blk][idx, :]
        elif op[0] == 'int':
            _, iA, iB, gg = op; vA.append(u['A'][iA, :].copy()); vB.append(u['B'][iB, :].copy()); gs.append(gg)
        elif op[0] == 'mark':
            PhiA = np.array(vA).reshape(-1, nA); PhiB = np.array(vB).reshape(-1, nB); g = np.array(gs, float)
            res = {}
            for name, (MA, MB) in observables.items():
                MAt = None if MA is None else u['A'].conj().T @ MA @ u['A']
                MBt = None if MB is None else u['B'].conj().T @ MB @ u['B']
                res[name] = _coeffs(np.array(occA), np.array(occB), PhiA, PhiB, g, MAt, MBt)
            out[op[1]] = res
    return out

def ops_from_gauss_blocks(occ, m, blocks, lo=0, hi=60):
    """Adapter: gauss.blocks() output (two 60-site chains) -> (occA, occB, ops).
    Chain 0 -> block A, chain 1 -> block B; step markers -> ('mark', step)."""
    L = hi - lo; occA = np.zeros(L); occB = np.zeros(L)
    for w in range(len(occ)):
        s, l = m[w]
        if lo <= s < hi: (occA if l == 0 else occB)[s - lo] = occ[w]
    ops = []; blk = {0: 'A', 1: 'B'}
    for b in blocks:
        if 'step' in b: ops.append(('mark', b['step'])); continue
        w = b['w']; U = b['U']; sl = [m[x] for x in w]
        if any(not (lo <= s < hi) for s, _ in sl): continue
        sl = [(s - lo, l) for s, l in sl]
        if len(w) == 1:
            s, l = sl[0]; ops.append(('u', blk[l], [s], [[U[1, 1] / U[0, 0]]]))
        elif sl[0][1] == sl[1][1]:
            (s1, l1), (s2, _) = sl; V = U[1:3, 1:3] / U[0, 0]
            ops.append(('u', blk[l1], [s1, s2], np.array([[V[1, 1], V[1, 0]], [V[0, 1], V[0, 0]]])))
        else:
            (s1, l1), (s2, l2) = sl; ph = np.angle(np.diag(U))
            g = ph[3] - ph[2] - ph[1] + ph[0]; g = (g + np.pi) % (2 * np.pi) - np.pi
            ops.append(('u', blk[l1], [s1], [[np.exp(1j * (ph[2] - ph[0]))]]))
            ops.append(('u', blk[l2], [s2], [[np.exp(1j * (ph[1] - ph[0]))]]))
            iA, iB = (s1, s2) if l1 == 0 else (s2, s1)
            ops.append(('int', iA, iB, g))
    return occA, occB, ops

if __name__ == '__main__':
    import os, json, gauss
    D = os.environ.get('SU2_CIRCUITS', 'circuits')
    eps = np.diag([(-1.0) ** r for r in range(60)])
    obs = {'stag': (eps, eps), 'Q': (np.eye(60), np.eye(60))}
    res = {}
    for circ in ('SCV', 'meson'):
        occ, m, bl = gauss.blocks(f'{D}/x_100_{circ}.qasm')
        oA, oB, ops = ops_from_gauss_blocks(occ, m, bl)
        res[circ] = pt2(oA, oB, ops, obs)
    for T in range(1, 21):
        s, me = res['SCV'][T]['stag'], res['meson'][T]['stag']
        print(f"step {T:2d} stag_SCV free {s[0]:.7f} a1 {s[1]:+.1e} a2 {s[2]:+.7f} PT2 {s[0]+s[2]:.7f} | "
              f"n_f free {me[0]-s[0]:.7f} a2 {me[2]-s[2]:+.7f} PT2 {me[0]+me[2]-s[0]-s[2]:.7f} | Q_SCV a2 {res['SCV'][T]['Q'][2]:+.0e}")
