#!/usr/bin/env python3
"""Clifford-disentangling diagnostic for chain-sweep boundary vectors.

Input: raw complex128 files from `chain_sweep dumpcut` (bit j = bond j of the
cut edge, bonds in time order). For each boundary psi on m bits:
  * raw Schmidt entropies (von Neumann, Renyi-2, log2 rank) at every time cut;
  * a lower bound on the stabilizer nullity nu (m - log2|A| - log2|B0|);
  * greedy 2-qubit Clifford disentangling (all 720 unsigned 2-qubit
    Cliffords on every neighbouring pair, sweeps L->R->L, maximising the
    Renyi-2 purity of the cut between the pair), then the same entropies.
"""
import sys, time, numpy as np, stim

def load(f):
    v = np.fromfile(f, dtype=np.float64).view(np.complex128)
    m = int(round(np.log2(len(v)))); v = v / np.linalg.norm(v)
    # axis j <-> bit j (time order)
    T = v.reshape([2] * m).transpose(list(range(m))[::-1]).copy()
    return m, T

def cut_stats(T):
    m = T.ndim; out = []
    for t in range(1, m):
        M = T.reshape(2 ** t, -1)
        s = np.linalg.svd(M, compute_uv=False); p = s ** 2; p = p[p > 1e-14]; p /= p.sum()
        out.append((-(p * np.log2(p)).sum(), -np.log2((p ** 2).sum()), np.log2(len(p))))
    return np.array(out)  # rows t=1..m-1: S_vN, S_2, log2 rank

CLIFF = None
def cliffords():
    global CLIFF
    if CLIFF is None:
        CLIFF = np.array([t.to_unitary_matrix(endian='big') for t in stim.Tableau.iter_all(2, unsigned=True)])
    return CLIFF  # (720,4,4), index (a,b) with a = first qubit

def best_gate(T, k):
    """Gate on axes (k,k+1) maximising purity of the cut after axis k."""
    m = T.ndim; L = 2 ** k; R = 2 ** (m - k - 2)
    A = T.reshape(L, 4, R)
    if L <= R:  # W[l,s,l',s'] = sum_r A A*
        Am = A.reshape(L * 4, R); W = (Am @ Am.conj().T).reshape(L, 4, L, 4)
        # Q[s1,s2,s3,s4] = sum_{l,l'} W[l,s1,l',s2] W[l',s3,l,s4]
        Q = np.einsum('ixjy,jziw->xyzw', W, W, optimize=True)
    else:       # V[r,s,r',s'] = sum_l A(l,s,r) A*(l,s',r'); Q = sum V[r,s1,r,s2]...
        Am = A.reshape(L, 4 * R); V = (Am.T @ Am.conj()).reshape(4, R, 4, R)
        # Q[s1,s2,s3,s4] = sum_{r,r'} V[s1,r,s4,r'] conj? -> derive directly
        # sum_l T(l,s1,r)T*(l,s4,r') = V[s1,r,s4,r'];  sum_l' T*(l',s2,r)T(l',s3,r') = conj(V[s2,r,s3,r'])
        Q = np.einsum('xiwj,yizj->xyzw', V, V.conj(), optimize=True)
    G = cliffords()  # G[g, (a,b), s]
    Ga = G.reshape(-1, 2, 2, 4)  # g,a,b,s
    # purity = sum_{a,b,a',b'} G[ab,s1] G*[a'b,s2] G[a'b',s3] G*[ab',s4] Q
    X = np.einsum('gabs,gcbt->gacst', Ga, Ga.conj())          # (a,a',s1,s2) shared b
    Y = np.einsum('gcdu,gadv->gcauv', Ga, Ga.conj())          # (a',a,s3,s4) shared b'
    pur = np.einsum('gacst,gcauv,stuv->g', X, Y, Q, optimize=True).real
    g = int(np.argmax(pur)); return g, pur[g], pur[0]

def apply(T, k, U):
    m = T.ndim
    T2 = np.tensordot(U.reshape(4, 2, 2), T, axes=([1, 2], [k, k + 1]))  # (4, rest)
    T2 = T2.reshape((2, 2) + T2.shape[1:])
    return np.moveaxis(T2, [0, 1], [k, k + 1])

def disentangle(T, sweeps=4, tol=1e-12):
    m = T.ndim; G = cliffords(); nap = 0
    for sw in range(sweeps):
        changed = 0
        order = list(range(m - 1)) + list(range(m - 2, -1, -1))
        for k in order:
            g, pg, p0 = best_gate(T, k)
            if pg > p0 * (1 + tol) + tol:
                T = apply(T, k, G[g]); changed += 1; nap += 1
        if changed == 0: break
    return T, nap

def nullity_lb(T):
    m = T.ndim; v = T.transpose(list(range(m))[::-1]).reshape(-1)  # back to bit-index order
    p = np.abs(v.astype(np.complex128)) ** 2
    def wht(a):
        a = a.copy(); h = 1
        while h < len(a):
            a = a.reshape(-1, 2, h); a = np.stack([a[:, 0] + a[:, 1], a[:, 0] - a[:, 1]], 1).reshape(-1); h *= 2
        return a
    F = wht(p); R = wht(F * F) / len(p)
    nA = int(np.sum(R >= R[0] * (1 - 1e-7)))
    nB = int(np.sum(np.abs(F) >= F[0] * (1 - 1e-7)))
    return m - np.log2(nA) - np.log2(nB), np.log2(nA), np.log2(nB)

if __name__ == '__main__':
    for f in sys.argv[1:]:
        t0 = time.time(); m, T = load(f)
        raw = cut_stats(T); nu = nullity_lb(T)
        Td, nap = disentangle(T)
        dis = cut_stats(Td)
        mid = m // 2 - 1
        print(f"{f} m={m} nullity_lb={nu[0]:.1f} (log2|A|={nu[1]:.1f}, log2|B0|={nu[2]:.1f}) gates={nap} {time.time()-t0:.1f}s")
        print("   t  raw:S1   S2  log2rk |  dis:S1   S2  log2rk")
        for t in range(m - 1):
            print(f"  {t+1:2d}  {raw[t,0]:6.3f} {raw[t,1]:6.3f} {raw[t,2]:5.1f} | {dis[t,0]:6.3f} {dis[t,1]:6.3f} {dis[t,2]:5.1f}")
        print(f"SUMMARY {f} m={m} mid_raw_S1={raw[mid,0]:.3f} mid_dis_S1={dis[mid,0]:.3f} max_raw_S1={raw[:,0].max():.3f} max_dis_S1={dis[:,0].max():.3f} max_dis_rank={dis[:,2].max():.1f} nullity_lb={nu[0]:.1f}", flush=True)
