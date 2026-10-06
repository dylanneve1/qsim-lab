"""Matching-based unswap for MPOU (our method; statistic suggested and validated densely by the parent agent):
S[w,v] = (1/3) sum_{P,Q in XYZ} |Tr(W P_w W^dag Q_v)|^2 / Tr(W W^dag)^2  for input wire w, output wire v.
Hungarian assignment on S gives the wire permutation; the output legs are then sorted (adjacent SWAPs on the upper legs,
pure relabelling) so that output wire pi(w) sits on the same site as input wire w.
Environments are (D, D') matrices; transfers are applied, never formed (O(D^3) per step)."""
import numpy as np
from scipy.optimize import linear_sum_assignment
PAULI = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.diag([1., -1.]).astype(complex)]
I2 = np.eye(2, dtype=complex)

def stepL(L, A, X, Y):
    """L'[r,r'] = sum L[l,l'] A[l,u,d,r] X[d,e] conj(A[l',v,e,r']) Y[v,u]."""
    B = np.einsum('ludr,de->luer', A, X); B = np.einsum('vu,luer->lver', Y, B)
    return np.einsum('lm,lver,mven->rn', L, B, A.conj(), optimize=True)
def stepR(R, A, X, Y):
    B = np.einsum('ludr,de->luer', A, X); B = np.einsum('vu,luer->lver', Y, B)
    return np.einsum('rn,lver,mven->lm', R, B, A.conj(), optimize=True)

def score_matrix(W):
    n = W.n; A = W.A
    one = np.ones((1, 1), complex)
    Lp = [one]
    for s in range(n): Lp.append(stepL(Lp[-1], A[s], I2, I2))
    Rp = [None] * (n + 1); Rp[n] = one
    for s in range(n - 1, -1, -1): Rp[s] = stepR(Rp[s + 1], A[s], I2, I2)
    norm = Lp[n][0, 0]
    S_site = np.zeros((n, n))
    for a in range(n):
        for P in PAULI:
            La = [None] * (n + 1); La[a + 1] = stepL(Lp[a], A[a], P, I2)
            for s in range(a + 1, n - 1): La[s + 1] = stepL(La[s], A[s], I2, I2)
            Ra = [None] * (n + 1); Ra[a] = stepR(Rp[a + 1], A[a], P, I2)
            for s in range(a - 1, 0, -1): Ra[s] = stepR(Ra[s + 1], A[s], I2, I2)
            for b in range(n):
                for Q in PAULI:
                    if b == a: E = stepL(Lp[a], A[a], P, Q); v = np.sum(E * Rp[a + 1])
                    elif b > a: E = stepL(La[b], A[b], I2, Q); v = np.sum(E * Rp[b + 1])
                    else: E = stepL(Lp[b], A[b], I2, Q); v = np.sum(E * Ra[b + 1])
                    S_site[a, b] += abs(v / norm) ** 2 / 3
    return np.array([[S_site[W.pd[w], W.pu[v]] for v in range(n)] for w in range(n)])

def match_unswap(W, log=None):
    S = score_matrix(W)
    r, c = linear_sum_assignment(-S)
    pi = dict(zip(r, c))
    sc = np.array([S[w, pi[w]] for w in range(W.n)])
    target = [None] * W.n
    for w in range(W.n): target[W.pd[w]] = pi[w]
    nsw = 0
    for pos in range(W.n):
        want = target[pos]; cur = W.pu[want]
        while cur > pos:
            W.swap_legs(cur - 1, 'up'); cur -= 1; nsw += 1
    if log: log(f'    match-unswap: {nsw} swaps, assignment scores min {sc.min():.3f} median {np.median(sc):.3f} -> {W.stats()}')
    return pi, sc, nsw
