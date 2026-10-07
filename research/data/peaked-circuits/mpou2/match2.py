"""Matching unswap for MPO2 (statistic S[w,v] = (1/3) sum_{P,Q in XYZ} |Tr(W P_w W^+ Q_v)|^2 / Tr(W W^+)^2,
validated densely in /tmp/unswap-match/proto2.py; Hungarian assignment on S; output legs then bubble-sorted
with 'up' relabelling swaps so that output wire pi(w) shares the site of input wire w)."""
import numpy as np
from scipy.optimize import linear_sum_assignment
PAULI = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.diag([1., -1.]).astype(complex)]


def _env_step_L(L, A, X=None, Y=None):
    """L'[r,s] = sum L[l,m] A[l,u,d,r] X[d,e] conj(A[m,v,e,s]) Y[v,u]   (X, Y act on dn / up legs)."""
    B = A
    if X is not None:
        B = np.einsum('ludr,de->luer', B, X)
    if Y is not None:
        B = np.einsum('vu,luer->lver', Y, B)
    T = np.tensordot(L, B, axes=(0, 0))                      # (m, v, e, r)
    return np.tensordot(T, A.conj(), axes=([0, 1, 2], [0, 1, 2]))   # (r, s)


def _env_step_R(R, A, X=None, Y=None):
    B = A
    if X is not None:
        B = np.einsum('ludr,de->luer', B, X)
    if Y is not None:
        B = np.einsum('vu,luer->lver', Y, B)
    T = np.tensordot(B, R, axes=(3, 0))                      # (l, v, e, s)
    return np.tensordot(T, A.conj(), axes=([1, 2, 3], [1, 2, 3]))   # (l, m)


def score_sites(W, stat='sum'):
    """stat 'sum': (1/3) sum_{P,Q} |T_PQ|^2 ; 'smax': largest singular value^2 of the 3x3 matrix T_PQ
    (T_PQ = Tr(W P_a W^+ Q_b)/Tr(W W^+)); 'smax' is blind to local unitaries and to diagonal (CZ-like) residues."""
    n = W.n; A = W.A
    one = np.ones((1, 1), complex)
    Lp = [one]
    for s in range(n):
        Lp.append(_env_step_L(Lp[-1], A[s]))
    Rp = [None] * (n + 1); Rp[n] = one
    for s in range(n - 1, -1, -1):
        Rp[s] = _env_step_R(Rp[s + 1], A[s])
    norm = Lp[n][0, 0]
    T = np.zeros((n, n, 3, 3), complex)
    for a in range(n):           # site of the input-leg Pauli
        for ip, P in enumerate(PAULI):
            La = [None] * (n + 1); La[a + 1] = _env_step_L(Lp[a], A[a], P)
            for s in range(a + 1, n - 1):
                La[s + 1] = _env_step_L(La[s], A[s])
            Ra = [None] * (n + 1); Ra[a] = _env_step_R(Rp[a + 1], A[a], P)
            for s in range(a - 1, 0, -1):
                Ra[s] = _env_step_R(Ra[s + 1], A[s])
            for b in range(n):
                for iq, Q in enumerate(PAULI):
                    if b == a:
                        v = np.sum(_env_step_L(Lp[a], A[a], P, Q) * Rp[a + 1].T) if False else \
                            np.einsum('rs,rs->', _env_step_L(Lp[a], A[a], P, Q), Rp[a + 1])
                    elif b > a:
                        v = np.einsum('rs,rs->', _env_step_L(La[b], A[b], None, Q), Rp[b + 1])
                    else:
                        v = np.einsum('rs,rs->', _env_step_L(Lp[b], A[b], None, Q), Ra[b + 1])
                    T[a, b, ip, iq] = v / norm
    if stat == 'sum':
        return np.sum(np.abs(T) ** 2, axis=(2, 3)) / 3
    sv = np.linalg.svd(T, compute_uv=False)
    return sv[:, :, 0] ** 2      # S[site of input leg, site of output leg]


def score_wires(W, stat='sum'):
    S = score_sites(W, stat)
    n = W.n
    return np.array([[S[W.pd[w], W.pu[v]] for v in range(n)] for w in range(n)])


def match_unswap(W, log=None, min_median=0.6, stat='sum'):
    S = score_wires(W, stat)
    r, c = linear_sum_assignment(-S)
    pi = dict(zip(r, c))
    sc = np.array([S[w, pi[w]] for w in range(W.n)])
    if np.median(sc) < min_median:
        return pi, sc, -1
    target = [None] * W.n                 # site -> desired output wire
    for w in range(W.n):
        target[W.pd[w]] = pi[w]
    nsw = 0
    for pos in range(W.n):
        want = target[pos]; cur = W.pu[want]
        while cur > pos:
            W.swap(cur - 1, 'up'); cur -= 1; nsw += 1
    if log:
        log(f'    match-unswap: {nsw} swaps, scores min {sc.min():.3f} median {np.median(sc):.3f} -> {W.stats()}')
    return pi, sc, nsw
