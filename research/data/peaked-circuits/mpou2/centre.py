"""Exact centre block for mpou2: the operator of a band of blocks whose interaction graph splits into small
connected components (groups) is contracted densely per group, the in->out wire pairing of each group is chosen by
the dense Pauli-transfer score + Hungarian assignment, the group's sites are laid out contiguously (best site order
by exhaustive search for m <= 6) and decomposed into MPO site tensors by successive SVD."""
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from mpou2 import MPO2

PAULI = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.diag([1., -1.]).astype(complex)]


def components(n, pairs):
    par = list(range(n))

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in pairs:
        par[f(a)] = f(b)
    comp = {}
    for q in range(n):
        comp.setdefault(f(q), []).append(q)
    return sorted(comp.values(), key=lambda g: g[0])


def dense_group(g, ops):
    """ops: [(a, b, M)] in time order, M on kron(a, b).  -> U (2^m x 2^m), bit order = g (first = most significant)."""
    m = len(g)
    U = np.eye(2 ** m, dtype=complex).reshape([2] * m + [2 ** m])
    for a, b, M in ops:
        ia, ib = g.index(a), g.index(b)
        T = np.tensordot(M.reshape(2, 2, 2, 2), U, axes=([2, 3], [ia, ib]))
        U = np.moveaxis(T, [0, 1], [ia, ib])
    return U.reshape(2 ** m, 2 ** m)


def op1(m, q, P):
    out = np.ones((1, 1), complex)
    for k in range(m):
        out = np.kron(out, P if k == q else np.eye(2))
    return out


def pauli_scores(U, m):
    d = 2 ** m; S = np.zeros((m, m))
    Ud = U.conj().T
    for i in range(m):
        for P in PAULI:
            WP = U @ op1(m, i, P) @ Ud
            for j in range(m):
                for Q in PAULI:
                    S[i, j] += abs(np.trace(WP @ op1(m, j, Q))) ** 2 / d ** 2
    return S / 3


def _chain(T, m, eps):
    """T with axes (o_0, i_0, o_1, i_1, ...) (already in site order) -> list of site tensors (Dl,u,d,Dr), ranks."""
    sites = []; rest = T.reshape(1, -1); ranks = []
    for k in range(m - 1):
        Dl = rest.shape[0]
        M = rest.reshape(Dl * 4, -1)
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        w = S ** 2; tail = np.cumsum(w[::-1])[::-1]
        r = max(1, int(np.sum(tail > eps * tail[0])))
        sites.append(U[:, :r].reshape(Dl, 2, 2, r)); ranks.append(r)
        rest = S[:r, None] * Vh[:r]
    sites.append(rest.reshape(rest.shape[0], 2, 2, 1))
    return sites, ranks


def build(n, groups, group_ops, eps=1e-12, log=print):
    """returns MPO2 W (eps for later growth set by the caller) representing the product of the groups' operators."""
    W = MPO2(n)
    A = []; su = []; sd = []
    lognorm = 0.0
    for g in groups:
        m = len(g)
        if m == 1:
            A.append(np.eye(2, dtype=complex).reshape(1, 2, 2, 1)); su.append(g[0]); sd.append(g[0]); continue
        U = dense_group(g, group_ops[tuple(g)])
        S = pauli_scores(U, m)
        r, c = linear_sum_assignment(-S)
        pi = dict(zip(r, c))                       # input index i -> output index pi[i] (indices into g)
        T = U.reshape([2] * (2 * m))               # axes: out bits (g order), in bits (g order)
        best = None
        orders = itertools.permutations(range(m)) if m <= 6 else [tuple(range(m))]
        for order in orders:                        # order[k] = input index placed on site k
            ax = []
            for i in order:
                ax += [pi[i], m + i]
            sites, ranks = _chain(np.transpose(T, ax), m, eps)
            cost = sum(ranks)
            if best is None or cost < best[0]:
                best = (cost, order, sites, ranks)
        cost, order, sites, ranks = best
        nrm = np.linalg.norm(sites[-1]); sites[-1] = sites[-1] / nrm; lognorm += np.log(nrm)
        A += sites
        for i in order:
            sd.append(g[i]); su.append(g[pi[i]])
        sc = [S[i, pi[i]] for i in range(m)]
        log(f'  group {g}: pairing {[ (g[i], g[pi[i]]) for i in range(m)]} scores min {min(sc):.3f} ranks {ranks}')
    W.A = A; W.su = su; W.sd = sd
    W.pu = [None] * n; W.pd = [None] * n
    for s in range(n):
        W.pu[su[s]] = s; W.pd[sd[s]] = s
    W.c = n - 1; W.lognorm = lognorm
    return W
