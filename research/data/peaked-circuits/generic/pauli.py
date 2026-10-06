"""Sparse Heisenberg-picture Pauli algebra on up to ~128 qubits (bitmask ints).

A Pauli string is (x, z) with the Hermitian convention  P = prod_q i^{x_q z_q} X^{x_q} Z^{z_q}
(so (1,1) on a qubit is Y). A Pauli sum is a dict {(x, z): complex coef}.
"""
import numpy as np

_P1 = {(0, 0): np.eye(2, dtype=complex), (1, 0): np.array([[0, 1], [1, 0]], dtype=complex),
       (0, 1): np.diag([1, -1]).astype(complex), (1, 1): np.array([[0, -1j], [1j, 0]])}
# local 2-qubit index: s = xa | za<<1 | xb<<2 | zb<<3
_LOC = []
for s in range(16):
    xa, za, xb, zb = s & 1, (s >> 1) & 1, (s >> 2) & 1, (s >> 3) & 1
    _LOC.append(np.kron(_P1[xa, za], _P1[xb, zb]))
_LOC1 = [_P1[s & 1, s >> 1] for s in range(4)]


def mul(p1, p2):
    """(x1,z1)*(x2,z2) -> (phase, (x3,z3)); phase in {1, i, -1, -i}."""
    x1, z1 = p1; x2, z2 = p2
    x3, z3 = x1 ^ x2, z1 ^ z2
    e = ((x1 & z1).bit_count() + (x2 & z2).bit_count() + 2 * (z1 & x2).bit_count() - (x3 & z3).bit_count()) & 3
    return (1, 1j, -1, -1j)[e], (x3, z3)


def sum_mul(A, B, eps=0.0):
    out = {}
    for pa, ca in A.items():
        for pb, cb in B.items():
            ph, p = mul(pa, pb)
            out[p] = out.get(p, 0) + ph * ca * cb
    return {p: c for p, c in out.items() if abs(c) > eps}


def conj_table2(G):
    """T[s] = list of (s', c) with  G^dag sigma_s G = sum c sigma_s'  (G 4x4, first qubit = a)."""
    Gd = G.conj().T
    T = []
    for s in range(16):
        M = Gd @ _LOC[s] @ G
        row = []
        for t in range(16):
            c = np.trace(_LOC[t].conj().T @ M) / 4
            if abs(c) > 1e-13:
                row.append((t, complex(c)))
        T.append(row)
    return T


def conj_table1(V):
    Vd = V.conj().T
    T = []
    for s in range(4):
        M = Vd @ _LOC1[s] @ V
        T.append([(t, complex(np.trace(_LOC1[t].conj().T @ M) / 2)) for t in range(4)
                  if abs(np.trace(_LOC1[t].conj().T @ M)) > 2e-13])
    return T


def conj_sum2(S, T, a, b, eps=0.0):
    """Apply phi_G (G on wires a, b) to Pauli sum S."""
    ma, mb = 1 << a, 1 << b
    mask = ma | mb
    out = {}
    for (x, z), c in S.items():
        if not ((x | z) & mask):
            out[(x, z)] = out.get((x, z), 0) + c
            continue
        s = (1 if x & ma else 0) | (2 if z & ma else 0) | (4 if x & mb else 0) | (8 if z & mb else 0)
        x0, z0 = x & ~mask, z & ~mask
        for t, ct in T[s]:
            xx = x0 | (ma if t & 1 else 0) | (mb if t & 4 else 0)
            zz = z0 | (ma if t & 2 else 0) | (mb if t & 8 else 0)
            k = (xx, zz)
            out[k] = out.get(k, 0) + c * ct
    return {p: c for p, c in out.items() if abs(c) > eps}


def local_sum2(T, s, a, b):
    """The Pauli sum G^dag sigma_s G as a dict over full strings."""
    out = {}
    for t, ct in T[s]:
        xx = ((1 << a) if t & 1 else 0) | ((1 << b) if t & 4 else 0)
        zz = ((1 << a) if t & 2 else 0) | ((1 << b) if t & 8 else 0)
        out[(xx, zz)] = ct
    return out


def norm2(S):
    return sum(abs(c) ** 2 for c in S.values())
