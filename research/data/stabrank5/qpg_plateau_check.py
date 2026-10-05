"""Falsification check for chi(T^5) = 6 using the explicit 6-term decomposition of T^{⊗6} of
Qassim-Pashayan-Gosset (Quantum 5, 606 (2021), Eqs. (4)-(5)).

If chi(T^5) = chi(T^6) = 6 (a plateau), Theorem 4(b) of research/theory/stabrank-lower.md says
that restricting ANY qubit of ANY optimal decomposition of T^6 by ANY single-qubit stabilizer bra
gives 6 non-zero, linearly independent vectors spanning T^5.  A degenerate restriction (a zero,
or a dependency) would exhibit a decomposition of T^5 with at most 5 terms, i.e. refute
chi(T^5) >= 6.  This script checks all 6 qubits x 6 bras.  numpy only.
"""
import itertools
import numpy as np

n = 6
w = np.exp(1j * np.pi / 4)
T = np.array([1, w]) / np.sqrt(2)
Tp = np.array([1, -w]) / np.sqrt(2)  # Z|T>


def kron_all(vs):
    out = np.array([1.0 + 0j])
    for v in vs:
        out = np.kron(out, v)  # first factor = most significant bit
    return out


def basis_index(bits):
    return int("".join(map(str, bits)), 2)


dim = 2**n
ghz = np.zeros(dim, complex)
ghz[0] = 1
ghz[dim - 1] = -1j
ghz /= np.sqrt(2)
E = np.zeros(dim, complex)
for x in range(dim):
    if bin(x).count("1") % 2 == 0:
        E[x] = 1
E /= np.linalg.norm(E)
K = E.copy()
for x in range(dim):
    bits = [(x >> (n - 1 - q)) & 1 for q in range(n)]
    s = sum(bits[i] * bits[j] for i in range(n) for j in range(i + 1, n))
    K[x] *= (-1) ** s
cat6 = 2 ** -1.5 * (np.sqrt(2) * ghz) + 2 ** -0.5 * np.exp(3j * np.pi / 4) * (E + 1j * K)
cat_ref = (kron_all([T] * n) + kron_all([Tp] * n)) / np.sqrt(2)
assert np.allclose(cat6, cat_ref), "QPG Eq. (5) not reproduced"

S = np.diag([1, 1j])
X = np.array([[0, 1], [1, 0]])
A = np.exp(-1j * np.pi / 4) * S @ X
assert np.allclose(np.outer(T, T.conj()), (np.eye(2) + A) / 2)
A1 = np.kron(A, np.eye(2 ** (n - 1)))  # A on the first (most significant) qubit
terms = [np.sqrt(2) * ghz, E, K]  # cat6 = sum c_j terms_j
coef = [2 ** -1.5, 2 ** -0.5 * np.exp(3j * np.pi / 4), 2 ** -0.5 * np.exp(3j * np.pi / 4) * 1j]
terms = [t / np.linalg.norm(t) for t in terms]
coef = [c * np.linalg.norm(t0) for c, t0 in zip(coef, [np.sqrt(2) * ghz, E, K])]
six = terms + [A1 @ t for t in terms]
target = kron_all([T] * n)
M = np.array(six).T
c, *_ = np.linalg.lstsq(M, target, rcond=None)
res = np.linalg.norm(M @ c - target)
print(f"QPG: T^6 = sum of 6 stabilizer states, residual {res:.2e}, coefficients |c| =",
      np.round(np.abs(c), 6))
assert res < 1e-12

bras = {
    "0": np.array([1, 0]), "1": np.array([0, 1]),
    "+": np.array([1, 1]) / np.sqrt(2), "-": np.array([1, -1]) / np.sqrt(2),
    "+i": np.array([1, 1j]) / np.sqrt(2), "-i": np.array([1, -1j]) / np.sqrt(2),
}
target5 = kron_all([T] * (n - 1))
worst_sv = 1.0
worst_norm = 1.0
for q in range(n):
    for name, s in bras.items():
        rest = []
        for v in six:
            t = v.reshape([2] * n)
            r = np.tensordot(s.conj(), t, axes=([0], [q])).reshape(-1)
            rest.append(r)
        norms = [np.linalg.norm(r) for r in rest]
        worst_norm = min(worst_norm, min(norms))
        R = np.array(rest).T
        sv = np.linalg.svd(R, compute_uv=False)
        worst_sv = min(worst_sv, sv[-1] / sv[0])
        cc, *_ = np.linalg.lstsq(R, target5, rcond=None)
        assert np.linalg.norm(R @ cc - target5) < 1e-10
        assert min(norms) > 1e-9 and sv[-1] / sv[0] > 1e-9, f"degenerate restriction q={q} s={name}"
print(f"all 36 restrictions (6 qubits x 6 stabilizer bras): 6 non-zero independent terms spanning T^5;"
      f" min term norm {worst_norm:.4f}, min relative singular value {worst_sv:.4f}")
