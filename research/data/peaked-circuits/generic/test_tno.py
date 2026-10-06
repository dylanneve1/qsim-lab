import numpy as np, itertools, tno as TN, gparse as G
rng = np.random.default_rng(3)
def haar(d):
    z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2); q, r = np.linalg.qr(z)
    return q * (np.diag(r) / abs(np.diag(r)))
def dense_from_tno(T, n):
    # contract all tensors; output order o0..o_{n-1}, i0..i_{n-1}
    tids = list(T.T)
    A, la = T.T[tids[0]]
    for t in tids[1:]:
        A, la = T._contract(A, la, *T.T[t])
    order = [la.index(('o', q)) for q in range(n)] + [la.index(('i', q)) for q in range(n)]
    return np.transpose(A, order).reshape(2 ** n, 2 ** n)
def embed(Gm, a, b, n):
    # G acts on (a,b) with a first; qubit 0 most significant
    I = np.eye(2 ** n, dtype=complex).reshape([2] * (2 * n))
    U = np.zeros((2 ** n, 2 ** n), dtype=complex)
    Gt = Gm.reshape(2, 2, 2, 2)
    Ut = np.einsum('abcd,...c...d...->...', Gt, I) if False else None
    full = np.zeros((2,) * (2 * n), dtype=complex)
    M = np.eye(2 ** n, dtype=complex)
    for col in range(2 ** n):
        v = M[:, col].reshape([2] * n)
        v = np.moveaxis(v, [a, b], [0, 1])
        sh = v.shape
        v = (Gm @ v.reshape(4, -1)).reshape(sh)
        v = np.moveaxis(v, [0, 1], [a, b])
        U[:, col] = v.reshape(-1)
    return U
ok = True
for trial in range(4):
    n = 4
    gates = []
    for _ in range(8):
        a, b = rng.choice(n, 2, replace=False); gates.append((haar(4), a, b))
    Uref = np.eye(2 ** n, dtype=complex)
    for Gm, a, b in gates: Uref = embed(Gm, a, b, n) @ Uref
    T = TN.TNO(n, cutoff=0.0)
    c = 4
    order = []
    lo, hi = c, c
    # alternate absorbing after / before (serial order is a valid DAG order)
    while lo > 0 or hi < len(gates):
        if hi < len(gates): Gm, a, b = gates[hi]; T.gate(Gm, a, b, 'after'); hi += 1
        if lo > 0: Gm, a, b = gates[lo - 1]; T.gate(Gm, a, b, 'before'); lo -= 1
    Ut = dense_from_tno(T, n)
    err = np.linalg.norm(Ut - Uref) / np.linalg.norm(Uref)
    print(f"random 4q/8 gates: rel err {err:.2e}, max bond {T.max_bond_dim()}"); ok &= err < 1e-10
# mirror with a hidden permutation: V ; SWAPs ; V^dag(relabelled) must collapse to a bond-free permutation
n = 6; V = [(haar(4), *rng.choice(n, 2, replace=False)) for _ in range(20)]
pi = rng.permutation(n)
SW = np.eye(4)[[0, 2, 1, 3]]
# explicit SWAP network realising pi (selection sort into place), then V^dag relabelled
cur = list(range(n)); swaps = []
for t in range(n):
    s_ = cur.index(pi[t]) if False else None
perm_gates = []
pos = list(range(n))            # pos[q] = where the content of wire q currently is
target = {q: int(pi[q]) for q in range(n)}
arr = list(range(n))            # arr[w] = original wire whose state sits on w
for w in range(n):
    want = [q for q in range(n) if target[q] == w][0]
    j = arr.index(want)
    if j != w:
        perm_gates.append((SW, w, j)); arr[w], arr[j] = arr[j], arr[w]
gates = list(V) + perm_gates + [(Gm.conj().T, int(pi[a]), int(pi[b])) for Gm, a, b in reversed(V)]
T = TN.TNO(n, cutoff=1e-10)
c = len(V) + len(perm_gates) // 2; lo, hi = c, c
while lo > 0 or hi < len(gates):
    if hi < len(gates): Gm, a, b = gates[hi]; T.gate(Gm, a, b, 'after'); hi += 1
    if lo > 0: Gm, a, b = gates[lo - 1]; T.gate(Gm, a, b, 'before'); lo -= 1
T.drop_trivial_bonds()
print("mirror with hidden relabelling: max bond", T.max_bond_dim(), "size", T.size(), "(n identity tensors = %d)" % (4 * n))
m = T.perm(); print("  recovered out->in map", m, " true pi", {int(pi[q]): q for q in range(n)})
print("ALL OK" if ok else "FAIL")
# unswap_pass must leave the operator unchanged (exact, cutoff 0)
for trial in range(3):
    n = 4; gates = [(haar(4), *rng.choice(n, 2, replace=False)) for _ in range(6)]
    Uref = np.eye(2 ** n, dtype=complex)
    for Gm, a, b in gates: Uref = embed(Gm, a, b, n) @ Uref
    T = TN.TNO(n, cutoff=0.0)
    for Gm, a, b in gates: T.gate(Gm, a, b, 'after', try_swap=False)
    acc = TN.unswap_pass(T)
    err = np.linalg.norm(dense_from_tno(T, n) - Uref) / np.linalg.norm(Uref)
    print(f"unswap_pass on random 4q: accepted {acc}, rel err {err:.2e}"); ok &= err < 1e-10
# a pure SWAP network built from 3 CZ-class gates per swap, interleaved: unswap should reduce it to bond 1
n = 4
def swap_as_3(a, b):  # SWAP = CX(a,b) CX(b,a) CX(a,b), each CX = (I x H) CZ (I x H)
    H = G.H; CX = np.kron(np.eye(2), H) @ G.CZ @ np.kron(np.eye(2), H); CXr = np.kron(H, np.eye(2)) @ G.CZ @ np.kron(H, np.eye(2))
    return [(CX, a, b), (CXr, a, b), (CX, a, b)]
s1, s2 = swap_as_3(0, 1), swap_as_3(2, 3)
gates = [s1[0], s2[0], s1[1], s2[1], s1[2], s2[2]]
T = TN.TNO(n, cutoff=1e-10)
for Gm, a, b in gates: T.gate(Gm, a, b, 'after', try_swap=False)
before = T.max_bond_dim(); TN.unswap_pass(T); T.drop_trivial_bonds()
print("interleaved 3-CZ swaps: max bond before unswap", before, "after", T.max_bond_dim(), "map", T.perm())
print("ALL OK" if ok else "FAIL")
