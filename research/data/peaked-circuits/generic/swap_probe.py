"""Find same-pair CZ runs (consecutive CZs on a pair with no other CZ on either wire in between) and test if
a run of 3 is a SWAP up to local gates (Schmidt/KAK: SWAP-class iff the 4x4 run unitary has 3 maximal KAK coeffs)."""
import sys, collections, numpy as np, gparse as G, struct_probe as SP
f = sys.argv[1]
n, units, tail = G.parse(f); L = SP.layers(n, units)
CZ = np.diag([1,1,1,-1]).astype(complex)
last = {}
runs = []   # list of lists of unit idx
cur = {}
for k, (a, b, Pa, Pb) in enumerate(units):
    p = (min(a,b), max(a,b))
    if last.get(a) == p and last.get(b) == p:
        cur[p].append(k)
    else:
        for q in (a, b):
            pp = last.get(q)
            if pp and pp in cur:
                runs.append(cur.pop(pp))
        cur[p] = [k]
    last[a] = last[b] = p
runs += list(cur.values())
cnt = collections.Counter(len(r) for r in runs)
print(f.split('/')[-1], "run length histogram:", sorted(cnt.items()))
# KAK invariants via magic basis: gamma = M^T M eigenvalues
Bm = np.array([[1,0,0,1j],[0,1j,1,0],[0,1j,-1,0],[1,0,0,-1j]])/np.sqrt(2)
def kak_class(U):
    U = U / np.linalg.det(U)**0.25
    Um = Bm.conj().T @ U @ Bm
    ev = np.linalg.eigvals(Um.T @ Um)
    return np.sort(np.round(np.angle(ev), 4))
swapev = kak_class(np.eye(4)[[0,2,1,3]])
nswap = 0; where = []
for r in runs:
    if len(r) >= 3:
        a, b = units[r[0]][:2]
        U = np.eye(4, dtype=complex)
        for k in r:
            aa, bb, Pa, Pb = units[k]
            M = CZ @ (np.kron(Pa, Pb) if aa == a else np.kron(Pb, Pa))
            U = M @ U
        U = U  # interior local gates included; outer ones excluded -> equivalence class unaffected
        if np.allclose(kak_class(U), swapev, atol=1e-6):
            nswap += 1; where.append(L[r[0]])
print("  runs of >=3 that are SWAP-class:", nswap, "layers:", sorted(where)[:80])
