# Pauli-path / Fourier-spectrum probe: minimal # of anticommuting-T crossings for a
# path from a Z-type Pauli at t=0 to a Z-type Pauli at the end (=> |<Z_S>| contribution 2^{-a/2}).
import re, sys, numpy as np
F = sys.argv[1]; D = int(sys.argv[2]) if len(sys.argv) > 2 else 999
ops = []
for line in open(F):
    line = line.strip().rstrip(';')
    if not line or line.startswith(('OPENQASM','include','qreg','creg','barrier','measure')): continue
    m = re.match(r'([a-z]+)(\(([^)]*)\))?\s+(.*)', line)
    ops.append((m.group(1), [int(x) for x in re.findall(r'q\[(\d+)\]', m.group(4))]))
n = 70
# truncate to CZ depth D (keep 1q gates until a qubit's next CZ beyond D)
czl = [0]*n; keep = []
for nm, qs in ops:
    if nm == 'cz':
        d = max(czl[q] for q in qs) + 1
        if d > D: 
            for q in qs: czl[q] = 10**9
            continue
        for q in qs: czl[q] = d
    else:
        if czl[qs[0]] >= 10**9: continue
    keep.append((nm, qs))
ops = keep
Tsites = [i for i,(nm,qs) in enumerate(ops) if nm == 'rz']
nT = len(Tsites); NS = n + nT
X = np.zeros((NS, n), np.uint8); Z = np.zeros((NS, n), np.uint8)
for q in range(n): Z[q, q] = 1
M0 = np.zeros((nT, NS), np.uint8)
tq = []
k = 0
for nm, qs in ops:
    if nm == 'h': q = qs[0]; X[:, q], Z[:, q] = Z[:, q].copy(), X[:, q].copy()
    elif nm in ('sx', 'sxdg'): q = qs[0]; X[:, q] ^= Z[:, q]
    elif nm in ('s', 'sdg', 'z', 'x', 'y'):
        if nm in ('s','sdg'): q = qs[0]; Z[:, q] ^= X[:, q]
    elif nm == 'cz': a, b = qs; Z[:, a] ^= X[:, b]; Z[:, b] ^= X[:, a]
    elif nm == 'rz':
        q = qs[0]; M0[k] = X[:, q]; Z[n + k, q] = 1; tq.append(q); k += 1
    else: raise Exception(nm)
Lx = X.T.copy()  # n x NS : final x-part
Lz = Z.T.copy()
def gf2_solve_mat(A, B):
    # solve A Y = B over GF(2), A square invertible
    A = A.copy() % 2; B = B.copy() % 2; m = A.shape[0]
    for c in range(m):
        p = next(r for r in range(c, m) if A[r, c])
        A[[c, p]] = A[[p, c]]; B[[c, p]] = B[[p, c]]
        for r in range(m):
            if r != c and A[r, c]: A[r] ^= A[c]; B[r] ^= B[c]
    return B
LA = Lx[:, :n]; LT = Lx[:, n:]
Ainv_LT = gf2_solve_mat(LA, LT)                    # A = Ainv_LT y
M = (M0[:, :n].astype(np.int64) @ Ainv_LT + M0[:, n:]) % 2   # anticommutation vector = M y
Sm = (Lz[:, :n].astype(np.int64) @ Ainv_LT + Lz[:, n:]) % 2  # final Z-string S = Sm y
M = M.astype(np.uint8); Sm = Sm.astype(np.uint8)
np.save('M_D%d.npy' % D, M); np.save('S_D%d.npy' % D, Sm)
print(f"D={D} T={nT} rank(M)?")
# rank
def rank(A):
    A = A.copy(); r = 0
    for c in range(A.shape[1]):
        p = [i for i in range(r, A.shape[0]) if A[i, c]]
        if not p: continue
        A[[r, p[0]]] = A[[p[0], r]]
        for i in range(A.shape[0]):
            if i != r and A[i, c]: A[i] ^= A[r]
        r += 1
    return r
print("rank M", rank(M))
# singles
best = []
for j in range(nT):
    y = np.zeros(nT, np.uint8); y[j] = 1
    v = M[:, j]
    best.append((int(v.sum()), bool(v[j]), int(Sm[:, j].sum()), j))
best.sort()
print("single-branch paths (a, valid, |S|, j) best 10:", best[:10])
print("valid singles:", sum(b[1] for b in best), " min a over valid:", min([b[0] for b in best if b[1]] or [None]))
