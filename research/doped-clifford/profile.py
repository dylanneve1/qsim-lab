#!/usr/bin/env python3
"""Profile IBM's doped random-graph-state circuit (tracker issue 228).
Pure numpy: own stabilizer tableau (Aaronson-Gottesman, no phases needed for entanglement)."""
import re, sys, collections
import numpy as np

F = sys.argv[1]
ops = []
for line in open(F):
    line = line.strip().rstrip(';')
    if not line or line.startswith(('OPENQASM', 'include', 'qreg', 'creg', 'barrier', 'measure')):
        continue
    m = re.match(r'([a-z]+)(\(([^)]*)\))?\s+(.*)', line)
    name, arg, qs = m.group(1), m.group(3), [int(x) for x in re.findall(r'q\[(\d+)\]', m.group(4))]
    ops.append((name, arg, qs))
n = 1 + max(q for _, _, qs in ops for q in qs)

# ---- layering (ASAP) and gate stats
cnt = collections.Counter(o[0] for o in ops)
czs = [qs for nm, _, qs in ops if nm == 'cz']
dist = collections.Counter(abs(a - b) for a, b in czs)
edges = collections.Counter(tuple(sorted(e)) for e in czs)
level = [0] * n; czlevel = [0] * n
T_info = []  # (cz-depth at which the T sits, qubit)
cz_layers = collections.Counter()
for nm, arg, qs in ops:
    if nm == 'cz':
        d = max(czlevel[q] for q in qs) + 1
        for q in qs: czlevel[q] = d
        cz_layers[d] += 1
    elif nm == 'rz':
        T_info.append((czlevel[qs[0]], qs[0]))
czdepth = max(czlevel)
print(f"qubits {n}; gates {dict(cnt)}")
print(f"CZ: {len(czs)} total, distance histogram {dict(dist)}, distinct edges {len(edges)}, CZ-depth {czdepth}")
print(f"CZ per layer: min {min(cz_layers.values())} max {max(cz_layers.values())}")
Tl = collections.Counter(d for d, _ in T_info); Tq = collections.Counter(q for _, q in T_info)
print(f"T gates: {len(T_info)}; per CZ-layer min/mean/max {min(Tl.values())}/{len(T_info)/len(Tl):.1f}/{max(Tl.values())} over {len(Tl)} layers; "
      f"per qubit min/max {min(Tq.get(q,0) for q in range(n))}/{max(Tq.values())}")
first_T = min(d for d, _ in T_info); print(f"first T at CZ-layer {first_T}, last at {max(d for d,_ in T_info)}")
# T in left/right halves of the chain, and in the time halves
print("T by chain decile:", [sum(1 for _, q in T_info if 7*k <= q < 7*(k+1)) for k in range(10)])
print("T by depth decile:", [sum(1 for d, _ in T_info if 7*k < d <= 7*(k+1)) for k in range(10)])

# ---- stabilizer tableau of the UNDOPED circuit (rz(pi/4) dropped; also try rz->S i.e. Clifford proxy)
def tableau(ops, rz_as=None):
    X = np.eye(n, dtype=np.uint8); Z = np.zeros((n, n), dtype=np.uint8)  # stabilizers Z_i: x=0,z=e_i
    X, Z = np.zeros((n, n), np.uint8), np.eye(n, dtype=np.uint8)
    for nm, arg, qs in ops:
        if nm == 'rz':
            if rz_as is None: continue
            nm = rz_as
        if nm == 'h':
            q = qs[0]; X[:, q], Z[:, q] = Z[:, q].copy(), X[:, q].copy()
        elif nm in ('s', 'sdg'):
            q = qs[0]; Z[:, q] ^= X[:, q]
        elif nm in ('sx', 'sxdg'):
            q = qs[0]; X[:, q] ^= Z[:, q]
        elif nm == 'cz':
            a, b = qs; Z[:, a] ^= X[:, b]; Z[:, b] ^= X[:, a]
        else:
            raise ValueError(nm)
    return X, Z

def gf2_rank(M):
    M = M.copy() % 2; r = 0; rows, cols = M.shape
    for c in range(cols):
        piv = np.nonzero(M[r:, c])[0]
        if piv.size == 0: continue
        p = r + piv[0]; M[[r, p]] = M[[p, r]]
        nz = np.nonzero(M[:, c])[0]; nz = nz[nz != r]
        M[nz] ^= M[r]
        r += 1
        if r == rows: break
    return r

def ent(X, Z, A):
    A = np.array(A); sub = np.concatenate([X[:, A], Z[:, A]], axis=1)
    return gf2_rank(sub) - len(A)

for label, rz_as in (("undoped (T removed)", None), ("T->S Clifford proxy", 's')):
    X, Z = tableau(ops, rz_as)
    cuts = [ent(X, Z, list(range(k))) for k in range(1, n)]
    print(f"\n[{label}] entanglement (bits) across chain cut k|n-k, k=1..{n-1}:")
    print(" ", cuts)
    print(f"  max {max(cuts)} at k={1+int(np.argmax(cuts))}; MPS bond dim needed 2^{max(cuts)}")
    rng = np.random.default_rng(0)
    rb = [ent(X, Z, list(rng.permutation(n)[:n//2])) for _ in range(200)]
    print(f"  random half-bipartitions (200): min {min(rb)} max {max(rb)}")
