#!/usr/bin/env python3
"""Exact-amplitude cost model for 'push T gates through the Clifford'.

Push every T to the END: U = (prod_j exp(-i pi/8 P_j)) C, with P_j = Z_q conjugated through the rest of the circuit.
Every P_j maps |y> -> phase |y xor X_j>, so V = prod exp(..) preserves cosets of span{X_j}. Hence
<x|U|0> = <x| V Proj_coset(x) C|0>: a state-vector simulation in a space of dim 2^rx, rx = rank_GF2{X_j},
seeded by stabilizer amplitudes. Symmetric statement pushing to the START (X-parts acting on |0>).
Split: T gates in set E go to the end, the rest S to the start -> cost ~ 2^(rank X_E + rank X_S) (inner product
through the Clifford). This script reports those ranks."""
import re, sys
import numpy as np
ops = []
for line in open(sys.argv[1]):
    line = line.strip().rstrip(';')
    if not line or line.startswith(('OPENQASM', 'include', 'qreg', 'creg')): continue
    m = re.match(r'([a-z]+)(\(([^)]*)\))?\s+(.*)', line)
    ops.append((m.group(1), [int(x) for x in re.findall(r'q\[(\d+)\]', m.group(4))]))
n = 70
def step(x, z, nm, qs):
    if nm == 'h': q = qs[0]; x[q], z[q] = z[q], x[q]
    elif nm in ('s', 'sdg'): q = qs[0]; z[q] ^= x[q]
    elif nm in ('sx', 'sxdg'): q = qs[0]; x[q] ^= z[q]
    elif nm == 'cz': a, b = qs; z[a] ^= x[b]; z[b] ^= x[a]
def rank(rows):
    M = np.array(rows, np.uint8).copy() if len(rows) else np.zeros((0, n), np.uint8)
    r = 0
    for c in range(M.shape[1] if M.size else 0):
        piv = np.nonzero(M[r:, c])[0]
        if piv.size == 0: continue
        p = r + piv[0]; M[[r, p]] = M[[p, r]]
        nz = np.nonzero(M[:, c])[0]; nz = nz[nz != r]; M[nz] ^= M[r]; r += 1
        if r == M.shape[0]: break
    return r
tidx = [i for i, (nm, _) in enumerate(ops) if nm == 'rz']
XE, XS = [], []
for i in tidx:
    q = ops[i][1][0]
    x = np.zeros(n, np.uint8); z = np.zeros(n, np.uint8); z[q] = 1
    for nm2, qs2 in ops[i+1:]:
        if nm2 != 'rz': step(x, z, nm2, qs2)
    XE.append(x.copy())
    x = np.zeros(n, np.uint8); z = np.zeros(n, np.uint8); z[q] = 1
    for nm2, qs2 in reversed(ops[:i]):
        if nm2 != 'rz': step(x, z, nm2, qs2)
    XS.append(x.copy())
m = len(tidx)
print(f"T={m}; rank X (all pushed to END) = {rank(XE)}; rank X (all pushed to START) = {rank(XS)}")
# best split by time order: first k gates to START, rest to END
best = None
for k in range(0, m + 1, 4):
    rs, re_ = rank(XS[:k]), rank(XE[k:])
    if best is None or rs + re_ < best[0]: best = (rs + re_, k, rs, re_)
    if k % 40 == 0: print(f"  first {k:3d} -> start (rank {rs:2d}), last {m-k:3d} -> end (rank {re_:2d}); total {rs+re_}")
print(f"best time-ordered split: k={best[1]} start-rank {best[2]} end-rank {best[3]} total {best[0]}")
# end-only, nested by end-weight: how fast does the rank grow as we include heavier gates?
w = [int(v.sum()) for v in XE]; order = np.argsort(w)
for cut in (50, 100, 150, 200, 250, 300, 364, 400, 468):
    print(f"  {cut:3d} lightest-at-end T: X-rank {rank([XE[j] for j in order[:cut]])} (max X-weight {w[order[cut-1]]})")
