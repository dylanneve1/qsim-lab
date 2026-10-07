#!/usr/bin/env python3
"""For every T gate: weight of its Pauli generator Z_q conjugated (a) forward to the measurement and
(b) backward to the |0..0> input. Low weight = cheap to absorb at that end (Clifford-frame / CAMPS)."""
import re, sys, collections
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
czl = [0]*n; rows = []
for i, (nm, qs) in enumerate(ops):
    if nm == 'cz':
        d = max(czl[q] for q in qs) + 1
        for q in qs: czl[q] = d
    if nm != 'rz': continue
    q = qs[0]
    x = np.zeros(n, np.uint8); z = np.zeros(n, np.uint8); z[q] = 1
    for nm2, qs2 in ops[i+1:]:
        if nm2 != 'rz': step(x, z, nm2, qs2)
    w_end = int(np.count_nonzero(x | z)); xw_end = int(np.count_nonzero(x))  # x-part flips the Z-basis outcome
    x = np.zeros(n, np.uint8); z = np.zeros(n, np.uint8); z[q] = 1
    for nm2, qs2 in reversed(ops[:i]):
        if nm2 != 'rz': step(x, z, nm2, qs2)
    w_start = int(np.count_nonzero(x | z)); xw_start = int(np.count_nonzero(x))  # z-only part is a phase on |0>
    rows.append((czl[q], q, w_end, xw_end, w_start, xw_start))
rows = np.array(rows)
print("T count", len(rows))
for name, col in (("weight pushed to END", 2), ("X-weight at END", 3), ("weight pushed to START", 4), ("X-weight at START (0 = trivial phase on |0>)", 5)):
    v = rows[:, col]; print(f"{name}: min {v.min()} median {int(np.median(v))} max {v.max()}; hist(<=4,<=10,<=20,>20) "
          f"{(v<=4).sum()},{((v>4)&(v<=10)).sum()},{((v>10)&(v<=20)).sum()},{(v>20).sum()}")
best = np.minimum(rows[:, 2], rows[:, 4])
print("min(end,start) weight: hist(<=4,<=10,<=20,>20)", (best<=4).sum(), ((best>4)&(best<=10)).sum(), ((best>10)&(best<=20)).sum(), (best>20).sum())
print("T trivially absorbed at start (no X part):", int((rows[:,5]==0).sum()))
np.savetxt(sys.argv[1] + ".tweights.tsv", rows, fmt="%d", delimiter="\t", header="czlayer\tqubit\tw_end\txw_end\tw_start\txw_start")
