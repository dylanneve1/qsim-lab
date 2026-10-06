"""ASAP layer sizes and pair-sequence mirror score (heavy-hex: fixed wires) per circuit."""
import sys, collections, numpy as np, gparse as G
from struct_probe import layers
for f in sys.argv[1:]:
    n, units, tail = G.parse(f)
    L = layers(n, units); D = max(L) + 1
    E = [set() for _ in range(D)]
    for k, u in enumerate(units): E[L[k]].add((min(u[0], u[1]), max(u[0], u[1])))
    sz = [len(e) for e in E]
    print(f"== {f.split('/')[-1]} depth {D}")
    print("  layer sizes:", sz)
    # serialization monotone in layer?
    inv = sum(1 for k in range(1, len(units)) if L[k] < L[k-1] - 5)
    print("  serialization: big backward layer jumps:", inv)
    # mirror score around centre c (half-integer allowed): sum_d |E[c-d] & E[c+d]|
    sc = []
    for c2 in range(2, 2 * D - 2):
        s = 0; tot = 0
        for d2 in range(1 if c2 % 2 else 2, 2 * D, 2):
            lo, hi = (c2 - d2) // 2, (c2 + d2) // 2
            if lo < 0 or hi >= D: break
            s += len(E[lo] & E[hi]); tot += min(len(E[lo]), len(E[hi]))
        sc.append((c2 / 2, s, tot))
    top = sorted(sc, key=lambda x: -x[1] / max(x[2], 1) if x[2] > 50 else 0)[:8]
    print("  best mirror centres (centre, shared pairs, possible):", [(c, s, t) for c, s, t in top])
