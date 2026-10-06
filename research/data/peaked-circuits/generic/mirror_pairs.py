import sys, numpy as np, gparse as G, struct_probe as SP
f = sys.argv[1]; c2 = int(round(2*float(sys.argv[2])))
n, units, tail = G.parse(f); L = SP.layers(n, units); D = max(L)+1
E = [set() for _ in range(D)]
for k, u in enumerate(units): E[L[k]].add((min(u[0], u[1]), max(u[0], u[1])))
row = []
for d2 in range(1 if c2 % 2 else 2, 2*D, 2):
    lo, hi = (c2-d2)//2, (c2+d2)//2
    if lo < 0 or hi >= D: break
    row.append(f"{lo}/{hi}:{len(E[lo]&E[hi])}/{min(len(E[lo]),len(E[hi]))}")
print(" ".join(row))
