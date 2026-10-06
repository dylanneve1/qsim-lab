"""Pauli-image greedy growth from serial-prefix and layer cuts; score = gates absorbed under a term budget."""
import sys, time, numpy as np, gparse as G, struct_probe as SP, mirror as MI
f = sys.argv[1]; budget = int(sys.argv[2]); eps = float(sys.argv[3]); cuts = [float(x) for x in sys.argv[4:]]
n, units, tail = G.parse(f); L = SP.layers(n, units)
for c in cuts:
    t0 = time.time()
    W = MI.grow(n, units, c, budget, eps, layer=L)
    ks = sorted(W.inside); ls = [L[k] for k in ks]
    pm = W.perm(); single = sum(1 for q, (w, p) in pm.items() if p > 0.9); moved = sum(1 for q, (w, p) in pm.items() if p > 0.9 and w != q)
    print(f"c={c}: absorbed {len(ks)}, layers {min(ls) if ls else '-'}..{max(ls) if ls else '-'}, mean weight {W.weight()/(2*n):.2f}, single {single}/{n}, moved {moved}, {time.time()-t0:.1f}s", flush=True)
