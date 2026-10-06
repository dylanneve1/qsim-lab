import sys, time, numpy as np, gparse as G, struct_probe as SP, mirror as MI
f = sys.argv[1]; budget = int(sys.argv[2]); eps = float(sys.argv[3])
n, units, tail = G.parse(f); L = SP.layers(n, units)
for c in [float(x) for x in sys.argv[4:]]:
    t0 = time.time()
    W = MI.grow(n, units, c, budget, eps, log=100, layer=L)
    ks = sorted(W.inside); ls = [L[k] for k in ks]
    pm = W.perm(); single = sum(1 for q,(w,p) in pm.items() if p > 0.9); moved = sum(1 for q,(w,p) in pm.items() if p > 0.9 and w != q)
    print(f"c={c}: absorbed {len(ks)} gates, layer span {min(ls) if ls else None}..{max(ls) if ls else None}, size {W.size()}, weight {W.weight():.1f}, loss {W.loss():.3e}, {time.time()-t0:.1f}s; Z single-wire {single}/{n} (moved {moved})", flush=True)
