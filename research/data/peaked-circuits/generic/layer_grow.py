"""Symmetric layer-synchronous absorption around centre c (ASAP layers): window = layers [lo, hi).
Report mean image weight / size after each layer pair."""
import sys, time, numpy as np, gparse as G, struct_probe as SP, mirror as MI
f = sys.argv[1]; c = float(sys.argv[2]); eps = float(sys.argv[3]); maxsize = int(sys.argv[4])
n, units, tail = G.parse(f); L = SP.layers(n, units)
D = max(L) + 1
W = MI.Window(n, units, c, eps, layer=L)
bylayer = [[] for _ in range(D)]
for k in range(len(units)): bylayer[L[k]].append(k)
lo = hi = int(np.ceil(c))  # window layers [lo, hi)
t0 = time.time()
while lo > 0 or hi < D:
    for side in ('after', 'before'):
        if side == 'after' and hi < D:
            for k in bylayer[hi]: W.commit(W.cand_after(k), 'after', k)
            hi += 1
        elif side == 'before' and lo > 0:
            for k in bylayer[lo - 1]: W.commit(W.cand_before(k), 'before', k)
            lo -= 1
    W.cache.clear()
    wt = W.weight() / (2 * n)
    print(f"  layers [{lo},{hi}) gates {len(W.inside)} mean weight {wt:.3f} size {W.size()} ({time.time()-t0:.1f}s)", flush=True)
    if W.size() > maxsize: break
