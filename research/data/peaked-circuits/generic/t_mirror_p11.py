import sys, time, numpy as np, gparse as G, struct_probe as SP, mirror as MI
f = sys.argv[1]; budget = int(sys.argv[2]); eps = float(sys.argv[3])
n, units, tail = G.parse(f)
segs = SP.segments(n, units); anc, _ = SP.anchors(segs, False)
cen = np.array([(segs[i][2] + segs[j][1]) / 2 for i, j in anc])
print("anchor centre (serial idx) quantiles:", np.percentile(cen, [0, 10, 25, 50, 75, 90, 100]).round(1).tolist())
h = np.histogram(cen, bins=40, range=(0, len(units)))[0]; print("hist40:", h.tolist())
for c in [int(x) for x in sys.argv[4:]]:
    t0 = time.time()
    W = MI.grow(n, units, c, budget, eps, log=100)
    ks = sorted(W.inside)
    print(f"c={c}: absorbed {len(ks)} gates, serial span {ks[0] if ks else None}..{ks[-1] if ks else None}, size {W.size()}, loss {W.loss():.3e}, {time.time()-t0:.1f}s", flush=True)
    pm = W.perm(); single = sum(1 for q,(w,p) in pm.items() if p > 0.9)
    print(f"   Z images dominated (>0.9) by single-wire op: {single}/{n}")
