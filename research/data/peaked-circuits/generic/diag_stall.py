import sys, numpy as np, gparse as G, struct_probe as SP, mirror as MI
f=sys.argv[1]; c=float(sys.argv[2]); budget=int(sys.argv[3]); eps=float(sys.argv[4])
n, units, tail = G.parse(f); L = SP.layers(n, units)
W = MI.grow(n, units, c, budget, eps, layer=L)
print("absorbed", len(W.inside), "size", W.size(), "weight", round(W.weight(),2))
lo = [L[W.seq[q][W.prv[q]]] if W.prv[q] >= 0 else -1 for q in range(n)]
hi = [L[W.seq[q][W.nxt[q]]] if W.nxt[q] < len(W.seq[q]) else 999 for q in range(n)]
print("per-wire window boundary layers (before-side last outside):", lo)
print("per-wire (after-side first outside):", hi)
aft, bef = W.frontier()
rows = []
for side, ks in (('after', aft), ('before', bef)):
    for k in ks:
        dw, dt, new = W.evaluate(side, k)
        rows.append((dw, dt, side, k, L[k], units[k][:2]))
rows.sort()
for r in rows[:12]: print("  cand", r)
# heavy images
ws = sorted(((W.wsum(S), g) for g, S in enumerate(W.img)), reverse=True)[:8]
print("heaviest images (weight, gen):", [(round(w,2), g) for w, g in ws])
