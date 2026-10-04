#!/usr/bin/env python3
"""Pool chunks per tag; per-round LER with Wilson CI; ratios vs a reference tag (log-ratio normal CI).
usage: summary.py <jsonl...> [--ref-map a=b,...]"""
import json, math, sys, collections
agg = collections.OrderedDict()
meta = {}
for f in [a for a in sys.argv[1:] if not a.startswith("--")]:
    for l in open(f):
        j = json.loads(l)
        k = j["tag"]
        a = agg.setdefault(k, [0, 0])
        a[0] += j["fails"]; a[1] += j["shots"]
        meta[k] = j
def per_round(P, R):
    return (1 - max(0.0, 1 - 2 * P) ** (1 / R)) / 2
def wilson(f, n, z=1.96):
    p = f / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h
rows = {}
for k, (f, n) in agg.items():
    R = meta[k]["rounds"]
    lo, hi = wilson(f, n)
    rows[k] = (f, n, per_round(f / n, R), per_round(lo, R), per_round(hi, R))
    print(f"{k:40s} fails {f:6d} / {n:9d}  p_L/round {rows[k][2]:.3e} [{rows[k][3]:.3e}, {rows[k][4]:.3e}]")
refs = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--ref-map=")]
if refs:
    for pair in refs[0].split(","):
        a, b = pair.split(":")
        if a in rows and b in rows and rows[a][0] and rows[b][0]:
            fa, na, ra = rows[a][:3]; fb, nb, rb = rows[b][:3]
            r = ra / rb
            s = math.sqrt(1 / fa + 1 / fb)
            print(f"ratio {a} / {b}: {r:.3f} [{r*math.exp(-1.96*s):.3f}, {r*math.exp(1.96*s):.3f}]")
