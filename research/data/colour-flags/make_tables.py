#!/usr/bin/env python3
"""Markdown LER tables from campaign jsonl files (pooled per tag, deduplicated by seed).
Per-round p_L = (1 - (1-2P)^(1/R))/2 with Wilson 95% CI; ratio vs the K-F arm of the same (d, p)
with the log-ratio normal CI.  usage: make_tables.py <prefix> <jsonl...>"""
import json, math, sys, collections
prefix = sys.argv[1]
agg, meta, seen = collections.OrderedDict(), {}, set()
for f in sys.argv[2:]:
    for l in open(f):
        j = json.loads(l)
        if not j["tag"].startswith(prefix) or (j["tag"], j["seed"]) in seen:
            continue
        seen.add((j["tag"], j["seed"]))
        a = agg.setdefault(j["tag"], [0, 0]); a[0] += j["fails"]; a[1] += j["shots"]; meta[j["tag"]] = j
pr = lambda P, R: (1 - max(0.0, 1 - 2 * P) ** (1 / R)) / 2
def wilson(f, n, z=1.96):
    p = f / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h
NAMES = {"kf": "K–F", "hf": "HF (flags + hook-free schedule)", "kfflag": "K–F + boundary flags", "d8": "global D8 (colour-global)"}
rows = collections.defaultdict(dict)
for t, (f, n) in agg.items():
    _, d, p, arm = t.split("_", 3)
    rows[(int(d[1:]), float(p[1:]))][arm] = (f, n, meta[t]["rounds"])
print("| d | p | arm | fails / shots | p_L per round [95% CI] | ratio vs K–F [95% CI] |")
print("|---|---|---|---|---|---|")
for (d, p) in sorted(rows):
    r = rows[(d, p)]
    for arm in ["kf", "hf", "kfflag", "d8"]:
        if arm not in r:
            continue
        f, n, R = r[arm]
        lo, hi = wilson(f, n)
        s = f"| {d} | {p*100:.1f}% | {NAMES[arm]} | {f} / {n:,} | {pr(f/n,R):.2e} [{pr(lo,R):.2e}, {pr(hi,R):.2e}] |"
        if arm != "kf" and "kf" in r and f and r["kf"][0]:
            fk, nk, _ = r["kf"]
            q = pr(f / n, R) / pr(fk / nk, R)
            sd = math.sqrt(1 / f + 1 / fk)
            s += f" **{q:.2f}** [{q*math.exp(-1.96*sd):.2f}, {q*math.exp(1.96*sd):.2f}] |"
        elif arm != "kf" and "kf" in r and not f:
            s += " 0 fails |"
        else:
            s += " — |"
        print(s)
