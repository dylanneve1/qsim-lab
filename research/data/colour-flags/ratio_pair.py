#!/usr/bin/env python3
"""Ratio of per-round p_L between two arms of the same (d, p): A / B with log-ratio normal 95% CI.
usage: ratio_pair.py <prefix> <armA> <armB> <jsonl...>"""
import json, sys, math, collections
prefix, A, B = sys.argv[1:4]
agg, seen, R = collections.defaultdict(lambda: [0, 0]), set(), {}
for f in sys.argv[4:]:
    for l in open(f):
        j = json.loads(l)
        if not j["tag"].startswith(prefix) or (j["tag"], j["seed"]) in seen:
            continue
        seen.add((j["tag"], j["seed"]))
        a = agg[j["tag"]]; a[0] += j["fails"]; a[1] += j["shots"]; R[j["tag"]] = j["rounds"]
pr = lambda P, r: (1 - max(0.0, 1 - 2 * P) ** (1 / r)) / 2
keys = sorted({t.rsplit("_", 1)[0] for t in agg})
for k in keys:
    a, b = agg.get(f"{k}_{A}"), agg.get(f"{k}_{B}")
    if not a or not b or not a[0] or not b[0]:
        continue
    r = pr(a[0] / a[1], R[f"{k}_{A}"]) / pr(b[0] / b[1], R[f"{k}_{B}"])
    s = math.sqrt(1 / a[0] + 1 / b[0])
    print(f"{k}: {A}/{B} = {r:.2f} [{r*math.exp(-1.96*s):.2f}, {r*math.exp(1.96*s):.2f}]  ({a[0]} vs {b[0]} fails)")
