#!/usr/bin/env python3
"""Aggregate interleaved LER chunks (Mac, ler_chunks.sh) per (p, schedule); ratio with 95% CI
(log-ratio normal approximation, as research/data/qec-r4/ler_compare.py).
usage: ler_summary.py cg_ler_d9.jsonl"""
import sys, json, math, collections
agg = collections.defaultdict(lambda: [0, 0, 0])
for l in open(sys.argv[1]):
    if not l.startswith("{"):
        continue
    r = json.loads(l)["r"]
    a = agg[(r["p"], r["schedule"])]
    a[0] += r["shots"]; a[1] += r["fails"]; a[2] = r["rounds"]
def per_round(P, R):
    return (1 - (1 - 2 * P) ** (1 / R)) / 2
ps = sorted({p for p, _ in agg}, reverse=True)
for p in ps:
    k = agg[(p, "kf")]; n = agg[(p, "d9_global_D8.sched")]
    if not k[0] or not n[0]:
        continue
    Pk, Pn = k[1] / k[0], n[1] / n[0]
    lr = math.log(Pn / Pk); se = math.sqrt(1 / k[1] + 1 / n[1])
    print(f"| {p*100:.2f}% | {per_round(Pk, k[2]):.3e} ({k[1]} / {k[0]}) | {per_round(Pn, n[2]):.3e} ({n[1]} / {n[0]}) | "
          f"**{math.exp(lr):.3f} [{math.exp(lr-1.96*se):.3f}, {math.exp(lr+1.96*se):.3f}]** |")
