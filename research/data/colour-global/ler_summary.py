#!/usr/bin/env python3
"""Aggregate interleaved LER chunks per (p, schedule): ratio new / K-F with 95% CI (log-ratio
normal approximation, as research/data/qec-r4/ler_compare.py). Accepts color_ler chunk lines
({"chunk":..,"r":{...}}, ler_chunks.sh / ler2.sh) and tesseract_ler.py lines.
usage: ler_summary.py file.jsonl [file2.jsonl ...]"""
import sys, json, math, collections, os
agg = collections.defaultdict(lambda: [0, 0, 0])
for fn in sys.argv[1:]:
    for l in open(fn):
        if not l.startswith("{"):
            continue
        r = json.loads(l)
        r = r.get("r", r)
        a = agg[(r["p"], os.path.basename(r["schedule"]))]
        a[0] += r["shots"]; a[1] += r["fails"]; a[2] = r["rounds"]
def per_round(P, R):
    return (1 - (1 - 2 * P) ** (1 / R)) / 2
for p in sorted({p for p, _ in agg}, reverse=True):
    k = agg[(p, "kf")]; n = agg[(p, "d9_global_D8.sched")]
    if not k[0] or not n[0] or not k[1] or not n[1]:
        continue
    Pk, Pn = k[1] / k[0], n[1] / n[0]
    lr = math.log(Pn / Pk); se = math.sqrt(1 / k[1] + 1 / n[1])
    print(f"| {p*100:.2f}% | {per_round(Pk, k[2]):.3e} ({k[1]} / {k[0]}) | {per_round(Pn, n[2]):.3e} ({n[1]} / {n[0]}) | "
          f"**{math.exp(lr):.3f} [{math.exp(lr-1.96*se):.3f}, {math.exp(lr+1.96*se):.3f}]** |")
