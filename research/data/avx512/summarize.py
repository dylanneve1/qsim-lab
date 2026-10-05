#!/usr/bin/env python3
"""Summarises baselines_raw.csv: min seconds per framework per (workload, n, prec, threads),
the best baseline, qsim-lab, and the load during the cell.

usage: summarize.py <raw.csv> [more.csv ...] [--md out.md]
A framework's entry is the min over all its configs and file variants (the chosen one
is printed); "loaded" marks cells whose load1 before the cell exceeded 16 or where other
processes used > 4 cores on average during the cell.
"""
import csv
import sys
from collections import defaultdict

FWS = ["qsim", "qulacs", "aer", "lightning", "qsimlab"]


def main():
    args = sys.argv[1:]
    md = None
    if "--md" in args:
        md = args[args.index("--md") + 1]
        args = args[:args.index("--md")]
    rows = []
    for p in args:
        rows += list(csv.DictReader(open(p)))
    best = {}
    load = defaultdict(lambda: [0.0, 0.0, 0.0])
    for r in rows:
        cell = (r["workload"], int(r["n"]), r["prec"], int(r["threads"]))
        t = float(r["seconds"])
        k = cell + (r["framework"],)
        variant = "dense" if r["file"].endswith(".dense.txt") else "kak" if r["workload"] in ("qv", "brick_su4") else ""
        if k not in best or t < best[k][0]:
            best[k] = (t, r["config"] + (f" {variant}" if variant else ""))
        L = load[cell]
        L[0] = max(L[0], float(r["load1_before"]))
        L[1] = max(L[1], float(r["others_cores"]))
        L[2] = max(L[2], float(r.get("others_idle") or 0.0))
    cells = sorted({k[:4] for k in best}, key=lambda c: (c[1], c[2], c[3], c[0]))
    out = ["| workload | n | prec | thr | " + " | ".join(FWS) + " | best baseline | qsim-lab / best | load1 max | others cores max |",
           "|---|---|---|---|" + "---|" * len(FWS) + "---|---|---|---|"]
    for c in cells:
        vals = []
        bb = None
        for fw in FWS:
            v = best.get(c + (fw,))
            if v is None:
                vals.append("—")
                continue
            vals.append(f"{v[0]:.3f} ({v[1]})")
            if fw != "qsimlab" and (bb is None or v[0] < bb[0]):
                bb = (v[0], fw)
        ql = best.get(c + ("qsimlab",))
        ratio = f"{bb[0] / ql[0]:.2f}x" if (bb and ql) else "—"
        L = load[c]
        flag = " loaded" if (L[0] > 16 or L[1] > 4) else ""
        bbs = f"{bb[1]} {bb[0]:.3f}" if bb else "—"
        out.append(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | " + " | ".join(vals) +
                   f" | {bbs} | {ratio} | {L[0]:.1f} | {L[1]:.1f}{flag} |")
    text = "\n".join(out) + "\n"
    print(text)
    if md:
        open(md, "w").write(text)


if __name__ == "__main__":
    main()
