#!/usr/bin/env python3
"""Summarises baselines_raw.csv: min seconds per framework per (workload, n, prec, threads),
the best baseline, qsim-lab, and the load during the runs.

usage: summarize.py <raw.csv> [more.csv ...] [--md out.md] [--others-max 4] [--load-max 16]

A row is "clean" when the 1-min load before its run was <= 16 (MACHINE.md) and other
processes used <= --others-max cores on average during the run. A framework's entry is the
min over its clean rows (all configs and file variants; the chosen one is printed); if it
has no clean row, the min over all rows is shown with a "*" (loaded).
"""
import csv
import sys
from collections import defaultdict

FWS = ["qsim", "qulacs_src", "qulacs", "aer", "lightning", "qsimlab"]


def main():
    args = sys.argv[1:]
    opt = {}
    for k in ("--md", "--others-max", "--load-max"):
        if k in args:
            i = args.index(k)
            opt[k] = args[i + 1]
            del args[i:i + 2]
    others_max = float(opt.get("--others-max", 4))
    load_max = float(opt.get("--load-max", 16))
    rows = []
    for p in args:
        rows += list(csv.DictReader(open(p)))
    best = {}
    for r in rows:
        cell = (r["workload"], int(r["n"]), r["prec"], int(r["threads"]))
        t = float(r["seconds"])
        clean = float(r["load1_before"]) <= load_max and float(r["others_cores"]) <= others_max
        k = cell + (r["framework"],)
        su4 = r["workload"] in ("qv", "brick_su4")
        variant = ("dense" if r["file"].endswith(".dense.txt") else "kak") if su4 else ""
        cand = (not clean, t, r["config"] + (f" {variant}" if variant else ""),
                float(r["load1_before"]), float(r["others_cores"]))
        if k not in best or cand < best[k]:
            best[k] = cand
    cells = sorted({k[:4] for k in best}, key=lambda c: (c[1], c[2], c[3], c[0]))
    head = ("| workload | n | prec | thr | " + " | ".join(FWS) +
            " | best baseline | best / qsim-lab | load1 / others (best baseline run) | load1 / others (qsim-lab run) |")
    out = [head, "|---|---|---|---|" + "---|" * len(FWS) + "---|---|---|---|"]
    for c in cells:
        vals, bb = [], None
        for fw in FWS:
            v = best.get(c + (fw,))
            if v is None:
                vals.append("—")
                continue
            vals.append(f"{v[1]:.3f}{'*' if v[0] else ''} ({v[2]})")
            if fw != "qsimlab" and (bb is None or (v[0], v[1]) < (bb[1][0], bb[1][1])):
                bb = (fw, v)
        ql = best.get(c + ("qsimlab",))
        ratio = f"{bb[1][1] / ql[1]:.2f}x" if (bb and ql) else "—"
        bbs = f"{bb[0]} {bb[1][1]:.3f}{'*' if bb[1][0] else ''}" if bb else "—"
        lb = f"{bb[1][3]:.1f} / {bb[1][4]:.1f}" if bb else "—"
        lq = f"{ql[3]:.1f} / {ql[4]:.1f}" if ql else "—"
        out.append(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | " + " | ".join(vals) +
                   f" | {bbs} | {ratio} | {lb} | {lq} |")
    text = ("\n".join(out) + "\n\n* = no clean run (load1 > %g or other processes > %g cores during "
            "every run of that framework in that cell); ratio > 1 means qsim-lab is faster.\n"
            % (load_max, others_max))
    print(text)
    if "--md" in opt:
        open(opt["--md"], "w").write(text)


if __name__ == "__main__":
    main()
