#!/usr/bin/env python3
"""Picks each framework's fastest knob setting per (workload, precision) from a sweep CSV.

usage: pick_best.py <sweep.csv> <best.json> [--report sweep_summary.md]

For every (framework, workload, prec, file variant) the config with the smallest min time
is kept; for SU(4) workloads both file variants (decomposed `.txt`, dense `.dense.txt`)
keep their best config, so the matrix runs each baseline on both forms. qsim also keeps
its second-best fusion size for the best variant (the best size can shift with n).
"""
import csv
import json
import sys
from collections import defaultdict


def suffix(fname):
    return ".dense.txt" if fname.endswith(".dense.txt") else ".txt"


def knobs(fw, config):
    if fw == "qsimlab" or config in ("default", ""):
        return {}
    out = {}
    for kv in config.split(","):
        k, v = kv.split("=", 1)
        out[k] = int(v) if v.lstrip("-").isdigit() else v
    return out


def main():
    rows = list(csv.DictReader(open(sys.argv[1])))
    mins = defaultdict(lambda: float("inf"))
    for r in rows:
        key = (r["framework"], r["workload"], r["prec"], suffix(r["file"]), r["config"])
        mins[key] = min(mins[key], float(r["seconds"]))
    by = defaultdict(list)
    for (fw, wl, prec, suf, cfg), t in mins.items():
        by[(fw, wl, prec, suf)].append((t, cfg))
    best = defaultdict(list)
    lines = ["| framework | workload | prec | file | config: min s (sorted) |", "|---|---|---|---|---|"]
    for (fw, wl, prec, suf), lst in sorted(by.items()):
        lst.sort()
        lines.append(f"| {fw} | {wl} | {prec} | {suf} | " +
                     ", ".join(f"{c}: {t:.4f}" for t, c in lst) + " |")
        if fw == "qsimlab":
            continue
        best[f"{fw}|{wl}|{prec}"].append((lst[0][0], suf, knobs(fw, lst[0][1]), lst))
    out = {}
    for key, entries in best.items():
        entries.sort(key=lambda e: e[0])
        sel = [[e[1], e[2]] for e in entries]
        fw = key.split("|")[0]
        if fw == "qsim" and len(entries[0][3]) > 1:
            sel.insert(1, [entries[0][1], knobs(fw, entries[0][3][1][1])])
        out[key] = sel
    json.dump(out, open(sys.argv[2], "w"), indent=1, sort_keys=True)
    if "--report" in sys.argv:
        open(sys.argv[sys.argv.index("--report") + 1], "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
