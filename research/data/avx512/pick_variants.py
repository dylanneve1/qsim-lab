#!/usr/bin/env python3
"""From the n = 24 matrix rows, the faster SU(4) file variant per (framework, workload,
precision) for the slow frameworks, used at n >= 26 (make_matrix.py --variants).

usage: pick_variants.py <baselines_raw.csv> <variants.json>
"""
import csv
import json
import sys
from collections import defaultdict

mins = defaultdict(lambda: float("inf"))
for r in csv.DictReader(open(sys.argv[1])):
    if r["n"] != "24" or r["workload"] not in ("qv", "brick_su4"):
        continue
    if r["framework"] not in ("aer", "lightning", "qulacs_src"):
        continue
    v = ".dense.txt" if r["file"].endswith(".dense.txt") else ".txt"
    k = (r["framework"], r["workload"], r["prec"], v)
    mins[k] = min(mins[k], float(r["seconds"]))
out = {}
for (fw, wl, prec, v), t in mins.items():
    key = f"{fw}|{wl}|{prec}"
    other = ".txt" if v == ".dense.txt" else ".dense.txt"
    if t <= mins.get((fw, wl, prec, other), float("inf")):
        out[key] = v
json.dump(out, open(sys.argv[2], "w"), indent=1, sort_keys=True)
print(json.dumps({k: v for k, v in sorted(out.items())}, indent=1))
for k in sorted(mins):
    print(k, f"{mins[k]:.3f}")
