#!/usr/bin/env python3
"""Recompute the features column of driver CSVs with the current binary
(features are deterministic in spec+seed; engine timings are untouched).
    refeature.py BIN IN.csv OUT.csv [OBS]"""
import csv, json, subprocess, sys
binp, src, dst = sys.argv[1:4]
obs = sys.argv[4] if len(sys.argv) > 4 else "all"
rows = list(csv.DictReader(open(src)))
cache = {}
for r in rows:
    k = (r["spec"], r["seed"])
    if k not in cache:
        out = subprocess.run([binp, "features", r["spec"], r["seed"], obs], capture_output=True,
                             text=True, check=True).stdout
        cache[k] = json.dumps(json.loads(out.strip().splitlines()[-1]))
    r["features"] = cache[k]
w = csv.DictWriter(open(dst, "w", newline=""), fieldnames=list(rows[0].keys()))
w.writeheader()
w.writerows(rows)
print(f"{len(cache)} instances refeatured -> {dst}")
