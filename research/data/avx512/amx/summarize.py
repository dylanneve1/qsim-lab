#!/usr/bin/env python3
"""Summarise run.sh output: per (k, n, threads, method) the min over processes
of the per-process min time, Gamp/s, and the ratio to avx512-f32."""
import sys
from collections import defaultdict

best = defaultdict(lambda: float("inf"))
for line in open(sys.argv[1]):
    f = [x.strip() for x in line.strip().strip("|").split("|")]
    if len(f) < 6 or not f[0].isdigit():
        continue
    k, n, t, m, s = int(f[0]), int(f[1]), int(f[2]), f[3], float(f[4])
    best[(k, n, t, m)] = min(best[(k, n, t, m)], s)
print("| threads | n | k | avx512-f32 Gamp/s | amx-bf16x3 | amx-bf16x2 | amx-bf16x1 |")
print("|---|---|---|---|---|---|---|")
for (t, n, k) in sorted({(t, n, k) for (k, n, t, _) in best}):
    base = best[(k, n, t, "avx512-f32")]
    cells = []
    for m in ["avx512-f32", "amx-bf16x3", "amx-bf16x2", "amx-bf16x1"]:
        v = best.get((k, n, t, m))
        if v is None:
            cells.append("")
            continue
        g = (1 << n) / v / 1e9
        cells.append(f"{g:.3f}" if m == "avx512-f32" else f"{g:.3f} ({base / v:.2f}x)")
    print(f"| {t} | {n} | {k} | " + " | ".join(cells) + " |")
