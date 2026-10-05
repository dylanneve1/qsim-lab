"""Circuit (exact, every branch) against the paper's success model (exact
evaluation of main1's model) at the same mask widths, N = 3127, g = 3122
(the paper's Figure 4 instance). Reads out/sweep_mask_f10_shor.csv and
out/model_n12.txt; prints a markdown table."""

from __future__ import annotations

import csv
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"

model = {}
for line in (OUT / "model_n12.txt").read_text().splitlines():
    m = re.search(r"W=(\d+) S=W/N=([0-9.]+) success=([0-9.]+)", line)
    if m:
        model[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
rows = list(csv.DictReader(open(OUT / "sweep_mask_f10_shor.csv")))
print("| mask bits | W (full units) | S = W/N | paper's model | circuit, exact arithmetic | circuit, approximate | approx. / model |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    mask = int(r["mask"])
    t = 2  # dropped bits at f = 10, n = 12
    wf = (1 << mask) << t
    s, mod = model.get(wf, (wf / 3127, float("nan")))
    a, i = float(r["succ_actual"]), float(r["succ_ideal"])
    print(f"| {mask} | {wf} | {s:.4f} | {mod:.4f} | {i:.4f} | {a:.4f} | {a / mod:.3f} |")
