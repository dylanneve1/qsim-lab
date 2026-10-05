"""Circuit (exact, every branch) against the paper's success model (exact
evaluation of main1's model) at the same mask widths, N = 3127, g = 3122
(the paper's Figure 4 instance). Reads out/sweep_mask_f10_shor.csv,
out/model_n12.txt and the unmasked circuit value from out/dist.txt; prints a
markdown table with absolute success and suppression (masked / unmasked)."""

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
model_unmasked = model[1][1]
text = (OUT / "dist.txt").read_text()
blk = text[text.index("regs=[(22, 3122)]"):]
circ_unmasked = float(re.search(r"success\[paper\] .*unmasked=([0-9.]+)", blk).group(1))
rows = list(csv.DictReader(open(OUT / "sweep_mask_f10_shor.csv")))
print("| mask bits | W (units of N) | S = W/N | model | circuit, exact arith. | circuit, approx. | suppression: model / exact / approx. | 1 − S |")
print("|---|---|---|---|---|---|---|---|")
print(f"| unmasked | 1 | 0 | {model_unmasked:.4f} | {circ_unmasked:.4f} | — | 1 / 1 / — | 1 |")
for r in rows:
    mask = int(r["mask"])
    wf = (1 << mask) << 2  # t = 2 dropped bits at f = 10, n = 12
    s, mod = model.get(wf, (wf / 3127, float("nan")))
    a, i = float(r["succ_actual"]), float(r["succ_ideal"])
    print(
        f"| {mask} | {wf} | {s:.4f} | {mod:.4f} | {i:.4f} | {a:.4f} | "
        f"{mod / model_unmasked:.3f} / {i / circ_unmasked:.3f} / {a / circ_unmasked:.3f} | {1 - s:.3f} |"
    )
