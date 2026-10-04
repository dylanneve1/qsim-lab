#!/usr/bin/env python3
"""Roofline plot for research/metal.md (reads summary.csv, writes roofline.png).

x: operational intensity = amplitude-updates of the fused op list per DRAM byte,
   DRAM bytes = passes * 16 * 2^n (one read + one write of every float2 per pass).
y: achieved amplitude-updates per second.
Ceilings: measured streaming bandwidth (in-place scale kernel / rayon loop) and,
for the GPU, the measured in-register 1q-gate rate (synthetic H layers).
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(here, "summary.csv"))))

BW_GPU = 178e9  # bytes/s, scale4 kernel n=29
BW_CPU = 162e9  # bytes/s, rayon in-place scale n=29, 8 threads
GPU_COMPUTE = 137e9  # amp-updates/s, register-local 1q gates (prof5.out stages 1-2)

fig, ax = plt.subplots(figsize=(7.5, 5))
xs = [10 ** (k / 40) for k in range(-40, 29)]
ax.plot(xs, [min(BW_GPU * x, GPU_COMPUTE) for x in xs], color="#1f6feb", lw=1.5,
        label="GPU roof: 178 GB/s, 137 G upd/s (in-register 1q)")
ax.plot(xs, [BW_CPU * x for x in xs], color="#d9480f", lw=1.5, ls="--",
        label="CPU memory roof: 162 GB/s")
style = {
    ("gpu", "qft"): dict(color="#1f6feb", marker="o"),
    ("gpu", "brick"): dict(color="#1f6feb", marker="s"),
    ("cpu", "qft"): dict(color="#d9480f", marker="o", mfc="none"),
    ("cpu", "brick"): dict(color="#d9480f", marker="s", mfc="none"),
}
for (dev, wl), st in style.items():
    pts = [r for r in rows if r["mode"] == dev and r["workload"] == wl]
    pts.sort(key=lambda r: int(r["n"]))
    x = [float(r["amp_ops"]) / (int(r["passes"]) * 16 * 2 ** int(r["n"])) for r in pts]
    y = [float(r["amp_ops"]) / float(r["seconds"]) for r in pts]
    ax.plot(x, y, ls="none", ms=7, label=f"{dev.upper()} {wl} (n={pts[0]['n']}..{pts[-1]['n']})", **st)
    for xi, yi, r in zip(x, y, pts):
        ax.annotate(r["n"], (xi, yi), textcoords="offset points", xytext=(5, -3), fontsize=7,
                    color=st["color"])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("amplitude updates per DRAM byte")
ax.set_ylabel("amplitude updates per second")
ax.set_xlim(0.1, 5)
ax.set_ylim(5e9, 3e11)
ax.grid(True, which="both", alpha=0.25)
ax.set_title("qsim-lab on Apple M1 Pro (f32): Metal GPU vs NEON CPU, 8 threads")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(here, "roofline.png"), dpi=130)
print("wrote roofline.png")
