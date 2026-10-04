#!/usr/bin/env python3
"""Learning curves (validation logical error per shot vs training shots) -> research/neural-decoder-curves.png.
Reads results/models/*/log.jsonl; resumed runs are offset by the shots of the checkpoint they resumed from."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

R = os.path.join(os.path.dirname(__file__), "results", "models")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3 (light surface)


def curve(runs):
    xs, ys = [], []
    for name, off in runs:
        for line in open(os.path.join(R, name, "log.jsonl")):
            r = json.loads(line)
            if "val_fails" in r:
                xs.append((r["shots"] + off) / 1e6); ys.append(r["val_fails"] / r["val_shots"])
    return xs, ys


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF)
ax = axes[0]
surf = [("d = 3", [("m_s3", 0)], 6640e-6, 1112 / 2e5, C[0]),
        ("d = 5", [("m_s5_cls", 0), ("m_s5_cls2", 7.68e6)], 3404e-6, 357 / 2e5, C[1]),
        ("d = 7", [("m_s7b", 0), ("m_s7d", 4.48e6)], 1457e-6, 54 / 1e5, C[2])]
for lab, runs, pm, ts, col in surf:
    x, y = curve(runs)
    ax.plot(x, y, color=col, lw=2, marker="o", ms=4, label=f"NN {lab}")
    ax.axhline(pm, color=col, lw=1.2, ls="--")
    ax.axhline(ts, color=col, lw=1.2, ls=":")
    ax.annotate(f"NN {lab}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", color=INK, fontsize=9, va="center")
ax.set_title("Surface code, p = 0.3%: NN vs PyMatching (--) and Tesseract (··)", fontsize=10, color=INK, loc="left")
ax2 = axes[1]
for lab, run, bp, col in [("K–F (d_circ 7)", "m_c9kf", 538 / 300032, C[0]), ("D8 (d_circ 8)", "m_c9g8", 319 / 300032, C[1])]:
    x, y = curve([(run, 0)])
    ax2.plot(x, y, color=col, lw=2, marker="o", ms=4, label=f"NN {lab}")
    ax2.axhline(bp, color=col, lw=1.2, ls="--")
    ax2.annotate(f"NN {lab}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", color=INK, fontsize=9, va="center")
ax2.set_title("Colour code d = 9, 9 rounds: NN vs BP+OSD (--)", fontsize=10, color=INK, loc="left")
for a in axes:
    a.set_xscale("log"); a.set_yscale("log"); a.set_facecolor(SURF)
    a.set_xlabel("training shots (millions)", color=INK2); a.set_ylabel("logical error per shot (validation)", color=INK2)
    a.grid(True, which="major", color=GRID, lw=0.8); a.tick_params(colors=INK2)
    for s in a.spines.values():
        s.set_color(GRID)
    a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right" if a is axes[0] else "center left")
    a.set_xlim(right=a.get_xlim()[1] * 2.2)
ax2.set_xticks([0.5, 1, 2, 3])
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "..", "neural-decoder-curves.png")
fig.savefig(out, dpi=130, facecolor=SURF)
print(out)
