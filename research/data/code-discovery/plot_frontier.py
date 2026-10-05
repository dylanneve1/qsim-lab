"""Pareto plots: this search's frontier (exact d) vs the literature table.

usage: python plot_frontier.py frontier.json literature.md out.png
"""
import json
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ours = json.load(open(sys.argv[1]))
sec = open(sys.argv[2]).read().split("## Merged Pareto frontier")[1].split("## Weight-8")[0]
lit = [(int(a), int(b), int(c), s == "*") for a, b, c, s in re.findall(r"\[\[(\d+),(\d+),(\d+)(\*?)\]\]", sec)]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
ax.scatter([r["n"] for r in ours], [r["d"] for r in ours], s=[6 * r["k"] for r in ours],
           facecolors="none", edgecolors="#2a6fdb", label="this search (frontier, exact d; size ~ k)")
ax.scatter([x[0] for x in lit], [x[2] for x in lit], s=8, c="#d1495b", marker="x",
           label="literature frontier (x)")
ax.set_xlabel("n (data qubits)")
ax.set_ylabel("d")
ax.set_title("weight-6 two-block codes: (n, d) frontier points")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)
ax = axes[1]
best = {}
for r in ours:
    v = r["k"] * r["d"] ** 2 / r["n"]
    best[r["n"]] = max(best.get(r["n"], 0), v)
lbest = {}
for n, k, d, s in lit:
    lbest[n] = max(lbest.get(n, 0), k * d * d / n)
xs = sorted(best)
ax.plot(xs, [best[n] for n in xs], "o-", ms=3, color="#2a6fdb", label="this search: max k d^2/n")
xl = sorted(lbest)
ax.plot(xl, [lbest[n] for n in xl], "x", ms=5, color="#d1495b", label="literature: max k d^2/n")
ax.set_xlabel("n")
ax.set_ylabel("k d^2 / n")
ax.set_title("best k d^2 / n at each n")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(sys.argv[3], dpi=130)
