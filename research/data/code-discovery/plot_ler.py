"""Per-logical-qubit logical error per round vs p (Z memory, best schedules).

usage: python plot_ler.py ler.jsonl out.png
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rows = [json.loads(l) for l in open(sys.argv[1])]
best = {28: "-034521/452013-", 56: "-034521/452103-"}  # by group order l*m
series = {}
for r in rows:
    if r["basis"] != "z":
        continue
    nn = r["l"] * r["m"]
    if nn in best and r["sched"] != best[nn]:
        continue
    if nn not in best and r["n"] not in (72,):
        continue
    key = (f"[[{r['n']},{r['k']}]]", r["osd_order"])
    lo, hi = r["ci95_round"]
    series.setdefault(key, []).append((r["p"], r["p_L_round"] / r["k"], lo / r["k"], hi / r["k"]))
fig, ax = plt.subplots(figsize=(6, 4.4))
names = {"[[56,6]]": "[[56,6,8]] published", "[[112,12]]": "[[112,12,8]] (found)", "[[72,12]]": "[[72,12,6]] IBM"}
colors = {"[[56,6]]": "#d1495b", "[[112,12]]": "#2a6fdb", "[[72,12]]": "#888888"}
for (code, osd), v in sorted(series.items()):
    v.sort()
    ps = [x[0] for x in v]
    ys = [x[1] for x in v]
    err = [[x[1] - x[2] for x in v], [x[3] - x[1] for x in v]]
    ax.errorbar(ps, ys, yerr=err, marker="o" if osd == 10 else "s", ls="-" if osd == 10 else "--",
                color=colors[code], capsize=3, label=f"{names[code]}, OSD-{osd}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("physical error rate p (uniform circuit noise)")
ax.set_ylabel("logical error per logical qubit per round")
ax.set_title("same n/k and d: connected [[112,12,8]] vs 2x[[56,6,8]]")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=130)
