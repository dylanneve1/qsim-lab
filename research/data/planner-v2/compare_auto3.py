#!/usr/bin/env python3
"""Strategy::Auto with the ski-rental exploration past a switch (research/
planner-v2.md §5) against the round-4 Auto (auto2) on the same instances.
Control: frame and dense from the round-4 re-time session
(research/data/planner/mac/retime_skirental_*.csv). Mac, one thread.

    compare_auto3.py
"""
import csv, glob, os
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(H, "..", "planner", "mac")


def load(paths, eng):
    out = {}
    for p in paths:
        for r in csv.DictReader(open(p)):
            if r["engine"] in eng:
                out.setdefault((r["spec"], r["seed"]), {})[r["engine"]] = (
                    float(r["secs"]) if r["status"] == "ok" else None)
    return out


ctl = load(glob.glob(os.path.join(P, "retime_skirental_*.csv")), ("frame", "dense"))
a2 = load(glob.glob(os.path.join(P, "auto2_*.csv")), ("auto",))
a3 = load(glob.glob(os.path.join(H, "mac", "auto3", "*.csv")), ("auto",))
rows = []
for k in sorted(set(ctl) & set(a2) & set(a3)):
    b = [t for t in ctl[k].values() if t is not None]
    t2, t3 = a2[k].get("auto"), a3[k].get("auto")
    if not b or t2 is None or t3 is None:
        continue
    rows.append((k[0], min(b), t2, t3))
x2 = np.array([r[2] / r[1] for r in rows])
x3 = np.array([r[3] / r[1] for r in rows])
big = np.array([r[1] >= 1e-3 for r in rows])
out = []
for name, x in (("round-4 Auto (auto2)", x2), ("v2 Auto (auto3)", x3)):
    out.append(f"{name}: n={len(x)} geo={10 ** np.mean(np.log10(x)):.3f} median={np.median(x):.3f} "
               f"p90={np.percentile(x, 90):.2f} max={x.max():.1f} | best>=1ms n={big.sum()} "
               f"geo={10 ** np.mean(np.log10(x[big])):.3f} max={x[big].max():.1f}")
out.append("worst v2 Auto:")
for r in sorted(rows, key=lambda r: -r[3] / r[1])[:8]:
    out.append(f"  {r[3] / r[1]:8.1f}x {r[0]}  auto3 {r[3]:.4g}s  auto2 {r[2]:.4g}s  best(frame,dense) {r[1]:.4g}s")
out.append("largest changes (auto2/auto3):")
for r in sorted(rows, key=lambda r: -max(r[2] / r[3], r[3] / r[2]))[:10]:
    out.append(f"  {r[2] / r[3]:8.2f}x {r[0]}  auto2 {r[2]:.4g}s -> auto3 {r[3]:.4g}s  best {r[1]:.4g}s")
txt = "\n".join(out)
print(txt)
open(os.path.join(H, "auto3_compare.txt"), "w").write(txt + "\n")
