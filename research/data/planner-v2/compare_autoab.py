#!/usr/bin/env python3
"""Strategy::Auto A/B in one Mac session (research/planner-v2.md §5): `auto`
(v2: ski-rental exploration past a switch, explore_frac 0.3) vs `auto0`
(the round-4 logic, explore_frac 0), same binary (incl. the frame store's
small-rotation change), each instance back to back in one worker. Control
for "x the best": frame/dense of the round-4 re-time session.
    compare_autoab.py
"""
import csv, glob, os
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(H, "..", "planner", "mac")
ab, ctl = {}, {}
for p in glob.glob(os.path.join(H, "mac", "autoab", "*.csv")):
    for r in csv.DictReader(open(p)):
        ab.setdefault((r["spec"], r["seed"]), {})[r["engine"]] = (r["status"], float(r["secs"]) if r["status"] == "ok" else 20.0, r["note"])
for p in glob.glob(os.path.join(P, "retime_skirental_*.csv")):
    for r in csv.DictReader(open(p)):
        if r["engine"] in ("frame", "dense") and r["status"] == "ok":
            k = (r["spec"], r["seed"])
            ctl[k] = min(ctl.get(k, 1e9), float(r["secs"]))
rows = [(k[0], v["auto"][1], v["auto0"][1], ctl.get(k)) for k, v in ab.items() if "auto" in v and "auto0" in v]
r = np.array([a / b for _, a, b, _ in rows])
big = np.array([max(a, b) >= 1e-3 for _, a, b, _ in rows])
out = [f"auto/auto0 time ratio: n={len(r)} geo={10 ** np.mean(np.log10(r)):.3f} median={np.median(r):.3f} "
       f"min={r.min():.4f} max={r.max():.1f}; max(auto,auto0)>=1ms: n={big.sum()} geo={10 ** np.mean(np.log10(r[big])):.3f}"]
for name, i in (("auto (v2)", 1), ("auto0 (round 4)", 2)):
    x = np.array([row[i] / row[3] for row in rows if row[3]])
    b = np.array([row[3] >= 1e-3 for row in rows if row[3]])
    out.append(f"{name} vs best(frame,dense) of the round-4 session: n={len(x)} geo={10 ** np.mean(np.log10(x)):.3f} "
               f"max={x.max():.1f} | best>=1ms n={b.sum()} geo={10 ** np.mean(np.log10(x[b])):.3f} max={x[b].max():.1f}")
out.append("largest changes (auto0 / auto):")
for s, a, b, c in sorted(rows, key=lambda t: -max(t[1] / t[2], t[2] / t[1]))[:12]:
    out.append(f"  {b / a:8.2f}x  {s}  auto0 {b:.4g}s -> auto {a:.4g}s  (round-4 best frame/dense {c if c else float('nan'):.4g}s)")
txt = "\n".join(out)
print(txt)
open(os.path.join(H, "autoab_compare.txt"), "w").write(txt + "\n")
