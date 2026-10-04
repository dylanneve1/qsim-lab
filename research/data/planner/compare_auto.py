#!/usr/bin/env python3
"""Strategy::Auto before/after the ski-rental guard (research/planner.md §4).

Old: research/data/simulability/raw (Mac, same instances, old Auto).
New: Mac re-time of frame, dense and auto with the fixed Auto (same session
for all three, so frame/dense are the control).
    compare_auto.py NEW_CSV [NEW_CSV ...]
"""
import csv, glob, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "simulability", "raw")


def load(paths):
    out = {}
    for p in paths:
        for r in csv.DictReader(open(p)):
            k = (r["spec"], r["seed"])
            out.setdefault(k, {})[r["engine"]] = (r["status"], float(r["secs"]) if r["status"] == "ok" else None)
    return out


def ratios(d):
    res = []
    for k, v in d.items():
        a = v.get("auto")
        best = [x[1] for e, x in v.items() if e in ("frame", "dense") and x[0] == "ok"]
        if a and a[0] == "ok" and best:
            res.append((a[1] / min(best), k, a[1], min(best)))
    return res


def summ(name, rs):
    x = np.array([r[0] for r in rs])
    ge = 10 ** np.mean(np.log10(x))
    worst = sorted(rs, reverse=True)[:5]
    # only instances where the best takes >= 1 ms
    big = np.array([r[0] for r in rs if r[3] >= 1e-3])
    print(f"{name}: n={len(x)} geo={ge:.3f} median={np.median(x):.3f} p90={np.percentile(x, 90):.2f} "
          f"max={x.max():.1f} | best>=1ms: n={len(big)} geo={10 ** np.mean(np.log10(big)):.3f} max={big.max():.1f}")
    for w in worst:
        print(f"   {w[0]:8.1f}x  {w[1][0]}  auto {w[2]:.4g}s  best {w[3]:.4g}s")
    return dict(n=len(x), geo=float(ge), median=float(np.median(x)), max=float(x.max()),
                geo_ge1ms=float(10 ** np.mean(np.log10(big))), max_ge1ms=float(big.max()),
                worst=[(float(w[0]), w[1][0], w[2], w[3]) for w in worst])


def main():
    old = load([p for p in glob.glob(os.path.join(RAW, "*.csv")) if "mid2" not in p and "confirm" not in p])
    new = load(sys.argv[1:])
    common = set(old) & set(new)
    o = {k: old[k] for k in common}
    n = {k: new[k] for k in common}
    # old auto measured against the NEW session's frame/dense (control), and
    # against its own session
    rep = {"old_own_session": summ("old Auto vs old frame/dense", ratios(o)),
           "new": summ("new Auto vs new frame/dense", ratios(n))}
    mixed = {}
    for k in common:
        if "auto" in o[k]:
            mixed[k] = dict(n[k])
            mixed[k]["auto"] = o[k]["auto"]
    rep["old_vs_new_control"] = summ("old Auto vs new frame/dense", ratios(mixed))
    # per-instance speed-up of auto
    sp = [o[k]["auto"][1] / n[k]["auto"][1] for k in common
          if o[k].get("auto", ("",))[0] == "ok" and n[k].get("auto", ("",))[0] == "ok"
          and max(o[k]["auto"][1], n[k]["auto"][1]) >= 1e-3]
    sp = np.array(sp)
    print(f"auto old/new time ratio (>=1ms): n={len(sp)} geo={10 ** np.mean(np.log10(sp)):.3f} "
          f"min={sp.min():.3f} max={sp.max():.1f}")
    # control drift: frame and dense, new vs old
    for e in ("frame", "dense"):
        r = [o[k][e][1] / n[k][e][1] for k in common if o[k].get(e, ("",))[0] == "ok"
             and n[k].get(e, ("",))[0] == "ok" and o[k][e][1] >= 1e-3]
        print(f"control {e} old/new (>=1ms): n={len(r)} median={np.median(r):.3f}")
    import json
    json.dump(rep, open(os.path.join(HERE, "auto_compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
