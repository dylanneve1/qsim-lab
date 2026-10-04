#!/usr/bin/env python3
"""End-to-end Planner v2 evaluation on the Mac (research/planner-v2.md §4).

`planner_v2 plan VARIANT REQ` (plan + execute, in process; mac/e2e*.jsonl)
against the best engine measured for the same request in the read-out
session (mac/req.jsonl, `fit_v2.truth`). Variants: v1 (Planner v1 as
shipped, expectations only), v2nc (Planner v2, cache off: every instance is
new), rule (the old hand rules for samples and amplitudes).

    eval_v2.py OUTDIR req.jsonl feat.jsonl e2e.jsonl...
"""
import json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_v2 as fv  # noqa: E402


def stats(rows):
    if not rows:
        return {}
    x = np.array([r["reg"] for r in rows])
    e = np.array([r["eps"] for r in rows])
    c = np.array([r["choice_eps"] for r in rows])
    p = np.array([r["plan"] for r in rows])
    w = max(rows, key=lambda r: r["eps"])
    return dict(n=len(rows), geo=float(10 ** np.mean(np.log10(x))), within2=float(np.mean(x <= 2)),
                geo_eps=float(10 ** np.mean(np.log10(e))), worst=float(x.max()), worst_eps=float(e.max()),
                choice_geo_eps=float(10 ** np.mean(np.log10(c))),
                top1=float(np.mean([r["top1"] for r in rows])),
                plan_median_ms=float(np.median(p) * 1e3), plan_p90_ms=float(np.percentile(p, 90) * 1e3),
                plan_max_ms=float(p.max() * 1e3), aborts=int(sum(r["aborted"] for r in rows)),
                worst_case=f"{w['spec']} {w['variant']} -> {w['engine']} {w['t']:.4g}s best {w['best_e']} {w['best']:.4g}s")


def main():
    outdir, req_path, feat_path, *e2e = sys.argv[1:]
    os.makedirs(outdir, exist_ok=True)
    feat, runs = fv.load(req_path, feat_path)
    rows = defaultdict(list)
    for p in e2e:
        for l in open(p):
            r = json.loads(l)
            k = (r["spec"], str(r["seed"]))
            var, rq = r["job"].split(":")
            if k not in feat or k not in runs:
                continue
            tr = fv.truth(feat, runs, k, rq)
            if not tr:
                continue
            best_e, best = min(tr.items(), key=lambda x: x[1])
            if best >= fv.CENS:
                continue
            t = r["secs"] if r.get("ok") else fv.CENS
            chosen = r.get("chosen")
            tc = tr.get(chosen, fv.CENS)
            rows[(var, rq)].append(dict(
                spec=k[0], variant=var, reg=t / best, eps=(t + fv.EPS) / (best + fv.EPS),
                choice_eps=(tc + fv.EPS) / (best + fv.EPS), top1=chosen == best_e,
                plan=r.get("plan_secs", 0.0), aborted=len(r.get("aborted", [])) > 0,
                engine=r.get("engine", chosen), t=t, best=best, best_e=best_e, fam=fv.fam(k[0])))
    res = {}
    lines = []
    for (var, rq), rs in sorted(rows.items(), key=lambda x: (x[0][1], x[0][0])):
        s = {"all": stats(rs)}
        for lo, name in [(1e-3, ">=1ms"), (1e-2, ">=10ms"), (1e-1, ">=0.1s")]:
            s[name] = stats([r for r in rs if r["best"] >= lo])
        s["by_family"] = {fm: stats([r for r in rs if r["fam"] == fm]) for fm in fv.FAMS}
        res[f"{var}:{rq}"] = s
        a = s["all"]
        lines.append(
            f"{rq:6s} {var:5s} n={a['n']:3d} geo={a['geo']:.3f} <=2x={a['within2']:.2f} geo_eps={a['geo_eps']:.3f} "
            f"worst={a['worst']:.1f} worst_eps={a['worst_eps']:.2f} choice_eps={a['choice_geo_eps']:.3f} top1={a['top1']:.2f} "
            f"plan med/p90/max={a['plan_median_ms']:.3f}/{a['plan_p90_ms']:.3f}/{a['plan_max_ms']:.1f} ms aborts={a['aborts']}")
        for name in [">=1ms", ">=10ms", ">=0.1s"]:
            b = s[name]
            if b:
                lines.append(f"         {name:6s} n={b['n']:3d} geo={b['geo']:.3f} geo_eps={b['geo_eps']:.3f} worst={b['worst']:.1f}")
        lines.append(f"         worst: {a['worst_case']}")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(outdir, "stdout.txt"), "w").write(txt + "\n")
    json.dump(res, open(os.path.join(outdir, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
