#!/usr/bin/env python3
"""MPS cost prediction + planner evaluation (research/planner.md).

Inputs: the Mac timings of the simulability dataset (../simulability/raw)
and the deterministic MPS data from collect.py (mpsdata_*.jsonl: replayed
costs per bond bound, the real exact run's operation counts, capped-probe
predictions). Outputs a JSON report and plots in OUTDIR.

    fit_planner.py OUTDIR mpsdata_vps.jsonl
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "simulability"))
import fit  # noqa: E402  (the simulability fitting code)

RAW = os.path.join(HERE, "..", "simulability", "raw")
GRIDS = ["ct24", "ct32", "ctnn0", "brick", "arith", "qaoa"]
FAMS = ["ct", "brick", "arith", "qaoa"]
T_FLOOR = fit.T_FLOOR
L2 = math.log10(2)


def total(st, w, k):
    return (w * st["svd_work"] + st["qr_work"] + st["mm_work"] + st["oneq_work"]
            + k * (st["svd_calls"] + st["qr_calls"]))


def lg(x):
    return math.log2(max(x, 1.0))


def probe_stats(d, cap, key="pred"):
    """Probe prediction if the probe finished, else the best bound."""
    try:
        pr = d["m"]["trace"]["probes"][cap]
        if pr["done"]:
            return pr[key]["stats"]
    except (KeyError, TypeError):
        pass
    return d["m"]["cost"]["best"]["stats"]


def load(mpsdata):
    data = fit.load([os.path.join(RAW, g + ".csv") for g in GRIDS])
    md = {}
    for line in open(mpsdata):
        d = json.loads(line)
        md[(d["spec"], d["seed"])] = d
    for d in data:
        d["m"] = md.get((d["spec"], d["seed"]))
    return data


def mps_points(data):
    """(family, log2 t_mac, instance) for MPS runs >= T_FLOOR with data."""
    out = []
    for d in data:
        st, t, _ = d["runs"].get("mps", ("", None, None))
        if st == "ok" and t >= T_FLOOR and d["m"] and "stats" in d["m"].get("trace", {}):
            out.append((d["family"], math.log2(t), d))
    return out


def ols(x, y):
    X = np.c_[np.ones(len(x)), x]
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c


def score(pts, feat):
    """In-sample slope/RMSE and leave-one-family-out RMSE (log10)."""
    fams = [p[0] for p in pts]
    x = np.array([feat(p[2]) for p in pts])
    y = np.array([p[1] for p in pts])
    c = ols(x, y)
    r = y - (c[0] + c[1] * x)
    out = dict(n=len(pts), slope=float(c[1]), icpt=float(c[0]),
               rmse=float(np.sqrt(np.mean(r ** 2)) * L2), lofo={})
    allres = []
    for F in FAMS:
        tr = [i for i, f in enumerate(fams) if f != F]
        te = [i for i, f in enumerate(fams) if f == F]
        if len(te) == 0:
            continue
        c2 = ols(x[tr], y[tr])
        e = y[te] - (c2[0] + c2[1] * x[te])
        allres.extend(e.tolist())
        out["lofo"][F] = float(np.sqrt(np.mean(e ** 2)) * L2)
        out["lofo_max_" + F] = float(np.max(np.abs(e)) * L2)
    out["lofo_pooled"] = float(np.sqrt(np.mean(np.square(allres))) * L2)
    out["lofo_max"] = float(np.max(np.abs(allres)) * L2)
    return out


def main():
    outdir, mpsdata = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    data = load(mpsdata)
    pts = mps_points(data)
    rep = {"n_mps_points": len(pts)}
    # sanity: replay == engine and bound validity
    ms = [d["m"]["trace"] for d in data if d["m"] and "stats" in d["m"].get("trace", {})]
    rep["replay_matches"] = sum(1 for t in ms if t.get("replay_matches") is True)
    rep["replay_checked"] = sum(1 for t in ms if t.get("replay_matches") is not None)
    rep["bound_violations"] = sum(t.get("bound_violations") or 0 for t in ms)

    # 1. oracle: the real run's own operation counts -> which units?
    grid = {}
    for w in [1, 2, 4, 8, 16, 32]:
        for k in [0, 300, 1000, 2000, 4000, 8000, 16000]:
            s = score(pts, lambda d, w=w, k=k: lg(total(d["m"]["trace"]["stats"], w, k)))
            grid[(w, k)] = s
    best = min(grid, key=lambda wk: grid[wk]["rmse"])
    W, K = best
    rep["units"] = dict(svd_weight=W, call_overhead=K)
    rep["units_grid"] = {f"{w},{k}": round(v["rmse"], 3) for (w, k), v in grid.items()}
    feats = {
        # previous agent's features
        "mps_l (old: crossing+support, Σχ³)": lambda d: d["f"]["mps_l"],
        "mps_l0 (old: crossing only)": lambda d: d["f"]["mps_l0"],
        "oracle G·χmax_final³": lambda d: lg(d["f"]["gates"] * float(d["m"]["trace"]["max_bond_final"]) ** 3),
        "oracle trace work (svd only)": lambda d: lg(d["m"]["trace"]["stats"]["svd_work"]),
        "oracle trace work (units)": lambda d: lg(total(d["m"]["trace"]["stats"], W, K)),
    }
    for e in ["cut", "cross", "stab", "coset", "affine", "best"]:
        feats[f"replay[{e}]"] = (lambda d, e=e: lg(total(d["m"]["cost"][e]["stats"], W, K)))
    for cap in ["8", "16", "32"]:
        feats[f"probe χ≤{cap} + best"] = (
            lambda d, cap=cap: lg(total(probe_stats(d, cap), W, K)))
        feats[f"probe χ≤{cap} extrapolated"] = (
            lambda d, cap=cap: lg(total(probe_stats(d, cap, "predx"), W, K)))
    rep["features"] = {}
    for name, fn in feats.items():
        try:
            rep["features"][name] = score(pts, fn)
        except (KeyError, TypeError):
            pass
    # bond prediction quality: log2 predicted max bond vs real (all runs with a trace)
    bq = {}
    for e in ["cut", "cross", "stab", "coset", "affine", "best"]:
        errs = [lg(d["m"]["cost"][e]["max_bond"]) - lg(d["m"]["trace"]["trace_max"]) for d in data
                if d["m"] and d["m"]["trace"].get("done")]
        exact = sum(1 for x in errs if x == 0)
        bq[e] = dict(n=len(errs), mean_excess_bits=float(np.mean(errs)), exact=exact,
                     max_excess_bits=float(np.max(errs)))
    for cap in ["8", "16", "32"]:
        errs = [lg(d["m"]["trace"]["probes"][cap]["pred"]["max_bond"]) - lg(d["m"]["trace"]["trace_max"])
                for d in data if d["m"] and d["m"].get("trace", {}).get("done")]
        bq["probe" + cap] = dict(n=len(errs), mean_excess_bits=float(np.mean(errs)),
                                 exact=sum(1 for x in errs if x == 0),
                                 max_excess_bits=float(np.max(errs)),
                                 min_excess_bits=float(np.min(errs)))
    rep["max_bond_quality"] = bq
    # feature cost
    rep["replay_secs_best"] = dict(
        median=float(np.median([d["m"]["cost"]["best"]["secs"] for d in data if d["m"]])),
        max=float(np.max([d["m"]["cost"]["best"]["secs"] for d in data if d["m"]])))
    rep["probe_secs_median_over_mps"] = {
        cap: float(np.median([d["m"]["trace"]["probes"][cap]["secs"] / max(d["runs"]["mps"][1], 1e-6)
                              for d in data if d["m"] and "probes" in d["m"].get("trace", {})
                              and d["runs"].get("mps", ("",))[0] == "ok"
                              and d["runs"]["mps"][1] >= T_FLOOR]))
        for cap in ["8", "16", "32"]}

    # 2. planner decisions, leave one family out, old vs new MPS feature
    def with_mps(name, fn):
        for d in data:
            d["f"]["mps_r"] = fn(d) if d["m"] else d["f"]["mps_l"]
        orig = fit.resource

        def res(e, f):
            return f["mps_r"] if e == "mps" else orig(e, f)
        fit.resource = res
        out = {}
        try:
            for F in FAMS:
                tr = [d for d in data if d["family"] != F]
                te = [d for d in data if d["family"] == F]
                m = fit.Model()
                m.fit(tr)
                out[F] = fit.evaluate(m.choose, te, m.predict)
                for d in te:
                    d.setdefault("pred", {})[name] = m.choose(d["f"])
                    d.setdefault("ranked", {})[name] = sorted(
                        ((2 ** m.predict(e, d["f"]), e) for e in fit.ENGINES
                         if math.isfinite(m.predict(e, d["f"]))))
            full = fit.Model()
            full.fit(data)
            out["coef"] = {e: list(map(float, p[1])) for e, p in full.p.items()}
            out["in_sample"] = fit.evaluate(full.choose, data, full.predict)
        finally:
            fit.resource = orig
        n = sum(out[F]["n"] for F in FAMS)
        out["pooled"] = dict(
            acc=sum(out[F]["acc"] * out[F]["n"] for F in FAMS) / n,
            geo_regret=10 ** (sum(math.log10(out[F]["geo_regret"]) * out[F]["n"] for F in FAMS) / n),
            within2=sum(out[F]["within2"] * out[F]["n"] for F in FAMS) / n,
            geo_eps_regret=10 ** (sum(math.log10(out[F]["geo_eps_regret"]) * out[F]["n"] for F in FAMS) / n),
            max_regret=max(out[F]["max_regret"] for F in FAMS),
            max_eps_regret=max(out[F]["max_eps_regret"] for F in FAMS),
            mps_lofo_rmse={F: out[F]["rmse_log10"].get("mps", (0, None))[1] for F in FAMS})
        out["speculative"] = speculate(name)
        return out

    def speculate(name, kappa=1.0, floor=2e-3):
        """Regret if MPS/sparse choices are aborted after kappa x the
        runner-up's predicted time (then the runner-up runs)."""
        logs, worst, n = [], 1.0, 0
        for d in data:
            w = fit.winner(d)
            if not w or name not in d.get("ranked", {}):
                continue
            rk = d["ranked"][name]
            c = d["pred"][name]
            t = fit.actual_time(d, c) or fit.PENALTY * d["timeout"]
            if c in ("mps", "sparse"):
                others = [(p, e) for p, e in rk if e != c]
                if others:
                    pru, eru = others[0]
                    dl = max(kappa * pru, floor)
                    if t > dl:
                        tru = fit.actual_time(d, eru) or fit.PENALTY * d["timeout"]
                        t = dl + tru
            r = t / w[1]
            logs.append(math.log10(r))
            worst = max(worst, r)
            n += 1
        return dict(n=n, geo_regret=10 ** float(np.mean(logs)), max_regret=worst,
                    within2=float(np.mean([x <= math.log10(2) for x in logs])))

    rep["decisions"] = {
        "old mps_l": with_mps("old", lambda d: d["f"]["mps_l"]),
        "replay[best]": with_mps("best", lambda d: lg(total(d["m"]["cost"]["best"]["stats"], W, K))),
        "replay[cross]": with_mps("cross", lambda d: lg(total(d["m"]["cost"]["cross"]["stats"], W, K))),
        "probe16": with_mps("probe16", lambda d: lg(total(probe_stats(d, "16"), W, K))),
        "probe16x": with_mps("probe16x", lambda d: lg(total(probe_stats(d, "16", "predx"), W, K))),
        "probe8x": with_mps("probe8x", lambda d: lg(total(probe_stats(d, "8", "predx"), W, K))),
        "oracle": with_mps("oracle", lambda d: lg(total(d["m"]["trace"]["stats"], W, K))
                           if d["m"].get("trace", {}).get("done") else lg(total(d["m"]["cost"]["best"]["stats"], W, K))),
    }
    # worst cases
    worst = []
    for d in data:
        w = fit.winner(d)
        if not w or "pred" not in d:
            continue
        for k, e in d["pred"].items():
            t = fit.actual_time(d, e) or fit.PENALTY * d["timeout"]
            worst.append((t / w[1], k, d["spec"], e, w[0]))
    worst.sort(reverse=True)
    rep["worst"] = {k: [x for x in worst if x[1] == k][:8] for k in ["old", "best", "probe16", "oracle"]}
    json.dump(rep, open(os.path.join(outdir, "planner_report.json"), "w"), indent=1, default=str)
    # print summary
    print("points", len(pts), "units", rep["units"], "replay ok", rep["replay_matches"], "/",
          rep["replay_checked"], "violations", rep["bound_violations"])
    print(f"{'feature':38s} slope  rmse  lofo(ct/brick/arith/qaoa)  pooled  maxerr")
    for name, s in rep["features"].items():
        lo = "/".join(f"{s['lofo'].get(F, float('nan')):.2f}" for F in FAMS)
        print(f"{name:38s} {s['slope']:.2f}  {s['rmse']:.2f}  {lo:24s}  {s['lofo_pooled']:.2f}  {s['lofo_max']:.2f}")
    print("max-bond quality", json.dumps(bq))
    for k, v in rep["decisions"].items():
        p = v["pooled"]
        print(f"{k:14s} top1 {p['acc']:.3f} geo {p['geo_regret']:.3f} w2 {p['within2']:.3f} "
              f"eps {p['geo_eps_regret']:.3f} max {p['max_regret']:.1f} maxeps {p['max_eps_regret']:.1f} "
              f"mpsRMSE {p['mps_lofo_rmse']}")
        sp = v["speculative"]
        print(f"{'':14s} + speculation: geo {sp['geo_regret']:.3f} max {sp['max_regret']:.1f} w2 {sp['within2']:.3f}")
    for k, v in rep["worst"].items():
        print(k, [(round(x[0], 1), x[2], x[3], x[4]) for x in v[:5]])
    try:
        plots(rep, pts, feats, W, K, outdir)
    except ImportError:
        pass


def plots(rep, pts, feats, W, K, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    names = ["mps_l (old: crossing+support, Σχ³)", "replay[best]", "probe χ≤16 + best",
             "oracle trace work (units)"]
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.8), sharey=True)
    cols = dict(zip(FAMS, ["#2a78d6", "#eb6834", "#1baf7a", "#e34948"]))
    for ax, nm in zip(axes, names):
        fn = feats[nm]
        for F, mk in zip(FAMS, "o^sD"):
            xs = [fn(p[2]) for p in pts if p[0] == F]
            ys = [p[1] * L2 for p in pts if p[0] == F]
            ax.scatter(xs, ys, s=12, marker=mk, color=cols[F], label=F, alpha=0.8,
                       edgecolors="white", linewidths=0.4)
        s = rep["features"][nm]
        xr = np.linspace(*ax.get_xlim(), 20)
        ax.plot(xr, [(s["icpt"] + s["slope"] * x) * L2 for x in xr], color="#52514e", lw=1.2)
        ax.set_title(f"{nm}\nslope {s['slope']:.2f}, RMSE {s['rmse']:.2f}, held-out {s['lofo_pooled']:.2f}",
                     fontsize=8)
        ax.set_xlabel("log2 predicted work")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("log10 MPS seconds (Mac)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "mps_predictors.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
