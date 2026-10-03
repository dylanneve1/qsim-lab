#!/usr/bin/env python3
"""Fit per-engine cost models on the simulability dataset, validate them
leave-one-family-out, compare with single-statistic rules, draw the phase
diagrams.  Usage: fit.py OUTDIR CSV [CSV ...]"""
import csv, json, math, sys, os, collections, itertools
import numpy as np

ALL_ENGINES = ["sv", "sparse", "mps", "hsf", "tableau", "cstate", "frame", "dense", "auto"]
# State engines build an exact representation of U|0>; frame/auto are
# Heisenberg (observable-specific) engines and are analysed separately.
STATE_ENGINES = ["sv", "sparse", "mps", "hsf", "tableau", "cstate"]
ENGINES = list(STATE_ENGINES)
COLORS = dict(zip(ALL_ENGINES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                "#4a3aa7", "#008300", "#8a8986", "#e34948"]))
FAMILIES = ["ct", "brick", "arith", "qaoa"]
PENALTY = 2.0
T_FLOOR = float(os.environ.get("T_FLOOR", "1e-3"))  # s: below this, times are overhead
EPS = 1e-3       # s: additive slack in the epsilon-regret metric  # censored (timeout / too large) runs count as PENALTY x timeout


V0 = False  # --v0: the nominal (unpruned / uncapped) MPS and HSF features


def resource(e, f):
    """log2 work estimate of engine e from the cheap features f."""
    if V0 and e == "mps":
        return f["mps_l0"]
    if V0 and e == "hsf":
        return f["hsf_l0"]
    if e == "sv":
        return f["sv_l"]
    if e == "sparse":
        return f["sparse_l"]
    if e == "mps":
        return f["mps_l"]
    if e == "hsf":
        return f["hsf_l"]
    if e == "frame":
        return f["frame_l"]
    if e in ("dense", "cstate"):
        return f["dense_l"]
    if e == "auto":
        return min(f["frame_l"], f["dense_l"])
    if e == "tableau":
        return math.log2(max(f["gates"], 1)) + math.log2(max(f["n"], 1))
    raise KeyError(e)


def applicable(e, f, mem_bytes=1 << 30):
    """Hard feasibility from the features alone (memory / gate set)."""
    cap = int(math.log2(mem_bytes / 16))
    if e == "tableau":
        return f["rotations"] == 0
    if e == "sv":
        return f["n"] <= cap
    if e in ("dense", "cstate"):
        return f["d"] <= cap
    if e == "hsf":
        return f["n"] <= cap
    if e == "sparse":
        return f["n"] <= 64
    return True


def load(paths):
    inst = {}
    for p in paths:
        for r in csv.DictReader(open(p)):
            key = (r["spec"], r["seed"])
            d = inst.setdefault(key, dict(family=r["family"], spec=r["spec"], seed=r["seed"],
                                          params=json.loads(r["params"]),
                                          f=json.loads(r["features"]), runs={}, timeout=0.0))
            st = r["status"]
            d["runs"][r["engine"]] = (st, float(r["secs"]) if st == "ok" else None,
                                      float(r["abs_err"]) if r["abs_err"] else None)
            if st == "timeout":
                d["timeout"] = max(d["timeout"], float(r["wall"]))
    for d in inst.values():
        d["timeout"] = d["timeout"] or 10.0
    return list(inst.values())


def actual_time(d, e):
    st, t, _ = d["runs"].get(e, ("missing", None, None))
    if st == "ok":
        return t
    if st in ("timeout", "skipped", "toolarge", "error"):
        return PENALTY * d["timeout"]
    return None  # na / missing


def winner(d):
    best = None
    for e in ENGINES:
        st, t, _ = d["runs"].get(e, ("missing", None, None))
        if st == "ok" and (best is None or t < best[1]):
            best = (e, t)
    return best


class Model:
    """log2 t_e = max(floor_e, a_e + b_e * R_e)  [+ c_e * log2 gates if two=True]."""

    def __init__(self, two=False):
        self.two = two
        self.p = {}

    def x(self, e, f):
        r = resource(e, f)
        return [1.0, r] + ([math.log2(max(f["gates"], 1))] if self.two else [])

    def fit(self, data):
        """OLS on the runs above the common overhead floor T_FLOOR. No
        per-engine floor: a floor learned on other families transfers badly
        (an engine's cheapest runs depend on the family), so predictions
        below T_FLOOR are kept as extrapolations; the epsilon-regret metric
        makes sub-millisecond differences irrelevant anyway."""
        for e in ENGINES:
            pts = [(self.x(e, d["f"]), math.log2(d["runs"][e][1])) for d in data
                   if d["runs"].get(e, ("",))[0] == "ok" and d["runs"][e][1] >= T_FLOOR]
            if len(pts) < 3:
                continue
            X = np.array([x for x, _ in pts])
            Y = np.array([y for _, y in pts])
            coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
            self.p[e] = (math.log2(T_FLOOR), coef)

    def predict(self, e, f):
        if not applicable(e, f):
            return math.inf
        if e == "tableau":
            # polynomial engine: a prior, not a fit (it only runs on
            # Clifford circuits, which a held-out family may not contain)
            return -1e9  # polynomial vs exponential: always first
        if e not in self.p:
            return math.inf
        return float(np.dot(self.p[e][1], self.x(e, f)))

    def choose(self, f, engines=ENGINES):
        return min(engines, key=lambda e: self.predict(e, f))


def evaluate(model_choose, data, predict=None):
    """accuracy, geo-mean regret, frac within 2x, per-engine log10 RMSE."""
    acc = n = 0
    logreg = []
    epsreg = []
    within2 = 0
    for d in data:
        w = winner(d)
        if w is None:
            continue
        c = model_choose(d["f"])
        t = actual_time(d, c)
        if t is None:
            t = PENALTY * d["timeout"]
        n += 1
        acc += c == w[0]
        r = t / w[1]
        logreg.append(math.log10(r))
        epsreg.append(math.log10((t + EPS) / (w[1] + EPS)))
        within2 += r <= 2.0
    out = dict(n=n, acc=acc / max(n, 1), geo_regret=10 ** np.mean(logreg) if logreg else math.nan,
               within2=within2 / max(n, 1), max_regret=10 ** max(logreg) if logreg else math.nan,
               geo_eps_regret=10 ** np.mean(epsreg) if epsreg else math.nan,
               max_eps_regret=10 ** max(epsreg) if epsreg else math.nan)
    if predict is not None:
        rm = {}
        for e in ENGINES:
            errs = []
            for d in data:
                st, t, _ = d["runs"].get(e, ("", None, None))
                if st == "ok" and t >= T_FLOOR:
                    p = predict(e, d["f"])
                    if math.isfinite(p):
                        errs.append((p - math.log2(t)) * math.log10(2))
            if errs:
                rm[e] = (len(errs), float(np.sqrt(np.mean(np.square(errs)))))
        out["rmse_log10"] = rm
    return out


SINGLE = ["n", "gates", "t_count", "rotations", "d", "chi_bits", "hsf_k", "sup", "depth2"]


def fit_single(data, feat, k=3):
    """Best rule 'feature in interval i -> engine e_i' with <= k intervals,
    minimising total log regret on data."""
    pts = sorted([(d["f"][feat], d) for d in data if winner(d)], key=lambda x: x[0])
    vals = sorted(set(v for v, _ in pts))
    cuts = [(a + b) / 2 for a, b in zip(vals, vals[1:])]

    def cost(lo, hi):
        seg = [d for v, d in pts if lo <= v < hi]
        best = (0.0, "sv")
        bestc = math.inf
        for e in ENGINES:
            c = 0.0
            for d in seg:
                t = actual_time(d, e)
                t = PENALTY * d["timeout"] if t is None else t
                c += math.log10(t / winner(d)[1])
            if c < bestc:
                bestc, best = c, e
        return bestc, best

    best = (math.inf, None)
    for m in range(0, k):
        for cs in itertools.combinations(cuts, m):
            bounds = [-math.inf, *cs, math.inf]
            tot = 0.0
            rule = []
            for lo, hi in zip(bounds, bounds[1:]):
                c, e = cost(lo, hi)
                tot += c
                rule.append((lo, hi, e))
            if tot < best[0]:
                best = (tot, rule)
    rule = best[1]

    def choose(f):
        v = f[feat]
        for lo, hi, e in rule:
            if lo <= v < hi:
                return e
        return rule[-1][2]
    return choose, rule


def main():
    global ENGINES, V0
    args = sys.argv[1:]
    while args[0].startswith("--"):
        if args[0] == "--all":
            ENGINES[:] = ALL_ENGINES
        elif args[0] == "--v0":
            V0 = True
        args = args[1:]
    outdir = args[0]
    data = load(args[1:])
    os.makedirs(outdir, exist_ok=True)
    fams = [f for f in FAMILIES if any(d["family"] == f for d in data)]
    rep = {"n_instances": len(data)}
    # exactness
    errs = [(d["spec"], e, r[2]) for d in data for e, r in d["runs"].items()
            if r[0] == "ok" and r[2] is not None]
    rep["max_abs_err"] = max(x[2] for x in errs) if errs else None
    rep["runs_ok"] = len(errs)
    # full fit
    full = Model(); full.fit(data)
    full2 = Model(two=True); full2.fit(data)
    rep["coef"] = {e: (p[0], None if p[1] is None else list(map(float, p[1])))
                   for e, p in full.p.items()}
    rep["coef_two"] = {e: (p[0], None if p[1] is None else list(map(float, p[1])))
                       for e, p in full2.p.items()}
    rep["in_sample"] = evaluate(full.choose, data, full.predict)
    # leave one family out
    lofo = {}
    for F in fams:
        train = [d for d in data if d["family"] != F]
        test = [d for d in data if d["family"] == F]
        m = Model(); m.fit(train)
        m2 = Model(two=True); m2.fit(train)
        res = {"model": evaluate(m.choose, test, m.predict),
               "model_two": evaluate(m2.choose, test, m2.predict)}
        for feat in SINGLE:
            ch, rule = fit_single(train, feat)
            res["single_" + feat] = evaluate(ch, test)
            res["single_" + feat]["rule"] = [(lo, hi, e) for lo, hi, e in rule]
        res["always_sv"] = evaluate(lambda f: "sv" if f["n"] <= 26 else "auto", test)
        # store held-out predictions for plots
        for d in test:
            d["pred_lofo"] = m.choose(d["f"])
        lofo[F] = res
    rep["lofo"] = lofo
    # pooled held-out summary
    pooled = {}
    for key in ["model", "model_two", "always_sv"] + ["single_" + s for s in SINGLE]:
        n = sum(lofo[F][key]["n"] for F in fams)
        pooled[key] = dict(
            acc=sum(lofo[F][key]["acc"] * lofo[F][key]["n"] for F in fams) / max(n, 1),
            geo_regret=10 ** (sum(math.log10(lofo[F][key]["geo_regret"]) * lofo[F][key]["n"]
                                  for F in fams) / max(n, 1)),
            within2=sum(lofo[F][key]["within2"] * lofo[F][key]["n"] for F in fams) / max(n, 1),
            geo_eps_regret=10 ** (sum(math.log10(lofo[F][key]["geo_eps_regret"]) * lofo[F][key]["n"]
                                      for F in fams) / max(n, 1)),
            max_eps_regret=max(lofo[F][key]["max_eps_regret"] for F in fams))
    rep["lofo_pooled"] = pooled
    json.dump(rep, open(os.path.join(outdir, "fit_report.json"), "w"), indent=1, default=str)
    # winners table
    with open(os.path.join(outdir, "winners.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "spec", "seed", "winner", "best_secs", "pred_lofo", "pred_regret"]
                   + [f"t_{e}" for e in ENGINES])
        for d in data:
            b = winner(d)
            if not b:
                continue
            p = d.get("pred_lofo", "")
            pt = actual_time(d, p) if p else None
            w.writerow([d["family"], d["spec"], d["seed"], b[0], f"{b[1]:.3g}", p,
                        f"{(pt or PENALTY * d['timeout']) / b[1]:.3g}" if p else ""]
                       + [d["runs"].get(e, ("",))[0] if d["runs"].get(e, ("",))[0] != "ok"
                          else f"{d['runs'][e][1]:.3g}" for e in ENGINES])
    print(json.dumps({k: rep[k] for k in ["n_instances", "max_abs_err", "runs_ok", "lofo_pooled"]},
                     indent=1, default=str))
    for F in fams:
        print(F, {k: (round(v["acc"], 2), round(v["geo_regret"], 2), round(v["geo_eps_regret"], 2)) for k, v in lofo[F].items()
                  if k in ("model", "model_two", "always_sv", "single_d", "single_n", "single_t_count")})
    for e, (fl, c) in rep["coef"].items():
        print(f"  {e:8s} floor 2^{fl:.1f}s  coef {c}")
    try:
        plots(data, outdir, full)
    except ImportError:
        pass


def plots(data, outdir, full):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    # 1. predicted vs actual
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex=False)
    for ax in axes.flat[len(ENGINES):]:
        ax.axis("off")
    for ax, e in zip(axes.flat, ENGINES):
        for F, mk in zip(FAMILIES, "o^sD"):
            xs, ys = [], []
            for d in data:
                if d["family"] == F and d["runs"].get(e, ("",))[0] == "ok":
                    xs.append(resource(e, d["f"]))
                    ys.append(math.log10(d["runs"][e][1]))
            ax.scatter(xs, ys, s=12, marker=mk, label=F, alpha=0.75, color=COLORS[e],
                       edgecolors="white", linewidths=0.4)
        if e in full.p and full.p[e][1] is not None:
            fl, c = full.p[e]
            xr = np.linspace(*ax.get_xlim(), 50)
            ax.plot(xr, [(c[0] + c[1] * x) * math.log10(2) for x in xr], color="#52514e",
                    lw=1.5)
            ax.set_title(f"{e}: log2 t = {c[0]:.1f} + {c[1]:.2f}·R", fontsize=9)
        else:
            ax.set_title(e, fontsize=9)
        ax.set_xlabel("R = cheap log2 work estimate")
        ax.set_ylabel("log10 seconds")
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=7, title="family (marker)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "cost_models.png"), dpi=130)
    plt.close(fig)
    # 2. phase diagrams
    def phase(F, fixed, xk, yk, fname, title):
        sel = [d for d in data if d["family"] == F and all(d["params"].get(k) == v
                                                             for k, v in fixed.items())]
        if not sel:
            return
        xs = sorted(set(d["params"][xk] for d in sel))
        ys = sorted(set(d["params"][yk] for d in sel))
        fig, ax = plt.subplots(figsize=(max(6.5, 1.6 + 0.75 * len(xs)), 1.3 + 0.6 * len(ys)))
        used = set()
        for d in sel:
            b = winner(d)
            if not b:
                continue
            i, j = xs.index(d["params"][xk]), ys.index(d["params"][yk])
            used.add(b[0])
            ax.add_patch(plt.Rectangle((i - 0.47, j - 0.47), 0.94, 0.94, color=COLORS[b[0]]))
            p = d.get("pred_lofo")
            if p:
                used.add(p)
                ax.add_patch(plt.Rectangle((i - 0.2, j - 0.2), 0.4, 0.4, color=COLORS[p],
                                           ec="white", lw=1.2))
            txt = b[0] + ("" if p == b[0] else f"\n(pred {p})")
            ax.text(i, j + 0.33, f"{b[1]:.2g}s", ha="center", va="center", fontsize=6,
                    color="white")
            ax.text(i, j - 0.33, txt, ha="center", va="center", fontsize=6, color="white")
        ax.set_xticks(range(len(xs)), [f"{v:g}" for v in xs])
        ax.set_yticks(range(len(ys)), [f"{v:g}" for v in ys])
        ax.set_xlim(-0.5, len(xs) - 0.5)
        ax.set_ylim(-0.5, len(ys) - 0.5)
        ax.set_xlabel(xk)
        ax.set_ylabel(yk)
        ax.set_title(title + "\nbig square: measured winner; inner square: predicted (held-out fit)",
                     fontsize=8)
        ax.legend(handles=[Patch(color=COLORS[e], label=e) for e in ENGINES if e in used],
                  bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=130)
        plt.close(fig)
    universal(data, outdir, plt, Patch)
    phase("ct", {"n": 24, "nn": 1}, "t", "L", "phase_ct24.png", "Clifford+T, n=24, NN CNOT layers")
    phase("ct", {"n": 32, "nn": 1}, "t", "L", "phase_ct32.png", "Clifford+T, n=32, NN CNOT layers")
    phase("ct", {"n": 16, "nn": 1}, "t", "L", "phase_ct16.png", "Clifford+T, n=16")
    phase("brick", {"nn": 1}, "D", "n", "phase_brick.png", "Random brickwork (Haar 1q + CZ), NN")
    for b in (6, 8, 10, 12):
        phase("arith", {"bits": b}, "h", "reps", f"phase_arith{b}.png",
              f"Cuccaro adders, bits={b} (n={2*b+1})")
    for nn in (0, 1):
        phase("qaoa", {"deg": 3, "nn": nn}, "p", "n", f"phase_qaoa_nn{nn}.png",
              f"QAOA deg 3, {'ring' if nn else 'random'} graph")


def universal(data, outdir, plt, Patch):
    """All families in normalised resource coordinates: magic (d/n),
    entanglement (χ bits / (n/2)), support (sup/n). Colour = measured winner
    among the state engines, marker = family."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    pairs = [("d", "chi_bits", "active dimension d / n  (magic)",
              "bond bound log2 χ / (n/2)  (entanglement)"),
             ("sup", "chi_bits", "support bound / n  (superposition)",
              "bond bound log2 χ / (n/2)  (entanglement)"),
             ("d", "sup", "active dimension d / n  (magic)", "support bound / n  (superposition)")]
    norm = {"d": lambda f: f["d"] / f["n"], "sup": lambda f: f["sup"] / f["n"],
            "chi_bits": lambda f: f["chi_bits"] / max(1, f["n"] // 2)}
    rng = np.random.default_rng(0)
    used = set()
    for ax, (xa, ya, xl, yl) in zip(axes, pairs):
        for F, mk in zip(FAMILIES, "o^sD"):
            for d in data:
                if d["family"] != F:
                    continue
                w = winner(d)
                if not w:
                    continue
                used.add(w[0])
                jx, jy = rng.uniform(-0.012, 0.012, 2)
                ax.scatter(norm[xa](d["f"]) + jx, norm[ya](d["f"]) + jy, s=26, marker=mk,
                           color=COLORS[w[0]], edgecolors="white", linewidths=0.5)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
    h = [Patch(color=COLORS[e], label=e) for e in ENGINES if e in used]
    from matplotlib.lines import Line2D
    h += [Line2D([], [], marker=mk, ls="", color="#52514e", label=F)
          for F, mk in zip(FAMILIES, "o^sD")]
    axes[-1].legend(handles=h, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8,
                    title="colour: winner / marker: family", title_fontsize=8)
    fig.suptitle("Phase diagram of exact simulability in normalised resource coordinates "
                 "(measured winner, all families)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "phase_universal.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
