#!/usr/bin/env python3
"""Planner v2 cost models and leave-one-family-out decision study
(research/planner-v2.md §2).

Inputs: the Mac read-out timings (`collect_req.py` -> mac/req.jsonl) and the
planner features (`planner_v2 feat` -> feat.jsonl, deterministic).

Models (seconds; log2 fits on runs >= T_FLOOR):
  E      total_e  = 2^(a + b R_e)                  (v1 form: evolve + parity)
  state  state_e  = 2^(a + b R_e)                  (evolve only; HSF: + full output)
  read-out (linear in op counts, fitted on the read-out alone):
    sv/hsf sampling   c1 2^n + c2 S
    sparse sampling   c1 2^sup + c2 S
    mps               canon c U_canon ; shot c S U_shot ; amplitude c m U_amp
    cstate sampler    c1 (2^d (d+1) + n^2 W) + c2 S (d + n) W
    tableau           c1 n^2 W + c2 S n (1 + W)
    sv/sparse amps    c m
    hsf amplitudes    c1 G n + c2 2^keff G 2^max(nA,nB)  (set-up + path sums)

Decision: argmin over applicable engines of the predicted total; regret =
measured total of the choice / best measured total; censored = 2 x 10 s.

    fit_v2.py OUTDIR feat.jsonl req.jsonl [hsfamp.jsonl ...]
"""
import json, math, os, sys
from collections import defaultdict
import numpy as np
from scipy.optimize import least_squares

T_FLOOR = 1e-3
CENS = 20.0
EPS = 1e-3
ENG = ["sv", "sparse", "mps", "hsf", "cstate", "tableau"]
REQS = ["e", "s1", "s1k", "s100k", "a1", "a1k"]
SHOTS = {"s1": 1, "s1k": 1000, "s100k": 100000}
AMPS = {"a1": 1, "a1k": 1000}
FAMS = ["ct", "brick", "arith", "qaoa", "hea", "qft"]
L2 = math.log10(2)


def fam(spec):
    return spec.split(":")[0]


def load(req_path, feat_path):
    feat = {}
    for l in open(feat_path):
        r = json.loads(l)
        feat[(r["spec"], str(r["seed"]))] = r
    runs = defaultdict(dict)
    paths = [req_path] if isinstance(req_path, str) else req_path
    hsfamp = {}
    for p in paths:
        for l in open(p):
            r = json.loads(l)
            k = (r["spec"], str(r["seed"]))
            if r["job"] == "hsfamp":
                hsfamp[k] = r
            else:
                runs[k][r["job"]] = r
    # HSF amplitudes of instances whose HSF full output was skipped (prior
    # sweep censored / slow) were timed separately (`req hsfamp`)
    for k, r in hsfamp.items():
        h = runs[k].get("hsf")
        if h is None or h.get("skipped"):
            runs[k]["hsf"] = dict(r, job="hsf", amp_only=True)
    return feat, runs


def words(n):
    return max(1, -(-n // 64))


def measured(run, e, rq):
    """Measured total seconds of engine e for request rq (None: censored)."""
    if run is None or not run.get("ok"):
        return None
    ev = run["evolve"]
    if rq == "e":
        if e == "tableau":
            return ev
        if e == "hsf":
            return None if run.get("e") is None else ev + run["e"]
        return None if run.get("e") is None else ev + run["e"]
    if rq in SHOTS:
        if run.get(rq) is None or run.get("prep") is None:
            return None
        return ev + run["prep"] + run[rq]
    if rq in AMPS:
        if e in ("cstate", "tableau"):
            return None
        return None if run.get(rq) is None else ev + run[rq]


def applicable(e, f, rq):
    n = f["n"]
    if e == "tableau":
        return f["clifford"] and rq[0] != "a"
    if e == "cstate":
        return rq[0] != "a" and f["d"] <= 26
    if e == "sv":
        return n <= 26
    if e == "hsf":
        return (n <= 52 and n < 64) if rq[0] == "a" else n < 26
    if e == "sparse":
        return n <= 64
    return True


def resource(e, f):
    return {"sv": f["sv_l"], "sparse": f["sparse_l"], "mps": f["mps_r"], "hsf": f["hsf_l"],
            "cstate": f["dense_l"]}.get(e)


def hsf_amp_r(f, m):
    k = f["hsf_keff"]
    x = k + math.log2(max(f["gates"], 1)) + max(f["hsf_na"], f["hsf_nb"])
    y = k + math.log2(max(m, 1))
    mx = max(x, y)
    return mx + math.log2(2 ** (x - mx) + 2 ** (y - mx))


# ---------------------------------------------------------------- fitting

def fit_loglin(xs, ts):
    """log2 t = a + b x on t >= T_FLOOR."""
    x = np.array([a for a, t in zip(xs, ts) if t >= T_FLOOR])
    y = np.log2([t for t in ts if t >= T_FLOOR])
    if len(x) < 3:
        return None
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def fit_linear_terms(T, ts, floor=2e-5):
    """t ≈ Σ c_i T_i, c >= 0, least squares in log space."""
    T = np.array(T, float)
    ts = np.array(ts, float)
    keep = ts >= floor
    T, ts = T[keep], ts[keep]
    if len(ts) < 3:
        return None
    k = T.shape[1]
    # init per term from median ratio
    x0 = []
    for i in range(k):
        m = T[:, i] > 0
        x0.append(math.log(max(np.median(ts[m] / T[m, i]) / k, 1e-15)) if m.any() else -30)

    def res(lc):
        p = T @ np.exp(lc)
        return np.log(np.maximum(p, 1e-300)) - np.log(ts)

    r = least_squares(res, np.array(x0), bounds=(-60, 5))
    return [float(v) for v in np.exp(r.x)]


def readout_terms(e, f, run=None, real=False):
    """Op-count features per read-out component; `real` uses the measured
    sizes (nnz, bonds, d) instead of the planner's predictions."""
    n = f["n"]
    W = words(n)
    if e == "sparse":
        nnz = run.get("nnz") if (real and run) else 2.0 ** min(f["sup"], n)
        return dict(samp=lambda S: [nnz, S])
    if e in ("sv", "hsf"):
        return dict(samp=lambda S: [2.0 ** n, S])
    if e == "mps":
        if real and run and "canon_u" in run:
            cu, su, au = run["canon_u"], run["shot_u"], run["amp_u"]
        else:
            cu, su, au = f["mps_canon_u"], f["mps_shot_u"], f["mps_amp_u"]
        # op counts plus a per-site overhead (loop, RNG draw, normalisation)
        return dict(canon=[cu, n], samp=lambda S: [S * su, S * n], amp=lambda m: [m * au, m * n])
    if e == "cstate":
        d = run.get("d", f["d"]) if (real and run) else f["d"]
        return dict(build=[2.0 ** d * (d + 1) + n * n * W], samp=lambda S: [S * (d + n) * W])
    if e == "tableau":
        return dict(samp=lambda S: [n * n * W, S * n * (1 + W)])
    return {}


def fit_all(keys, feat, runs, real=False):
    M = {"E": {}, "state": {}, "ro": {}}
    for e in ENG:
        xe, te, xs, tsx = [], [], [], []
        for k in keys:
            f, run = feat[k], runs[k].get(e)
            if e == "tableau" or not run or not run.get("ok"):
                continue
            R = resource(e, f)
            t = measured(run, e, "e")
            if t is not None:
                xe.append(R); te.append(t)
            st = run["evolve"] + (run["prep"] if e == "hsf" and run.get("prep") is not None else 0)
            if e == "hsf" and run.get("prep") is None:
                continue
            xs.append(R); tsx.append(st)
        if e != "tableau":
            M["E"][e] = fit_loglin(xe, te)
            M["state"][e] = fit_loglin(xs, tsx)
    # sampling read-outs
    for e in ENG:
        rows_T, rows_t = [], []
        canon_T, canon_t, amp_T, amp_t = [], [], [], []
        for k in keys:
            f, run = feat[k], runs[k].get(e)
            if not run or not run.get("ok"):
                continue
            terms = readout_terms(e, f, run, real)
            if not terms:
                continue
            for rq, S in SHOTS.items():
                if run.get(rq) is None:
                    continue
                t = run[rq]
                if e == "cstate":
                    continue
                rows_T.append(terms["samp"](S)); rows_t.append(t)
            if e == "mps" and run.get("prep") is not None:
                canon_T.append(terms["canon"]); canon_t.append(run["prep"])
            if e == "mps":
                for rq, m in AMPS.items():
                    if run.get(rq) is not None:
                        amp_T.append(terms["amp"](m)); amp_t.append(run[rq])
            if e == "cstate":
                if run.get("prep") is not None:
                    canon_T.append(terms["build"]); canon_t.append(run["prep"])
                for rq, S in SHOTS.items():
                    if run.get(rq) is not None:
                        rows_T.append(terms["samp"](S)); rows_t.append(run[rq])
        if rows_T:
            M["ro"][e + "_samp"] = fit_linear_terms(rows_T, rows_t)
        if canon_T:
            M["ro"][e + "_prep"] = fit_linear_terms(canon_T, canon_t, floor=1e-6)
        if amp_T:
            M["ro"][e + "_amp"] = fit_linear_terms(amp_T, amp_t, floor=1e-6)
    # sv/sparse amplitude look-ups
    for e in ("sv", "sparse"):
        T, t = [], []
        for k in keys:
            run = runs[k].get(e)
            if run and run.get("ok") and run.get("a1k") is not None:
                T.append([1000.0]); t.append(run["a1k"])
        M["ro"][e + "_amp"] = fit_linear_terms(T, t, floor=1e-7)
    # HSF amplitudes: set-up (KL partition, segments) + path sums, linear in
    # [G n, 2^keff G 2^max(nA, nB)] (a log-linear fit in R_amp has slope 0.39
    # and badly under-predicts large path counts)
    Ta, ta = [], []
    for k in keys:
        f, run = feat[k], runs[k].get("hsf")
        if run and run.get("ok"):
            for rq, m in AMPS.items():
                if run.get(rq) is not None:
                    Ta.append(hsf_amp_terms(f)); ta.append(run["evolve"] + run[rq])
    M["hsf_amp"] = fit_linear_terms(Ta, ta, floor=1e-6)
    return M


def hsf_amp_terms(f):
    g = max(f["gates"], 1)
    return [g * f["n"], 2.0 ** f["hsf_keff"] * g * 2.0 ** max(f["hsf_na"], f["hsf_nb"])]


def lin(c, T):
    return float(np.dot(c, T)) if c is not None else float("inf")


def predict(M, e, f, rq):
    if not applicable(e, f, rq):
        return None
    n = f["n"]
    if e == "tableau":
        if rq == "e":
            return 0.0
        c = M["ro"].get("tableau_samp")
        return lin(c, readout_terms("tableau", f)["samp"](SHOTS[rq])) if c else 1e-4
    if rq == "e":
        m = M["E"].get(e)
        return 2 ** (m[0] + m[1] * resource(e, f)) if m else None
    if rq in AMPS and e == "hsf":
        return lin(M["hsf_amp"], hsf_amp_terms(f))
    m = M["state"].get(e)
    if not m:
        return None
    st = 2 ** (m[0] + m[1] * resource(e, f))
    terms = readout_terms(e, f)
    ro = M["ro"]
    if rq in SHOTS:
        S = SHOTS[rq]
        if e in ("sv", "hsf"):
            return st + lin(ro.get("sv_samp"), terms["samp"](S))
        if e == "sparse":
            return st + lin(ro.get("sparse_samp"), terms["samp"](S))
        if e == "mps":
            return st + lin(ro.get("mps_prep"), terms["canon"]) + lin(ro.get("mps_samp"), terms["samp"](S))
        if e == "cstate":
            return st + lin(ro.get("cstate_prep"), terms["build"]) + lin(ro.get("cstate_samp"), terms["samp"](S))
    if rq in AMPS:
        m_ = AMPS[rq]
        if e in ("sv", "sparse"):
            return st + lin(ro.get(e + "_amp"), [m_])
        if e == "mps":
            return st + lin(ro.get("mps_amp"), terms["amp"](m_))
    return None


# ---------------------------------------------------------------- scoring

def truth(feat, runs, k, rq):
    out = {}
    f = feat[k]
    for e in ENG:
        if not applicable(e, f, rq):
            continue
        if e == "tableau" and not f["clifford"]:
            continue
        run = runs[k].get(e)
        t = measured(run, e, rq) if run else None
        out[e] = t if t is not None else CENS
    return out


def score(rows):
    if not rows:
        return {}
    reg = np.array([r[0] for r in rows])
    eps = np.array([r[1] for r in rows])
    top = np.array([r[2] for r in rows])
    worst = max(rows, key=lambda r: r[0])
    return dict(n=len(rows), top1=float(top.mean()), geo=float(10 ** np.mean(np.log10(reg))),
                within2=float(np.mean(reg <= 2)), geo_eps=float(10 ** np.mean(np.log10(eps))),
                worst=float(reg.max()), worst_eps=float(eps.max()), worst_case=worst[3])


def evaluate(feat, runs, keys, choose):
    """choose(k, rq) -> engine. Returns per-request scores."""
    out = {}
    for rq in REQS:
        rows = []
        for k in keys:
            tr = truth(feat, runs, k, rq)
            if not tr:
                continue
            best_e, best = min(tr.items(), key=lambda x: x[1])
            if best >= CENS:
                continue
            e = choose(k, rq)
            t = tr.get(e, CENS)
            rows.append((t / best, (t + EPS) / (best + EPS), e == best_e,
                         f"{k[0]} -> {e} {t:.4g}s (best {best_e} {best:.4g}s)", fam(k[0])))
        out[rq] = score(rows)
        out[rq]["by_family"] = {fm: score([r for r in rows if r[4] == fm]) for fm in FAMS}
        for fm in FAMS:
            out[rq]["by_family"][fm].pop("worst_case", None)
    return out


def main():
    outdir, feat_path, *req_path = sys.argv[1:]
    os.makedirs(outdir, exist_ok=True)
    feat, runs = load(req_path, feat_path)
    keys = sorted(k for k in feat if k in runs)
    print(f"{len(keys)} instances with data")
    res = {}

    def pick(M):
        def choose(k, rq):
            f = feat[k]
            c = [(predict(M, e, f, rq), e) for e in ENG]
            c = [(p, e) for p, e in c if p is not None]
            if f["clifford"] and rq != "e" and rq[0] != "a":
                return "tableau"
            if f["clifford"] and rq == "e":
                return "tableau"
            return min(c)[1] if c else "sv"
        return choose

    # baselines recorded by the binary
    res["rule"] = evaluate(feat, runs, keys, lambda k, rq: feat[k]["choices"][rq]["rule"])
    res["v1_reused"] = evaluate(feat, runs, keys, lambda k, rq: feat[k]["choices"][rq]["v1"])
    # in-sample v2
    M_all = fit_all(keys, feat, runs)
    res["v2_insample"] = evaluate(feat, runs, keys, pick(M_all))
    M_real = fit_all(keys, feat, runs, real=True)
    # LOFO v2
    lofo_choice = {}
    for fm in FAMS:
        train = [k for k in keys if fam(k[0]) != fm]
        test = [k for k in keys if fam(k[0]) == fm]
        if not test:
            continue
        M = fit_all(train, feat, runs)
        ch = pick(M)
        for k in test:
            for rq in REQS:
                lofo_choice[(k, rq)] = ch(k, rq)
    res["v2_lofo"] = evaluate(feat, runs, keys, lambda k, rq: lofo_choice[(k, rq)])
    json.dump(dict(models=M_all, models_real_sizes=M_real, results=res), open(os.path.join(outdir, "report.json"), "w"),
              indent=1, default=str)
    # table
    lines = []
    for rq in REQS:
        lines.append(f"\n== request {rq}")
        lines.append(f"{'rule':12s} {'n':>4s} {'top1':>6s} {'geo':>7s} {'<=2x':>6s} {'geo_eps':>8s} {'worst':>8s}  worst case")
        for name in ["rule", "v1_reused", "v2_insample", "v2_lofo"]:
            s = res[name][rq]
            if not s:
                continue
            lines.append(f"{name:12s} {s['n']:4d} {s['top1']:6.3f} {s['geo']:7.3f} {s['within2']:6.3f} {s['geo_eps']:8.3f} {s['worst']:8.1f}  {s['worst_case']}")
        bf = res["v2_lofo"][rq]["by_family"]
        lines.append("   v2_lofo by family: " + "  ".join(
            f"{fm} {bf[fm].get('top1', 0):.2f}/{bf[fm].get('geo', 0):.2f}/{bf[fm].get('worst', 0):.1f}" for fm in FAMS if bf[fm]))
    lines.append("\nmodels (all data): " + json.dumps(M_all, default=str))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(outdir, "stdout.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
