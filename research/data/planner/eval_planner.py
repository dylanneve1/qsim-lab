#!/usr/bin/env python3
"""End-to-end planner evaluation on the Mac (research/planner.md §5).

1. The 314 dataset instances: `planx` (Planner v0 as shipped: plan + run,
   speculation on, no certificate), `planp` (+ probe-or-solve), `plan`
   (+ certificate) measured end to end, against the best state engine
   measured in the original sweep (research/data/simulability/raw).
2. The held-out families hea / qft (never used for fitting): every engine
   plus the planner, measured in one session; and the MPS time predictors
   fitted on the 314 dataset, tested on these families.

    eval_planner.py OUTDIR
"""
import csv, glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fit_planner as fp  # noqa: E402

STATE = ["sv", "sparse", "mps", "hsf", "tableau", "cstate"]
L2 = math.log10(2)


def load(paths):
    out = {}
    for p in paths:
        for r in csv.DictReader(open(p)):
            k = (r["spec"], r["seed"])
            d = out.setdefault(k, dict(family=r["family"], f=json.loads(r["features"]), runs={}))
            d["runs"][r["engine"]] = (r["status"], float(r["secs"]) if r["status"] == "ok" else None,
                                      float(r["wall"]) if r["wall"] else None, r["note"])
    return out


def best(d, engines=STATE):
    c = [(d["runs"][e][1], e) for e in engines if d["runs"].get(e, ("",))[0] == "ok"]
    return min(c) if c else None


def chosen(note):
    for part in note.split():
        if part.startswith("engine="):
            return part[7:]
    return None


def regret_table(rows):
    if not rows:
        return {}
    x = np.array([r[0] for r in rows])
    eps = np.array([r[1] for r in rows])
    return dict(n=len(x), geo=float(10 ** np.mean(np.log10(x))), within2=float(np.mean(x <= 2)),
                max=float(x.max()), geo_eps=float(10 ** np.mean(np.log10(eps))),
                max_eps=float(eps.max()))


def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    rep = {}
    raw = load([p for p in glob.glob(os.path.join(fp.RAW, "*.csv"))
                if "mid2" not in p and "confirm" not in p])
    plan = load(glob.glob(os.path.join(HERE, "mac", "plan_*.csv")))
    # --- 1. dataset, end to end -------------------------------------------
    for eng in ["planx", "planp", "plan", "mpsb"]:
        rows, choice_rows, worst = [], [], []
        for k, d in plan.items():
            r = d["runs"].get(eng)
            b = best(raw[k])
            if not r or not b:
                continue
            t = r[1] if r[0] == "ok" else 2 * 10.0
            if eng == "mpsb":
                m = raw[k]["runs"].get("mps")
                if m and m[0] == "ok" and r[0] == "ok" and m[1] >= 1e-3:
                    rows.append((m[1] / r[1], m[1] / r[1]))
                continue
            rows.append((t / b[0], (t + 1e-3) / (b[0] + 1e-3)))
            worst.append((t / b[0], k[0], chosen(r[3]), b[1], t, b[0]))
            c = chosen(r[3])
            if c and c in raw[k]["runs"] and c != "zero":
                rc = raw[k]["runs"][c]
                tc = rc[1] if rc[0] == "ok" else 20.0
                choice_rows.append((tc / b[0], (tc + 1e-3) / (b[0] + 1e-3), c == b[1]))
        if eng == "mpsb":
            x = np.array([r[0] for r in rows])
            rep["mpsb_speedup_vs_mps"] = dict(n=len(x), geo=float(10 ** np.mean(np.log10(x))),
                                              median=float(np.median(x)), min=float(x.min()),
                                              max=float(x.max()))
            continue
        rep[eng] = dict(end_to_end=regret_table(rows),
                        choice=regret_table([(a, b) for a, b, _ in choice_rows]),
                        top1=float(np.mean([c for _, _, c in choice_rows])) if choice_rows else None,
                        worst=sorted(worst, reverse=True)[:6])
    # planning overhead
    ps = []
    for k, d in plan.items():
        r = d["runs"].get("planx")
        if r and r[0] == "ok":
            for part in r[3].split():
                if part.startswith("plan_secs="):
                    ps.append(float(part[10:]))
    rep["plan_secs"] = dict(median=float(np.median(ps)), p90=float(np.percentile(ps, 90)),
                            max=float(np.max(ps)))
    # --- 2. held-out families --------------------------------------------
    new = load(glob.glob(os.path.join(HERE, "mac", "new_*.csv")))
    fams = {}
    for k, d in new.items():
        b = best(d)
        if not b:
            continue
        F = d["family"]
        for eng in ["planx", "planp"]:
            r = d["runs"].get(eng)
            if not r:
                continue
            t = r[1] if r[0] == "ok" else 20.0
            c = chosen(r[3])
            rc = d["runs"].get(c)
            tc = (rc[1] if rc and rc[0] == "ok" else 20.0)
            fams.setdefault((F, eng), []).append((t / b[0], (t + 1e-3) / (b[0] + 1e-3), tc / b[0],
                                                   c == b[1], k[0], c, b[1]))
    rep["heldout"] = {}
    for (F, eng), v in sorted(fams.items()):
        rep["heldout"][f"{F}/{eng}"] = dict(
            end_to_end=regret_table([(a, b) for a, b, *_ in v]),
            choice=regret_table([(c, c) for _, _, c, *_ in v]),
            top1=float(np.mean([x[3] for x in v])),
            misses=[(round(x[2], 2), x[4], x[5], x[6]) for x in sorted(v, key=lambda y: -y[2])[:5]])
    # winners per held-out family
    rep["heldout_winners"] = {}
    for k, d in new.items():
        b = best(d)
        if b:
            rep["heldout_winners"].setdefault(d["family"], {}).setdefault(b[1], 0)
            rep["heldout_winners"][d["family"]][b[1]] += 1
    # --- MPS predictors: fit on the 314 dataset, test on hea / qft ---------
    data = fp.load(os.path.join(HERE, "mpsdata_vps.jsonl"))
    tr = fp.mps_points(data)
    md = {}
    for line in open(os.path.join(HERE, "mpsdata_new_vps.jsonl")):
        x = json.loads(line)
        md[(x["spec"], x["seed"])] = x
    te = []
    for k, d in new.items():
        m = d["runs"].get("mps")
        if m and m[0] == "ok" and m[1] >= fp.T_FLOOR and k in md and "stats" in md[k].get("trace", {}):
            dd = dict(f=d["f"], m=md[k], family=d["family"])
            te.append((d["family"], math.log2(m[1]), dd))
    W, K = 8, 1000
    feats = {
        "old mps_l": lambda d: d["f"]["mps_l"],
        "replay[best]": lambda d: fp.lg(fp.total(d["m"]["cost"]["best"]["stats"], W, K)),
        "probe16": lambda d: fp.lg(fp.total(fp.probe_stats(d, "16"), W, K)),
        "oracle": lambda d: fp.lg(fp.total(d["m"]["trace"]["stats"], W, K)),
    }
    rep["mps_heldout_families"] = {}
    for name, fn in feats.items():
        x = np.array([fn(p[2]) for p in tr])
        y = np.array([p[1] for p in tr])
        c = fp.ols(x, y)
        res = {}
        for F in sorted(set(p[0] for p in te)):
            xs = np.array([fn(p[2]) for p in te if p[0] == F])
            ys = np.array([p[1] for p in te if p[0] == F])
            e = ys - (c[0] + c[1] * xs)
            res[F] = dict(n=len(xs), rmse=float(np.sqrt(np.mean(e ** 2)) * L2),
                          max=float(np.max(np.abs(e)) * L2))
        rep["mps_heldout_families"][name] = res
    json.dump(rep, open(os.path.join(outdir, "eval_report.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("heldout",)}, indent=1, default=str)[:6000])
    for k, v in rep["heldout"].items():
        print(k, json.dumps(v["end_to_end"]), "choice", json.dumps(v["choice"]), "top1", round(v["top1"], 2))
        print("   misses", v["misses"])


if __name__ == "__main__":
    main()
