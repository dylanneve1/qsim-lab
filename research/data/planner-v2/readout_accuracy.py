#!/usr/bin/env python3
"""Accuracy of the read-out models (research/simulability/planner-v2.md §2): per
component, RMSE (log10) of the linear op-count model, fitted on all
instances and leave-one-family-out, on read-outs >= 20 µs.
    readout_accuracy.py feat.jsonl req.jsonl hsfamp.jsonl
"""
import sys, math
import numpy as np
import fit_v2 as fv

feat, runs = fv.load(sys.argv[2:], sys.argv[1])
keys = sorted(k for k in feat if k in runs)


def rows(component, keys):
    out = []
    for k in keys:
        f = feat[k]
        for e in fv.ENG:
            run = runs[k].get(e)
            if not run or not run.get("ok"):
                continue
            t = fv.readout_terms(e, f, run)
            if component == "sv_samp" and e in ("sv", "hsf") or component == "sparse_samp" and e == "sparse" \
                    or component == "mps_samp" and e == "mps" or component == "tableau_samp" and e == "tableau":
                for rq, S in fv.SHOTS.items():
                    if run.get(rq) is not None:
                        out.append((t["samp"](S), run[rq], fv.fam(k[0])))
            if component == "cstate_samp" and e == "cstate":
                for rq, S in fv.SHOTS.items():
                    if run.get(rq) is not None:
                        out.append((t["samp"](S), run[rq], fv.fam(k[0])))
            if component == "mps_prep" and e == "mps" and run.get("prep") is not None:
                out.append((t["canon"], run["prep"], fv.fam(k[0])))
            if component == "cstate_prep" and e == "cstate" and run.get("prep") is not None:
                out.append((t["build"], run["prep"], fv.fam(k[0])))
            if component == "mps_amp" and e == "mps":
                for rq, m in fv.AMPS.items():
                    if run.get(rq) is not None:
                        out.append((t["amp"](m), run[rq], fv.fam(k[0])))
            if component == "hsf_amp" and e == "hsf":
                for rq, m in fv.AMPS.items():
                    if run.get(rq) is not None:
                        out.append((fv.hsf_amp_terms(f), run["evolve"] + run[rq], fv.fam(k[0])))
    return [r for r in out if r[1] >= 2e-5]


for comp in ["sv_samp", "sparse_samp", "mps_samp", "mps_prep", "mps_amp", "cstate_samp", "cstate_prep",
             "tableau_samp", "hsf_amp"]:
    R = rows(comp, keys)
    if len(R) < 5:
        continue
    T = [r[0] for r in R]; t = np.array([r[1] for r in R])
    c = fv.fit_linear_terms(T, t, floor=2e-5)
    pred = np.array(T) @ np.array(c)
    ins = math.sqrt(np.mean(np.log10(pred / t) ** 2))
    errs = []
    for fm in fv.FAMS:
        tr = [r for r in R if r[2] != fm]; te = [r for r in R if r[2] == fm]
        if not te or len(tr) < 5:
            continue
        cc = fv.fit_linear_terms([r[0] for r in tr], [r[1] for r in tr], floor=2e-5)
        if cc is None:
            continue
        p = np.array([r[0] for r in te]) @ np.array(cc)
        errs += list(np.log10(p / np.array([r[1] for r in te])))
    lofo = math.sqrt(np.mean(np.array(errs) ** 2))
    print(f"{comp:13s} n={len(R):4d} coef={['%.3g' % x for x in c]} in-sample RMSE {ins:.3f}  LOFO {lofo:.3f} "
          f"(log10), median {np.median(t) * 1e3:.3g} ms")
