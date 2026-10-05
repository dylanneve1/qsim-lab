#!/usr/bin/env python3
"""Fit and validate the planner's tensor-network time model.

    python3 fit_tn.py calib.jsonl [--kappa K]

Model (seconds of one amplitude through `planner::tn_contract`, i.e. network
build + simplification + contraction along a planned tree):

    t = 2^(a + b log2 C) + c0 + c1 * tensors

with C the sliced contraction cost (complex multiply-adds). Fitted in log space
(least squares on log10 t) on every instance with a measured time; validated
leave-one-family-out (fit on three families, predict the fourth). Prints the
constants, per-family held-out RMSE (decades) and the Rust `TnModel` literal;
with --kappa (x86 time / M1 time of the state vector, measured by `svcal`) the
intercept is converted to M1 units (a - log2 kappa).
"""
import json
import math
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares


def load(path):
    path_lines, runs = {}, defaultdict(list)
    for line in open(path):
        d = json.loads(line)
        if d.get("stage") == "path":
            path_lines[d["label"]] = d
        elif d.get("stage") == "run" and "secs" in d:
            runs[d["label"]].append(d["secs"])
    rows = []
    for lab, p in path_lines.items():
        if lab not in runs:
            continue
        fam = lab.split(":")[0]
        t = min(runs[lab]) + p["simplify_secs"]
        rows.append((fam, lab, p["log10_sliced_flops"], p["tensors"], t))
    return rows


def model(x, l10c, tens):
    a, b, lc0, lc1 = x
    l2 = l10c / math.log10(2)
    return np.exp2(a + b * l2) + np.exp(lc0) + np.exp(lc1) * tens


def fit(rows):
    l10c = np.array([r[2] for r in rows])
    tens = np.array([r[3] for r in rows], dtype=float)
    t = np.array([r[4] for r in rows])
    res = least_squares(
        lambda x: np.log10(model(x, l10c, tens)) - np.log10(t),
        x0=[-30.0, 1.0, math.log(2e-4), math.log(2e-6)],
    )
    return res.x


def rmse(x, rows):
    l10c = np.array([r[2] for r in rows])
    tens = np.array([r[3] for r in rows], dtype=float)
    t = np.array([r[4] for r in rows])
    e = np.log10(model(x, l10c, tens)) - np.log10(t)
    return float(np.sqrt(np.mean(e * e))), float(np.max(np.abs(e)))


def main():
    rows = load(sys.argv[1])
    kappa = 1.0
    if "--kappa" in sys.argv:
        kappa = float(sys.argv[sys.argv.index("--kappa") + 1])
    fams = sorted({r[0] for r in rows})
    print(f"{len(rows)} instances, families {fams}")
    x = fit(rows)
    r, w = rmse(x, rows)
    print(f"all: a={x[0]:.4f} b={x[1]:.4f} c0={math.exp(x[2]):.3e} c1={math.exp(x[3]):.3e}  "
          f"in-sample RMSE {r:.3f} dec, worst {w:.3f}")
    pooled = []
    for f in fams:
        tr = [q for q in rows if q[0] != f]
        te = [q for q in rows if q[0] == f]
        xf = fit(tr)
        rf, wf = rmse(xf, te)
        pooled += [(rf, len(te))]
        print(f"held out {f:7s} n={len(te):3d}: RMSE {rf:.3f} dec, worst {wf:.3f} "
              f"(a={xf[0]:.3f} b={xf[1]:.3f})")
    tot = sum(n for _, n in pooled)
    print(f"pooled held-out RMSE {math.sqrt(sum(r*r*n for r, n in pooled)/tot):.3f} dec")
    a_m1 = x[0] - math.log2(kappa)
    print("Rust literal (M1 units):")
    print(f"TnModel {{ a: {a_m1:.4f}, b: {x[1]:.4f}, c0: {math.exp(x[2]) / kappa:.4e}, "
          f"c1: {math.exp(x[3]) / kappa:.4e} }}")


if __name__ == "__main__":
    main()
