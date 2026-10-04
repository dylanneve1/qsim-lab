"""Fit the growth of the misplaced fraction δ̄(k)·2^c and the unfaithful
fraction with the number of windows k (31-bit EH MC logs)."""
import re, sys, glob
import numpy as np
for f in sorted(glob.glob("mc31_eh_a*_c*.log")):
    rows = [l.split() for l in open(f) if re.match(r"^\s+\d+ \(", l)]
    k = np.array([int(r[0]) for r in rows], float)
    # columns after the offset "(ox,oy)" token(s): find numbers
    nums = [[float(x) for x in re.findall(r"-?\d+\.\d+", " ".join(r[1:]))] for r in rows]
    d = np.array([n[0] for n in nums]); unf = np.array([n[3] for n in nums]); var = np.array([n[5] for n in nums])
    # model d = a*sqrt(k) + b*k
    A = np.vstack([np.sqrt(k), k]).T
    (a, b), res, *_ = np.linalg.lstsq(A, d, rcond=None)
    rl = np.polyfit(k, d, 1); rs = np.polyfit(np.sqrt(k), d, 1)
    ru = np.polyfit(k, unf, 1); rv = np.polyfit(k, var, 1)
    lin_err = np.sqrt(np.mean((np.polyval(rl, k) - d) ** 2))
    sq_err = np.sqrt(np.mean((np.polyval(rs, np.sqrt(k)) - d) ** 2))
    mix_err = np.sqrt(np.mean((A @ [a, b] - d) ** 2))
    print(f"{f}: dbar*2^c(k): linear {rl[0]:.3f}k{rl[1]:+.2f} (rms {lin_err:.2f}) | sqrt {rs[0]:.2f}√k{rs[1]:+.2f} (rms {sq_err:.2f}) | {a:.2f}√k+{b:.3f}k (rms {mix_err:.2f}) | unfaithful*2^c ≈ {ru[0]:.3f}/window | var(dJ) ≈ {rv[0]:.2f}/window")
