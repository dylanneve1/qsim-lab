#!/usr/bin/env python3
"""Paired comparison table from the per-shot fail vectors of every decoder on one test file.
usage: summarize.py <out-prefix> <rounds> <reference-decoder> [decoders...]
Rows: decoder, fails/shots, p_L/round [95% CI], ratio vs reference on the common shots [paired 95% CI]."""
import sys, glob, os
import numpy as np
from nd_common import wilson, per_round, paired_ratio

out, rounds, ref = sys.argv[1], int(sys.argv[2]), sys.argv[3]
names = sys.argv[4:] or sorted(os.path.basename(f)[len(os.path.basename(out)) + 1:-len(".fails.npy")]
                               for f in glob.glob(out + ".*.fails.npy"))
F = {n: np.load(f"{out}.{n}.fails.npy") for n in names + [ref]}
print("| decoder | fails / shots | p_L per round [95% CI] | ratio vs " + ref + " (same shots) [95% CI] |")
print("|---|---|---|---|")
for n in names:
    f = F[n]; k = int(f.sum()); N = len(f); lo, hi = wilson(k, N)
    m = min(N, len(F[ref]))
    r, rl, rh, cells = paired_ratio(f[:m], F[ref][:m])
    rr = "—" if n == ref else f"{r:.3f} [{rl:.3f}, {rh:.3f}] (n={m})"
    print(f"| {n} | {k} / {N} | {per_round(k / N, rounds):.3e} [{per_round(lo, rounds):.3e}, {per_round(hi, rounds):.3e}] | {rr} |")
