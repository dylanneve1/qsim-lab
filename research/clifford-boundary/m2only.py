#!/usr/bin/env python3
"""M2 (stabilizer Renyi-2 entropy) and raw middle-cut entropy of boundary dumps; no disentangling."""
import sys, numpy as np, diag, diag2
rng = np.random.default_rng(11); S = int(sys.argv[1])
for f in sys.argv[2:]:
    m, T = diag.load(f); M2, se, haar = diag2.m2(T, S, rng)
    print(f"M2ONLY {f} m={m} M2={M2:.2f}+-{se:.2f} haar={haar:.2f} gap={haar-M2:.2f}", flush=True)
