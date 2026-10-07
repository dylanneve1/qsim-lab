#!/usr/bin/env python3
"""Calibration: the restarted greedy disentangler on Haar-random states (writes dumps in dumpcut's format)."""
import numpy as np, sys
rng = np.random.default_rng(5)
for m in map(int, sys.argv[1:]):
    v = rng.normal(size=2 ** m) + 1j * rng.normal(size=2 ** m); v /= np.linalg.norm(v)
    v.astype(np.complex128).view(np.float64).tofile(f'haar_m{m}.bin')
