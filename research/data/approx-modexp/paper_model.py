"""The paper's masked success-rate model, evaluated exactly.

`facto/algorithm/sim/main1_sample_masked_success_rates.py` (Gidney 2025
release) estimates the success rate of masked period finding by Monte Carlo:
measured = (signal[r] + u) mod N for random r < P (the period) and u < W (the
mask width), the kept set K = {r : (measured - signal[r]) mod N < W}, an FFT
of the uniform superposition over K modulo P (the idealised mod-P QFT), and
success iff the continued-fraction denominator d of k/P gives
1 < gcd(g^(d/2) + 1, N) < N (the paper's `C.success_mask`).

Here the same model (the paper's own class C for the signal and the success
mask) is averaged EXACTLY over every measured value: P(measured) = |K|/(P W).

Usage: python3 paper_model.py N g W1 [W2 ...]   (W in units of N, i.e. full width)
"""

from __future__ import annotations

import math
import sys

import numpy as np

import gidney_env  # noqa: F401  (puts the release on sys.path)
from facto.algorithm.sim.main1_sample_masked_success_rates import C


def exact_model(n_mod: int, g: int, w: int) -> tuple[float, int]:
    c = C(modulus=n_mod, g=g, use_randomization=False)
    p = c.period
    sig = c.signal.astype(np.int64)
    ms = np.unique(((sig[:, None] + np.arange(w)[None, :]) % n_mod).ravel())
    total = 0.0
    mask = c.success_mask.astype(np.float64)
    for m in ms:
        kept = ((m - sig) % n_mod) < w
        k = int(kept.sum())
        amps = kept.astype(np.float64) / math.sqrt(k)
        dist = np.abs(np.fft.fft(amps, norm="ortho")) ** 2
        total += k / (p * w) * float(np.dot(dist, mask))
    return total, p


def main(argv: list[str]) -> int:
    n_mod, g = int(argv[0]), int(argv[1])
    for w in map(int, argv[2:]):
        s, p = exact_model(n_mod, g, w)
        print(f"model N={n_mod} g={g} period={p} W={w} S=W/N={w / n_mod:.6f} success={s:.10f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
