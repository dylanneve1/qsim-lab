"""Precision of the D=70 calibration against SUTD's exact batches (Zenodo 10.5281/zenodo.21912448).
K calibration sweeps, each returning B = 2^m tail amplitudes that are all in SUTD's set.
Estimators: pooled amplitude fidelity F_hat = |sum e* l|^2 / (sum|e|^2 sum|l|^2) and the
exact expected XEB of our batched sampler on those prefixes, X_hat = mean_a sum_b q~(b|a) z(a,b) - 1.
Noise model as sampler_mc.py; uses REAL SUTD exact amplitudes for e."""
import numpy as np, sys
rng = np.random.default_rng(2)
d = np.load('../sutd/data-for-doped-clifford-tn-simulation/data/amplitude_batches.npz')
E = d['raw_vectors'].astype(np.complex128) * float(d['recovery_factor']) * 2.0 ** 35  # |E|^2 = z
print('| F | m | B | K sweeps | F_hat mean ± sd | X_hat mean ± sd (exact-sampler X for same prefixes) |')
print('|---|---|---|---|---|---|')
for F in [0.43, 0.53, 0.85, 0.96]:
    for m in [4, 6, 8]:
        B = 2 ** m
        for K in [4, 8, 16]:
            fh, xh, x0 = [], [], []
            for rep in range(400):
                rows = rng.choice(2051, K, replace=False)
                sub = rng.integers(0, 256 // B, K)
                e = np.stack([E[r, s * B:(s + 1) * B] for r, s in zip(rows, sub)])
                g = (rng.standard_normal(e.shape) + 1j * rng.standard_normal(e.shape)) / np.sqrt(2)
                l = np.sqrt(F) * e + np.sqrt(1 - F) * g
                fh.append(abs((e.conj() * l).sum()) ** 2 / ((abs(e) ** 2).sum() * (abs(l) ** 2).sum()))
                z = abs(e) ** 2; zt = abs(l) ** 2
                xh.append(((zt / zt.sum(1, keepdims=True)) * z).sum(1).mean() - 1)
                x0.append(((z / z.sum(1, keepdims=True)) * z).sum(1).mean() - 1)
            print(f'| {F} | {m} | {B} | {K} | {np.mean(fh):.3f} ± {np.std(fh):.3f} | {np.mean(xh):.3f} ± {np.std(xh):.3f} ({np.mean(x0):.3f}) |')
