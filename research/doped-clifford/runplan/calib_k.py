"""Sweeps K needed for sd(F_hat) <= 0.05 / 0.02 using SUTD exact amplitudes; B = 2^m amplitudes per sweep."""
import numpy as np
rng = np.random.default_rng(3)
d = np.load('../sutd/data-for-doped-clifford-tn-simulation/data/amplitude_batches.npz')
E = d['raw_vectors'].astype(np.complex128) * float(d['recovery_factor']) * 2.0 ** 35
def sd(F, B, K, reps=300):
    out = []
    for _ in range(reps):
        rows = rng.choice(2051, K, replace=False); sub = rng.integers(0, 256 // B, K)
        e = np.concatenate([E[r, s*B:(s+1)*B] for r, s in zip(rows, sub)])
        g = (rng.standard_normal(e.shape) + 1j*rng.standard_normal(e.shape)) / np.sqrt(2)
        l = np.sqrt(F)*e + np.sqrt(1-F)*g
        out.append(abs(np.vdot(e, l))**2 / (np.vdot(e, e).real*np.vdot(l, l).real))
    return np.std(out), np.mean(out)
print('| F | m (amps/sweep) | sweeps for sd<=0.05 | sweeps for sd<=0.02 |'); print('|---|---|---|---|')
for F in [0.43, 0.85, 0.96]:
    for m in [0, 1, 4, 6, 8]:
        B = 2**m; res = []
        for tgt in [0.05, 0.02]:
            K = 1
            while K < 2000:
                s, _ = sd(F, B, K)
                if s <= tgt: break
                K = max(K+1, int(K*1.25))
            res.append(K)
        print(f'| {F} | {m} ({B}) | {res[0]} | {res[1]} |')
