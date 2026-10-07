"""Does K-run dithered averaging (F_K = 1/(1+(1/F1-1)/K)) ever beat nearest rounding at 64 GB?
Total sweeps = K * N5(X), X = F_K * 63/65 (tail m=6 sampler), N5 = 25(1+2X-X^2)/(X-0.044)^2."""
import math
def N(X, k=5): return k*k*(1+2*X-X*X)/(X-0.044)**2 if X > 0.044 else math.inf
R = 66
# r per rounding: nearest from LOWPREC; dz assumed 1.2x nearest (int4:b64: dz F1 0.33 vs nearest 0.40 at R=70)
fmts = {'int4:b64': (0.012, 34), 'int4:b16:h': (0.009, 36), 'int5:b64': (0.0026, 42), 'int5:b16:h': (0.002, 44), 'int6:b64': (6.2e-4, 50)}
print('| format | store | nearest F | nearest sweeps (5σ) | dz F1 | best K | F_K | dz total sweeps (5σ) |')
print('|---|---|---|---|---|---|---|---|')
for f, (r, gib) in fmts.items():
    Fn = math.exp(-r*R); sn = N(Fn*63/65)
    F1 = math.exp(-1.2*r*R)
    best = min(((K*N(1/(1+(1/F1-1)/K)*63/65), K) for K in range(1, 400)))
    FK = 1/(1+(1/F1-1)/best[1])
    print(f'| {f} | {gib} GiB | {Fn:.3f} | {sn:.0f} | {F1:.3f} | {best[1]} | {FK:.3f} | {best[0]:.0f} |')
# RNS exact: 55 channel-sweeps per batch of 2^m exact amplitudes
print()
for m in [0, 6]:
    X = 1.0*(2**m-1)/(2**m+1) if m else 1.0
    print(f'RNS as sampler, tail m={m}: XEB {X:.3f}, N5 {N(X):.0f}, channel-sweeps {55*N(X):.0f}')
