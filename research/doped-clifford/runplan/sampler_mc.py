"""Monte Carlo for sampler designs on an approximate amplitude engine.
Model (validated in LOWPREC.md at D<=48, xeb ratio ~ F): l = sqrt(F) e + sqrt(1-F) g,
e, g iid CN(0,1) per bitstring (Porter-Thomas; checked at D=70 on SUTD amplitudes).
For each design: XEB = E[z(sample)] - 1 against IDEAL z, Var of z(sample), sweeps per sample,
N for 3 sigma over 0.044 and over IBM's 0.342 (+-0.028), total sweeps."""
import numpy as np, sys
rng = np.random.default_rng(1)
T = 400_000  # proposals per design

def cn(shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)

def design(F, B, mode, M=None):
    T = max(20_000, 400_000 // max(1, B // 4))
    e = cn((T, B)); l = np.sqrt(F) * e + np.sqrt(1 - F) * cn((T, B))
    z = np.abs(e) ** 2; zt = np.abs(l) ** 2
    St = zt.sum(1)
    # pick suffix b with prob zt_b / St (batched / pair); B=1 trivially b=0
    u = rng.random(T)
    c = np.cumsum(zt, 1) / St[:, None]
    b = (c < u[:, None]).sum(1).clip(max=B - 1)
    zs = z[np.arange(T), b]
    if mode == 'batch':          # uniform prefix, no rejection: 1 sweep/sample
        w = np.ones(T)
    else:                        # rejection on prefix weight St/B (mean 1), cutoff M
        w = np.minimum(1, (St / B) / M)
    acc = w.mean()
    xeb = (w * zs).sum() / w.sum() - 1
    var = (w * zs ** 2).sum() / w.sum() - (xeb + 1) ** 2
    return xeb, var, 1 / acc

def nreq(xeb, var, bar, bar_se=0.0, k=3):
    d = xeb - bar
    if d <= k * bar_se: return float('inf')
    return k * k * var / (d * d - k * k * bar_se ** 2)

rows = []
Fs = [float(x) for x in sys.argv[1:]] or [0.43, 0.53, 0.83, 0.87, 0.96, 0.996]
for F in Fs:
    cands = []
    for M in [2, 3, 4, 6, 8, 10]:
        cands.append(('single rejection', 1, 'rej', M))
    for M in [1.5, 2, 3, 4, 6]:
        cands.append(('pair (last qubit open) rejection', 2, 'rej', M))
    for m in [1, 4, 6, 8]:
        cands.append((f'tail-open m={m} batched, uniform prefix', 2 ** m, 'batch', None))
    for m in [4, 6]:
        for M in [1.2, 1.5, 2]:
            cands.append((f'tail-open m={m} + prefix rejection', 2 ** m, 'rej', M))
    for name, B, mode, M in cands:
        x, v, s = design(F, B, mode, M)
        n3 = nreq(x, v, 0.044); n5 = nreq(x, v, 0.044, k=5); nibm = nreq(x, v, 0.342, 0.028)
        rows.append((F, name, M, x, v, s, n3, s * n3, n5, s * n5, nibm, s * nibm))
print('| F | design | M | XEB | Var z | sweeps/sample | N 3σ>0.044 | sweeps | N 5σ>0.044 | sweeps | N 3σ>IBM 0.342±0.028 | sweeps |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for r in rows:
    F, name, M, x, v, s, n3, w3, n5, w5, ni, wi = r
    f = lambda q: '∞' if q == float('inf') else f'{q:.0f}'
    print(f'| {F} | {name} | {M if M else "-"} | {x:.3f} | {v:.2f} | {s:.2f} | {f(n3)} | {f(w3)} | {f(n5)} | {f(w5)} | {f(ni)} | {f(wi)} |')
