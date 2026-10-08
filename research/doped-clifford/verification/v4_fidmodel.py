#!/usr/bin/env python3
"""Task 4: fidelity of the production engine (int5:b16:h, R=65, m=6, tail-open run loop) at depths where the
exact register fits on a 16 GB Mac, against exact f64 tail batches of the same run loop (cpu64 backend), and its
extrapolation to D=70 (F = exp(-r R)). Inputs: verification/mac/{ref,gpu5,cpu5}_dD.jsonl, mac/run2.log."""
import json, os, re, glob, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); M = os.path.join(H, 'mac')
rng = np.random.default_rng(3)
def load(p):
    d = {}
    if not os.path.exists(p): return d
    for l in open(p):
        r = json.loads(l)
        if r['kind'] == 'sample':
            a = np.array(r['amps']); d[r['i']] = (r, a[:, 0] + 1j * a[:, 1])
    return d
def F(es, ls):
    num = sum(np.vdot(e, l) for e, l in zip(es, ls))
    return abs(num) ** 2 / (sum(np.vdot(e, e).real for e in es) * sum(np.vdot(l, l).real for l in ls))
def Fal(es, ls):  # per-batch phase aligned (what the sampler sees)
    return F(es, [l * np.exp(-1j * np.angle(np.vdot(e, l))) for e, l in zip(es, ls)])
out = {}
for D in (48, 52, 56):
    ref = load(f'{M}/ref_d{D}.jsonl')
    for tag in ('gpu5', 'cpu5'):
        t = load(f'{M}/{tag}_d{D}.jsonl'); keys = sorted(set(ref) & set(t))
        if not keys: continue
        assert all(ref[k][0]['prefix_bits'] == t[k][0]['prefix_bits'] for k in keys)
        E = [ref[k][1] for k in keys]; L = [t[k][1] for k in keys]; K = len(keys)
        f = F(E, L)
        jk = np.array([F(E[:j] + E[j+1:], L[:j] + L[j+1:]) for j in range(K)]) if K > 2 else np.array([f])
        jse = float(np.sqrt((K - 1) / K * ((jk - jk.mean()) ** 2).sum())) if K > 2 else float('nan')
        bs = [F([E[i] for i in ix], [L[i] for i in ix]) for ix in (rng.integers(0, K, K) for _ in range(2000))]
        R = t[keys[0]][0]['R']
        z = lambda e: abs(e) ** 2 * 2.0 ** 70
        xo = np.mean([(abs(l) ** 2 / (abs(l) ** 2).sum() * z(e)).sum() - 1 for e, l in zip(E, L)])
        xe = np.mean([(z(e) / z(e).sum() * z(e)).sum() - 1 for e in E])
        r = -np.log(f) / R
        out[f'D{D}_{tag}'] = {'batches': K, 'amplitudes': K * 64, 'R': R, 'F': float(f), 'F_se_jk': jse, 'F_se_boot_batches': float(np.std(bs)),
                              'F_phase_aligned': float(Fal(E, L)), 'F_per_batch_mean': float(np.mean([F([e], [l]) for e, l in zip(E, L)])),
                              'r=-lnF/R': float(r), 'r_se': float(np.std(bs) / f / R), 'pred_F_D70_R65': float(np.exp(-r * 65)),
                              'xeb_ratio': float(xo / xe), 't_sweep_s_mean': float(np.mean([t[k][0]['t_sweep_s'] for k in keys])),
                              'underflow+overflow': int(sum(t[k][0]['underflow'] + t[k][0]['overflow'] for k in keys)),
                              'backend': t[keys[0]][0]['backend']}
    g, c = load(f'{M}/gpu5_d{D}.jsonl'), load(f'{M}/cpu5_d{D}.jsonl'); ks = sorted(set(g) & set(c))
    if ks:
        diff = max(float(np.max(abs(g[k][1] - c[k][1])) / np.sqrt(np.mean(abs(c[k][1]) ** 2))) for k in ks)
        out[f'D{D}_gpu_vs_cpu_packed'] = {'batches': len(ks), 'bit_identical_batches': sum(bool(np.array_equal(g[k][1], c[k][1])) for k in ks),
                                         'max_abs_diff_over_rms': diff, 'F_between': float(F([c[k][1] for k in ks], [g[k][1] for k in ks]))}
    if ref:
        rr = [v[0] for v in ref.values()]
        out[f'D{D}_ref'] = {'batches': len(rr), 'R': rr[0]['R'], 'backend': rr[0]['backend'], 't_sweep_s_mean': float(np.mean([x['t_sweep_s'] for x in rr]))}
# tailcheck D=40 (per-completion exact chain-sweep reference)
log = os.path.join(M, 'run2.log')
if os.path.exists(log):
    fs = {}
    for l in open(log):
        m = re.search(r'tailcheck n=70 d=(\d+) m=6 trial=(\d+) backend=(\S+) .*R=(\d+) .*F=([0-9.]+)', l)
        if m: fs.setdefault((int(m[1]), m[3], int(m[4])), []).append(float(m[5]))
    for (d, b, R), v in fs.items():
        v = np.array(v); out[f'tailcheck_D{d}_{b}'] = {'batches': len(v), 'R': R, 'F_mean_of_batches': float(v.mean()), 'se': float(v.std(ddof=1) / np.sqrt(len(v))),
                                                     'r': float(-np.log(v.mean()) / R), 'pred_F_D70_R65': float(np.exp(65 * np.log(v.mean()) / R))}
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(H, 'v4_fidmodel.json'), 'w'), indent=1)
