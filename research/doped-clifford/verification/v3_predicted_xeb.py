#!/usr/bin/env python3
"""Verification task 3: predicted XEB of the 74 production samples from first principles.

Porter-Thomas (PT) model: ideal amplitudes of a batch (one uniform prefix, B = 2^m completions) are iid complex
Gaussians, z = 2^n |e|^2 ~ Exp(1). The tail sampler draws j with prob |l_j|^2 / sum|l|^2. Its linear XEB is
E[z_j] - 1. Noise models for l (all fitted to the four D=70 calibration records, scale-free):
  A  absolute noise:  l = e + g, g iid CN(0, c) in z units (rounding noise set by the register, not the batch)
  R  relative noise:  l = sqrt(F) e + sqrt(1-F) |e|_rms g  (fixed per-batch fidelity F)
  E  empirical:       l = e + r, r = a residual vector of a real calibration record (scaled to z units), permuted
Outputs: predicted XEB, Var z, and the distribution of the 74-sample mean (exact MC, no Gaussian approximation).
"""
import json, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(2026)
B, NS, n = 64, 74, 70
Z = np.load(os.path.join(ROOT, 'sutd/data-for-doped-clifford-tn-simulation/data/amplitude_batches.npz'))
EX = Z['raw_vectors'].astype(np.complex128) * float(Z['recovery_factor'])
recs = [json.loads(l) for l in open(os.path.join(ROOT, 'production/session2/out_final74.jsonl'))]
cal = [r for r in recs if r['kind'] == 'calib']
Es, Ls = [], []
for r in cal:
    sub = int(r['prefix_bits'][6:8], 2); e = EX[r['row'], sub::4] * 2 ** 35  # z-units: |e|^2 = z
    a = np.array(r['amps']); l = a[:, 0] + 1j * a[:, 1]
    alpha = np.vdot(l, e) / np.vdot(l, l)            # best complex scale mapping l onto e
    Es.append(e); Ls.append(alpha * l)
out = {}
# exact-sampler XEB for B=64: analytic (B-1)/(B+1)
out['exact_sampler_xeb_analytic'] = (B - 1) / (B + 1)

def cn(shape, var=1.0):
    return np.sqrt(var / 2) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

def sample_z(model, par, T):
    """T batches -> sampled ideal z for each (one sample per batch, as in production)."""
    e = cn((T, B))
    if model == 'exact':
        l = e
    elif model == 'A':
        l = e + cn((T, B), par)
    elif model == 'R':
        rms = np.sqrt((abs(e) ** 2).mean(1, keepdims=True))
        l = np.sqrt(par) * e + np.sqrt(1 - par) * rms * cn((T, B))
    elif model == 'E':
        res = par  # list of residual vectors
        k = rng.integers(0, len(res), T)
        l = e + np.stack([res[i][rng.permutation(B)] * np.exp(2j * np.pi * rng.random()) for i in k])
    w = abs(l) ** 2; w /= w.sum(1, keepdims=True)
    u = rng.random(T)[:, None]
    j = (np.cumsum(w, 1) > u).argmax(1)
    return abs(e[np.arange(T), j]) ** 2, (w * abs(e) ** 2).sum(1) - 1

def fidelity_model(model, par, T=200000):
    e = cn((T, B))
    if model == 'A':
        l = e + cn((T, B), par)
    elif model == 'R':
        rms = np.sqrt((abs(e) ** 2).mean(1, keepdims=True)); l = np.sqrt(par) * e + np.sqrt(1 - par) * rms * cn((T, B))
    num = (np.conj(e) * l).sum(); return abs(num) ** 2 / ((abs(e) ** 2).sum() * (abs(l) ** 2).sum())

# fit model A: pooled F = SW/(SW + K c) with W = mean z of the record's 64 ideal amplitudes
W = np.array([(abs(e) ** 2).mean() for e in Es])
num = sum(np.vdot(e, l) for e, l in zip(Es, Ls)); Fp = abs(num) ** 2 / (sum((abs(e) ** 2).sum() for e in Es) * sum((abs(l) ** 2).sum() for l in Ls))
c_fit = W.sum() * (1 / Fp - 1) / len(W)
res = [l - (np.vdot(e, l) / np.vdot(e, e)) * e for e, l in zip(Es, Ls)]
c_emp = np.mean([(abs(r) ** 2).mean() for r in res])
out['calib'] = {'F_pooled': float(Fp), 'W_per_record': W.round(4).tolist(), 'c_fit_modelA': float(c_fit), 'c_resid_mean': float(c_emp),
                'F_per_record': [float(abs(np.vdot(e, l)) ** 2 / ((abs(e) ** 2).sum() * (abs(l) ** 2).sum())) for e, l in zip(Es, Ls)],
                'modelA_pred_F_per_record': (W / (W + c_fit)).round(4).tolist()}
# bootstrap c over amplitudes (stratified by record) for the calibration uncertainty
cb = []
for _ in range(2000):
    ix = [rng.integers(0, B, B) for _ in Es]
    e2 = [e[i] for e, i in zip(Es, ix)]; l2 = [l[i] for l, i in zip(Ls, ix)]
    nm = sum(np.vdot(a, b) for a, b in zip(e2, l2))
    F2 = abs(nm) ** 2 / (sum((abs(a) ** 2).sum() for a in e2) * sum((abs(b) ** 2).sum() for b in l2))
    cb.append(np.mean([(abs(a) ** 2).mean() for a in e2]) * (1 / F2 - 1))
cb = np.array(cb)
out['calib']['c_boot_se'] = float(cb.std())
T = 400000
res_models = {}
for name, model, par in [('exact', 'exact', None), ('A_fit', 'A', c_fit), ('R_F=Fpooled', 'R', Fp), ('E_empirical_residuals', 'E', res)]:
    z, xb = sample_z(model, par, T)
    res_models[name] = {'xeb': float(z.mean() - 1), 'xeb_mc_se': float(z.std() / np.sqrt(T)), 'xeb_expected_per_batch': float(xb.mean()),
                        'var_z': float(z.var()), 'var_formula_1+2X-X2': float(1 + 2 * (z.mean() - 1) - (z.mean() - 1) ** 2)}
out['models'] = res_models
out['modelA_production_F_uniform_prefix'] = float(fidelity_model('A', c_fit))
# propagate calibration uncertainty for model A (draw c from its bootstrap) and the 74-sample sampling noise exactly
reps = 100000
xs = []
for cpar in rng.choice(cb, 200):
    z, _ = sample_z('A', cpar, 74 * 500)
    xs.append(z.reshape(500, 74).mean(1) - 1)
xs = np.concatenate(xs)
mu, sd = xs.mean(), xs.std()
from math import erf, sqrt
from statistics import NormalDist
def sig(p): return NormalDist().inv_cdf(1 - p) if p > 0 else float('inf')
pA = (xs < 0.044).mean(); pI = (xs < 0.342).mean()
out['74_samples_modelA_with_calib_unc'] = {'mean': float(mu), 'sd': float(sd), 'P(xeb74<0.044)': float(pA), 'P(xeb74<0.342)': float(pI),
    'gaussian_sigma_over_0.044': float((mu - 0.044) / sd), 'tail_equiv_sigma_over_0.044': sig(pA),
    'gaussian_sigma_over_0.342_incl_IBM_se': float((mu - 0.342) / np.hypot(sd, 0.028)), 'q01_q05_q50': np.quantile(xs, [0.01, 0.05, 0.5]).round(4).tolist()}
# the analyze.py formulas, reproduced
def an(x, xse):
    v = 1 + 2 * x - x * x; ss = np.sqrt(v / NS); st = np.hypot(ss, xse)
    return {'se_sampling': float(ss), 'se_total': float(st), 'sigma_0.044': float((x - 0.044) / st), 'sigma_IBM': float((x - 0.342) / np.hypot(st, 0.028))}
out['analyze_py_formula'] = {'xratio 0.822+-0.0616': an(0.822024, 0.061607), 'Fbased 0.8524+-0.0152(jk)': an(0.85241, 0.0152 * 63 / 65),
                             'Fbased 0.8524+-0.015 (as in SESSION2)': an(0.852, 0.015)}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v3_predicted_xeb.json'), 'w'), indent=1)
