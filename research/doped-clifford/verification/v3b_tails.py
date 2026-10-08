#!/usr/bin/env python3
"""Task 3 addendum: (i) is the measured X ratio (0.848) consistent with model A on the 4 actual calibration prefixes?
(ii) exact lower-tail probabilities for the 74-sample mean (Chernoff bound + Lugannani-Rice saddlepoint), model A and
the ratio-based mean, with the calibration uncertainty folded in by worst-case shift."""
import json, os
import numpy as np
from statistics import NormalDist
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(7)
B = 64
Z = np.load(os.path.join(ROOT, 'sutd/data-for-doped-clifford-tn-simulation/data/amplitude_batches.npz'))
EX = Z['raw_vectors'].astype(np.complex128) * float(Z['recovery_factor'])
cal = [r for r in map(json.loads, open(os.path.join(ROOT, 'production/session2/out_final74.jsonl'))) if r['kind'] == 'calib']
v3 = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v3_predicted_xeb.json')))
c = v3['calib']['c_fit_modelA']; cse = v3['calib']['c_boot_se']
out = {}
# (i)
Xo_pred, Xo_obs, Xe = [], [], []
for r in cal:
    e = EX[r['row'], int(r['prefix_bits'][6:8], 2)::4] * 2 ** 35; z = abs(e) ** 2
    a = np.array(r['amps']); l = a[:, 0] + 1j * a[:, 1]
    g = np.sqrt(c / 2) * (rng.standard_normal((20000, B)) + 1j * rng.standard_normal((20000, B)))
    w = abs(e + g) ** 2; w /= w.sum(1, keepdims=True)
    xs = (w * z).sum(1) - 1
    Xo_pred.append((xs.mean(), xs.std())); wl = abs(l) ** 2; Xo_obs.append((wl / wl.sum() * z).sum() - 1); Xe.append((z / z.sum() * z).sum() - 1)
out['per_record_X_ours_obs_vs_modelA'] = [{'obs': round(o, 4), 'modelA_mean': round(p[0], 4), 'modelA_sd': round(p[1], 4), 'X_exact': round(x, 4)}
                                          for o, p, x in zip(Xo_obs, Xo_pred, Xe)]
out['X_ratio_obs'] = float(np.mean(Xo_obs) / np.mean(Xe))
out['X_ratio_modelA_on_these_prefixes'] = float(np.mean([p[0] for p in Xo_pred]) / np.mean(Xe))
out['X_ratio_modelA_sd_on_these_prefixes'] = float(np.sqrt(sum(p[1] ** 2 for p in Xo_pred)) / 4 / np.mean(Xe))
# (ii) z distribution of one production sample under model A (c and c + 2 se), large MC
def zs(cc, T=1_000_000):
    out = []
    for _ in range(T // 100000):
        e = np.sqrt(.5) * (rng.standard_normal((100000, B)) + 1j * rng.standard_normal((100000, B)))
        l = e + np.sqrt(cc / 2) * (rng.standard_normal((100000, B)) + 1j * rng.standard_normal((100000, B)))
        w = abs(l) ** 2; cs = np.cumsum(w, 1); u = rng.random(100000)[:, None] * cs[:, -1:]
        j = (cs > u).argmax(1); out.append(abs(e[np.arange(100000), j]) ** 2)
    return np.concatenate(out)
def tails(z, N, thr):
    a = 1 + thr  # mean z threshold
    ts = np.linspace(-6, -1e-4, 3000)
    K = np.array([np.log(np.mean(np.exp(t * z))) for t in ts])
    ch = N * (K - ts * a); k = ch.argmin(); t = ts[k]
    # Lugannani-Rice
    w2 = np.exp(t * z); m0 = w2.mean(); m1 = (z * w2).mean() / m0; m2 = (z * z * w2).mean() / m0 - m1 ** 2
    wv = np.sign(t) * np.sqrt(max(-2 * ch[k], 0)); uu = t * np.sqrt(N * m2)
    nd = NormalDist(); lr = nd.cdf(wv) + nd.pdf(wv) * (1 / wv - 1 / uu)
    return {'chernoff_bound': float(np.exp(ch[k])), 'lugannani_rice': float(lr), 'equiv_sigma_LR': (float(-NormalDist().inv_cdf(lr)) if 0 < lr < 1 else None),
            'gaussian_sigma': float((z.mean() - a) / (z.std() / np.sqrt(N)))}
for label, cc in [('modelA_c_fit', c), ('modelA_c_fit+2se', c + 2 * cse)]:
    z = zs(cc)
    out[label] = {'c': cc, 'xeb': float(z.mean() - 1), 'var_z': float(z.var()), 'P74(<0.044)': tails(z, 74, 0.044), 'P74(<0.342)': tails(z, 74, 0.342)}
# ratio-based central value: emulate with a mixture sampler of XEB 0.822 (model A with larger c tuned to 0.822)
for cc in np.linspace(0.15, 0.30, 16):
    z = zs(cc, 200000)
    if z.mean() - 1 < 0.822:
        z = zs(cc); out['modelA_tuned_to_xeb_0.822'] = {'c': float(cc), 'xeb': float(z.mean() - 1), 'P74(<0.044)': tails(z, 74, 0.044), 'P74(<0.342)': tails(z, 74, 0.342)}
        break
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v3b_tails.json'), 'w'), indent=1)
