#!/usr/bin/env python3
"""Task 5: model-free consistency checks on the 74 production batches (no exact amplitudes exist for them).
- Common scale: production and calibration records come from the same engine, so the calibration fixes the conversion
  |l|^2 -> z units (Sum|l|^2 / Sum|e|^2 per calibration record).
- Under model A (l = e + g, both circular Gaussian) the production amplitudes are themselves Porter-Thomas with mean
  1 + c (in exact-z units, times the engine's norm factor). Check moments, batch-weight spread and the u/j draws.
"""
import json, os
import numpy as np
from statistics import NormalDist
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z = np.load(os.path.join(ROOT, 'sutd/data-for-doped-clifford-tn-simulation/data/amplitude_batches.npz'))
EX = Z['raw_vectors'].astype(np.complex128) * float(Z['recovery_factor'])
recs = [json.loads(l) for l in open(os.path.join(ROOT, 'production/session2/out_final74.jsonl'))]
cal = [r for r in recs if r['kind'] == 'calib']; smp = [r for r in recs if r['kind'] == 'sample']
A = lambda r: (lambda a: a[:, 0] + 1j * a[:, 1])(np.array(r['amps']))
nr = []
for r in cal:
    e = EX[r['row'], int(r['prefix_bits'][6:8], 2)::4]; l = A(r)
    nr.append((abs(l) ** 2).sum() / (abs(e) ** 2).sum())
# convert production |l|^2 to "z units" using the pooled calibration ratio of norms of l over the exact SUTD amplitudes
k = np.sum([(abs(A(r)) ** 2).sum() for r in cal]) / np.sum([(abs(EX[r['row'], int(r['prefix_bits'][6:8], 2)::4]) ** 2).sum() for r in cal])
zt = np.array([abs(A(r)) ** 2 * 2.0 ** 70 / k for r in smp])        # 74 x 64
Wt = zt.mean(1)
out = {'norm_ratio_l_over_e_per_calib_record': np.round(nr, 4).tolist(), 'pooled_norm_ratio': float(k),
       'production_mean_ztilde': float(zt.mean()), 'production_mean_ztilde_se': float(Wt.std(ddof=1) / np.sqrt(len(Wt))),
       'E[z~^2]/E[z~]^2 (PT: 2)': float((zt ** 2).mean() / zt.mean() ** 2),
       'E[z~^3]/E[z~]^3 (PT: 6)': float((zt ** 3).mean() / zt.mean() ** 3),
       'batch_weight_sd (Gamma(64)/64: 0.125)': float((Wt / Wt.mean()).std(ddof=1)),
       'per_session_mean_ztilde': {'s0-41': float(zt[:42].mean()), 's42-73': float(zt[42:].mean())}}
# sampled index statistics: the drawn j has q_j*64 with E = 2B/(B+1) under PT
qj = np.array([(lambda w: w[r['j_tail']] / w.sum() * 64)(abs(A(r)) ** 2) for r in smp])
out['mean_64q_of_drawn_j (PT: 2B/(B+1)=1.969)'] = float(qj.mean()); out['se'] = float(qj.std(ddof=1) / np.sqrt(len(qj)))
u = np.array([r['u_tail'] for r in smp]); us = np.sort(u)
ks = max(np.max(np.arange(1, 75) / 74 - us), np.max(us - np.arange(74) / 74))
out['u_tail KS statistic (n=74; 5% crit 0.155)'] = float(ks)
# prefix uniformity: fraction of ones per qubit q6..q69 across 74 prefixes
P = np.array([[int(c) for c in r['prefix_bits'][6:]] for r in smp])
out['prefix_ones_fraction_mean'] = float(P.mean()); out['prefix_ones_fraction_min_max_per_qubit'] = [float(P.mean(0).min()), float(P.mean(0).max())]
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v5_production_stats.json'), 'w'), indent=1)
