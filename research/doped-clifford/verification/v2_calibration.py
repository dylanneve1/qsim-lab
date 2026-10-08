#!/usr/bin/env python3
"""Independent recomputation of the D=70 calibration (verification task 2). Does not import analyze.py."""
import json, os, sys, itertools
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z = np.load(os.path.join(ROOT, 'sutd/data-for-doped-clifford-tn-simulation/data/amplitude_batches.npz'))
EX = Z['raw_vectors'].astype(np.complex128) * float(Z['recovery_factor'])   # physical amplitudes
ASSIGN = Z['assignments_q8_q69']; BS = Z['bitstrings_q0_first']; SEL = Z['selected_vector_indices']
# sanity of SUTD conventions: selected index == int(bitstring[:8],2) and q8..q69 == assignment
assert all(int(BS[r][:8], 2) == SEL[r] and BS[r][8:] == ASSIGN[r] for r in range(len(BS)))
z_all = np.abs(EX) ** 2 * 2.0 ** 70
print(f'SUTD sanity: 2051x256 amps, E z = {z_all.mean():.4f}, E z^2 = {(z_all**2).mean():.4f}, E z^3 = {(z_all**3).mean():.3f} (PT: 1,2,6)')
zibm = z_all[np.arange(len(BS)), SEL]
print(f'IBM samples: linear XEB = {zibm.mean()-1:.4f} +- {zibm.std(ddof=1)/np.sqrt(len(zibm)):.4f}, log-XEB = {np.euler_gamma+np.log(zibm).mean():.4f}')

recs = [json.loads(l) for l in open(os.path.join(ROOT, 'production/session2/out_final74.jsonl'))]
cal = [r for r in recs if r['kind'] == 'calib']
M = 6; B = 2 ** M
E, L, keys = [], [], []
for r in cal:
    p = r['prefix_bits']; row = r['row']
    assert p[:M] == '.' * M and p[8:] == ASSIGN[row]
    sub = int(p[M:8], 2)
    # SUTD index j = 4*t + sub where t = int(q0..q5) (q0 MSB), sub = int(q6 q7)
    e = np.array([EX[row, 4 * t + sub] for t in range(B)])
    a = np.array(r['amps']); l = a[:, 0] + 1j * a[:, 1]
    E.append(e); L.append(l); keys.append(f'c{row}')

def F(es, ls):
    num = sum(np.vdot(e, l) for e, l in zip(es, ls))
    return abs(num) ** 2 / (sum(np.vdot(e, e).real for e in es) * sum(np.vdot(l, l).real for l in ls))

def Xsamp(e, q):  # expected linear XEB of a batch sampler drawing j with prob q_j/sum q, scored by ideal z
    z = np.abs(e) ** 2 * 2.0 ** 70; w = np.abs(q) ** 2
    return float((w / w.sum() * z).sum() - 1)

rng = np.random.default_rng(12345)
out = {'per_record': {}}
for k, e, l in zip(keys, E, L):
    bs = []
    for _ in range(4000):
        ix = rng.integers(0, B, B); bs.append(F([e[ix]], [l[ix]]))
    out['per_record'][k] = {'F': F([e], [l]), 'F_boot_se': float(np.std(bs)), 'phase': float(np.angle(np.vdot(e, l))),
                            'X_ours': Xsamp(e, l), 'X_exact': Xsamp(e, e), 'norm_ratio_l_over_e': float(np.vdot(l, l).real / np.vdot(e, e).real),
                            'prefix_weight_W': float((np.abs(EX[int(k[1:])]) ** 2).sum() * 2.0 ** 62)}
K = len(E)
Fp = F(E, L)
# bootstrap over amplitudes (pooled), stratified bootstrap by record, and jackknife over records
Ea, La = np.concatenate(E), np.concatenate(L)
b1 = [F([Ea[ix]], [La[ix]]) for ix in (rng.integers(0, len(Ea), len(Ea)) for _ in range(4000))]
b2 = []
for _ in range(4000):
    ix = [rng.integers(0, B, B) for _ in range(K)]
    b2.append(F([E[i][ix[i]] for i in range(K)], [L[i][ix[i]] for i in range(K)]))
jk = np.array([F([E[i] for i in range(K) if i != j], [L[i] for i in range(K) if i != j]) for j in range(K)])
jk_se = float(np.sqrt((K - 1) / K * ((jk - jk.mean()) ** 2).sum()))
Xo = np.array([Xsamp(e, l) for e, l in zip(E, L)]); Xe = np.array([Xsamp(e, e) for e in E])
ratio = Xo.mean() / Xe.mean()
jkr = np.array([np.delete(Xo, j).mean() / np.delete(Xe, j).mean() for j in range(K)])
jkr_se = float(np.sqrt((K - 1) / K * ((jkr - jkr.mean()) ** 2).sum()))
# per-amplitude bootstrap of the ratio too (resampling suffixes within each record)
br = []
for _ in range(4000):
    ix = [rng.integers(0, B, B) for _ in range(K)]
    br.append(np.mean([Xsamp(E[i][ix[i]], L[i][ix[i]]) for i in range(K)]) / np.mean([Xsamp(E[i][ix[i]], E[i][ix[i]]) for i in range(K)]))
# convention controls: conjugate, bit-reversed tail index, wrong q6q7 slice, wrong row
rev = np.array([int(format(t, '06b')[::-1], 2) for t in range(B)])
ctrl = {'conjugated': F(E, [np.conj(l) for l in L]), 'tail_bitreversed': F(E, [l[rev] for l in L]),
        'both': F(E, [np.conj(l[rev]) for l in L])}
alt = {}
for k, e, l, r in zip(keys, E, L, cal):
    row = r['row']; sub = int(r['prefix_bits'][6:8], 2)
    alt[k] = {f'sub{s}': round(abs(np.vdot(EX[row, s::4], l)) ** 2 / (np.vdot(EX[row, s::4], EX[row, s::4]).real * np.vdot(l, l).real), 4) for s in range(4)}
    alt[k]['contiguous_block_sub'] = round(abs(np.vdot(EX[row, 64 * sub:64 * sub + 64], l)) ** 2 / (np.vdot(EX[row, 64*sub:64*sub+64], EX[row, 64*sub:64*sub+64]).real * np.vdot(l, l).real), 4)
    alt[k]['used_sub'] = sub
out.update({'records': K, 'amplitudes': K * B, 'F_pooled': Fp, 'F_se_boot_amplitudes': float(np.std(b1)),
            'F_se_boot_stratified': float(np.std(b2)), 'F_se_jackknife_records': jk_se,
            'F_mean_of_records': float(np.mean([out['per_record'][k]['F'] for k in keys])),
            'X_ours_mean': float(Xo.mean()), 'X_exact_mean': float(Xe.mean()), 'X_ratio': float(ratio),
            'X_ratio_se_jackknife': jkr_se, 'X_ratio_se_boot_amplitudes': float(np.std(br)),
            'pred_xeb_ratio_based': float(ratio * (B - 1) / (B + 1)), 'pred_xeb_F_based': float(Fp * (B - 1) / (B + 1)),
            'convention_controls_pooled_F': ctrl, 'slice_controls_per_record': alt})
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(__file__), 'v2_calibration.json'), 'w'), indent=1)
