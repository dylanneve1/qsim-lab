"""Profiles of free/Hartree Delta rho(r) = rho_MID - rho_SCV, to size the n_f window; symmetry checks."""
import numpy as np, gauss, os
res = {}
for mode in ('free', 'hartree'):
    for circ in ('SCV', 'meson'):
        rec, _, _ = gauss.run(circ, mode); res[(mode, circ)] = rec
for mode in ('free', 'hartree'):
    for k in (10, 20):
        nS = np.array(res[(mode, 'SCV')][k - 1]['n']); nM = np.array(res[(mode, 'meson')][k - 1]['n'])
        d = (nM - nS) * np.array([(-1) ** r for r in range(60)])[:, None]
        tot = d.sum()
        print(mode, 'step', k, 'n_f', round(tot, 6))
        for R in (6, 8, 10, 12, 14, 16, 18, 20, 24):
            sel = [r for r in range(60) if abs(r - 29.5) <= R + 0.5]
            print(f'   R={R:2d} sites {sel[0]}..{sel[-1]} partial n_f {d[sel].sum():.6f} miss {tot - d[sel].sum():+.2e}')
        if k == 20:
            print('  chain asym SCV max|n_i-n_o|', abs(nS[:, 0] - nS[:, 1]).max(), ' MID', abs(nM[:, 0] - nM[:, 1]).max())
            print('  SCV refl+PH max|n(r) - (1-n(59-r))|', abs(nS[:, 0] - (1 - nS[::-1, 0])).max())
            print('  SCV bulk n_i r=20..39:', np.round(nS[20:40, 0], 6))
nM0 = np.array(res[('free', 'meson')][0]['n'])
print('MID step1 occupations r=24..35 chain i', np.round(nM0[24:36, 0], 3), 'chain o', np.round(nM0[24:36, 1], 3))
