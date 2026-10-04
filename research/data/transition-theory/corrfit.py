#!/usr/bin/env python3
"""Scaling in h with a leading correction: rho = h^a F(x) (1 + c h^w), x = (p - p_c) h^-b.
F: degree-4 polynomial (weighted LSQ inside the cost). Scans the correction exponent w and
the h window; prints (p_c, a, b, nu = a/b, y_h = 1/a, nu_eff = 1/b). Bootstrap errors over cells."""
import os, sys, math, json
import numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import steady_all, cells, HERE
rng = np.random.default_rng(7)
NB = int(os.environ.get('NB', 20))
C = cells(steady_all())
P = C[(C.pattern == 'poisson') & (C.init == 'zero')].copy()
P['neta'] = P.n * P.eta
P['err'] = np.maximum(P.err.fillna(0), 0.003 * P.rho)

def cost(th, D, w, fixed_pc=None):
    if fixed_pc is None:
        pc, a, b, c = th
    else:
        a, b, c = th; pc = fixed_pc
    x = (D.p_m.values - pc) * D.h.values ** (-b)
    corr = 1 + c * D.h.values ** w
    y = D.rho.values * D.h.values ** (-a) / corr
    e = D.err.values * D.h.values ** (-a) / corr
    wt = 1 / e**2
    cf = np.polyfit(x, y, 4, w=np.sqrt(wt))
    return (((np.polyval(cf, x) - y) ** 2) * wt).sum() / (len(x) - 5 - len(th))

def fit(D, w, fixed_pc=None):
    best = None
    for pc0 in ([0.155, 0.16, 0.165] if fixed_pc is None else [None]):
        for c0 in [-1.0, 0.0, 1.0]:
            x0 = ([pc0] if fixed_pc is None else []) + [0.5, 0.41, c0]
            r = minimize(cost, x0, args=(D, w, fixed_pc), method='Nelder-Mead', options=dict(xatol=1e-7, fatol=1e-9, maxiter=8000))
            if best is None or r.fun < best.fun:
                best = r
    v = list(best.x)
    if fixed_pc is not None:
        v = [fixed_pc] + v
    return dict(pc=v[0], a=v[1], b=v[2], c=v[3], chi2dof=best.fun)

out = []
for pwin in [(0.14, 0.20), (0.145, 0.18)]:
    for hmax in [1/16, 1/32, 1/64]:
        D = P[(P.p_m >= pwin[0]) & (P.p_m <= pwin[1]) & (P.h <= hmax) & (P.neta >= 16)]
        for w in [0.25, 0.5, 1.0]:
            for fpc in [None, 0.15995]:
                f = fit(D, w, fpc)
                bs = []
                for _ in range(NB):
                    Db = D.copy(); Db['rho'] = D.rho + rng.normal(0, 1, len(D)) * D.err
                    bs.append(fit(Db, w, fpc))
                e = {k: float(np.std([r[k] for r in bs], ddof=1)) if NB > 1 else float('nan') for k in ('pc', 'a', 'b')}
                nus = [r['a'] / r['b'] for r in bs]; yhs = [1 / r['a'] for r in bs]; nes = [1 / r['b'] for r in bs]
                rec = dict(pwin=pwin, hmax=hmax, w=w, pc_fixed=fpc, points=len(D), **f, pc_err=e['pc'], a_err=e['a'], b_err=e['b'],
                           nu=f['a'] / f['b'], nu_err=float(np.std(nus, ddof=1)) if NB > 1 else float('nan'),
                           y_h=1 / f['a'], y_h_err=float(np.std(yhs, ddof=1)) if NB > 1 else float('nan'),
                           nu_eff=1 / f['b'], nu_eff_err=float(np.std(nes, ddof=1)) if NB > 1 else float('nan'))
                out.append(rec)
                print(f"pwin={pwin} hmax=1/{round(1/hmax)} w={w} pc{'=' if fpc else '~'}{f['pc']:.4f}±{e['pc']:.4f} a={f['a']:.3f}±{e['a']:.3f} b={f['b']:.3f}±{e['b']:.3f} c={f['c']:.2f} "
                      f"nu={rec['nu']:.3f}±{rec['nu_err']:.3f} y_h={rec['y_h']:.3f}±{rec['y_h_err']:.3f} nu_eff={rec['nu_eff']:.3f}±{rec['nu_eff_err']:.3f} chi2={f['chi2dof']:.2f} pts={len(D)}", flush=True)
json.dump(out, open(os.path.join(HERE, 'corrfit.json'), 'w'), indent=1, default=float)
