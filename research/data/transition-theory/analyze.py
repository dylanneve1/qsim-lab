#!/usr/bin/env python3
"""Analysis + figures for research/transition-theory.md.

usage: python3 analyze.py           (reads parts/ via jobs.txt, ../magic-transition/raw.csv)
Writes fits.json, cells.csv and the PNGs next to this file.
"""
import json, math, os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize, curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import steady_all, cells, parts, HERE

rng = np.random.default_rng(2026)
NB = int(os.environ.get("NB", 30))
SKIP = os.environ.get("SKIP", "").split(",")
OUT = {}
# literature (1+1)D Clifford MIPT, brickwork + Z measurements w.p. p after each layer
PC_LIT, NU_LIT, XB_LIT = 0.15995, 1.260, 0.129 / 1.260   # Sierant et al. PRB 106, 214316: p_c, nu, beta -> x_1 = beta/nu
YH_LIT = 2 - XB_LIT

df = steady_all()
C = cells(df)
C.to_csv(os.path.join(HERE, 'cells.csv'), index=False)

# ------------------------------------------------------------------ collapse machinery
def master_cost(x, y, e, npar, deg=4):
    w = 1 / e**2
    cf = np.polyfit(x, y, deg, w=np.sqrt(w))
    return (((np.polyval(cf, x) - y) ** 2) * w).sum() / max(len(x) - deg - 1 - npar, 1)

def h_cost(th, D, fixed):
    p = dict(fixed); free = [k for k in ('pc', 'a', 'b') if k not in fixed]
    p.update(zip(free, th))
    x = (D.p_m - p['pc']) * D.h ** (-p['b'])
    y = D.rho * D.h ** (-p['a'])
    e = D.err * D.h ** (-p['a'])
    return master_cost(x.values, y.values, e.values, len(free))

def h_fit(D, fixed={}, starts=None):
    free = [k for k in ('pc', 'a', 'b') if k not in fixed]
    base = dict(pc=[0.155, 0.16, 0.165], a=[0.45, 0.53], b=[0.40])
    best = None
    import itertools
    for st in itertools.product(*[base[k] for k in free]):
        r = minimize(h_cost, st, args=(D, fixed), method='Nelder-Mead',
                     options=dict(xatol=1e-7, fatol=1e-9, maxiter=6000))
        if best is None or r.fun < best.fun:
            best = r
    out = dict(fixed); out.update(zip(free, best.x)); out['chi2dof'] = best.fun
    return out

def boot(D, fitter, nb=40):
    """Bootstrap over cells (resample cell means within their errors)."""
    res = []
    if nb < 2:
        return {k: float('nan') for k in fitter(D) if k != 'chi2dof'}
    for _ in range(nb):
        Db = D.copy()
        Db['rho'] = D.rho + rng.normal(0, 1, len(D)) * D.err
        res.append(fitter(Db))
    keys = [k for k in res[0] if k != 'chi2dof']
    return {k: float(np.std([r[k] for r in res], ddof=1)) for k in keys}

def floor_err(D, rel=0.003):
    D = D.copy()
    D['err'] = np.maximum(D.err.fillna(0), rel * D.rho)
    return D

# ------------------------------------------------------------------ 1. rho depends on h only
P = C[(C.pattern == 'poisson') & (C.init == 'zero')].copy()
P['neta'] = P.n * P.eta
pairs = []
for (pm, h), grp in P[P.neta >= 16].groupby(['p_m', 'h']):
    if len(grp) > 1:
        r = grp.rho.values; e = grp.err.fillna(0).values
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                pairs.append(dict(p=pm, h=h, n1=int(grp.n.values[i]), eta1=grp.eta.values[i], n2=int(grp.n.values[j]), eta2=grp.eta.values[j],
                                  rel=(r[i] - r[j]) / ((r[i] + r[j]) / 2), z=(r[i] - r[j]) / math.hypot(e[i], e[j]) if (e[i] + e[j]) > 0 else np.nan))
pairs = pd.DataFrame(pairs)
pairs.to_csv(os.path.join(HERE, 'same_h_pairs.csv'), index=False)
OUT['same_h'] = dict(npairs=len(pairs), median_abs_rel=float(pairs.rel.abs().median()) if len(pairs) else None,
                     max_abs_rel=float(pairs.rel.abs().max()) if len(pairs) else None,
                     chi2_per_pair=float(np.nanmean(pairs.z**2)) if len(pairs) else None)
print('same-h pairs', OUT['same_h'])

# ------------------------------------------------------------------ 2. scaling in h (all poisson sources)
def hdata(pwin, hmax, hmin=0.0, neta_min=16):
    D = P[(P.p_m >= pwin[0]) & (P.p_m <= pwin[1]) & (P.h <= hmax) & (P.h >= hmin) & (P.neta >= neta_min)]
    return floor_err(D)

hfits = []
for pwin in [(0.13, 0.22), (0.14, 0.20)]:
    for hmax in [1/16, 1/32, 1/64, 1/128, 1/256, 1/512]:
        D = hdata(pwin, hmax)
        if len(D) < 25:
            continue
        f = h_fit(D)
        e = boot(D, h_fit, nb=NB)
        rec = dict(pwin=pwin, hmax=hmax, points=len(D), **{k: float(v) for k, v in f.items()},
                   **{k + '_err': v for k, v in e.items()})
        rec['nu'] = rec['a'] / rec['b']; rec['y_h'] = 1 / rec['a']; rec['nu_eff'] = 1 / rec['b']
        # error propagation through bootstrap-correlated a,b is approximated by the larger relative error
        rec['nu_err'] = rec['nu'] * math.hypot(rec['a_err'] / rec['a'], rec['b_err'] / rec['b'])
        hfits.append(rec)
        print(f"free  pwin={pwin} hmax=1/{round(1/hmax)} pts={len(D)} pc={f['pc']:.4f}±{e['pc']:.4f} a={f['a']:.3f}±{e['a']:.3f} "
              f"b={f['b']:.3f}±{e['b']:.3f} -> y_h={1/f['a']:.3f} nu={f['a']/f['b']:.3f} nu_eff={1/f['b']:.2f} chi2={f['chi2dof']:.2f}")
OUT['h_free'] = hfits

# p_c fixed to the literature value: a, b free
hfix = []
for pwin in [(0.13, 0.22), (0.14, 0.20)]:
    for hmax in [1/32, 1/64, 1/128, 1/256, 1/512]:
        D = hdata(pwin, hmax)
        if len(D) < 20:
            continue
        fx = lambda DD: h_fit(DD, fixed=dict(pc=PC_LIT))
        f = fx(D); e = boot(D, fx, nb=NB)
        rec = dict(pwin=pwin, hmax=hmax, points=len(D), **{k: float(v) for k, v in f.items()}, **{k + '_err': v for k, v in e.items()})
        rec['nu'] = rec['a'] / rec['b']; rec['y_h'] = 1 / rec['a']; rec['nu_eff'] = 1 / rec['b']
        rec['nu_err'] = rec['nu'] * math.hypot(rec['a_err'] / rec['a'], rec['b_err'] / rec['b'])
        hfix.append(rec)
        print(f"pc=lit pwin={pwin} hmax=1/{round(1/hmax)} pts={len(D)} a={f['a']:.3f}±{e['a']:.3f} b={f['b']:.3f}±{e['b']:.3f} "
              f"-> y_h={1/f['a']:.3f} nu={f['a']/f['b']:.3f} nu_eff={1/f['b']:.2f} chi2={f['chi2dof']:.2f}")
OUT['h_pc_fixed'] = hfix

# all exponents fixed to the literature: how good is the collapse?
for pwin in [(0.13, 0.22)]:
    for hmax in [1/64, 1/256]:
        D = hdata(pwin, hmax)
        c = h_cost([], D, dict(pc=PC_LIT, a=1 / YH_LIT, b=1 / (NU_LIT * YH_LIT)))
        print(f"literature exponents (pc={PC_LIT}, a={1/YH_LIT:.3f}, b={1/(NU_LIT*YH_LIT):.3f}) hmax=1/{round(1/hmax)}: chi2/dof={c:.2f}")
        OUT.setdefault('h_lit_chi2', []).append(dict(hmax=hmax, chi2dof=c, points=len(D)))

# ------------------------------------------------------------------ 3. local exponent at p ~ p_c: dln rho / dln h
loc = []
for pm in sorted(P.p_m.unique()):
    Q = P[(P.p_m == pm) & (P.neta >= 16)].groupby('h').apply(
        lambda g: pd.Series(dict(rho=np.average(g.rho, weights=1 / np.maximum(g.err.fillna(1), 1e-6)**2),
                                 err=1 / math.sqrt((1 / np.maximum(g.err.fillna(1), 1e-6)**2).sum())))).reset_index().sort_values('h')
    for i in range(len(Q) - 1):
        h1, h2 = Q.h.values[i], Q.h.values[i + 1]
        if h2 / h1 > 4.1:
            continue
        r1, r2 = Q.rho.values[i], Q.rho.values[i + 1]
        e1, e2 = Q.err.values[i], Q.err.values[i + 1]
        s = math.log(r2 / r1) / math.log(h2 / h1)
        se = math.hypot(e1 / r1, e2 / r2) / math.log(h2 / h1)
        loc.append(dict(p=pm, h=math.sqrt(h1 * h2), slope=s, err=se))
loc = pd.DataFrame(loc)
loc.to_csv(os.path.join(HERE, 'local_slope.csv'), index=False)

# ------------------------------------------------------------------ 4. per-eta finite-size scaling in n (the original ansatz)
def n_cost(th, D):
    pc, nu, bn = th
    x = (D.p_m - pc) * D.n ** (1 / nu)
    y = D.rho * D.n ** bn
    e = D.err * D.n ** bn
    return master_cost(x.values, y.values, e.values, 3)

def n_fit(D):
    best = None
    for pc0 in [0.145, 0.155, 0.165]:
        for nu0 in [1.3, 2.5]:
            r = minimize(n_cost, (pc0, nu0, 0.5), args=(D,), method='Nelder-Mead', options=dict(xatol=1e-7, fatol=1e-9, maxiter=6000))
            if best is None or r.fun < best.fun:
                best = r
    return dict(pc=best.x[0], nu=best.x[1], bn=best.x[2], chi2dof=best.fun)

pereta = []
for eta in sorted(P.eta.unique()):
    if eta not in (0.5, 1.0, 2.0, 4.0):
        continue
    ns_all = sorted(P[(P.eta == eta)].n.unique())
    for nmin in [64, 128, 256]:
        D = floor_err(P[(P.eta == eta) & (P.n >= nmin) & (P.n <= 1024) & (P.p_m >= 0.13) & (P.p_m <= 0.22) & (P.neta >= 16)])
        if D.n.nunique() < 3 or len(D) < 20:
            continue
        f = n_fit(D); e = boot(D, n_fit, nb=NB)
        rec = dict(eta=eta, nmin=int(nmin), sizes=sorted(map(int, D.n.unique())), points=len(D), **f, **{k + '_err': v for k, v in e.items()})
        rec['h_max'] = eta / nmin
        pereta.append(rec)
        print(f"eta={eta} n>={nmin} (h<={eta/nmin:.4f}) pc={f['pc']:.4f}±{e['pc']:.4f} nu_eff={f['nu']:.2f}±{e['nu']:.2f} "
              f"beta/nu_eff={f['bn']:.3f}±{e['bn']:.3f} -> nu = nu_eff*beta/nu = {f['nu']*f['bn']:.3f} chi2={f['chi2dof']:.2f}")
OUT['per_eta'] = pereta

json.dump(OUT, open(os.path.join(HERE, 'fits.json'), 'w'), indent=1, default=float)

# ------------------------------------------------------------------ 5. single-injection survival (x_h)
S = parts('survival')
if len(S):
    S = S[S.activated == 1]
    surv = []
    for (n, pm), g in S.groupby(['n', 'p_m']):
        tau = g.tau.values.astype(float); cens = g.censored.values.astype(bool)
        tmax = int(tau.max())
        ts = np.unique(np.round(np.logspace(0, math.log10(tmax), 40)).astype(int))
        # Kaplan–Meier: P(T > t)
        order = np.argsort(tau)
        P_ = 1.0; km = {}
        deaths = pd.Series(tau[~cens]).value_counts().sort_index()
        at_risk = lambda t: (tau >= t).sum()
        cur = 1.0; kmv = []
        for t in range(1, tmax + 1):
            dth = deaths.get(t, 0)
            r = at_risk(t)
            if r > 0:
                cur *= 1 - dth / r
            kmv.append(cur)
        kmv = np.array(kmv)
        for t in ts:
            surv.append(dict(n=n, p=pm, t=t, P=kmv[t - 1], at_risk=int(at_risk(t)), injections=len(tau)))
    surv = pd.DataFrame(surv); surv.to_csv(os.path.join(HERE, 'survival_km.csv'), index=False)
    xs = []
    for (n, pm), g in surv.groupby(['n', 'p']):
        for lo, hi in [(8, 64), (16, 128), (32, 256), (64, 512)]:
            q = g[(g.t >= lo) & (g.t <= hi) & (g.t <= n / 4) & (g.at_risk >= 30)]
            if len(q) < 4:
                continue
            sl, ic = np.polyfit(np.log(q.t), np.log(q.P), 1)
            # error: bootstrap over injections would be better; use the spread between windows (reported below)
            xs.append(dict(n=int(n), p=pm, tlo=lo, thi=hi, x=-sl, P_lo=float(q.P.iloc[0]), P_hi=float(q.P.iloc[-1])))
            print(f"survival n={n} p={pm} t in [{lo},{hi}]: P ~ t^-{-sl:.3f}  (P({lo})={q.P.iloc[0]:.3f}, injections={g.injections.iloc[0]})")
    OUT['survival'] = xs

# ------------------------------------------------------------------ 6. decay of the maximally mixed state: S(t) vs n/t
Dd = parts('decay')
if len(Dd):
    rows = []
    for (n, pm), g in Dd.groupby(['n', 'p_m']):
        seeds = g.seed.unique(); T = int(g.t.max())
        ts = np.unique(np.round(np.logspace(0, math.log10(2 * n), 50)).astype(int))
        mat = np.zeros((len(seeds), len(ts)))
        for i, s in enumerate(seeds):
            gs = g[g.seed == s].sort_values('t')
            idx = np.searchsorted(gs.t.values, ts, side='right') - 1
            vals = np.where(idx >= 0, gs.d.values[np.maximum(idx, 0)], n)
            # trajectories that hit d = 0 stop recording: they stay 0
            mat[i] = vals
        for j, t in enumerate(ts):
            rows.append(dict(n=n, p=pm, t=t, S=mat[:, j].mean(), err=mat[:, j].std(ddof=1) / math.sqrt(len(seeds))))
    dec = pd.DataFrame(rows); dec.to_csv(os.path.join(HERE, 'decay.csv'), index=False)
    for (n, pm), g in dec.groupby(['n', 'p']):
        q = g[(g.t >= 8) & (g.t <= n / 4)]
        sl, ic = np.polyfit(np.log(q.t), np.log(q.S), 1)
        tl = g[(g.t >= n / 8) & (g.t <= n / 4)]
        print(f"decay n={n} p={pm}: S ~ t^{sl:.3f} on [8, n/4]; S*t/n at t in [n/8,n/4] = {np.mean(tl.S * tl.t / n):.3f}")
        OUT.setdefault('decay', []).append(dict(n=int(n), p=pm, slope=sl, c=float(np.mean(tl.S * tl.t / n))))

json.dump(OUT, open(os.path.join(HERE, 'fits.json'), 'w'), indent=1, default=float)

# ------------------------------------------------------------------ 7. fixed-site (line-defect) injection: ordinary FSS
F = C[(C.pattern == 'fixed') & (C.init == 'zero')].copy()
def f_cost(th, D):
    pc, nu, A = th
    x = (D.p_m - pc) * D.n ** (1 / nu)
    y = D.d - A * np.log(D.n)
    return master_cost(x.values, y.values, D.d_err.values, 3)
def f_fit(D):
    best = None
    for pc0 in [0.15, 0.16, 0.17]:
        for nu0 in [1.3, 2.5]:
            r = minimize(f_cost, (pc0, nu0, 1.0), args=(D,), method='Nelder-Mead', options=dict(xatol=1e-7, fatol=1e-9, maxiter=6000))
            if best is None or r.fun < best.fun:
                best = r
    return dict(pc=best.x[0], nu=best.x[1], A=best.x[2], chi2dof=best.fun)
if len(F) and 'fixed' not in SKIP:
    F['d_err'] = np.maximum(F.d_err.fillna(0), 0.005 * F.d)
    OUT['fixed'] = []
    for pwin in [(0.14, 0.18), (0.13, 0.20)]:
        for nmin in [64, 128]:
            D = F[(F.p_m >= pwin[0]) & (F.p_m <= pwin[1]) & (F.n >= nmin)]
            if D.n.nunique() < 3:
                continue
            f = f_fit(D)
            def fb(DD):
                return f_fit(DD.assign(d=DD.d + rng.normal(0, 1, len(DD)) * DD.d_err))
            res = [fb(D) for _ in range(max(NB // 2, 2))]
            e = {k: float(np.std([r[k] for r in res], ddof=1)) for k in ('pc', 'nu', 'A')}
            OUT['fixed'].append(dict(pwin=pwin, nmin=nmin, points=len(D), **f, **{k + '_err': v for k, v in e.items()}))
            print(f"fixed-site d = A ln n + G((p-pc) n^(1/nu)): pwin={pwin} n>={nmin} pc={f['pc']:.4f}±{e['pc']:.4f} nu={f['nu']:.2f}±{e['nu']:.2f} A={f['A']:.2f}±{e['A']:.2f} chi2={f['chi2dof']:.2f}")
    # the same data under the poisson-style ansatz (d/n) n^{beta/nu}
    D = floor_err(F[(F.p_m >= 0.13) & (F.p_m <= 0.2) & (F.n >= 64)], 0.005)
    f = n_fit(D)
    OUT['fixed_rho_ansatz'] = f
    print(f"fixed-site, d/n ansatz: pc={f['pc']:.4f} nu={f['nu']:.2f} beta/nu={f['bn']:.3f} chi2={f['chi2dof']:.2f}")

json.dump(OUT, open(os.path.join(HERE, 'fits.json'), 'w'), indent=1, default=float)

# ------------------------------------------------------------------ figures
if 'figs' not in SKIP:
    cmap = plt.get_cmap('viridis')
    pms_show = [0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.2, 0.22]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    mk = {0.5: 'v', 1.0: 'o', 2.0: 's', 4.0: 'D'}
    Pp = P[P.neta >= 16]
    for i, pm in enumerate(pms_show):
        col = cmap(i / (len(pms_show) - 1))
        q = Pp[Pp.p_m == pm]
        for eta, g in q.groupby('eta'):
            m = mk.get(round(eta, 6), 'x' if g.n.nunique() > 1 and g.h.nunique() == 1 else '^')
            if round(eta, 6) not in mk:
                m = '^' if eta < 0.5 else 'x'
            ax[0].errorbar(g.h, g.rho, g.err, fmt=m, color=col, ms=4, alpha=0.85, label=f'p={pm}' if eta == 1.0 else None)
    ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('h = η/n  (dephasing per site per layer)'); ax[0].set_ylabel('ρ = d̄/n')
    hh = np.logspace(-5, -1, 10)
    ax[0].plot(hh, 1.75 * hh ** (1 / YH_LIT), 'k--', lw=1, label=f'∝ h^(1/y_h), y_h={YH_LIT:.2f}')
    ax[0].set_title('d̄/n depends only on h = η/n\n(▽ η=½, ○ 1, □ 2, ◇ 4, △ η<½, × constant p_T)')
    ax[0].legend(fontsize=7, ncol=2)
    # collapse with literature exponents
    a, b = 1 / YH_LIT, 1 / (NU_LIT * YH_LIT)
    for i, pm in enumerate(sorted(Pp[(Pp.p_m >= 0.13) & (Pp.p_m <= 0.22)].p_m.unique())):
        q = Pp[(Pp.p_m == pm) & (Pp.h <= 1 / 64)]
        ax[1].plot((pm - PC_LIT) * q.h ** (-b), q.rho * q.h ** (-a), 'o', ms=3, color=cmap((pm - 0.13) / 0.09))
    ax[1].set_xlabel('(p − p_c) h^(−1/(ν y_h))'); ax[1].set_ylabel('ρ h^(−1/y_h)')
    ax[1].set_title(f'collapse with literature Clifford-MIPT exponents\np_c={PC_LIT}, ν={NU_LIT}, y_h=2−β/ν={YH_LIT:.3f} (no fit), h ≤ 1/64')
    if len(loc):
        for pm, col in [(0.155, 'tab:blue'), (0.16, 'k'), (0.165, 'tab:red')]:
            q = loc[(loc.p == pm)]
            ax[2].errorbar(q.h, q.slope, q.err, fmt='o-', color=col, ms=4, label=f'p={pm}')
        ax[2].axhline(1 / YH_LIT, color='gray', ls='--', label=f'1/y_h = {1/YH_LIT:.3f} (x_h = β/ν = {XB_LIT:.3f})')
        ax[2].axhline(0.5, color='gray', ls=':', label='1/2 (x_h = 0)')
        ax[2].set_xscale('log'); ax[2].set_xlabel('h'); ax[2].set_ylabel('d ln ρ / d ln h')
        ax[2].set_ylim(0.3, 0.8); ax[2].legend(fontsize=8); ax[2].set_title('local exponent of ρ(h) near p_c')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'h_scaling.png'), dpi=110); plt.close(fig)

    if OUT.get('per_eta'):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for r in OUT['per_eta']:
            ax[0].errorbar(r['h_max'], r['nu'], r['nu_err'], fmt=mk.get(r['eta'], 'o'), color='C0')
            ax[0].errorbar(r['h_max'], r['nu'] * r['bn'], 0, fmt=mk.get(r['eta'], 'o'), color='C2', mfc='none')
            ax[1].errorbar(r['h_max'], r['pc'], r['pc_err'], fmt=mk.get(r['eta'], 'o'), color='C1')
        ax[0].axhline(NU_LIT * YH_LIT, color='C0', ls='--', label=f'ν·y_h = {NU_LIT*YH_LIT:.2f}')
        ax[0].axhline(NU_LIT, color='C2', ls='--', label=f'ν = {NU_LIT}')
        ax[0].set_xscale('log'); ax[0].set_xlabel('h_max = η / n_min'); ax[0].set_ylabel('fitted ν_eff (filled), ν_eff·(β/ν)_eff (open)'); ax[0].legend(fontsize=8)
        ax[1].axhline(PC_LIT, color='k', ls='--', label='p_c (Clifford MIPT)'); ax[1].set_xscale('log'); ax[1].set_xlabel('h_max = η / n_min'); ax[1].set_ylabel('fitted p_c')
        ax[1].legend(fontsize=8)
        fig.suptitle('per-η finite-size scaling in n: ▽ η=½, ○ 1, □ 2, ◇ 4 — results depend on η/n_min only')
        fig.tight_layout(); fig.savefig(os.path.join(HERE, 'per_eta.png'), dpi=110); plt.close(fig)

    if len(F):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for i, pm in enumerate(sorted(F.p_m.unique())):
            q = F[F.p_m == pm].sort_values('n')
            ax[0].errorbar(q.n, q.d, q.d_err, fmt='o-', color=cmap(i / max(F.p_m.nunique() - 1, 1)), ms=3, label=f'p={pm}')
        ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('n'); ax[0].set_ylabel('d̄ (fixed site, η = 1)'); ax[0].legend(fontsize=7, ncol=2)
        fx = [r for r in OUT.get('fixed', []) if r['pwin'] == (0.14, 0.18) and r['nmin'] == 64]
        if fx:
            r = fx[0]
            for i, pm in enumerate(sorted(F.p_m.unique())):
                q = F[(F.p_m == pm) & (F.n >= 64)]
                ax[1].plot((pm - r['pc']) * q.n ** (1 / r['nu']), q.d - r['A'] * np.log(q.n), 'o', ms=3, color=cmap(i / max(F.p_m.nunique() - 1, 1)))
            ax[1].set_xlim(-3, 3); ax[1].set_ylim(-30, 60)
            ax[1].set_title(f"d − A ln n vs (p−p_c) n^(1/ν): p_c={r['pc']:.4f}, ν={r['nu']:.2f}, A={r['A']:.2f}")
        fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fixed_site.png'), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    if os.path.exists(os.path.join(HERE, 'survival_km.csv')):
        sv = pd.read_csv(os.path.join(HERE, 'survival_km.csv'))
        for (n, pm), g in sv.groupby(['n', 'p']):
            g = g[g.at_risk >= 20]
            ax[0].plot(g.t, g.P, 'o-' if n > 256 else 'x--', ms=3, label=f'n={n}, p={pm}')
        tt = np.logspace(0.5, 2.7, 10); ax[0].plot(tt, 0.8 * tt ** (-XB_LIT), 'k--', lw=1, label=f't^(−{XB_LIT:.3f}) (β/ν)')
        ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('t (layers after one dephasing)'); ax[0].set_ylabel('P(entropy survives)'); ax[0].legend(fontsize=7)
    if os.path.exists(os.path.join(HERE, 'decay.csv')):
        dc = pd.read_csv(os.path.join(HERE, 'decay.csv'))
        for (n, pm), g in dc.groupby(['n', 'p']):
            ax[1].plot(g.t / n, g.S * g.t / n, '-', label=f'n={n}, p={pm}')
        ax[1].set_xscale('log'); ax[1].set_xlabel('t / n'); ax[1].set_ylabel('S(t) · t / n'); ax[1].legend(fontsize=7)
        ax[1].set_title('maximally mixed start, no injection: S ∝ n/t')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'survival_decay.png'), dpi=110); plt.close(fig)
