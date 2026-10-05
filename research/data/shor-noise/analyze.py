#!/usr/bin/env python3
"""Analysis of the noisy gate-level Shor trajectories (research/shor/shor-noise.md).

Input: raw/*.csv[.gz] written by `examples/shor_noise.rs` (one row per
trajectory). Output: tables (stdout + CSV) and PNG plots in this directory.

Success metric ("peak"): the recorded y satisfies |y/2^t - s/r| < 1/(2 r^2)
for the nearest integer s, i.e. s/r (in lowest terms) is a continued-fraction
convergent of y/2^t and r follows with the standard small-multiple / lcm
post-processing. `order_strict` (r itself is a convergent denominator) and
`factor_ok` (the repo's shor::postprocess finds a factor) are also tallied.

Capping: a trajectory whose support ever exceeds C_VCAP x (noiseless peak
support of that instance) is "v-capped" (its real run may have been
stopped at the hard cap). Its outcome is counted as a failure in the
lower estimate S_k^lo; the calibrated estimate adds c_hat x (#v-capped),
with c_hat = success rate of v-capped trajectories measured on the
instances that were simulated without any real cap (n = 10, 11, 12).
"""
import csv, glob, gzip, math, os, random, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
C_VCAP = 8.0
random.seed(12345)


def opener(fn):
    return gzip.open(fn, 'rt') if fn.endswith('.gz') else open(fn)


def load_all():
    files = sorted(glob.glob(os.path.join(HERE, 'raw', '*.csv*')))
    rows, meta = [], {}
    for fn in files:
        with opener(fn) as f:
            lines = f.read().splitlines()
        hdr = [l for l in lines if l.startswith('# N=')]
        info = {}
        if hdr:
            for tok in hdr[0][2:].split():
                if '=' in tok:
                    k, v = tok.split('=', 1)
                    info[k] = v
        body = [l for l in lines if not l.startswith('#')]
        for r in csv.DictReader(body):
            r['file'] = os.path.basename(fn)
            r['L'] = int(info.get('locations', 0))
            r['mode'] = info.get('mode', 'strat')
            r['p'] = info.get('p', 'None')
            r['realcap'] = int(info.get('cap', 0))
            if os.path.basename(fn).startswith('rdep'):
                r['kind'] = r['kind'] + '@r=' + r['r']
            if info.get('reset_ancillas') == 'true':
                r['kind'] = r['kind'] + '+reset'
            rows.append(r)
        meta[os.path.basename(fn)] = info
    return rows, meta


def peak_ok(y, r, t):
    y = int(y)
    if y < 0:
        return None
    s = (y * r + (1 << (t - 1))) >> t
    return abs(y * r - (s << t)) * 2 * r < (1 << t)


def maxsupp(r):
    return max(int(v) for v in r['support'].split(';'))


def wilson(k, n, z=1.0):
    if n == 0:
        return (0, 0, 1)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0, c - h), min(1, c + h))


def binom_pmf(L, p, k):
    if p <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(math.lgamma(L + 1) - math.lgamma(k + 1) - math.lgamma(L - k + 1)
                    + k * math.log(p) + (L - k) * math.log1p(-p))


def main():
    rows, meta = load_all()
    for r in rows:
        r['n'] = int(r['n']); r['k'] = int(r['k']); r['r'] = int(r['r'])
        r['t'] = 2 * r['n']
        r['ok'] = peak_ok(r['measured'], r['r'], r['t'])
        r['ms'] = maxsupp(r)
    # noiseless peak support per instance (from k = 0 rows, any kind)
    base = {}
    for r in rows:
        if r['mode'] == 'strat' and r['k'] == 0:
            base[(r['N'], r['a'])] = max(base.get((r['N'], r['a']), 0), r['ms'])
    for r in rows:
        b = base.get((r['N'], r['a']))
        r['vcap'] = b is not None and r['ms'] > C_VCAP * b
        r['really_capped'] = r['capped_round'] != '-1'
        assert not (r['really_capped'] and not r['vcap']), (r['file'], 'real cap below the virtual cap')

    strat = [r for r in rows if r['mode'] == 'strat']
    # ---- calibration of v-capped success on uncapped instances
    cal = defaultdict(lambda: [0, 0])
    for r in strat:
        if r['vcap'] and r['realcap'] >= (1 << (r['t'])) and not r['really_capped']:
            cal[(r['n'], r['kind'])][0] += 1
            cal[(r['n'], r['kind'])][1] += bool(r['ok'])
    out = []
    out.append('## calibration: success among v-capped trajectories (instances run without a binding cap)')
    tot = [0, 0]
    for key in sorted(cal):
        m, s = cal[key]
        w = wilson(s, m)
        out.append(f'n={key[0]} {key[1]}: {s}/{m} = {w[0]:.4f} [{w[1]:.4f}, {w[2]:.4f}]')
        if key[1] == 'depol':
            tot[0] += m; tot[1] += s
    c_hat = tot[1] / tot[0] if tot[0] else 0.0
    out.append(f'c_hat (depol, pooled) = {c_hat:.4f} from {tot[0]} trajectories')
    chk = defaultdict(lambda: [0, 0])
    for (n_, kind_), (m, s_) in cal.items():
        chk[kind_][0] += m; chk[kind_][1] += s_
    c_hats = {k_: v[1] / v[0] for k_, v in chk.items() if v[0] >= 20}
    out.append('c_hat per kind (pooled over n): ' + ', '.join(f'{k_} {v:.4f}' for k_, v in sorted(c_hats.items())))
    CH = lambda kind_: c_hats.get(kind_, c_hat)

    # ---- strata
    cells = defaultdict(list)
    for r in strat:
        cells[(r['kind'], r['n'], r['k'])].append(r)
    table = []
    for (kind, n, k), v in sorted(cells.items()):
        m = len(v)
        ok_unc = sum(1 for r in v if r['ok'] and not r['vcap'])
        ok_all = sum(1 for r in v if r['ok'])  # true where nothing was really capped
        vc = sum(1 for r in v if r['vcap'])
        rc = sum(1 for r in v if r['really_capped'])
        strict = sum(1 for r in v if r['order_strict'] == '1')
        fac = sum(1 for r in v if r['factor_ok'] == '1')
        exact = rc == 0
        s_lo = ok_unc / m
        s_est = ok_all / m if exact else (ok_unc + CH(kind) * vc) / m
        table.append(dict(kind=kind, n=n, k=k, M=m, ok_unc=ok_unc, ok_all=ok_all, vcap=vc,
                          realcap=rc, exact=exact, S_lo=s_lo, S_est=s_est,
                          strict=strict / m, factor=fac / m, L=v[0]['L'], r=v[0]['r'],
                          N=v[0]['N'], secs=sum(float(x['secs']) for x in v) / m))
    with open(os.path.join(HERE, 'strata.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        for t_ in table:
            w.writerow(t_)
    out.append('\n## strata (S_k = P(peak success | exactly k faults))')
    out.append('kind n k M S_lo S_est(±1σ) vcapped realcapped strict factor')
    for t_ in table:
        p, lo, hi = wilson(round(t_['S_est'] * t_['M']), t_['M'])
        out.append(f"{t_['kind']} {t_['n']} {t_['k']} {t_['M']} {t_['S_lo']:.3f} {t_['S_est']:.3f}±{(hi-lo)/2:.3f} "
                   f"{t_['vcap']} {t_['realcap']} {t_['strict']:.3f} {t_['factor']:.3f}")

    # ---- per instance: d (per-fault damage), G_eff, P(p) curve, bootstrap
    inst = defaultdict(dict)
    for (kind, n, k), v in cells.items():
        inst[(kind, n)][k] = v
    summary = []
    for (kind, n), ks in sorted(inst.items()):
        if 1 not in ks:
            continue
        if 0 not in ks:
            # no-fault stratum is independent of the noise kind: borrow it
            donor = [v for (kd, nd), kv in inst.items() for kk, v in kv.items()
                     if nd == n and kk == 0 and kd.split('@')[0] in ('depol', 'depol+reset', 'bitflip', 'phaseflip')
                     and v[0]['a'] == ks[1][0]['a']]
            if not donor:
                continue
            ks = dict(ks)
            ks[0] = donor[0]
        L = ks[1][0]['L']
        kmax = max(ks)

        def est(v):
            if all(not r['really_capped'] for r in v):
                return sum(1 for r in v if r['ok']) / len(v)
            return (sum(1 for r in v if r['ok'] and not r['vcap']) + CH(kind) * sum(1 for r in v if r['vcap'])) / len(v)

        def est_lo(v):
            return sum(1 for r in v if r['ok'] and not r['vcap']) / len(v)

        def curve(S, p):
            P = sum(binom_pmf(L, p, k) * S[k] for k in range(kmax + 1))
            tail = 1 - sum(binom_pmf(L, p, k) for k in range(kmax + 1))
            return P, tail

        def fit(S):
            # S_k = q + (S0 - q)(1-d)^k, q = random-guess floor ~ 0 (fixed);
            # weighted least squares on k = 1..kmax for d
            S0 = S[0]
            best = None
            for i in range(1, 2000):
                d = i / 2000
                err = sum((S[k] - S0 * (1 - d) ** k) ** 2 for k in range(1, kmax + 1))
                if best is None or err < best[0]:
                    best = (err, d)
            return best[1]

        def p_half(S):
            # p where P(p) = S0/2 (bisection on log p)
            lo, hi = 1e-12, 50.0 / L
            for _ in range(80):
                mid = math.sqrt(lo * hi)
                if curve(S, mid)[0] > S[0] / 2:
                    lo = mid
                else:
                    hi = mid
            return math.sqrt(lo * hi)

        S = {k: est(v) for k, v in ks.items()}
        Slo = {k: est_lo(v) for k, v in ks.items()}
        d1 = 1 - S[1] / S[0]
        dfit = fit(S)
        ph = p_half(S)
        boots = []
        for _ in range(300):
            Sb = {k: est([random.choice(v) for _ in v]) for k, v in ks.items()}
            if Sb[0] == 0:
                continue
            boots.append((1 - Sb[1] / Sb[0], fit(Sb), p_half(Sb)))
        def sd(i):
            xs = [b[i] for b in boots]
            mu = sum(xs) / len(xs)
            return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))
        r0 = ks[0][0]['r']
        summary.append(dict(kind=kind, n=n, N=ks[0][0]['N'], r=r0, L=L, kmax=kmax,
                            M1=len(ks[1]), S0=S[0], S1=S[1], S1_lo=Slo[1],
                            d1=d1, d1_sd=sd(0), dfit=dfit, dfit_sd=sd(1),
                            Geff1=L * d1, Geff1_sd=L * sd(0), Gefffit=L * dfit, Gefffit_sd=L * sd(1),
                            p_half=ph, p_half_sd=sd(2),
                            slack=r0 and (2 * n - 2 * math.log2(r0)),
                            exact=all(not r['really_capped'] for v in ks.values() for r in v)))
    with open(os.path.join(HERE, 'summary.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for s in summary:
            w.writerow(s)
    out.append('\n## per instance: per-fault damage d = 1 - S1/S0, G_eff = L d, p_1/2')
    out.append('kind n r L S0 S1 d1±sd dfit±sd Geff1 Gefffit p_half slack exact')
    for s in summary:
        out.append(f"{s['kind']} {s['n']} {s['r']} {s['L']} {s['S0']:.3f} {s['S1']:.3f} "
                   f"{s['d1']:.3f}±{s['d1_sd']:.3f} {s['dfit']:.3f}±{s['dfit_sd']:.3f} "
                   f"{s['Geff1']:.3g} {s['Gefffit']:.3g} {s['p_half']:.3g}±{s['p_half_sd']:.2g} {s['slack']:.2f} {s['exact']}")

    # ---- breakdown of single faults (k = 1, depol): class damage
    out.append('\n## single-fault breakdown (k=1): class share of locations, P(success|fault in class)/S0')
    s0 = {(s['kind'], s['n']): s['S0'] for s in summary}
    bd = defaultdict(lambda: [0, 0.0])
    bd_n = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    for r in strat:
        if r['k'] != 1:
            continue
        f = r['faults'].split('/')
        rnd, site, gname, q, role, pauli = int(f[0]), f[1], f[2], f[3], f[4], f[5]
        okv = (1.0 if r['ok'] else 0.0) if not r['vcap'] else (CH(r['kind']) if r['really_capped'] else (1.0 if r['ok'] else 0.0))
        cls = (r['kind'], f'{site}:{role}:{pauli}' if site == 'gate' else f'{site}:{pauli}')
        bd[cls][0] += 1; bd[cls][1] += okv
        bd_n[(r['kind'], r['n'])][('pauli', pauli)][0] += 1
        bd_n[(r['kind'], r['n'])][('pauli', pauli)][1] += okv
        bd_n[(r['kind'], r['n'])][('role', role)][0] += 1
        bd_n[(r['kind'], r['n'])][('role', role)][1] += okv
        rel = rnd / r['t']
        bd_n[(r['kind'], r['n'])][('rounddec', int(rel * 10))][0] += 1
        bd_n[(r['kind'], r['n'])][('rounddec', int(rel * 10))][1] += okv
        bd_n[(r['kind'], r['n'])][('dirty', r['dirty_from'] != '-1')][0] += 1
        bd_n[(r['kind'], r['n'])][('dirty', r['dirty_from'] != '-1')][1] += okv
    tot_by_kind = defaultdict(int)
    for (kind, c), (m, s) in bd.items():
        tot_by_kind[kind] += m
    rowsb = []
    for (kind, c), (m, s) in sorted(bd.items(), key=lambda x: (x[0][0], -x[1][0])):
        rowsb.append(dict(kind=kind, cls=c, count=m, share=m / tot_by_kind[kind], success=s / m))
        out.append(f'{kind} {c:28s} share {m / tot_by_kind[kind]:.3f} (n={m}) P(ok|fault) {s / m:.3f}')
    with open(os.path.join(HERE, 'breakdown_classes.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(rowsb[0].keys()))
        w.writeheader()
        for x in rowsb:
            w.writerow(x)
    rowsn = []
    for (kind, n), d in sorted(bd_n.items()):
        for key, (m, s) in sorted(d.items(), key=lambda x: str(x[0])):
            rowsn.append(dict(kind=kind, n=n, group=key[0], value=key[1], count=m, success=s / m,
                              rel=(s / m) / s0[(kind, n)] if s0.get((kind, n)) else ''))
    with open(os.path.join(HERE, 'breakdown_by_n.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(rowsn[0].keys()))
        w.writeheader()
        for x in rowsn:
            w.writerow(x)

    # ---- window model: P(success | 1 fault) by fault family and window
    # start window: rounds i < floor(t - 2 log2 r) (spare low bits);
    # end window: rounds i >= t - nu2(r) (bits only needed mod 2^(t - nu2(r)))
    def nu2(x):
        return (x & -x).bit_length() - 1
    win = defaultdict(lambda: [0, 0.0])
    win_n = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    for r in strat:
        if r['k'] != 1 or r['kind'] not in ('depol', 'bitflip', 'phaseflip', 'depol+reset'):
            continue
        f = r['faults'].split('/')
        rnd, site, pauli = int(f[0]), f[1], f[5]
        t = r['t']; rr = r['r']
        sw = math.floor(t - 2 * math.log2(rr))
        ew = t - nu2(rr)
        w_ = 'start' if rnd < sw else ('end' if rnd >= ew else 'middle')
        fam = 'Z' if pauli == 'Z' else 'XY'
        if site in ('prep', 'meas'):
            fam = 'XY'
        okv = (1.0 if r['ok'] else 0.0) if not r['really_capped'] else CH(r['kind'])
        win[(r['kind'], fam, w_)][0] += 1
        win[(r['kind'], fam, w_)][1] += okv
        win_n[(r['kind'], r['n'])][(fam, w_)][0] += 1
        win_n[(r['kind'], r['n'])][(fam, w_)][1] += okv
    out.append('\n## window model: P(success | 1 fault) by family x window (pooled over n)')
    rates = {}
    for key in sorted(win):
        m, s_ = win[key]
        w_ = wilson(round(s_), m)
        rates[key] = s_ / m
        out.append(f'{key[0]:12s} {key[1]:2s} {key[2]:6s} n={m:5d} P(ok) = {s_ / m:.3f} [{w_[1]:.3f}, {w_[2]:.3f}]')
    # predicted damage per instance from the pooled middle rates + the
    # instance's window sizes (each family's locations ~ uniform over rounds)
    out.append('\n## window-model prediction of d vs measured (depol)')
    out.append('n t nu2 start_window d_measured d_model')
    model_rows = []
    for s_ in summary:
        if s_['kind'] != 'depol':
            continue
        t = 2 * s_['n']; rr = s_['r']
        sw = max(0, math.floor(t - 2 * math.log2(rr))); nu = nu2(rr)
        fz = 1 / 3  # depolarizing: Z on 1/3 of gate locations (prep/meas are X; < 0.01 % of L)
        mid = (t - sw - nu) / t
        e_xy = rates.get(('depol', 'XY', 'middle'), 0.0)
        z_mid = rates.get(('depol', 'Z', 'middle'), 0.0)
        e_xy_s = rates.get(('depol', 'XY', 'start'), 0.0)
        z_s = rates.get(('depol', 'Z', 'start'), 1.0)
        e_xy_e = rates.get(('depol', 'XY', 'end'), 1.0)
        z_e = rates.get(('depol', 'Z', 'end'), 1.0)
        ok_pred = ((1 - fz) * (sw / t * e_xy_s + mid * e_xy + nu / t * e_xy_e)
                   + fz * (sw / t * z_s + mid * z_mid + nu / t * z_e))
        d_model = 1 - ok_pred / max(s_['S0'], 1e-9)
        model_rows.append(dict(n=s_['n'], t=t, nu2=nu, start_window=sw, d=s_['d1'], d_sd=s_['d1_sd'], d_model=d_model))
        out.append(f"{s_['n']} {t} {nu} {sw} {s_['d1']:.3f}±{s_['d1_sd']:.3f} {d_model:.3f}")
    e_xy = rates.get(('depol', 'XY', 'middle'), 0.0); z_mid = rates.get(('depol', 'Z', 'middle'), 0.0)
    d_inf = 1 - ((2 / 3) * e_xy + (1 / 3) * z_mid)
    out.append(f'large-n limit (windows -> 0 rounds / t): d_inf = 1 - (2/3) e_XY - (1/3) z_mid = {d_inf:.3f}')
    if model_rows:
        with open(os.path.join(HERE, 'window_model.csv'), 'w') as f:
            w = csv.DictWriter(f, fieldnames=list(model_rows[0].keys()))
            w.writeheader()
            for x in model_rows:
                w.writerow(x)

    # ---- direct-sampling cross-checks
    direct = [r for r in rows if r['mode'] == 'direct']
    if direct:
        out.append('\n## direct sampling at fixed p vs the stratified prediction')
        dg = defaultdict(list)
        for r in direct:
            dg[(r['kind'], r['n'], r['p'])].append(r)
        for (kind, n, p), v in sorted(dg.items()):
            pv = float(p.strip('Some()'))
            m = len(v)
            ok_unc = sum(1 for r in v if r['ok'] and not r['vcap'])
            vc = sum(1 for r in v if r['vcap'])
            exact = all(not r['really_capped'] for r in v)
            est_ = (sum(1 for r in v if r['ok']) / m) if exact else (ok_unc + c_hat * vc) / m
            w_ = wilson(round(est_ * m), m)
            pred = None
            for s in summary:
                if s['kind'] == kind and s['n'] == n:
                    ks = inst[(kind, n)]
                    L = s['L']
                    Sx = {k: (sum(1 for r in vv if r['ok']) / len(vv)) if all(not r['really_capped'] for r in vv)
                          else (sum(1 for r in vv if r['ok'] and not r['vcap']) + c_hat * sum(1 for r in vv if r['vcap'])) / len(vv)
                          for k, vv in ks.items()}
                    pred = sum(binom_pmf(L, pv, k) * Sx[k] for k in Sx)
                    tail = (1 - sum(binom_pmf(L, pv, k) for k in Sx)) * Sx[max(Sx)]  # S_k <= S_kmax
            out.append(f'{kind} n={n} p={pv:.3g} (pL={pv * v[0]["L"]:.2f}) direct {est_:.3f} [{w_[1]:.3f},{w_[2]:.3f}] M={m}; '
                       f'stratified {pred:.3f} (+ tail ≤ {tail:.3f})')
    txt = '\n'.join(out)
    print(txt)
    with open(os.path.join(HERE, 'analysis.txt'), 'w') as f:
        f.write(txt + '\n')
    return summary, table, inst, c_hat


if __name__ == '__main__':
    main()
