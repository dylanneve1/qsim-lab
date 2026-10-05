#!/usr/bin/env python3
"""Analysis of the noise-oracles campaign (research/shor/noise-oracles.md).

Input: raw/*.csv.gz from `examples/noise_oracles.rs` (one row per
trajectory; columns ok, weight, support trace, faults with block tags).
Output: analysis.txt, strata.csv, summary.csv, fatality_blocks.csv,
fatality_roles.csv, windows.csv, t3_rounds.csv in this directory.

Estimators (as research/shor/shor-noise.md, plus importance weights):
* S_k = (sum over non-v-capped trajectories of weight*ok + c_hat * sum of
  weights of v-capped ones) / M.  v-capped = support exceeded C_VCAP x the
  noiseless peak support (k = 0 rows) of the instance; c_hat = weighted
  success rate of v-capped trajectories in the calibration runs (n = 10, 11,
  no binding cap), per oracle.
* d = 1 - S1/S0, G_eff = Lbar * d, p_1/2 from P(p) = sum_k Binom(k; Lbar, p) S_k
  (k <= 3; tail bounded by S_k <= S_3 as in shor-noise; Lbar = mean L of the
  oracle's resolved streams), 1-sigma errors by bootstrap over trajectories.
"""
import csv, glob, gzip, math, os, random, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
C_VCAP = 8.0
random.seed(4242)
ORDER = ['Windowed(4)', 'WindowedOpt(4)', 'WindowedMbuLookup(4)', 'WindowedMbu(4)', 'Ge(2;4)']
SHORT = {'Windowed(4)': 'windowed', 'WindowedOpt(4)': 'windowed-opt',
         'WindowedMbuLookup(4)': 'mbu-lookup', 'WindowedMbu(4)': 'mbu', 'Ge(2;4)': 'ge(2;4)'}


def load():
    rows = []
    for fn in sorted(glob.glob(os.path.join(HERE, 'raw', '*.csv.gz'))):
        with gzip.open(fn, 'rt') as f:
            lines = f.read().splitlines()
        hdr = [l for l in lines if l.startswith('# N=')]
        info = dict(tok.split('=', 1) for tok in hdr[0][2:].split() if '=' in tok)
        base = os.path.basename(fn)
        for r in csv.DictReader(l for l in lines if not l.startswith('#')):
            r['file'] = base
            r['realcap'] = int(info['cap'])
            r['variant'] = r['oracle'] + ('+reset' if info.get('reset_ancillas') == 'true' else '')
            if info.get('design', 'None') != 'None':
                r['variant'] += '+' + info['design'].lower()
            for tag in ('+design', ):
                if tag in base:
                    r['variant'] += tag
            r['cal'] = 'cal' in base.split('_')[0]
            rows.append(r)
    for r in rows:
        r['n'] = int(r['n']); r['k'] = int(r['k']); r['r'] = int(r['r'])
        r['t'] = 2 * r['n']
        r['ok'] = r['ok'] == '1'
        r['w'] = float(r['weight'])
        r['L'] = int(r['L'])
        r['ms'] = max(int(v) for v in r['support'].split(';'))
        r['really_capped'] = r['capped_round'] != '-1'
        r['dirty'] = r['dirty_from'] != '-1'
        r['flist'] = [f.split('/') for f in r['faults'].split('|')] if r['faults'] else []
    return rows


def wspan(r, k):
    """Rounds (lo, hi) of engine round k (one window of w_e rounds for GE)."""
    o = r['oracle']
    we = int(o[3:].split(';')[0]) if o.startswith('Ge(') else 1
    return k * we, k * we + we - 1


def nu2(x):
    v = 0
    while x % 2 == 0:
        x //= 2; v += 1
    return v


def binom_pmf(L, p, k):
    return math.exp(math.lgamma(L + 1) - math.lgamma(k + 1) - math.lgamma(L - k + 1)
                    + k * math.log(p) + (L - k) * math.log1p(-p))


def wmean_sd(vals):
    m = len(vals)
    if m == 0:
        return float('nan'), float('nan')
    mu = sum(vals) / m
    if m < 2:
        return mu, float('nan')
    var = sum((v - mu) ** 2 for v in vals) / (m - 1)
    return mu, math.sqrt(var / m)


def main():
    rows = load()
    out = []
    # noiseless peak support per (variant, N)
    base = {}
    for r in rows:
        if r['k'] == 0:
            key = (r['oracle'], r['N'])
            base[key] = max(base.get(key, 0), r['ms'])
    for r in rows:
        # no k = 0 rows (calibration-only instances): the noiseless peak
        # support is max_i B_i = r / gcd(r, 2) (theory-shor T1)
        b = base.get((r['oracle'], r['N']), r['r'] // math.gcd(r['r'], 2))
        r['vcap'] = b is not None and r['ms'] > C_VCAP * b
        if r['really_capped'] and not r['vcap']:
            print('warning: real cap below the virtual cap', r['file'], file=sys.stderr)
            r['vcap'] = True

    # ---- calibration
    out.append('## calibration: weighted success among v-capped trajectories (no binding cap)')
    # theory-shor T3(a): a trajectory whose faults all act in the last
    # nu2(r) rounds succeeds with probability exactly S0, so a capped one is
    # counted with S0 (not c_hat), and it is left out of the calibration
    for r in rows:
        t = r['t']; ew = t - nu2(r['r'])
        r['endwin'] = bool(r['flist']) and all(wspan(r, int(f[0]))[0] >= ew for f in r['flist'])
    # success of trajectories that exceeded the v-cap but completed: the
    # calibration runs (n = 10, 11, no binding cap) and, for the MBU
    # oracles, every run with cap 16r at n >= 14 (their dirty supports stay
    # small; at n = 10, 12 the v-capped success is 12-13 %, at n >= 14 it
    # is 1-3 %: the large-n regime of n = 24)
    # calibration classes: oracle family (reversible: dirt persists; meas:
    # MBU / GE / reset variants, dirt is removed) x where the faults act
    # (start window only, or some fault in the middle)
    def fam(r):
        return 'rev' if r['variant'] in ('WindowedOpt(4)', 'Windowed(4)') else 'meas'

    def cls(r):
        t = r['t']; rr = r['r']
        ew = t - nu2(rr); sw = max(0, math.floor(t - 2 * math.log2(rr)))
        for f in r['flist']:
            lo, hi = wspan(r, int(f[0]))
            if lo < ew and hi >= sw:
                return 'middle'
        return 'start'
    for r in rows:
        r['ccls'] = (fam(r), cls(r))
    cal = defaultdict(lambda: [0, 0.0, 0.0])
    caln = defaultdict(lambda: [0, 0.0])
    for r in rows:
        if not r['vcap'] or r['really_capped'] or r['endwin']:
            continue
        use = r['cal'] if fam(r) == 'rev' else (
            r['n'] >= 14 and r['realcap'] >= 16 * r['r'] and '+' not in r['variant'])
        if use:
            c = cal[r['ccls']]
            c[0] += 1; c[1] += r['w'] * r['ok']; c[2] += r['w']
        cn = caln[(r['oracle'], r['n'])]
        cn[0] += 1; cn[1] += r['w'] * r['ok']
    chat = {}
    for o, (m, s, wsum) in sorted(cal.items()):
        chat[o] = s / wsum if wsum else 0.0
        se = math.sqrt(max(chat[o] * (1 - chat[o]), 1e-4) / m)
        out.append(f'{o}: {s:.1f}/{wsum:.1f} (M={m}) = {chat[o]:.4f} ± {se:.4f}')
    out.append('per n (completed v-capped, not end-window): ' + ', '.join(
        f"{SHORT.get(o, o)} n={n}: {x[1]:.0f}/{x[0]}" for (o, n), x in sorted(caln.items())))
    CH = lambda r: chat.get(r['ccls'], 0.021)

    # ---- strata
    cells = defaultdict(list)
    for r in rows:
        if r['cal']:
            continue
        cells[(r['variant'], r['kind'], r['n'], r['k'])].append(r)

    s0inst = defaultdict(lambda: [0.0, 0])
    for r in rows:
        if r['k'] == 0 and not r['cal']:
            s0inst[r['N']][0] += r['w'] * r['ok']; s0inst[r['N']][1] += 1
    S0I = {N: a / m for N, (a, m) in s0inst.items()}

    def est(v, o, lo=False):
        s = 0.0
        for r in v:
            if r['really_capped']:
                if r['endwin']:
                    s += S0I.get(r['N'], 1.0) * r['w']
                elif not lo:
                    s += CH(r) * r['w']
            else:
                s += r['w'] * r['ok']
        return s / len(v)

    strata = []
    for (var, kind, n, k), v in sorted(cells.items()):
        o = v[0]['oracle']
        strata.append(dict(variant=var, kind=kind, n=n, k=k, M=len(v), S=est(v, o), S_lo=est(v, o, True),
                           vcap=sum(r['vcap'] for r in v), realcap=sum(r['really_capped'] for r in v),
                           w_ne1=sum(abs(r['w'] - 1) > 1e-9 for r in v),
                           w_mean=sum(r['w'] for r in v) / len(v),
                           dirty=sum(r['dirty'] for r in v) / len(v),
                           Lbar=sum(r['L'] for r in v) / len(v), secs=sum(float(r['secs']) for r in v) / len(v)))
    with open(os.path.join(HERE, 'strata.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(strata[0].keys()))
        w.writeheader(); [w.writerow(s) for s in strata]
    out.append('\n## strata: S_k (calibrated), S_lo (v-capped = failure), counts')
    out.append('variant kind n k M S S_lo vcapped realcapped w!=1 mean_w dirty_frac Lbar secs')
    for s in strata:
        out.append(f"{SHORT.get(s['variant'], s['variant'])} {s['kind']} {s['n']} {s['k']} {s['M']} {s['S']:.3f} {s['S_lo']:.3f} "
                   f"{s['vcap']} {s['realcap']} {s['w_ne1']} {s['w_mean']:.4f} {s['dirty']:.3f} {s['Lbar']:.0f} {s['secs']:.3f}")

    # ---- per instance
    inst = defaultdict(dict)
    for (var, kind, n, k), v in cells.items():
        inst[(var, kind, n)][k] = v
    s0pool = defaultdict(list)
    for r in rows:
        if r['k'] == 0 and not r['cal']:
            s0pool[r['N']].append(r)
    summary = []
    for (var, kind, n), ks in sorted(inst.items()):
        if 1 not in ks or 0 not in ks:
            continue
        o = ks[1][0]['oracle']
        Lbar = sum(r['L'] for v in ks.values() for r in v) / sum(len(v) for v in ks.values())
        kmax = max(ks)

        def curve(S, p):
            P = sum(binom_pmf(Lbar, p, k) * S[k] for k in range(kmax + 1))
            tail = 1 - sum(binom_pmf(Lbar, p, k) for k in range(kmax + 1))
            return P + tail * S[kmax], P

        def p_half(S):
            lo, hi = 1e-13, 50.0 / Lbar
            for _ in range(90):
                mid = math.sqrt(lo * hi)
                if curve(S, mid)[0] > S[0] / 2:
                    lo = mid
                else:
                    hi = mid
            return math.sqrt(lo * hi)

        def p_half_lo(S):  # tail counted as failure
            lo, hi = 1e-13, 50.0 / Lbar
            for _ in range(90):
                mid = math.sqrt(lo * hi)
                if curve(S, mid)[1] > S[0] / 2:
                    lo = mid
                else:
                    hi = mid
            return math.sqrt(lo * hi)

        S = {k: est(v, o) for k, v in ks.items()}
        if any(k not in S for k in range(kmax + 1)):
            continue
        # S0 is a property of the instance (every oracle is exact): pool the
        # k = 0 trajectories of all oracles and variants of this N
        pool0 = s0pool[ks[1][0]['N']]
        S[0] = sum(r['w'] * r['ok'] for r in pool0) / len(pool0)
        d1 = 1 - S[1] / S[0]
        ph, phl = p_half(S), p_half_lo(S)
        # systematic lower bound: capped trajectories counted as failures
        # (except the end-window ones, exactly S0 by T3(a))
        Slo = {k: est(v, o, lo=True) for k, v in ks.items()}
        Slo[0] = S[0]
        ph_c0 = p_half(Slo)
        d_c0 = 1 - Slo[1] / S[0]
        boots = []
        for _ in range(400):
            Sb = {k: est([random.choice(v) for _ in v], o) for k, v in ks.items() if k > 0}
            Sb[0] = sum(r['w'] * r['ok'] for r in (random.choice(pool0) for _ in pool0)) / len(pool0)
            if Sb[0] <= 0:
                continue
            boots.append((1 - Sb[1] / Sb[0], p_half(Sb)))

        def sd(i):
            xs = [b[i] for b in boots]
            mu = sum(xs) / len(xs)
            return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))
        r0 = ks[1][0]['r']
        summary.append(dict(variant=var, kind=kind, n=n, N=ks[1][0]['N'], r=r0, Lbar=Lbar, kmax=kmax,
                            M1=len(ks[1]), M0=len(pool0), S0=S[0], S1=S[1], S2=S.get(2, float('nan')), S3=S.get(3, float('nan')),
                            d=d1, d_sd=sd(0), Geff=Lbar * d1, Geff_sd=Lbar * sd(0),
                            p_half=ph, p_half_sd=sd(1), p_half_tail0=phl, pL_half=ph * Lbar,
                            p_half_c0=ph_c0, d_c0=d_c0,
                            dirty1=sum(r['dirty'] for r in ks[1]) / len(ks[1])))
    if summary:
        with open(os.path.join(HERE, 'summary.csv'), 'w') as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader(); [w.writerow(s) for s in summary]
    out.append('\n## per instance: d = 1 - S1/S0, G_eff = Lbar d, p_1/2 (bootstrap 1 sigma)')
    out.append('variant kind n r Lbar S0 S1 S2 S3 d±sd Geff p_half±sd p_half(tail=0) p_half*L dirty_frac(k=1)')
    for s in summary:
        out.append(f"{SHORT.get(s['variant'], s['variant'])} {s['kind']} {s['n']} {s['r']} {s['Lbar']:.0f} {s['S0']:.3f} {s['S1']:.3f} "
                   f"{s['S2']:.3f} {s['S3']:.3f} {s['d']:.3f}±{s['d_sd']:.3f} {s['Geff']:.3g} "
                   f"{s['p_half']:.3e}±{s['p_half_sd']:.1e} {s['p_half_tail0']:.3e} {s['pL_half']:.3f} {s['dirty1']:.3f}")

    # ---- fatality map (k = 1, depol, gate faults): by block tag and by role
    s0 = {(s['variant'], s['kind'], s['n']): s['S0'] for s in summary}

    def fmap(keyf, fname, title):
        acc = defaultdict(lambda: [0, 0.0, 0.0])  # count, sum w*ok (calibrated), sum w
        tot = defaultdict(int)
        for (var, kind, n, k), v in cells.items():
            if k != 1 or kind != 'depol' or (var, kind, n) not in s0:
                continue
            for r in v:
                f = r['flist'][0]
                key = keyf(f)
                if key is None:
                    continue
                tot[var] += 1
                a = acc[(var, key)]
                a[0] += 1
                okv = ((S0I.get(r['N'], 1.0) if r['endwin'] else CH(r)) if r['really_capped'] else float(r['ok'])) / s0[(var, kind, n)]
                a[1] += r['w'] * okv
                a[2] += r['w']
        rowsf = []
        for (var, key), (m, s, ws) in sorted(acc.items()):
            share = m / tot[var]
            pok = s / m
            se = math.sqrt(max(pok * (1 - pok), 1e-4) / m)
            rowsf.append(dict(variant=var, cls=key, M=m, share=share, P_ok_rel=pok, P_ok_se=se,
                              geff_share=share * (1 - pok)))
        # normalise G_eff contributions to fractions
        gs = defaultdict(float)
        for x in rowsf:
            gs[x['variant']] += x['geff_share']
        for x in rowsf:
            x['geff_frac'] = x['geff_share'] / gs[x['variant']] if gs[x['variant']] else 0
        with open(os.path.join(HERE, fname), 'w') as f:
            w = csv.DictWriter(f, fieldnames=list(rowsf[0].keys()))
            w.writeheader(); [w.writerow(x) for x in rowsf]
        out.append(f'\n## {title} (k = 1, depol, pooled over n; P(ok|fault)/S0, share of locations, share of G_eff)')
        for var in ORDER + sorted({x['variant'] for x in rowsf} - set(ORDER)):
            sel = [x for x in rowsf if x['variant'] == var]
            if not sel:
                continue
            out.append(f'-- {SHORT.get(var, var)}')
            for x in sorted(sel, key=lambda x: -x['share']):
                out.append(f"   {x['cls']:<28} M={x['M']:<5} share={x['share']:.3f} P_ok={x['P_ok_rel']:.3f}±{x['P_ok_se']:.3f} G_eff_frac={x['geff_frac']:.3f}")
        return rowsf

    def blk(f):
        if f[1] != 'gate':
            return 'control'
        t = f[5].replace('.inv', '')
        return t

    def blk_pauli(f):
        b = blk(f)
        fam = 'Z' if f[6] == 'Z' else 'XY'
        if f[2].startswith('measx'):
            fam = f[2]
        return f'{b}:{fam}'

    def role(f):
        return f[4] if f[1] == 'gate' else 'ctrl(site)'

    def coarse(f):
        if f[1] != 'gate' or f[5].startswith('ctrl'):
            return 'control sites'
        return f[5].split('.')[0]

    fmap(coarse, 'fatality_coarse.csv', 'fatality by block')
    fmap(blk, 'fatality_blocks.csv', 'fatality by block.part')
    fmap(blk_pauli, 'fatality_blocks_pauli.csv', 'fatality by block.part x Pauli family')
    fmap(role, 'fatality_roles.csv', 'fatality by register (role of the faulted qubit)')

    # ---- windows (T3) and clean vs dirty
    out.append('\n## T3 windows: P(ok | 1 fault)/S0 by window x family (depol, pooled over n)')
    out.append('start = rounds < floor(t - 2 log2 r); end = rounds >= t - nu2(r)')
    win = defaultdict(lambda: [0, 0.0])
    wrows = []
    for (var, kind, n, k), v in cells.items():
        if k != 1 or kind != 'depol' or (var, kind, n) not in s0:
            continue
        for r in v:
            f = r['flist'][0]
            lo, hi = wspan(r, int(f[0])); t = r['t']; rr = r['r']
            sw = max(0, math.floor(t - 2 * math.log2(rr))); ew = t - nu2(rr)
            wnd = 'start' if hi < sw else ('end' if lo >= ew else 'middle')
            fam = 'Z' if f[6] == 'Z' else 'XY'
            okv = ((S0I.get(r['N'], 1.0) if r['endwin'] else CH(r)) if r['really_capped'] else float(r['ok'])) * r['w'] / s0[(var, kind, n)]
            for key in [(var, wnd, fam), (var, wnd, 'clean' if not r['dirty'] else 'dirty'), (var, wnd, 'all')]:
                win[key][0] += 1; win[key][1] += okv
    for var in ORDER + sorted({k[0] for k in win} - set(ORDER)):
        for fam in ['XY', 'Z', 'clean', 'dirty', 'all']:
            cellsw = []
            for wnd in ['start', 'middle', 'end']:
                m, s = win.get((var, wnd, fam), [0, 0.0])
                cellsw.append(f'{wnd} {s / m:.3f} ({m})' if m else f'{wnd} - (0)')
                wrows.append(dict(variant=var, family=fam, window=wnd, M=m, P_ok_rel=s / m if m else float('nan')))
            if any(win.get((var, w_, fam), [0])[0] for w_ in ['start', 'middle', 'end']):
                out.append(f'{SHORT.get(var, var):<14} {fam:<6} ' + ' | '.join(cellsw))
    with open(os.path.join(HERE, 'windows.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(wrows[0].keys()))
        w.writeheader(); [w.writerow(x) for x in wrows]

    # window model per instance: rates pooled per variant (clean/dirty split)
    out.append('\n## T3 window model: d_model from (window sizes, pooled clean/dirty rates, clean fraction) vs measured d')
    out.append('variant n d d_model clean_frac')
    for s in summary:
        var = s['variant']
        if s['kind'] != 'depol':
            continue
        v = cells[(var, 'depol', s['n'], 1)]
        cf = sum(not r['dirty'] for r in v) / len(v)
        t = 2 * s['n']; rr = s['r']
        sw = max(0, math.floor(t - 2 * math.log2(rr))); nu = nu2(rr)
        fr = {'start': sw / t, 'end': nu / t}
        fr['middle'] = 1 - fr['start'] - fr['end']

        def rate(wnd, fam):
            m, x = win.get((var, wnd, fam), [0, 0.0])
            return x / m if m else 1.0
        surv = sum(fr[w_] * (cf * rate(w_, 'clean') + (1 - cf) * rate(w_, 'dirty')) for w_ in fr)
        s['d_model'] = 1 - surv
        out.append(f"{SHORT.get(var, var)} {s['n']} {s['d']:.3f} {1 - surv:.3f} {cf:.3f}")
    # large-n limit per variant
    for var in ORDER:
        sel = [s for s in summary if s['variant'] == var and s['kind'] == 'depol']
        if not sel:
            continue
        cfs = []
        for s in sel:
            v = cells[(var, 'depol', s['n'], 1)]
            cfs.append(sum(not r['dirty'] for r in v) / len(v))
        cf = sum(cfs) / len(cfs)
        mc = win.get((var, 'middle', 'clean'), [0, 0]); md = win.get((var, 'middle', 'dirty'), [0, 0])
        if mc[0] and md[0]:
            dinf = 1 - (cf * mc[1] / mc[0] + (1 - cf) * md[1] / md[0])
            out.append(f'{SHORT.get(var, var)}: clean fraction {cf:.3f}, middle clean {mc[1]/mc[0]:.3f}, middle dirty {md[1]/md[0]:.3f} -> d_inf = {dinf:.3f}')

    # ---- T3 (b) start-window lower bound for clean faults, (d) dephasing
    # upper bound for dirty faults, per round, pooled into bound buckets
    out.append('\n## T3(b): clean single faults in rounds with Delta_i >= 1: measured P(ok) vs mean bound L(Delta_i)')
    out.append('## T3(d): dirty single faults: measured P(ok) vs mean dephasing bound 1/r + r_odd 2^(i+1+nu-t) (rounds with bound < 0.1)')
    tb = defaultdict(lambda: [0, 0.0, 0.0])
    for (var, kind, n, k), v in cells.items():
        if k != 1 or kind != 'depol':
            continue
        for r in v:
            f = r['flist'][0]
            i = wspan(r, int(f[0]))[0]; t = r['t']; rr = r['r']; nu = nu2(rr); rodd = rr >> nu
            if r['really_capped']:
                okv = (S0I.get(r['N'], 1.0) if r['endwin'] else CH(r)) * r['w']
            else:
                okv = r['w'] * r['ok']
            if not r['dirty']:
                delta = 2.0 ** (t - i - 2) / rr ** 2
                if delta >= 1:
                    cd = math.ceil(delta)
                    Lb = 1 - 1 / (2 * (cd - 3)) if cd >= 4 else 8 / math.pi ** 2
                    a = tb[(var, 'b')]
                    a[0] += 1; a[1] += okv; a[2] += Lb
            else:
                if i < t - nu:
                    B = 1 / rr + rodd * 2.0 ** (i + 1 + nu - t)
                    if B < 0.1:
                        a = tb[(var, 'd')]
                        a[0] += 1; a[1] += okv; a[2] += B
    for var in ORDER:
        for th in ['b', 'd']:
            m, x, b = tb.get((var, th), [0, 0, 0])
            if m:
                se = math.sqrt(max(x / m * (1 - x / m), 1e-4) / m)
                out.append(f"{SHORT.get(var, var):<14} T3({th}) M={m:<5} measured {x / m:.3f}±{se:.3f}  mean bound {b / m:.4f}  ({'>=' if th == 'b' else '<='} expected)")

    open(os.path.join(HERE, 'analysis.txt'), 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
