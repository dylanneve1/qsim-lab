#!/usr/bin/env python3
"""Plots for research/shor-noise.md (run analyze.py first; reads raw/ again)."""
import math, os, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze as A

HERE = A.HERE
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
ORD = ['#86b6ef', '#6da7ec', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#0d366b']


def style(ax, title, xl, yl):
    ax.set_facecolor(SURF)
    ax.figure.set_facecolor(SURF)
    ax.set_title(title, color=INK, fontsize=11, loc='left')
    ax.set_xlabel(xl, color=INK2)
    ax.set_ylabel(yl, color=INK2)
    ax.tick_params(colors=INK2)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, lw=0.6)


def main():
    summary, table, inst, c_hat = A.main()
    random.seed(7)
    # 1. P_succ(p) for depol, all n with k up to 3, ordinal blue ramp.
    # Line: stratified estimate; band: bootstrap 16-84 % over trajectories,
    # widened by the unmeasured k > kmax tail bounded by S_k <= S_kmax.
    ns = sorted(n for (k, n) in inst if k == 'depol' and max(inst[(k, n)]) >= 3)
    pick = ns if len(ns) <= 8 else [10, 12, 14, 16, 18, 20, 22, 24]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    rows_csv = []
    ps = [10 ** (e / 40) for e in range(-320, -190)]

    def S_of(ks):
        S = {}
        for k, v in ks.items():
            if all(not r['really_capped'] for r in v):
                S[k] = sum(1 for r in v if r['ok']) / len(v)
            else:
                S[k] = (sum(1 for r in v if r['ok'] and not r['vcap']) + c_hat * sum(1 for r in v if r['vcap'])) / len(v)
        return S

    def curve(S, L, kmax, p):
        pk = [A.binom_pmf(L, p, k) for k in range(kmax + 1)]
        base = sum(pk[k] * S[k] for k in range(kmax + 1))
        return base, base + (1 - sum(pk)) * S[kmax]

    for ci, n in enumerate([x for x in pick if x in ns]):
        ks = inst[('depol', n)]
        L = ks[1][0]['L']
        kmax = max(ks)
        S = S_of(ks)
        boots = [S_of({k: [random.choice(v) for _ in v] for k, v in ks.items()}) for _ in range(200)]
        P, lo, hi = [], [], []
        for p in ps:
            b, u = curve(S, L, kmax, p)
            bb = sorted(curve(Sb, L, kmax, p)[0] for Sb in boots)
            bu = sorted(curve(Sb, L, kmax, p)[1] for Sb in boots)
            P.append(b)
            lo.append(bb[int(0.16 * len(bb))])
            hi.append(bu[int(0.84 * len(bu))])
            rows_csv.append((n, L, p, b, lo[-1], hi[-1]))
        ax.plot(ps, P, color=ORD[ci], lw=2, label=f'n = {n}')
        ax.fill_between(ps, lo, hi, color=ORD[ci], alpha=0.18, lw=0)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.02)
    ax.set_xlim(ps[0], ps[-1])
    style(ax, 'P(order recoverable) vs Pauli rate p per gate-qubit location (depolarizing)',
          'p', 'P_succ')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'psucc_vs_p.png'), dpi=150)
    with open(os.path.join(HERE, 'psucc_vs_p.csv'), 'w') as f:
        f.write('n,L,p,P_succ,P_lo_16pct,P_hi_84pct_incl_tail\n')
        for r in rows_csv:
            f.write(','.join(f'{x:.6g}' for x in r) + '\n')

    # 2. G_eff vs n, with L(n): log-log; one chart per measure kind is not
    # needed - same unit (count of locations), one axis.
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    kinds = [('depol', 'depolarizing'), ('bitflip', 'bit-flip'), ('phaseflip', 'phase-flip'),
             ('depol+reset', 'depolarizing + ancilla reset')]
    Ls = sorted({(s['n'], s['L']) for s in summary if s['kind'] == 'depol'})
    ax.plot([x[0] for x in Ls], [x[1] for x in Ls], color=INK2, lw=1.2, ls='--', label='L (all locations)')
    for ci, (k, lab) in enumerate(kinds):
        ss = sorted((s for s in summary if s['kind'] == k), key=lambda s: s['n'])
        if not ss:
            continue
        ax.errorbar([s['n'] for s in ss], [s['Geff1'] for s in ss], yerr=[s['Geff1_sd'] for s in ss],
                    color=CAT[ci], lw=2, marker='o', ms=5, capsize=2, label=f'G_eff, {lab}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([10, 12, 14, 16, 18, 20, 22, 24]))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    style(ax, 'Effective number of fatal locations G_eff = L(1 - S1/S0)', 'n (bits of N)', 'locations')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'geff_vs_n.png'), dpi=150)

    # 2b. per-fault damage d = G_eff / L vs n, with the window model
    import csv as _csv
    wm = list(_csv.DictReader(open(os.path.join(HERE, 'window_model.csv'))))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for ci, (k, lab) in enumerate(kinds):
        ss = sorted((s_ for s_ in summary if s_['kind'] == k), key=lambda s_: s_['n'])
        if not ss:
            continue
        ax.errorbar([s_['n'] for s_ in ss], [s_['d1'] for s_ in ss], yerr=[s_['d1_sd'] for s_ in ss],
                    color=CAT[ci], lw=2, marker='o', ms=5, capsize=2, label=lab)
    ax.plot([int(x['n']) for x in wm], [float(x['d_model']) for x in wm], color=INK2, lw=1.2, ls='--',
            marker='x', ms=6, label='window model (depolarizing)')
    ax.set_ylim(0, 1)
    style(ax, 'Per-fault damage d = 1 - S1/S0 = G_eff / L', 'n (bits of N)', 'd')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'damage_vs_n.png'), dpi=150)

    # 3. single-fault success vs round position, by Pauli (depol, pooled n)
    rows, _ = A.load_all()
    series = ['X or Y', 'Z', 'X or Y, ancilla reset', 'Z, ancilla reset']
    pos = {k_: [[0, 0.0] for _ in range(10)] for k_ in series}
    for r in rows:
        if r['kind'] not in ('depol', 'depol+reset') or r['mode'] != 'strat' or r['k'] != '1':
            continue
        f = r['faults'].split('/')
        fam = 'Z' if f[5] == 'Z' else 'X or Y'
        key = fam + (', ancilla reset' if r['kind'] == 'depol+reset' else '')
        t = 2 * int(r['n'])
        b = min(9, int(10 * int(f[0]) / t))
        okv = A.peak_ok(r['measured'], int(r['r']), t)
        if okv is None:
            okv = c_hat
        pos[key][b][0] += 1
        pos[key][b][1] += float(okv)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = [i / 10 + 0.05 for i in range(10)]
    for ci, pa in enumerate(series):
        ys = [s / m if m else float('nan') for m, s in pos[pa]]
        es = [math.sqrt(max(y * (1 - y), 1e-4) / m) if m else 0 for y, (m, s) in zip(ys, pos[pa])]
        ax.errorbar(xs, ys, yerr=es, color=CAT[ci], lw=2, marker='o', ms=5, capsize=2,
                    ls='-' if 'reset' not in pa else ':', label=pa)
    ax.set_ylim(0, 1.02)
    style(ax, 'One depolarizing fault: P(success) vs where it hits (round i of t), n = 10-24 pooled',
          'round position i / t (0 = first, least-significant bit)', 'P(success | 1 fault)')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='upper left', bbox_to_anchor=(0.22, 0.98))
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'single_fault_by_round.png'), dpi=150)
    with open(os.path.join(HERE, 'single_fault_by_round.csv'), 'w') as f:
        f.write('pauli,decile,count,success\n')
        for pa in pos:
            for i, (m, s) in enumerate(pos[pa]):
                f.write(f'{pa},{i},{m},{s / m if m else ""}\n')


if __name__ == '__main__':
    main()
