#!/usr/bin/env python3
"""Markdown tables for research/noise-oracles.md from summary.csv,
fatality_*.csv, windows.csv (written by analyze.py) -> tables.md."""
import csv, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ORD = ['WindowedOpt(4)', 'WindowedMbuLookup(4)', 'WindowedMbu(4)', 'Ge(2;4)']
NAME = {'WindowedOpt(4)': 'windowed-opt', 'WindowedMbuLookup(4)': 'mbu-lookup',
        'WindowedMbu(4)': 'mbu', 'Ge(2;4)': 'GE w_e=2'}


def f(x, d=3):
    return f'{float(x):.{d}f}'


def e(x, s):
    x, s = float(x), float(s)
    exp = int(f'{x:e}'.split('e')[1])
    return f'{x / 10**exp:.2f}±{s / 10**exp:.2f}e{exp}'


def main():
    S = list(csv.DictReader(open(os.path.join(HERE, 'summary.csv'))))
    old = {int(r['n']): r for r in csv.DictReader(open(os.path.join(HERE, '../shor-noise/summary.csv')))
           if r['kind'] == 'depol'}
    out = []
    out.append('### Per oracle and size (depolarizing)\n')
    out.append('| oracle | n | L̄ | S₀ | S₁ | S₂ | S₃ | d | G_eff | p½ | p½ (ĉ = 0) | p½·L̄ | p½ / p½(round 4) |')
    out.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for v in ORD:
        for r in sorted((r for r in S if r['variant'] == v and r['kind'] == 'depol'), key=lambda r: int(r['n'])):
            n = int(r['n'])
            ratio = float(r['p_half']) / float(old[n]['p_half']) if n in old else float('nan')
            out.append(f"| {NAME[v]} | {n} | {float(r['Lbar']):,.0f} | {f(r['S0'])} | {f(r['S1'])} | {f(r['S2'])} | {f(r['S3'])} | "
                       f"{f(r['d'])}±{f(r['d_sd'])} | {float(r['Geff']):.3g} | {e(r['p_half'], r['p_half_sd'])} | {float(r['p_half_c0']):.2e} | {f(r['pL_half'], 2)} | {ratio:.2f} |")
    out.append('\n### Design variants (resets of should-be-clean ancillas; no added gates, reset flips counted as locations)\n')
    out.append('| oracle | n | resets | L̄ | d | p½ | p½ / p½(no resets) |')
    out.append('|---|---|---|---|---|---|---|')
    base = {(r['variant'], int(r['n'])): r for r in S if r['kind'] == 'depol'}
    for v in ORD:
        for n in sorted({int(r['n']) for r in S}):
            b = base.get((v, n))
            if not b:
                continue
            rows = [('none', b)] + [(m, base[(v + '+' + m, n)]) for m in ('round', 'window') if (v + '+' + m, n) in base]
            if len(rows) == 1:
                continue
            for m, r in rows:
                ratio = float(r['p_half']) / float(b['p_half'])
                out.append(f"| {NAME[v]} | {n} | {m} | {float(r['Lbar']):,.0f} | {f(r['d'])}±{f(r['d_sd'])} | {e(r['p_half'], r['p_half_sd'])} | {ratio:.2f} |")
    for fn, title in [('fatality_coarse.csv', 'block'), ('fatality_blocks.csv', 'block.part')]:
        F = list(csv.DictReader(open(os.path.join(HERE, fn))))
        out.append(f'\n### Fatality map by {title} (k = 1, depolarizing, pooled over n; P(ok | fault)/S₀; share of G_eff)\n')
        cls = []
        for r in F:
            if r['variant'] in ORD and r['cls'] not in cls:
                cls.append(r['cls'])
        out.append('| ' + title + ' | ' + ' | '.join(f'{NAME[v]}: share L / P(ok) / share G_eff' for v in ORD) + ' |')
        out.append('|---|' + '---|' * len(ORD))
        idx = {(r['variant'], r['cls']): r for r in F}
        for c in cls:
            cells = []
            for v in ORD:
                r = idx.get((v, c))
                cells.append(f"{float(r['share']):.3f} / {float(r['P_ok_rel']):.2f}±{float(r['P_ok_se']):.2f} / {float(r['geff_frac']):.3f}" if r else '—')
            out.append(f'| {c} | ' + ' | '.join(cells) + ' |')
    W = list(csv.DictReader(open(os.path.join(HERE, 'windows.csv'))))
    out.append('\n### T3 windows: P(ok | one fault)/S₀ by window (pooled over n; trajectory counts in brackets)\n')
    out.append('| oracle | family | start | middle | end |')
    out.append('|---|---|---|---|---|')
    wi = defaultdict(dict)
    for r in W:
        wi[(r['variant'], r['family'])][r['window']] = r
    for v in ORD:
        for fam in ['XY', 'Z', 'clean', 'dirty']:
            d = wi.get((v, fam))
            if not d:
                continue
            cells = [f"{float(d[w]['P_ok_rel']):.3f} ({d[w]['M']})" if int(d[w]['M']) else '—' for w in ['start', 'middle', 'end']]
            out.append(f'| {NAME[v]} | {fam} | ' + ' | '.join(cells) + ' |')
    open(os.path.join(HERE, 'tables.md'), 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
