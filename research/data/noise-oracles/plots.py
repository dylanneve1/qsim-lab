#!/usr/bin/env python3
"""Plots for research/noise-oracles.md (reads summary.csv, fatality_blocks.csv
written by analyze.py, and ../shor-noise/summary.csv for the round-4 windowed
oracle). Palette: the dataviz reference categorical order (validated: CVD
adjacent dE >= 9.1); identity is also carried by marker shape and the legend,
and every number is in the tables of the note."""
import csv, math, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SERIES = [  # (variant key, label, color, marker) in fixed order
    ('windowed', 'windowed (round 4)', '#2a78d6', 'o'),
    ('WindowedOpt(4)', 'windowed-opt', '#eb6834', 's'),
    ('WindowedMbuLookup(4)', 'mbu-lookup', '#1baf7a', '^'),
    ('WindowedMbu(4)', 'mbu (full)', '#eda100', 'D'),
    ('Ge(2;4)', 'GE windowed exp. (w_e=2)', '#e87ba4', 'v'),
]
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def load():
    rows = list(csv.DictReader(open(os.path.join(HERE, 'summary.csv'))))
    data = {}
    for r in rows:
        if r['kind'] != 'depol':
            continue
        data.setdefault(r['variant'], []).append(r)
    old = list(csv.DictReader(open(os.path.join(HERE, '../shor-noise/summary.csv'))))
    data['windowed'] = [dict(n=r['n'], p_half=r['p_half'], p_half_sd=r['p_half_sd'], d=r['d1'],
                             d_sd=r['d1_sd'], Lbar=r['L']) for r in old if r['kind'] == 'depol']
    return data


def main():
    data = load()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF)
    ax = axes[0]
    for key, lab, col, mk in SERIES:
        v = sorted(data.get(key, []), key=lambda r: int(r['n']))
        if not v:
            continue
        ns = [int(r['n']) for r in v]
        ax.errorbar(ns, [float(r['p_half']) for r in v], yerr=[float(r['p_half_sd']) for r in v],
                    color=col, marker=mk, ms=6, lw=2, capsize=2, label=lab)
    ax.set_yscale('log')
    ax.set_xlabel('modulus bits n')
    ax.set_ylabel('p½ (per gate-qubit location)')
    ax.set_title('Depolarizing rate that halves the success probability')
    style(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    ax = axes[1]
    for key, lab, col, mk in SERIES:
        v = sorted(data.get(key, []), key=lambda r: int(r['n']))
        if not v:
            continue
        ns = [int(r['n']) for r in v]
        dk = 'd' if 'd' in v[0] else 'd1'
        ax.errorbar(ns, [float(r[dk]) for r in v], yerr=[float(r['d_sd']) for r in v],
                    color=col, marker=mk, ms=6, lw=2, capsize=2, label=lab)
    ax.set_xlabel('modulus bits n')
    ax.set_ylabel('d = P(one random fault is fatal)')
    ax.set_ylim(0.3, 0.85)
    ax.set_title('Per-location fatality')
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'phalf_d_vs_n.png'), dpi=130, facecolor=SURF)

    # fatality map: share of locations vs share of G_eff by block (coarse)
    fb = list(csv.DictReader(open(os.path.join(HERE, 'fatality_coarse.csv'))))
    blocks = ['lookup', 'unlookup', 'modadd', 'swap']
    vs = [s for s in SERIES[1:] if any(r['variant'] == s[0] for r in fb)]
    fig, axes = plt.subplots(1, len(vs), figsize=(3.2 * len(vs), 3.6), facecolor=SURF, sharey=True)
    if len(vs) == 1:
        axes = [axes]
    for ax, (key, lab, col, mk) in zip(axes, vs):
        sel = {r['cls']: r for r in fb if r['variant'] == key}
        xs = range(len(blocks))
        sh = [float(sel[b]['share']) if b in sel else 0 for b in blocks]
        pk = [1 - float(sel[b]['P_ok_rel']) if b in sel else 0 for b in blocks]
        ax.bar([x - 0.2 for x in xs], sh, width=0.38, color='#c3c2b7', label='share of locations')
        ax.bar([x + 0.2 for x in xs], pk, width=0.38, color=col, label='P(fatal | fault there)')
        ax.set_xticks(list(xs))
        ax.set_xticklabels(blocks, fontsize=8)
        ax.set_title(lab, fontsize=10)
        ax.set_ylim(0, 1)
        style(ax)
        ax.legend(frameon=False, fontsize=7, labelcolor=INK2, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fatality_blocks.png'), dpi=130, facecolor=SURF)


if __name__ == '__main__':
    main()
