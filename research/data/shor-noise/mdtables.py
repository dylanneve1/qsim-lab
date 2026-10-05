#!/usr/bin/env python3
"""Markdown tables for research/shor/shor-noise.md from summary.csv / strata.csv / window_model.csv."""
import csv, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
summ = list(csv.DictReader(open(os.path.join(HERE, 'summary.csv'))))
strata = list(csv.DictReader(open(os.path.join(HERE, 'strata.csv'))))
gates = {}
for line in open(os.path.join(HERE, 'gate_counts.txt')):
    if line.startswith('N='):
        kv = dict(tok.split('=') for tok in line.split() if '=' in tok)
        if kv['kind'] == 'depol':
            gates[int(kv['n'])] = int(kv['gates'])


def nu2(x):
    return (x & -x).bit_length() - 1


def f(x, d=3):
    return f'{float(x):.{d}f}'


print('### Main table (depolarizing)\n')
print('| n | N | r | ν₂(r) | spare low bits t−2log₂r | gates | L | S₀ | S₁ | S₂ | S₃ | d = 1−S₁/S₀ | G_eff = L·d | p½ (P_succ = S₀/2) |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for s in sorted((s for s in summ if s['kind'] == 'depol'), key=lambda s: int(s['n'])):
    n = int(s['n'])
    st = {int(x['k']): x for x in strata if x['kind'] == 'depol' and int(x['n']) == n}

    def sk(k):
        if k not in st:
            return '–'
        m = int(st[k]['M']); p = float(st[k]['S_est'])
        return f'{p:.3f}±{math.sqrt(max(p * (1 - p), 1 / m) / m):.3f} ({m})'
    r = int(s['r'])
    print(f"| {n} | {s['N']} | {r} | {nu2(r)} | {float(s['slack']):.1f} | {gates.get(n, '')} | {s['L']} | {sk(0)} | {sk(1)} | {sk(2)} | {sk(3)} | "
          f"{f(s['d1'])}±{f(s['d1_sd'])} | {float(s['Geff1']):.3g}±{float(s['Geff1_sd']):.2g} | {float(s['p_half']):.3g}±{float(s['p_half_sd']):.1g} |")

print('\n### Other noise channels and ancilla reset\n')
print('| channel | n | S₀ | S₁ | d | G_eff | p½ |')
print('|---|---|---|---|---|---|---|')
for s in sorted((s for s in summ if s['kind'] != 'depol' and '@' not in s['kind']), key=lambda s: (s['kind'], int(s['n']))):
    print(f"| {s['kind']} | {s['n']} | {f(s['S0'])} | {f(s['S1'])} | {f(s['d1'])}±{f(s['d1_sd'])} | {float(s['Geff1']):.3g} | {float(s['p_half']):.3g} |")

print('\n### Same N (n = 20), bases of different order\n')
print('| r | ν₂(r) | spare low bits | S₁ | d |')
print('|---|---|---|---|---|')
for s in sorted((s for s in summ if '@' in s['kind']), key=lambda s: int(s['r'])):
    r = int(s['r'])
    print(f"| {r} | {nu2(r)} | {float(s['slack']):.1f} | {f(s['S1'])} | {f(s['d1'])}±{f(s['d1_sd'])} |")
