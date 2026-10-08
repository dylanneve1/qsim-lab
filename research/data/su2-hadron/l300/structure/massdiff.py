"""Compare the mass-term (rz(+-0.03)) pattern on logical (site, chain) wires between circuits."""
import os, sys
from parse import load
import graph_util
def mass_pattern(path, zzp):
    graph_util.ZZP = zzp
    m = graph_util.logical_map(path)
    ops = load(path); k = 0
    while ops[k][0] == 'x': k += 1
    lab = list(range(120)); steps = []; cur = {}
    for name, p, qs in ops[k:]:
        if name == 'swap':
            a, b = qs; lab[a], lab[b] = lab[b], lab[a]
        elif name == 'rz' and abs(abs(p) - 0.03) < 1e-12:
            s, l = m[lab[qs[0]]]
            cur[(s, l)] = cur.get((s, l), 0) + p
            if len(cur) == 120 and all(True for _ in cur):
                pass
    # total per wire over whole circuit
    return cur, m
old, mo = mass_pattern(sys.argv[1], 0.005)
new, mn = mass_pattern(sys.argv[2], 0.1130625)
print('old site0..7 chain0', [round(old[(s, 0)], 3) for s in range(8)], 'chain1', [round(old[(s, 1)], 3) for s in range(8)])
print('new site0..7 chain0', [round(new[(s, 0)], 3) for s in range(8)], 'chain1', [round(new[(s, 1)], 3) for s in range(8)])
print('max |old-new| total mass phase per wire:', max(abs(old[k] - new[k]) for k in old))
print('logical maps equal:', mo == mn)
