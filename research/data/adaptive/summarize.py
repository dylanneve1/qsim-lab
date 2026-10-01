#!/usr/bin/env python3
"""Min / max time per (circuit, method) from interleaved ab.sh output."""
import sys, collections
rows = collections.OrderedDict()
for line in open(sys.argv[1]):
    if not line.startswith('| ') or line.startswith('| circuit'):
        continue
    c = [x.strip() for x in line.strip().strip('|').split('|')]
    key = (c[0], c[1])
    t = float(c[-1])
    r = rows.setdefault(key, {'t': [], 'row': c})
    r['t'].append(t)
print('| circuit | method | value | peak terms | switch | dense qubits | min time (s) | max time (s) | runs |')
print('|---|---|---|---|---|---|---|---|---|')
for (circ, m), r in rows.items():
    c = r['row']
    print(f"| {circ} | {m} | {c[2]} | {c[3]} | {c[4]} | {c[5]} | {min(r['t']):.4f} | {max(r['t']):.4f} | {len(r['t'])} |")
