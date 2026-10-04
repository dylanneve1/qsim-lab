#!/usr/bin/env python3
"""families.jsonl + scaling.jsonl -> tables.md (per-family branching rank)."""
import json, sys, os
here = os.path.dirname(os.path.abspath(__file__))
rows = []
seen = set()
for f in ['families.jsonl', 'scaling.jsonl']:
    for l in open(os.path.join(here, f)):
        d = json.loads(l)
        if 'error' in d or d['spec'] in seen:
            continue
        seen.add(d['spec'])
        rows.append(d)
out = ['| instance | n | gates | T-count | d | ν max / end (n≤12) | max r_k | r_end | log2 max r / d | pair merges | secs (VPS, loaded) |',
       '|---|---|---|---|---|---|---|---|---|---|---|']
import math
for d in sorted(rows, key=lambda d: (d['spec'].split(':')[0], d['n'])):
    mr = str(d['max_r']) if d['ok'] else f">1024 (stopped at gate {d['stopped_at']}/{d['gates']})"
    nu = f"{d['nu_max']:.0f} / {d['nu_end']:.0f}" if d['nu_max'] >= 0 else '—'
    lr = f"{math.log2(max(d['max_r'],1)):.1f} / {d['d']}" if d['ok'] else f"> 10 / {d['d']}"
    out.append(f"| `{d['spec']}` | {d['n']} | {d['gates']} | {d['t']} | {d['d']} | {nu} | {mr} | {d['r_end'] if d['ok'] else '—'} | {lr} | {d['pair_merges']} | {d['secs']:.3g} |")
open(os.path.join(here, 'tables.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
