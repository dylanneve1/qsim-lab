import json, collections
rows = []
for f in ['res_camps.jsonl', 'res_d70.jsonl', 'res_d70b.jsonl', 'res_mps.jsonl']:
    for l in open(f):
        rows.append(json.loads(l))
T = collections.defaultdict(dict)
meta = {}
for r in rows:
    T[(r['mode'], r['d'])][r['chi']] = r
chis = [16, 32, 64, 128, 256]
print("-lnF (truncation estimate)  [time s / peak RSS MB]")
print("mode         D   T  mpoT | " + " | ".join(f"chi={c:<4d}" for c in chis))
for (mode, d) in sorted(T, key=lambda k: (k[1], k[0])):
    rr = T[(mode, d)]
    any_r = next(iter(rr.values()))
    cells = []
    for c in chis:
        if c in rr:
            r = rr[c]
            cells.append(f"{-r['logfid'] if r['logfid'] is not None else float('inf'):6.2f} [{r['time']:.0f}/{r['rss_mb']:.0f}]")
        else:
            cells.append(" " * 14)
    print(f"{mode:12s} {d:2d} {any_r['T']:3d} {str(any_r.get('mpo','-')):>4s} | " + " | ".join(cells))
