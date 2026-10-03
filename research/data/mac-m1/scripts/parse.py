import sys, collections
# usage: parse.py file.raw [file.raw...]  -> markdown tables of min per config + ratios
D = collections.defaultdict(lambda: collections.defaultdict(list))
order = []
for fn in sys.argv[1:]:
    for line in open(fn):
        head, rest = line.split('|', 1)
        t, w = head.split()
        f = [x.strip() for x in rest.split('|')]
        wl, n, prec, cfg, sec = f[0], int(f[1]), f[2], f[3], float(f[4])
        key = (wl, prec, n, int(t))
        if key not in D: order.append(key)
        D[key][cfg].append(sec)
cfgs = []
for k in order:
    for c in D[k]:
        if c not in cfgs: cfgs.append(c)
print('| workload | prec | n | thr | ' + ' | '.join(f'{c} s' for c in cfgs) + ' | reps | ' + ' | '.join(f'main/{c}' for c in cfgs if c != 'main') + (' | fma/best-tile |' if 'fma' in cfgs else ' |'))
print('|' + '---|' * (5 + 2 * len(cfgs) - (1 if 'main' in cfgs else 0) + (1 if 'fma' in cfgs else 0)))
for k in sorted(order):
    m = {c: min(v) for c, v in D[k].items()}
    reps = min(len(v) for v in D[k].values())
    row = [k[0], k[1], str(k[2]), str(k[3])] + [f'{m[c]:.4f}' if c in m else '' for c in cfgs] + [str(reps)]
    if 'main' in m:
        row += [f'{m["main"]/m[c]:.2f}x' if c in m else '' for c in cfgs if c != 'main']
    tiles = [m[c] for c in m if c.startswith('t') or '_t' in c]
    if 'fma' in m and tiles:
        row.append(f'{m["fma"]/min(tiles):.3f}x')
    print('| ' + ' | '.join(row) + ' |')
