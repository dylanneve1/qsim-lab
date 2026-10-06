import sys
sys.path.insert(0,'/tmp/peaked-generic')
import gparse as G
for f in sys.argv[1:]:
    n, units, tail = G.parse(f)
    layers, cur, used = [], [], set()
    for k, x in enumerate(units):
        if x[0] in used or x[1] in used:
            layers.append(cur); cur, used = [], set()
        cur.append(k); used |= {x[0], x[1]}
    layers.append(cur)
    print(f.split('/')[-1], len(layers), 'serial layers; sizes (start idx:size):')
    print(' '.join(f"{l[0]}:{len(l)}" for l in layers))
