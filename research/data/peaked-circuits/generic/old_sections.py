import sys, collections, numpy as np
sys.path.insert(0, '/tmp/peaked-generic')
import solve_peaked as v1, gparse as G, struct_probe as SP
for f in sys.argv[1:]:
    n, units = v1.parse(f)
    secs = v1.sections(units)
    anc = v1.anchors(n, units)
    by = collections.Counter()
    for a, b in anc:
        c = (a[2] + b[1]) / 2
        for si, (lo, hi) in enumerate(secs):
            if lo <= c <= hi: by[si] += 1
    # ASAP layer range per section
    nn, un2, tl = G.parse(f); L = SP.layers(nn, un2)
    print(f.split('/')[-1], 'sections', [(lo, hi, hi-lo+1, (min(L[lo:hi+1]), max(L[lo:hi+1]))) for lo, hi in secs], 'anchors by centre section', dict(by))
