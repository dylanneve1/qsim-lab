"""Mutual-nearest inverse segment pairs and their consistency with the exact-anchor block involutions."""
import sys, collections, numpy as np, gparse as G, struct_probe as SP, blocks as BL
sys.path.insert(0, '/tmp/peaked-generic'); import solve_peaked as v1
for f in sys.argv[1:]:
    n, units, tail = G.parse(f); L = SP.layers(n, units)
    segs = SP.segments(n, units)
    Q = np.array([SP.quat(s[3]) for s in segs]); Qi = Q.copy(); Qi[:, 1:] *= -1
    D = 1 - np.abs(Q @ Qi.T); np.fill_diagonal(D, 9)
    nn = D.argmin(1)
    A = BL.anchor_list(n, units, L); B = BL.sweep_blocks(n, A); maps = [fm for _, fm in B]
    secs = v1.sections(v1.parse(f)[1])
    def sec_of(k):
        for si, (lo, hi) in enumerate(secs):
            if lo <= k <= hi: return si
    stats = collections.defaultdict(lambda: collections.Counter())
    for i in range(len(segs)):
        j = nn[i]
        if nn[j] != i or i > j: continue
        w, v = segs[i][0], segs[j][0]
        d = D[i, j]
        cons = [bi for bi, fm in enumerate(maps) if fm.get(w) == v]
        tag = 'exact' if d < 1e-10 else ('approx' if d < 1e-3 else 'far')
        key = 'A' if cons == [0] else ('B' if cons == [1] else ('AB' if cons else ('id' if w == v else 'none')))
        stats[(sec_of(segs[i][2]), sec_of(segs[j][1]))][(tag, key)] += 1
    print(f"== {f.split('/')[-1]}  sections {secs}")
    for k in sorted(stats):
        print("  sections", k, dict(stats[k]))
