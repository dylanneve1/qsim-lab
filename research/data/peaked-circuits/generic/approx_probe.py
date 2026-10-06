"""Nearest-inverse distance of each inter-CZ segment (SU(2) quaternion, phase-free); also same-pair CZ runs."""
import sys, collections, numpy as np, gparse as G
from struct_probe import segments, layers, quat
for f in sys.argv[1:]:
    n, units, tail = G.parse(f)
    segs = segments(n, units)
    Q = np.array([quat(s[3]) for s in segs])            # unit quaternions (sign-fixed)
    Qi = Q.copy(); Qi[:, 1:] *= -1                       # inverse
    D = np.abs(Q @ Qi.T)                                 # |<q_i, q_j^-1>| = 1 for exact inverse
    np.fill_diagonal(D, 0)
    best = 1 - D.max(1)
    bins = [0, 1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 1]
    print(f"== {f.split('/')[-1]}: segments {len(segs)}; 1-max|<q,q'^-1>| histogram:")
    print("   ", list(zip(bins[1:], np.histogram(best, bins=bins)[0].tolist())))
    # identity-ish segments
    idn = np.abs(Q[:, 0])
    print("   segments with |q0|>1-1e-9 (identity):", int((idn > 1 - 1e-9).sum()), " diag (Z-rot):", int((np.abs(Q[:,2])+np.abs(Q[:,3]) < 1e-9).sum()))
    # consecutive CZs on the same pair
    lastpair = {}; runs = 0
    nxt = {}
    for k, (a, b, *_ ) in enumerate(units):
        if lastpair.get(a) == (min(a,b),max(a,b)) and lastpair.get(b) == (min(a,b),max(a,b)): runs += 1
        lastpair[a] = lastpair[b] = (min(a,b), max(a,b))
    pairs = collections.Counter((min(u[0],u[1]), max(u[0],u[1])) for u in units)
    print(f"   back-to-back same-pair CZs: {runs}; distinct pairs {len(pairs)}; CZs/pair max {max(pairs.values())}")
