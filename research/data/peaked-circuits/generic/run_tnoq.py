import sys, time, resource, numpy as np, gparse as G, struct_probe as SP, tnoq
f = sys.argv[1]; c = float(sys.argv[2]); cutoff = float(sys.argv[3]); mb = int(sys.argv[4]) or None
n, units, tail = G.parse(f); L = SP.layers(n, units)
W, lo, hi, hist = tnoq.grow(n, units, L, c, cutoff=cutoff, max_bond=mb, max_elems=3e6)
print("final window layers", lo, hi, "depth", max(L)+1, "peak RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
