import sys, resource, gparse as G, struct_probe as SP, tnos
f = sys.argv[1]; c = float(sys.argv[2]); cutoff = float(sys.argv[3]); maxe = float(sys.argv[4])
n, units, tail = G.parse(f); L = SP.layers(n, units)
W, lo, hi, hist = (tnos.grow_adaptive if "--adaptive" in sys.argv else tnos.grow)(n, units, L, c, cutoff=cutoff, max_elems=maxe, log=True, **({"unswap": True} if "--unswap" in sys.argv else {}))
print("final", lo, hi, "RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
