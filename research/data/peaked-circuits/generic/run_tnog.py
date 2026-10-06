import sys, resource, gparse as G, struct_probe as SP, tnos
f = sys.argv[1]; c = float(sys.argv[2]); cutoff = float(sys.argv[3]); maxe = float(sys.argv[4])
n, units, tail = G.parse(f); L = SP.layers(n, units)
W, inside, nxt, prv, hist = tnos.grow_greedy(n, units, L, c, cutoff=cutoff, max_elems=maxe, log=True)
print("final absorbed", len(inside), "of", len(units), "RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
