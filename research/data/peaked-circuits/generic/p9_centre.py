import numpy as np, collections, gparse as G, struct_probe as SP
f='/tmp/peaked-gen/peaked_circuit_P9_Hqap_56x1917.qasm'
n, units, tail = G.parse(f); L = SP.layers(n, units)
segs = SP.segments(n, units)
key = lambda q: tuple(np.round(q, 4))
cnt = collections.Counter(key(SP.quat(s[3])) for s in segs)
special = {k for k, v in cnt.items() if v >= 5}
# per unit: is either of its incoming segments special?
inseg = collections.defaultdict(list)
for (w, kp, k, V) in segs: inseg[k].append((w, key(SP.quat(V)) in special))
byL = collections.defaultdict(list)
for k, u in enumerate(units):
    byL[L[k]].append((u[0], u[1], sum(s for _, s in inseg[k])))
for l in range(40, 62):
    row = byL[l]
    print(l, len(row), "special-seg units:", sum(1 for r in row if r[2] > 0), " pairs:", sorted((a, b) for a, b, s in row if s > 0)[:14])
# same-pair recurrence near centre
pairs = collections.Counter((min(u[0],u[1]), max(u[0],u[1])) for k,u in enumerate(units) if 42 <= L[k] <= 58)
print("pairs used >=2 times in layers 42..58:", sorted([(p,c) for p,c in pairs.items() if c>=2], key=lambda x:-x[1])[:20])
