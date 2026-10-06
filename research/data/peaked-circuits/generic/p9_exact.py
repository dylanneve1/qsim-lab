import numpy as np, collections, gparse as G, struct_probe as SP
f='/tmp/peaked-gen/peaked_circuit_P9_Hqap_56x1917.qasm'
n, units, tail = G.parse(f); L = SP.layers(n, units)
segs = SP.segments(n, units)
Q = np.array([SP.quat(s[3]) for s in segs]); Qi = Q.copy(); Qi[:,1:] *= -1
D = 1 - np.abs(Q @ Qi.T); np.fill_diagonal(D, 9)
ex = np.argwhere(D < 1e-10)
ex = [(i,j) for i,j in ex if i<j]
triv = lambda q: abs(abs(q[0])-1)<1e-9 or (abs(q[2])+abs(q[3])<1e-9)
nontriv = [(i,j) for i,j in ex if not triv(Q[i])]
print("exact inverse pairs:", len(ex), " non-trivial (not identity/diagonal):", len(nontriv))
for i,j in nontriv[:20]:
    (w,kp,k,_),(v,kp2,k2,_) = segs[i], segs[j]
    print(f"  wire {w} layer {L[k]} <-> wire {v} layer {L[kp2]}  quat {np.round(Q[i],4)}")
cnt = collections.Counter(tuple(np.round(q,6)) for q in Q)
print("most common segment fingerprints:", [(k,v) for k,v in cnt.most_common(6)])
