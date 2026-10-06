"""Layer-synchronous middle-out TNO growth around ASAP-layer centre c."""
import sys, time, resource, numpy as np, gparse as G, struct_probe as SP, tno as TN
def unitG(u):
    a, b, Pa, Pb = u
    return np.diag([1, 1, 1, -1]).astype(complex) @ np.kron(Pa, Pb)
def grow_layers(n, units, L, c, cutoff, max_elems, log=True, max_bond=None):
    D = max(L) + 1
    bylayer = [[] for _ in range(D)]
    for k in range(len(units)): bylayer[L[k]].append(k)
    T = TN.TNO(n, cutoff=cutoff, max_bond=max_bond)
    lo = hi = int(np.ceil(c)); t0 = time.time(); hist = []
    while lo > 0 or hi < D:
        if hi < D:
            for k in bylayer[hi]:
                a, b = units[k][:2]; T.gate(unitG(units[k]), a, b, 'after')
            hi += 1
        if lo > 0:
            for k in reversed(bylayer[lo - 1]):
                a, b = units[k][:2]; T.gate(unitG(units[k]), a, b, 'before')
            lo -= 1
        # clean-up sweep over bonded pairs
        for (x, y) in list(T.bond_graph()):
            T.compress_pair(x, y)
        T.drop_trivial_bonds(); T.normalise()
        E = T.bond_graph()
        rec = dict(lo=lo, hi=hi, elems=T.size(), maxbond=max(E.values()) if E else 1, nbonds=len(E),
                   moved=sum(1 for o, i in T.perm().items() if o != i), err=T.trunc_err, t=round(time.time() - t0, 1))
        hist.append(rec)
        if log: print("  ", rec, flush=True)
        if T.size() > max_elems: break
    return T, hist
if __name__ == "__main__":
    f = sys.argv[1]; c = float(sys.argv[2]); cutoff = float(sys.argv[3]); maxe = int(float(sys.argv[4]))
    n, units, tail = G.parse(f); L = SP.layers(n, units)
    T, hist = grow_layers(n, units, L, c, cutoff, maxe)
    print("peak RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
