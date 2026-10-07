"""Build raw-TN and ZX-TN hypergraphs (inputs, output, size_dict) for n=70 at depth D; pickle them."""
import sys, time, pickle, numpy as np, quimb.tensor as qtn
from circ import *
D = int(sys.argv[1]); n = 70
gates = load(D, n); x = np.random.default_rng(7).integers(0, 2, n)
nT = sum(g[0] == 't' for g in gates)
t = time.time(); tn = raw_tn(gates, n, x); nt0 = tn.num_tensors
tn.full_simplify_(output_inds=[], atol=1e-12)
raw = dict(inputs=[t_.inds for t_ in tn], output=(), size_dict={i: 2 for i in tn.ind_map}, ntens0=nt0, ntens=tn.num_tensors, ninds=tn.num_indices)
print(f'D={D} T={nT} raw: tensors {nt0} -> {tn.num_tensors} (full_simplify), indices {tn.num_indices}  {time.time()-t:.0f}s', flush=True)
t = time.time(); g, before = zx_graph(gates, n, x, 'full')
inp, arr, sd, sc = zx_to_arrays(g)
zx_counts = dict(before=before, after=(g.num_vertices(), g.num_edges()))
tz = qtn.TensorNetwork([qtn.Tensor(a, inds=i) for i, a in zip(inp, arr)]); nz0 = tz.num_tensors
tz.full_simplify_(output_inds=[], atol=1e-12)
zxd = dict(inputs=[t_.inds for t_ in tz], output=(), size_dict={i: 2 for i in tz.ind_map}, ntens0=nz0, ntens=tz.num_tensors, ninds=tz.num_indices, **zx_counts)
print(f'D={D} zx: spiders/edges {before} -> {zx_counts["after"]} ({time.time()-t:.0f}s); TN tensors {nz0} -> {tz.num_tensors} (full_simplify), indices {tz.num_indices}', flush=True)
# T count remaining in ZX graph (non-Clifford phases)
from fractions import Fraction
nonc = sum(1 for v in g.vertices() if Fraction(g.phase(v)).denominator > 2)
print(f'D={D} zx: non-Clifford spiders {nonc}', flush=True)
pickle.dump(dict(raw=raw, zx=zxd, D=D, T=nT, nonclifford=nonc), open(f'net_D{D}.pkl', 'wb'))
