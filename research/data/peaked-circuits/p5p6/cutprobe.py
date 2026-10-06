"""Balanced kahypar bipartitions of the amplitude TN (after full_simplify) for P6: report cut sizes (# bonds,
log2 of cut dimension) at several imbalance settings, plus a greedy/kahypar contraction width."""
import sys, time, numpy as np, quimb.tensor as qtn, cotengra as ctg
from cotengra.pathfinders.path_kahypar import kahypar_subgraph_find_membership
fn = sys.argv[1]
circ = qtn.Circuit.from_openqasm2_file(fn, gate_opts=dict(contract='split-gate'))
n = circ.N
tn = circ.amplitude_tn('0' * n); tn.full_simplify_(output_inds=(), seq='ADCR')
print('tensors', tn.num_tensors, flush=True)
inputs = [t.inds for t in tn]; size = {ix: tn.ind_size(ix) for ix in tn.ind_map}
for imb in [0.01, 0.1, 0.3, 0.6, 0.9]:
    best = None
    for seed in range(4):
        memb = kahypar_subgraph_find_membership(inputs, set(), size, parts=2, imbalance=imb, seed=seed, mode='recursive', objective='cut')
        A = {i for i, m in enumerate(memb) if m == 0}
        cut = [ix for ix, ts in tn.ind_map.items() if len(ts) == 2 and (len(set(tn._get_tids_from_inds([ix])) & {tn.tensor_map and 0}) or True)]
        # compute cut bonds
        tids = list(tn.tensor_map.keys())
        side = {tids[i]: memb[i] for i in range(len(tids))}
        cb = [ix for ix, ts in tn.ind_map.items() if len({side[t] for t in ts}) > 1]
        lc = sum(np.log2(size[ix]) for ix in cb)
        if best is None or lc < best[0]: best = (lc, len(cb), sum(1 for m in memb if m == 0), len(memb))
    print('imbalance', imb, 'cut log2 dim %.1f bonds %d sizes %d/%d' % best, flush=True)
opt = ctg.HyperOptimizer(methods=['greedy'], max_repeats=32, parallel=False, progbar=False, minimize='flops')
tree = tn.contraction_tree(optimize=opt, output_inds=())
print('greedy width', round(tree.contraction_width(), 1), 'log2 flops', round(tree.contraction_cost(log=2), 1), flush=True)
