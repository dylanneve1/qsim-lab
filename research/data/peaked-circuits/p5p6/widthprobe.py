"""Contraction width / flops of the single-amplitude network <x|C|0> (full simplify) for a QASM file."""
import sys, time, numpy as np, quimb.tensor as qtn, cotengra as ctg
fn = sys.argv[1]; secs = float(sys.argv[2]) if len(sys.argv) > 2 else 60
circ = qtn.Circuit.from_openqasm2_file(fn, gate_opts=dict(contract='split-gate'))
n = circ.N
tn = circ.amplitude_tn('0' * n)
tn.full_simplify_(output_inds=(), seq='ADCR')
print('tensors after simplify', tn.num_tensors, flush=True)
opt = ctg.HyperOptimizer(methods=['kahypar', 'greedy'], max_time=secs, max_repeats=10**6, parallel=False, progbar=False, minimize='flops')
tree = tn.contraction_tree(optimize=opt, output_inds=())
print(fn.split('/')[-1], 'width', round(tree.contraction_width(), 1), 'log2 flops', round(tree.contraction_cost(log=2), 1), flush=True)
