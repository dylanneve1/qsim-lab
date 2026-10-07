#!/usr/bin/env python3
"""Cost of one exact amplitude <x|U|0>, via cotengra path search (no contraction).
Reports width (log2 of the biggest tensor), log2 FLOPs, and the slicing needed to fit a target width."""
import sys, time
import quimb.tensor as qtn, cotengra as ctg
f, depth_cut = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 999
lines = [l for l in open(f).read().split('\n')]
# truncate to the first `depth_cut` CZ layers (for scaling studies)
out, czl = [], [0]*70
import re
for l in lines:
    m = re.match(r'cz q\[(\d+)\],q\[(\d+)\];', l.strip())
    if m:
        a, b = int(m.group(1)), int(m.group(2)); d = max(czl[a], czl[b]) + 1
        if d > depth_cut: continue
        czl[a] = czl[b] = d
    out.append(l)
circ = qtn.Circuit.from_openqasm2_str('\n'.join(out))
tn = circ.amplitude_tn('0' * 70)
tn.full_simplify_(output_inds=[], atol=1e-12)
print(f"depth<= {depth_cut}: tensors {tn.num_tensors}, indices {tn.num_indices}", flush=True)
for target in (None, 30, 28):
    opt = ctg.HyperOptimizer(methods=['kahypar', 'greedy'] if False else ['greedy', 'labels'], max_repeats=64,
                             max_time=60, minimize='flops', slicing_reconf_opts={'target_size': 2**target} if target else None,
                             parallel=False, progbar=False)
    t = time.time(); tree = tn.contraction_tree(optimize=opt)
    print(f"  target width {target}: width {tree.contraction_width():.1f}, log2 flops {__import__("math").log2(tree.total_flops()):.1f}, "
          f"slices {tree.multiplicity}, search {time.time()-t:.0f}s", flush=True)
