"""Lean slice+reconfigure: start from a tree (chain-sweep seed or pickled kahypar tree), then repeatedly slice one more
bit of width (cotengra SliceFinder, max_repeats=2 to bound memory) followed by in-place subtree reconfiguration.
Logs (width, log2 total cost, log2 slices) at every width. Usage: lean.py D wmin [seed|tree_pickle] [subtree]"""
import sys, re, time, math, json, pickle, numpy as np, cotengra as ctg
from circ import *
D, wmin = int(sys.argv[1]), int(sys.argv[2]); src = sys.argv[3] if len(sys.argv) > 3 else 'seed'
sub = int(sys.argv[4]) if len(sys.argv) > 4 else 6
if src == 'seed':
    n = 70; gates = load(D, n); x = np.random.default_rng(7).integers(0, 2, n)
    tn = raw_tn(gates, n, x); tn.full_simplify_(output_inds=[], atol=1e-12)
    def key(t):
        qs = [int(g[1:]) for g in t.tags if re.fullmatch(r'I\d+', g)]; gs = [int(g[5:]) for g in t.tags if re.fullmatch(r'GATE_\d+', g)]
        return (min(qs), min(gs))
    ts = sorted(tn, key=key); inputs = [t.inds for t in ts]; sd = {i: 2 for i in tn.ind_map}; N = len(inputs)
    tree = ctg.ContractionTree.from_path(inputs, (), sd, ssa_path=[(0, 1)] + [(N + k, k + 2) for k in range(N - 2)])
    del tn, ts
else:
    tree = pickle.load(open(src, 'rb'))
tree.subtree_reconfigure_(subtree_size=8)
def rep(r): print(json.dumps(r), flush=True); open('/tmp/doped-zx/lean.jsonl', 'a').write(json.dumps(r) + '\n')
t0 = time.time(); w = round(tree.contraction_width())
rep(dict(D=D, src=src, width=w, log2C=math.log2(tree.contraction_cost()), log2slices=0, secs=0))
while w > wmin:
    tree.slice_(target_size=2**(w - 1), max_repeats=2)
    tree.subtree_reconfigure_(subtree_size=sub)
    w = round(tree.contraction_width())
    rep(dict(D=D, src=src, width=w, log2C=math.log2(tree.contraction_cost()), log2slices=math.log2(tree.multiplicity),
             nsliced=len(tree.sliced_inds), secs=round(time.time() - t0)))
