"""Slicing overhead trend: chain-sweep seed tree (+subtree reconf), then cotengra slicing to width w0-k, k=1..7
(plain greedy slice; plus slice_and_reconfigure for small k). Usage: trend.py D [reconf_kmax]"""
import sys, re, time, math, json, numpy as np, cotengra as ctg
from circ import *
D = int(sys.argv[1]); kre = int(sys.argv[2]) if len(sys.argv) > 2 else 0
n = 70; gates = load(D, n); x = np.random.default_rng(7).integers(0, 2, n)
tn = raw_tn(gates, n, x); tn.full_simplify_(output_inds=[], atol=1e-12)
def key(t):
    qs = [int(g[1:]) for g in t.tags if re.fullmatch(r'I\d+', g)]; gs = [int(g[5:]) for g in t.tags if re.fullmatch(r'GATE_\d+', g)]
    return (min(qs), min(gs))
ts = sorted(tn, key=key); inputs = [t.inds for t in ts]; sd = {i: 2 for i in tn.ind_map}; N = len(inputs)
tree = ctg.ContractionTree.from_path(inputs, (), sd, ssa_path=[(0, 1)] + [(N + k, k + 2) for k in range(N - 2)])
tree = tree.subtree_reconfigure(subtree_size=8, progbar=False)
w0 = int(round(tree.contraction_width())); C0 = math.log2(tree.contraction_cost())
def rep(r): print(json.dumps(r), flush=True); open('/tmp/doped-zx/trend.jsonl', 'a').write(json.dumps(r) + '\n')
rep(dict(D=D, method='chain-seed', k=0, width=w0, log2C=C0, log2slices=0))
for k in range(1, 8):
    t0 = time.time(); tp = tree.slice(target_size=2**(w0 - k))
    rep(dict(D=D, method='chain-seed+slice', k=k, width=tp.contraction_width(), log2C=math.log2(tp.contraction_cost()),
             log2slices=math.log2(tp.multiplicity), secs=round(time.time()-t0)))
for k in range(1, kre + 1):
    t0 = time.time(); tp = tree.slice_and_reconfigure(target_size=2**(w0 - k), step_size=1, reconf_opts=dict(subtree_size=6))
    rep(dict(D=D, method='chain-seed+slice_and_reconfigure', k=k, width=tp.contraction_width(), log2C=math.log2(tp.contraction_cost()),
             log2slices=math.log2(tp.multiplicity), secs=round(time.time()-t0)))
