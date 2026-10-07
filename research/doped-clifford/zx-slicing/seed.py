"""Chain-sweep seed tree for the raw TN (sort tensors by (min qubit, min gate id), contract linearly), then
cotengra general slicing + subtree reconfiguration to target widths. Usage: seed.py D targets [reconf_rounds]"""
import sys, re, time, math, json, pickle, numpy as np, cotengra as ctg
from circ import *
D = int(sys.argv[1]); targets = [int(t) for t in sys.argv[2].split(',') if t] if len(sys.argv) > 2 else []
n = 70; gates = load(D, n); x = np.random.default_rng(7).integers(0, 2, n)
split = len(sys.argv) > 3 and sys.argv[3] == 'split'
tn = raw_tn(gates, n, x, split=split); tn.full_simplify_(output_inds=[], atol=1e-12)
def key(t):
    qs = [int(g[1:]) for g in t.tags if re.fullmatch(r'I\d+', g)]
    gs = [int(g[5:]) for g in t.tags if re.fullmatch(r'GATE_\d+', g)]
    return (min(qs) if qs else 0, min(gs) if gs else 0)
ts = list(tn); order = sorted(range(len(ts)), key=lambda i: key(ts[i]))
inputs = [ts[i].inds for i in order]; size_dict = {i: 2 for i in tn.ind_map}
path = [(0, 1)] + [(0, 1)] * (len(inputs) - 2)   # linear: (acc, next)
N = len(inputs); ssa = [(0, 1)] + [(N + k, k + 2) for k in range(N - 2)]   # accumulate in sweep order
tree = ctg.ContractionTree.from_path(inputs, (), size_dict, ssa_path=ssa)
def rep(tag, t, extra={}):
    r = dict(D=D, tag=tag + (' [split]' if split else ''), ntens=tn.num_tensors, width=t.contraction_width(), log2C=math.log2(t.contraction_cost()), slices=t.multiplicity,
             log2C_per_slice=math.log2(t.contraction_cost()/t.multiplicity), **extra)
    print(json.dumps(r), flush=True); open('/tmp/doped-zx/results_seed.jsonl', 'a').write(json.dumps(r) + '\n')
rep('chain-sweep linear', tree)
t0 = time.time(); tree2 = tree.subtree_reconfigure(subtree_size=8, progbar=False); rep('chain-sweep + subtree_reconf', tree2, dict(secs=time.time()-t0))
pickle.dump(dict(inputs=inputs, size_dict=size_dict, tree=tree2), open(f'/tmp/doped-zx/seed_D{D}.pkl', 'wb'))
for T in targets:
    t0 = time.time()
    ts_ = tree2.slice_and_reconfigure(target_size=2**T, step_size=2, reconf_opts=dict(subtree_size=6))
    rep(f'seed slice_and_reconfigure -> {T}', ts_, dict(secs=time.time()-t0))
    t0 = time.time(); tp = tree2.slice(target_size=2**T)
    rep(f'seed plain slice -> {T}', tp, dict(secs=time.time()-t0))
