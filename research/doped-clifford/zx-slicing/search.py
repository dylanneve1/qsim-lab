"""Hyper-optimised path search (cotengra + kahypar) on a pickled network. Usage: search.py D net target_width|none seconds [methods]
Reports width, log2 C (C = cotengra cost = sum over pairwise contractions of prod(index sizes) x #slices ~ complex MACs;
real FLOPs ~ 8 C), slice count."""
import sys, time, math, pickle, json, cotengra as ctg
D, net, tgt, secs = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
methods = sys.argv[5].split(',') if len(sys.argv) > 5 else ['kahypar', 'greedy']
d = pickle.load(open(f'/tmp/doped-zx/net_D{D}.pkl', 'rb'))[net]
kw = {}
mode = sys.argv[6] if len(sys.argv) > 6 else 'reconf'
if tgt != 'none':
    if mode == 'reconf': kw['slicing_reconf_opts'] = dict(target_size=2**int(tgt))
    elif mode == 'lean': kw['slicing_reconf_opts'] = dict(target_size=2**int(tgt), step_size=1, max_repeats=2, reconf_opts=dict(subtree_size=6))
    else: kw['slicing_opts'] = dict(target_size=2**int(tgt))
opt = ctg.HyperOptimizer(methods=methods, max_time=secs, max_repeats=10**6, minimize='flops', parallel=False, on_trial_error='ignore',
                         progbar=False, **kw)
t = time.time()
tree = opt.search(d['inputs'], d['output'], d['size_dict'])
res = dict(D=D, net=net, target=tgt, mode=mode if tgt != 'none' else '-', secs=secs, methods=methods, trials=len(opt.scores), ntens=d['ntens'], ninds=d['ninds'],
           width=tree.contraction_width(), log2C=math.log2(tree.contraction_cost()), slices=tree.multiplicity,
           log2C_per_slice=math.log2(tree.contraction_cost()/tree.multiplicity), wall=time.time()-t,
           sliced_inds=list(tree.sliced_inds)[:64])
print(json.dumps(res), flush=True)
open('/tmp/doped-zx/results.jsonl', 'a').write(json.dumps(res) + '\n')
pickle.dump(tree, open(f'/tmp/doped-zx/tree_D{D}_{net}_{tgt}.pkl', 'wb'))
