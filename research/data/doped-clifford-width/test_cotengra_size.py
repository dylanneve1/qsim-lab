import sys, time, math, pickle, json
import cotengra as ctg

# Load D=40 and D=70 raw networks
for D in [40, 70]:
    d = pickle.load(open(f'/tmp/doped-zx/net_D{D}.pkl', 'rb'))['raw']
    print(f"\n--- Testing cotengra search for D={D} (tensors={d['ntens']}, inds={d['ninds']}) ---")
    
    # Try different objectives: 'size' (width) and 'flops'
    for obj in ['size', 'flops']:
        opt = ctg.HyperOptimizer(
            methods=['kahypar', 'greedy'],
            max_time=30, # short search
            minimize=obj,
            parallel=False,
            on_trial_error='ignore',
            progbar=False
        )
        t0 = time.time()
        tree = opt.search(d['inputs'], d['output'], d['size_dict'])
        w = tree.contraction_width()
        cost = math.log2(tree.contraction_cost())
        dt = time.time() - t0
        print(f"D={D} obj={obj:5s} trials={len(opt.scores):2d} in {dt:.1f}s -> width={w:.1f}, log2(cost)={cost:.2f}")
