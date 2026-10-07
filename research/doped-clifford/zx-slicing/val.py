import sys, numpy as np, cotengra as ctg, opt_einsum as oe, quimb.tensor as qtn
from circ import *
n, D = int(sys.argv[1]), int(sys.argv[2])
gates = load(D, n); rng = np.random.default_rng(1)
psi = statevector(gates, n)
for trial in range(3):
    x = rng.integers(0, 2, n); ref = psi[tuple(x)]
    tn = raw_tn(gates, n, x).full_simplify(output_inds=[]); trr = tn.contraction_tree(optimize=ctg.HyperOptimizer(max_repeats=32, progbar=False), output_inds=())
    a_raw = tn.contract(all, output_inds=(), optimize=trr) if trr.contraction_width() <= 21 else float('nan')
    for red in ('full',):
        g, before = zx_graph(gates, n, x, red)
        inp, arr, sd, sc = zx_to_arrays(g)
        tnz = qtn.TensorNetwork([qtn.Tensor(a, inds=i) for i, a in zip(inp, arr)])
        if tnz.num_tensors:
            tr = tnz.contraction_tree(optimize=ctg.HyperOptimizer(max_repeats=16, progbar=False), output_inds=())
            wz = tr.contraction_width(); print(f'  zx network: {tnz.num_tensors} tensors, {tnz.num_indices} spider indices, width {wz:.0f}', flush=True)
            if wz > 21: print('  zx contraction skipped (too wide)'); continue
            val = tnz.contract(all, output_inds=(), optimize=tr)
        else: val = 1.0
        a = complex(val)*sc
        print(f'n={n} D={D} x#{trial} red={red}: ref={ref:.3e} raw={complex(a_raw):.3e} zx={a:.3e} |zx|/|ref|={abs(a)/abs(ref):.6f} '
              f'phase-diff={np.angle(a/ref):.4f}  spiders {before[0]}->{g.num_vertices()} edges {before[1]}->{g.num_edges()}', flush=True)
