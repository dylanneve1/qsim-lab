import numpy as np, time, json, sys
import lad, ent_free
circ = 'SCV'
for lam in (0.0, 1.0):
    init, steps = lad.gate_steps(circ, lam)
    print('lam', lam, 'gates per step', [len(s) for s in steps[:3]], 'two-site', sum(g[0] == '2' for g in steps[0]))
    psi = lad.MPS(init, 256)
    t0 = time.time()
    for k in range(6):
        for g in steps[k]: psi.apply(g)
    n = psi.occupations()
    print(' fwd step6 maxchi', psi.maxchi, 'maxdw', psi.maxdw, 't', time.time() - t0)
    if lam == 0:
        rho = None; recs = {}
        def rec(step, r): recs[step] = [np.real(np.diag(x)).copy() for x in r]
        ent_free.evolve_free(circ, record=rec)
        ex = np.array(recs[6]).T
        print(' free check max|n_mps - n_gauss| step 6:', abs(n - ex).max())
    else:
        nfwd = n
        for r in (29, 30):
            t0 = time.time()
            O = lad.heisenberg(steps, 6, {r: np.diag(lad.Zi).astype(complex)}, 256)
            z = O.expect(init).real
            print(f' heis site {r} chain i: <Z>={z:.10f}  fwd={1-2*nfwd[r,0]:.10f} diff={z-(1-2*nfwd[r,0]):.1e} maxchi {O.maxchi} t {time.time()-t0:.1f}s')
