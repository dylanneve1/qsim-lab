"""P9 attempt: swap-aware TNO from the centre with unswapping; fine cutoff inside the central swap band,
coarse cutoff + bond cap outside it (fidelity traded for reach); then bond-capped candidates and exact
amplitude checks on the least-compressed state that fits. Usage: run_p9.py C CUT_IN CUT_OUT CHI_OUT BAND_LO BAND_HI"""
import sys, time, copy, json, resource, numpy as np
import gparse as G, struct_probe as SP, tno as TN, tnos
import quimb.tensor as qtn, cotengra as ctg
import solve_generic as SGn
f = '/tmp/peaked-gen/portal/P9_hqap_1917.qasm'
c, cin, cout, chi, blo, bhi = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
tmax = float(sys.argv[7]) if len(sys.argv) > 7 else 3000
n, units, tail = G.parse(f); L = SP.layers(n, units); D = max(L) + 1
bylayer = [[] for _ in range(D)]
for k in range(len(units)): bylayer[L[k]].append(k)
W = TN.TNO(n, cutoff=1e-8); lo = hi = int(np.ceil(c)); t0 = time.time()
def absorb(Wx, side, layer, cut, mb):
    ks = bylayer[layer] if side == 'after' else list(reversed(bylayer[layer]))
    for k in ks: Wx.gate(tnos.unitG(units[k]), units[k][0], units[k][1], side)
    TN.canonical_compress(Wx, cutoff=cut, max_bond=mb); Wx.drop_trivial_bonds()
    if TN.unswap_pass(Wx, min_bond=4):
        TN.canonical_compress(Wx, cutoff=cut, max_bond=mb); Wx.drop_trivial_bonds()
    return Wx
while lo > 0 or hi < D:
    inside = lo > blo or hi < bhi
    cut, mb = (cin, None) if inside else (cout, chi)
    opts = []
    if hi < D: Wa = absorb(copy.deepcopy(W), 'after', hi, cut, mb); opts.append((Wa.size(), 0, 'after', Wa))
    if lo > 0: Wb = absorb(copy.deepcopy(W), 'before', lo - 1, cut, mb); opts.append((Wb.size(), 1, 'before', Wb))
    sz, _, side, W = min(opts, key=lambda x: (x[0], x[1]))
    if side == 'after': hi += 1
    else: lo -= 1
    E = W.bond_graph()
    print(dict(lo=lo, hi=hi, side=side, cut=cut, elems=W.size(), max_bond=max(E.values()) if E else 1, nbonds=len(E),
               moved=sum(1 for o, i in W.perm().items() if o != i), t=round(time.time() - t0, 1),
               rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024), flush=True)
    if time.time() - t0 > tmax or W.size() > 3e6:
        print("STOP: time/size limit", flush=True); break
if lo == 0 and hi == D:
    class _W: pass
    Wq = _W(); Wq.tn = TN.to_quimb(W)
    psi0 = SGn.compress_state(n, SGn.state_network(n, units, L, Wq, 0, D, tail), cout)
    print("state W|0>: max bond", psi0.max_bond(), flush=True)
    res = []
    for cap in (8, 4, 2, 1):
        try:
            psi = SGn.compress_state(n, psi0, cout, max_bond=cap)
            b, p, zs, nrm, w, cst = SGn.contract_peak(n, psi, log=lambda s: None)
            res.append(dict(cap=cap, peak=b, p=p, norm=nrm, min_abs_z=float(np.min(np.abs(zs))), width=w))
            print(f"cap {cap}: width {w:.1f} p {p:.4f} norm {nrm:.4f} min|Z| {np.min(np.abs(zs)):.3f} cand#{1 + [r['peak'] for r in res].index(b)}", flush=True)
        except (RuntimeError, MemoryError) as e:
            print(f"cap {cap}: rejected {str(e)[:70]}", flush=True)
    opt1 = ctg.ReusableHyperOptimizer(methods=['greedy', 'kahypar'], max_repeats=32, minimize='flops', parallel=False, progbar=False)
    def amp2(bits):
        t = psi0.copy()
        for q in range(n): t |= qtn.Tensor(np.array([1, 0] if bits[q] == '0' else [0, 1], dtype=complex), inds=(f"s{q}",))
        tr = t.contraction_tree(optimize=opt1, output_inds=())
        if tr.contraction_width() > 26: raise RuntimeError(f"amplitude width {tr.contraction_width():.1f}")
        return float(abs(t.contract(all, optimize=opt1)) ** 2)
    for r in res:
        try:
            r['p_full'] = amp2(r['peak']); print(f"cand from cap {r['cap']}: |<s|W0>|^2 (uncapped) {r['p_full']:.4f}", flush=True)
        except (RuntimeError, MemoryError) as e:
            print(f"amplitude check rejected: {e}", flush=True)
    json.dump(res, open('/tmp/peaked-generic/private/p9_run.json', 'w'), indent=1)
