#!/usr/bin/env python3
"""Generic peaked-circuit solver (operator-cancellation route).

  1. gparse: any QASM 2/3 -> CZ units (exact).
  2. Mirror centre: for every candidate ASAP-layer cut c (coarse grid, then refined), grow the window
     operator W layer-synchronously from c with canonised compression and record how many gates are
     absorbed before W exceeds a memory budget.  Centre = argmax (ties: smaller final W).  No structural
     thresholds; the budget is a resource limit only.
  3. Grow W from the centre over the whole circuit (or until the budget stops it); the part that is not
     absorbed is the reduced core R |> W |> P, contracted exactly with cotengra.
  4. Peak = sign of exact single-qubit marginals; core peak probability = exact |amplitude|^2.
The solver never reads a target string.
Usage: solve_generic.py FILE.qasm [--centre C] [--cutoff 1e-3] [--budget 2e5] [--json OUT]
"""
import argparse, hashlib, json, resource, sys, time
import numpy as np
import quimb.tensor as qtn
import cotengra as ctg
import gparse as G
import struct_probe as SP
import tnob as tnoq
import backend as BK


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def absorbed_gates(L, lo, hi):
    return sum(1 for l in L if lo <= l < hi)


def scan_centres(n, units, L, cutoff, budget, grid):
    out = []
    for c in grid:
        t = time.time()
        W, lo, hi, hist = tnoq.grow(n, units, L, c, cutoff=cutoff, max_elems=budget, log=False)
        st = hist[-1] if hist else dict(elems=0)
        g = absorbed_gates(L, lo, hi)
        out.append(dict(c=c, gates=g, lo=lo, hi=hi, elems=max(h['elems'] for h in hist) if hist else 0,
                        s=round(time.time() - t, 1)))
    return out


def find_centre(n, units, L, cutoff, budget, log=print):
    D = max(L) + 1
    step = max(1, D // 24)
    coarse = scan_centres(n, units, L, cutoff, budget, list(range(step, D - 1, step)))
    best = max(coarse, key=lambda r: (r['gates'], -r['elems']))
    log(f"  coarse centre scan (step {step}): " + ", ".join(f"{r['c']}:{r['gates']}" for r in coarse))
    fine = scan_centres(n, units, L, cutoff, budget,
                        [c for c in range(max(1, best['c'] - step // 2), min(D - 1, best['c'] + step // 2 + 1)) if c != best['c']])
    best = max(fine + [best], key=lambda r: (r['gates'], -r['elems'], -abs(r['c'] - best['c'])))
    log(f"  fine scan: " + ", ".join(f"{r['c']}:{r['gates']}({r['s']}s)" for r in fine))
    return best, coarse, fine


def state_network(n, units, L, W, lo, hi, tail):
    """psi = P |> W |> R |0>, with R = layers < lo, P = layers >= hi (exact gates), tails on outputs."""
    tn = W.tn.copy()
    # R gates: before W on the input side, in reverse time order composing towards |0>
    # Build as a circuit-style TN: inputs b{q} of W are fed by R's outputs.
    cur_in = {q: f"b{q}" for q in range(n)}     # index currently waiting to be fed (input side)
    R = [k for k in range(len(units)) if L[k] < lo]
    P = [k for k in range(len(units)) if L[k] >= hi]
    ts = []
    # feed backwards through R (last R gate is adjacent to W)
    for k in reversed(R):
        a, b = units[k][:2]
        Gt = tnoq.unitG(units[k]).reshape(2, 2, 2, 2)
        na, nb = tnoq._nid("x"), tnoq._nid("x")
        ts.append(qtn.Tensor(Gt, inds=(cur_in[a], cur_in[b], na, nb), tags={"R"}))
        cur_in[a], cur_in[b] = na, nb
    for q in range(n):
        ts.append(qtn.Tensor(np.array([1, 0], dtype=complex), inds=(cur_in[q],), tags={"ZERO"}))
    cur_out = {q: f"k{q}" for q in range(n)}
    for k in P:
        a, b = units[k][:2]
        Gt = tnoq.unitG(units[k]).reshape(2, 2, 2, 2)
        na, nb = tnoq._nid("y"), tnoq._nid("y")
        ts.append(qtn.Tensor(Gt, inds=(na, nb, cur_out[a], cur_out[b]), tags={"P"}))
        cur_out[a], cur_out[b] = na, nb
    for q in range(n):
        ts.append(qtn.Tensor(tail[q], inds=(f"s{q}", cur_out[q]), tags={"TAIL"}))
    psi = tn | qtn.TensorNetwork(ts)
    return psi


def compress_state(n, psi, cutoff, max_bond=None):
    """fuse every tensor into its nearest site (wire output index s{q}) and compress all bonds (canonised)."""
    from quimb.tensor.tensor_arbgeom_compress import tensor_network_ag_compress
    psi = psi.copy()
    # tag each tensor with the site whose output index it carries, else with a neighbour's site (BFS)
    site_of = {}
    for t in psi:
        for ix in t.inds:
            if ix.startswith('s') and ix[1:].isdigit():
                site_of[id(t)] = int(ix[1:])
    frontier = [t for t in psi if id(t) in site_of]
    while len(site_of) < psi.num_tensors and frontier:
        nxt = []
        for t in frontier:
            for ix in t.inds:
                for u in psi._inds_get(ix):
                    if id(u) not in site_of:
                        site_of[id(u)] = site_of[id(t)]; nxt.append(u)
        frontier = nxt
    for t in psi:
        t.drop_tags(); t.add_tag(f"Q{site_of.get(id(t), 0)}")
    tags = [f"Q{q}" for q in range(n)]
    psi = tensor_network_ag_compress(psi, max_bond=max_bond, cutoff=cutoff, method='local-late', site_tags=tags,
                                     canonize=True, equalize_norms=True)
    psi.squeeze_()
    psi.exponent = 0.0
    return psi


def contract_peak(n, psi, max_width=26, log=print):
    """exact marginals <Z_q> and peak amplitude; one hyper-optimised path reused for all marginals
    (the norm network carries a 2x2 operator on every qubit, identity except on q -> identical structure)."""
    Z = np.diag([1., -1.]).astype(complex); I = np.eye(2, dtype=complex)
    bra = psi.conj().reindex({f"s{q}": f"S{q}" for q in range(n)})
    base = psi | bra
    def net(q):
        tn = base.copy()
        for p in range(n):
            tn |= qtn.Tensor(Z if p == q else I, inds=(f"S{p}", f"s{p}"), tags={f"OP{p}"})
        return tn
    opt = ctg.ReusableHyperOptimizer(methods=['greedy', 'kahypar'], max_repeats=32, minimize='flops', parallel=False,
                                     progbar=False)
    tn0 = net(-1)
    tree = tn0.contraction_tree(optimize=opt, output_inds=())
    width, cost = tree.contraction_width(), tree.contraction_cost(log=2)
    log(f"  norm/marginal network: {tn0.num_tensors} tensors, width {width:.1f}, log2 flops {cost:.1f}")
    if width > max_width:
        raise RuntimeError(f"contraction width {width:.1f} > {max_width} (2.5 GB cap)")
    nrm = float(np.real(tn0.contract(all, optimize=opt)))
    zs = np.array([float(np.real(net(q).contract(all, optimize=opt))) for q in range(n)]) / nrm
    bits = "".join("0" if z > 0 else "1" for z in zs)
    amp_tn = psi.copy()
    for q in range(n):
        amp_tn |= qtn.Tensor(np.array([1, 0] if bits[q] == "0" else [0, 1], dtype=complex), inds=(f"s{q}",))
    amp = amp_tn.contract(all, optimize=opt)
    p = float(abs(amp) ** 2 / nrm)
    return bits, p, zs, nrm, width, cost


def solve(path, centre=None, cutoff=1e-3, budget=2e4, log=print, max_snap_tries=10):
    t0 = time.time()
    n, units, tail = G.parse(path)
    L = SP.layers(n, units); D = max(L) + 1
    log(f"  parsed: n={n}, {len(units)} CZ units, depth {D}  ({time.time() - t0:.1f}s)")
    if centre is None:
        best, coarse, fine = find_centre(n, units, L, cutoff, budget, log)
        centre = best['c']
    t1 = time.time()
    log(f"  centre layer {centre}  (scan {t1 - t0:.1f}s, RSS {rss_mb()} MB)")
    snaps = []
    W, lo, hi, hist = tnoq.grow(n, units, L, centre, cutoff=cutoff, max_elems=50 * budget, log=False, snapshots=snaps)
    st = hist[-1]
    log(f"  window layers [{lo},{hi}) of [0,{D}): {absorbed_gates(L, lo, hi)}/{len(units)} gates absorbed; "
        f"W elems {st['elems']}, max bond {st['max_bond']}  ({time.time() - t1:.1f}s)")
    # resource rule: the LARGEST absorbed window whose exact final contraction fits the memory cap
    # (the answer is invariant up to compression error, so this choice only trades cost)
    last_err = None
    for (wlo, whi, wtn) in list(reversed(snaps))[:max_snap_tries]:
        W.tn = wtn
        psi = state_network(n, units, L, W, wlo, whi, tail)
        try:
            if wlo == 0 and whi == D:          # pure W|0>: compress the state (cheap, no exact gates left)
                psi = compress_state(n, psi, cutoff)
            bits, p, zs, nrm, width, cost = contract_peak(n, psi, log=lambda s_: None)
            lo, hi = wlo, whi
            log(f"  final core: window [{lo},{hi}) ({absorbed_gates(L, lo, hi)} gates in W), state max bond "
                f"{psi.max_bond()}, contraction width {width:.1f}, log2 flops {cost:.1f}")
            break
        except (RuntimeError, MemoryError) as e:
            last_err = e
            log(f"    snapshot [{wlo},{whi}) rejected: {str(e)[:80]}")
            continue
    else:
        # fallback: the full W|0> state, recompressed with a decreasing bond cap until it fits (approximate;
        # every fitting cap is evaluated so that the stability of the peak can be checked)
        log(f"  no exact core fits the cap; recompressing W|0> with bond caps")
        W.tn = snaps[-1][2]; lo, hi = snaps[-1][0], snaps[-1][1]
        psi0 = compress_state(n, state_network(n, units, L, W, lo, hi, tail), cutoff)
        approx = []
        chi = psi0.max_bond()
        while chi >= 2:
            chi //= 2
            try:
                psi = compress_state(n, psi0, cutoff, max_bond=chi)
                b_, p_, zs_, nrm_, w_, c_ = contract_peak(n, psi, log=lambda s_: None)
                approx.append(dict(chi=chi, peak=b_, p=p_, norm=nrm_, min_abs_z=float(np.min(np.abs(zs_))), width=w_))
                log(f"    max bond {chi}: width {w_:.1f}, p {p_:.4f}, min|Z| {np.min(np.abs(zs_)):.3f}, "
                    f"peak#{1 + [a_['peak'] for a_ in approx].index(b_)}")
                if len(approx) >= 3:
                    break
            except (RuntimeError, MemoryError) as e:
                log(f"    max bond {chi}: rejected ({str(e)[:60]})")
        if not approx:
            raise RuntimeError(f"no window snapshot or bond cap fits the memory cap ({last_err})")
        # verification on the UNCAPPED state: single amplitudes need no norm doubling, so they fit
        opt1 = ctg.ReusableHyperOptimizer(methods=['greedy', 'kahypar'], max_repeats=32, minimize='flops',
                                          parallel=False, progbar=False)
        def amp2(bits_):
            t = psi0.copy()
            for q in range(n):
                t |= qtn.Tensor(np.array([1, 0] if bits_[q] == "0" else [0, 1], dtype=complex), inds=(f"s{q}",))
            tree_ = t.contraction_tree(optimize=opt1, output_inds=())
            if tree_.contraction_width() > 26:
                raise RuntimeError(f"amplitude width {tree_.contraction_width():.1f}")
            return float(abs(t.contract(all, optimize=opt1)) ** 2)
        for a_ in approx:
            a_['p_full'] = amp2(a_['peak'])
            log(f"    candidate from bond {a_['chi']}: |<s|W0>|^2 on the uncapped state = {a_['p_full']:.4f}")
        best = max(approx, key=lambda a_: a_['p_full'])
        flips = []
        for q in range(n):
            fb = best['peak'][:q] + ('1' if best['peak'][q] == '0' else '0') + best['peak'][q + 1:]
            flips.append(amp2(fb))
        log(f"    1-bit flips of the best candidate: max p {max(flips):.4f} (candidate {best['p_full']:.4f})")
        best = dict(best, p=best['p_full'], flip_max=max(flips))
        bits, p, zs, nrm, width, cost = best['peak'], best['p'], np.full(n, best['min_abs_z']), best['norm'], best['width'], float('nan')
    res = dict(file=path.split('/')[-1], n=n, gates=len(units), depth=D, centre=centre, window=[lo, hi], window_final=[hist[-1]['lo'], hist[-1]['hi']], approx=locals().get('approx'),
               absorbed=absorbed_gates(L, lo, hi), W_max_bond=st['max_bond'], width=round(width, 1),
               log2_flops=round(cost, 1), peak=bits, p=p, norm=nrm, min_abs_z=float(np.min(np.abs(zs))),
               seconds=round(time.time() - t0, 1), rss_mb=rss_mb(), cutoff=cutoff, budget=budget)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("qasm"); ap.add_argument("--centre", type=float); ap.add_argument("--cutoff", type=float, default=1e-3)
    ap.add_argument("--budget", type=float, default=2e4); ap.add_argument("--json")
    a = ap.parse_args()
    print(a.qasm.split('/')[-1], flush=True)
    r = solve(a.qasm, a.centre, a.cutoff, a.budget, log=lambda s: print(s, flush=True))
    print(f"  PEAK (qubit 0 leftmost): {r['peak']}\n  peak probability {r['p']:.4f} (norm {r['norm']:.4f}, "
          f"min|<Z>| {r['min_abs_z']:.3f}); {r['seconds']}s, peak RSS {r['rss_mb']} MB", flush=True)
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
