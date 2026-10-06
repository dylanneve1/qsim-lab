"""Driver: own MPO+unswap grower (mpou.py) on a QASM circuit. Never reads a target string.
usage: run_mpou.py QASM CUTOFF MAXBOND CENTRE_FRAC OUT.json [UNSWAP_BOND] [RSS_GB]"""
import sys, os, time, json, resource, threading, subprocess, numpy as np, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gparse as G
from mpou import MPOU, SWAP, mps_marginals, mps_amp
from match import match_unswap
import copy
match_bond = int(os.environ.get('MATCH_BOND', '64'))
qasm, cutoff, maxb, cfrac, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
ubond = int(sys.argv[6]) if len(sys.argv) > 6 else 32
rss_gb = float(sys.argv[7]) if len(sys.argv) > 7 else 5.5
def guard():
    while True:
        r = int(subprocess.run(['ps', '-o', 'rss=', '-p', str(os.getpid())], capture_output=True, text=True).stdout or 0)
        if r > rss_gb * 1e6: print('RSS GUARD abort', r, flush=True); os._exit(3)
        time.sleep(5)
threading.Thread(target=guard, daemon=True).start()
CZ = np.diag([1, 1, 1, -1]).astype(complex)
n, units, tail = G.parse(qasm)
U = [(a, b, CZ @ np.kron(Pa, Pb)) for a, b, Pa, Pb in units]
# layer-balanced growth: ASAP layers of CZ units; centre = cfrac * depth
depth = [0] * n; lay = []
for a, b, _ in U:
    l = max(depth[a], depth[b]); lay.append(l); depth[a] = depth[b] = l + 1
D = max(lay) + 1; cl = int(round(cfrac * D))
Lq = sorted([k for k in range(len(U)) if lay[k] < cl], key=lambda k: (-lay[k], -k))
Rq = sorted([k for k in range(len(U)) if lay[k] >= cl], key=lambda k: (lay[k], k))
c = len(Lq)
W = MPOU(n, cutoff=cutoff, max_bond=maxb)
W.route_both = os.environ.get('ROUTE') == 'both'
t0 = time.time(); last = 0; step = 0; flip = 0; last_el = 0; last_mb = 0
def dist(k, side):
    a, b = U[k][:2]; p = W.pu if side == 'up' else W.pd; return abs(p[a] - p[b])
print(f'n={n} units={len(U)} centre={c} cutoff={cutoff} maxbond={maxb} ubond={ubond}', flush=True)
while Lq or Rq:
    if Lq and Rq:
        if os.environ.get('SCHED') == 'old': dl, dr = cfrac * D - lay[Lq[0]], lay[Rq[0]] + 1 - cfrac * D
        else: dl, dr = cl - 1 - lay[Lq[0]], lay[Rq[0]] - cl
        side = 'dn' if (dl < dr or (dl == dr and flip)) else 'up'; flip ^= 1
    else: side = 'dn' if Lq else 'up'
    if os.environ.get('SCHED') == 'foot' and Lq and Rq:
        snap = (copy.deepcopy(W.A), W.c, list(W.pu), list(W.pd), list(W.su), list(W.sd), W.lognorm, W.trunc)
        W.absorb(*U[Lq[0]], 'dn'); el = W.stats()['elems']
        resL = (W.A, W.c, W.pu, W.pd, W.su, W.sd, W.lognorm, W.trunc)
        W.A, W.c, W.pu, W.pd, W.su, W.sd, W.lognorm, W.trunc = snap
        W.absorb(*U[Rq[0]], 'up'); er = W.stats()['elems']
        if el < er or (el == er and flip):
            W.A, W.c, W.pu, W.pd, W.su, W.sd, W.lognorm, W.trunc = resL; Lq.pop(0)
        else: Rq.pop(0)
        step += 1
    else:
        k = (Lq if side == 'dn' else Rq).pop(0)
        W.absorb(*U[k], side); step += 1
    st = W.stats()
    if st['max_bond'] > ubond and (st['elems'] > 1.3 * last_el or st['max_bond'] > last_mb):
        for _ in range(3):
            g = W.unswap_sweep(hot=0.5 if st['max_bond'] > 64 else None)
            if g == 0: break
        st2 = W.stats()
        if st2['max_bond'] >= match_bond and st2['max_bond'] > last_mb:
            snap = (copy.deepcopy(W.A), W.c, list(W.pu), list(W.pd), list(W.su), list(W.sd), W.lognorm, W.trunc)
            try:
                pi_m, sc_m, nsw_m = match_unswap(W)
                for _ in range(3):
                    if W.unswap_sweep(hot=0.5 if W.stats()['max_bond'] > 64 else None) == 0: break
                st3 = W.stats()
                if st3['elems'] > st2['elems']:
                    W.A, W.c, W.pu, W.pd, W.su, W.sd, W.lognorm, W.trunc = snap; tag = 'reverted'
                else: tag = 'kept'
                print(f"    match-unswap at step {step}: {nsw_m} swaps, scores median {np.median(sc_m):.3f} min {sc_m.min():.3f}; "
                      f"{st2['elems']} -> {st3['elems']} elems ({tag})", flush=True)
            except Exception as e:
                W.A, W.c, W.pu, W.pd, W.su, W.sd, W.lognorm, W.trunc = snap; print('    match-unswap failed', e, flush=True)
            st2 = W.stats()
        last_el, last_mb = st2['elems'], st2['max_bond']
    if step - last >= 10 or not (Lq or Rq):
        last = step; st = W.stats()
        print(f"[{step}/{len(U)}] left {c-len(Lq)} right {len(U)-c-len(Rq)} {st} t={time.time()-t0:.0f}s "
              f"rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/ (2**20 if sys.platform=='darwin' else 2**10):.0f}MB", flush=True)
for w in range(n):
    if w in tail: W.gate1(w, tail[w], 'up')
for _ in range(6):
    if W.unswap_sweep() == 0: break
M, su = W.state(); p0, nrm = mps_marginals(M)
bits_site = ['0' if x >= 0.5 else '1' for x in p0]
peak = ''.join(bits_site[W.pu[w]] for w in range(n))
def prob(bits_logical):
    sb = [None] * n
    for w in range(n): sb[W.pu[w]] = bits_logical[w]
    return float(abs(mps_amp(M, sb)) ** 2 / nrm)
p = prob(peak); flips = [prob(peak[:q] + ('1' if peak[q] == '0' else '0') + peak[q+1:]) for q in range(n)]
zs = [2 * p0[W.pu[w]] - 1 for w in range(n)]
res = dict(qasm=qasm, cutoff=cutoff, max_bond=maxb, centre_frac=cfrac, peak=peak, p=p, norm_rel=nrm, max_flip=max(flips),
           n_flips_higher=int(sum(f > p for f in flips)), min_abs_z=float(min(abs(z) for z in zs)), zs=[round(z, 4) for z in zs],
           trunc=W.trunc, final=W.stats(), seconds=round(time.time() - t0))
json.dump(res, open(out, 'w'), indent=1)
print(f"PEAK hash {hashlib.sha256(peak.encode()).hexdigest()[:12]} p={p:.5f} maxflip={max(flips):.5f} higher={res['n_flips_higher']} "
      f"min|Z|={res['min_abs_z']:.3f} trunc={W.trunc:.4f} {res['seconds']}s", flush=True)
