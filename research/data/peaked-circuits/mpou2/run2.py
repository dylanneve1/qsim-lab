"""Driver for mpou2 (own MPO + unswap grower).  Never reads a target string.
usage: run2.py QASM --m CENTRE [--band CL CH] [--eps0 1e-10] [--eps 1e-5] [--mode sum2] [--maxb 512] ...
Order: blocks (consecutive same-pair CZ units merged) keyed by the ASAP layer of their first unit.
Left = layer < m (absorbed on the input side, by m - layer), right = layer >= m (output side, by layer - m).
Phase 1 ("exact centre block"): only blocks with CL <= layer < CH, cutoff eps0 (near exact), full local unswap.
Phase 2: the rest, cutoff eps, side by layer distance (or footprint), hot-bond greedy unswap, matching unswap when
greedy stalls."""
import sys, os, time, json, pickle, argparse, resource, threading, subprocess, hashlib
import numpy as np, faulthandler, signal
faulthandler.register(signal.SIGUSR1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gparse as G
from mpou2 import MPO2, SWAP, mps_marginals, mps_amp
from match2 import match_unswap

ap = argparse.ArgumentParser()
ap.add_argument('qasm'); ap.add_argument('--m', type=float, default=None)
ap.add_argument('--order', default='asap', choices=['asap', 'file', 'cutlayers'])
ap.add_argument('--cut', type=float, default=0.5, help='file-order cut (fraction of units, or unit index if > 1)')
ap.add_argument('--band', type=int, nargs=2, default=None)
ap.add_argument('--eps0', type=float, default=1e-10); ap.add_argument('--eps', type=float, default=1e-5)
ap.add_argument('--mode', default='sum2'); ap.add_argument('--maxb', type=int, default=512)
ap.add_argument('--hot', type=float, default=0.5, help='hot bonds: >= hot*maxbond')
ap.add_argument('--ubond', type=int, default=8, help='global sweeps only when max bond > ubond')
ap.add_argument('--match_lo', type=int, default=24); ap.add_argument('--match_hi', type=int, default=96)
ap.add_argument('--match_every', type=int, default=20, help='min steps between matching attempts')
ap.add_argument('--sched', default='dist', choices=['dist', 'foot', 'pair'])
ap.add_argument('--pair_window', type=int, default=80)
ap.add_argument('--ckpt', default=None); ap.add_argument('--ckpt_every', type=float, default=600.0)
ap.add_argument('--resume', default=None)
ap.add_argument('--out', default=None); ap.add_argument('--rss_gb', type=float, default=5.5)
ap.add_argument('--log_every', type=int, default=10)
ap.add_argument('--lookahead', type=int, default=0)
ap.add_argument('--route', default='both', choices=['both', 'side'])
ap.add_argument('--sabre', type=int, default=0, help='Sabre-style swap selection with this extended-set size (0 = meeting-point routing)')
ap.add_argument('--sabre_w', type=float, default=0.5)
ap.add_argument('--lam', type=float, default=1.0)
ap.add_argument('--tau', type=float, default=2e5, help='full unswap sweeps when elems > tau')
ap.add_argument('--max_its', type=int, default=10)
ap.add_argument('--diag_every', type=int, default=0)
ap.add_argument('--match_stat', default='sum')
ap.add_argument('--stop_elems', type=float, default=3e7)
ap.add_argument('--max_time', type=float, default=1e9)
ap.add_argument('--no_state', action='store_true')
ap.add_argument('--end_left', type=float, default=None, help='endgame: keep input-side blocks with layer < this out of W')
ap.add_argument('--end_right', type=float, default=None, help='endgame: keep output-side blocks with layer >= this out of W')
ap.add_argument('--end_opt', default='auto-hq')
ap.add_argument('--tno_band', action='store_true', help='build the centre band with the swap-aware TNO, then convert to MPO')
ap.add_argument('--tno_cut', type=float, default=1e-3)
ap.add_argument('--tno_local', type=float, default=1e-8)
ap.add_argument('--state_left', type=int, default=0, help='apply the last N input-side blocks to |0> as an MPS instead of absorbing them')
ap.add_argument('--finish_left', type=int, default=0, help='absorb the last N input-side blocks before any further output-side block')
ap.add_argument('--eps_centre', type=float, default=None, help='sum2 cutoff of the dense centre decomposition (default eps0)')
ap.add_argument('--dense_centre', type=int, default=0, help='max group size for the dense exact centre block (0=off)')
args = ap.parse_args()


def guard():
    while True:
        try:
            r = int(subprocess.run(['ps', '-o', 'rss=', '-p', str(os.getpid())], capture_output=True, text=True).stdout or 0)
        except Exception:
            r = 0
        if r > args.rss_gb * 1e6:
            print('RSS GUARD abort', r, flush=True); os._exit(3)
        time.sleep(5)


threading.Thread(target=guard, daemon=True).start()
CZ = np.diag([1, 1, 1, -1]).astype(complex)
n, units, tail = G.parse(args.qasm)
K = len(units)
depth = [0] * n; ulay = []
for a, b, _, _ in units:
    l = max(depth[a], depth[b]); ulay.append(l); depth[a] = depth[b] = l + 1
# blocks
last = {}; blocks = []      # [a, b, M(4x4 on kron(a,b)), first layer, first unit index]
for k, (a, b, Pa, Pb) in enumerate(units):
    M = CZ @ np.kron(Pa, Pb)
    ba, bb = last.get(a), last.get(b)
    if ba is not None and ba == bb:
        x, y = blocks[ba][0], blocks[ba][1]
        if (x, y) == (b, a):
            M = SWAP @ M @ SWAP
        blocks[ba][2] = M @ blocks[ba][2]
    else:
        blocks.append([a, b, M, ulay[k], k]); last[a] = last[b] = len(blocks) - 1
lone = {}
for w, T in enumerate(tail):
    if w in last:
        bi = last[w]; a, b = blocks[bi][0], blocks[bi][1]
        blocks[bi][2] = (np.kron(T, np.eye(2)) if w == a else np.kron(np.eye(2), T)) @ blocks[bi][2]
    else:
        lone[w] = T
NB = len(blocks)
m = args.m
if args.order != 'asap':
    cut = int(round(args.cut * K)) if args.cut <= 1 else int(args.cut)
    Lb = [i for i in range(NB) if blocks[i][4] < cut]; Rb = [i for i in range(NB) if blocks[i][4] >= cut]
    if args.order == 'file':
        for i in Lb: blocks[i][3] = -(cut - 1 - blocks[i][4])
        for i in Rb: blocks[i][3] = 1 + (blocks[i][4] - cut)
    else:
        d = [0] * n
        for i in sorted(Rb, key=lambda i: blocks[i][4]):
            a, b = blocks[i][0], blocks[i][1]; l = max(d[a], d[b]); blocks[i][3] = 1 + l; d[a] = d[b] = l + 1
        d = [0] * n
        for i in sorted(Lb, key=lambda i: -blocks[i][4]):
            a, b = blocks[i][0], blocks[i][1]; l = max(d[a], d[b]); blocks[i][3] = -l; d[a] = d[b] = l + 1
    m = 0.5
    print(f'order {args.order}: cut at unit {cut}, left depth {max(-blocks[i][3] for i in Lb)+1} right depth {max(blocks[i][3] for i in Rb)}', flush=True)
left = sorted([i for i in range(NB) if blocks[i][3] < m], key=lambda i: (m - blocks[i][3], -blocks[i][4]))
right = sorted([i for i in range(NB) if blocks[i][3] >= m], key=lambda i: (blocks[i][3] - m, blocks[i][4]))
endL = [i for i in left if args.end_left is not None and blocks[i][3] < args.end_left]
endR = [i for i in right if args.end_right is not None and blocks[i][3] >= args.end_right]
if endL or endR:
    sL, sR = set(endL), set(endR)
    left = [i for i in left if i not in sL]; right = [i for i in right if i not in sR]
    print(f'endgame: {len(endL)} input-side and {len(endR)} output-side blocks kept out of W', flush=True)
# wire-neighbour blocks (time order) for frontier checks
nextb = {}; prevb = {}; lastw = {}
for i in sorted(range(NB), key=lambda i: blocks[i][4]):
    for w in (blocks[i][0], blocks[i][1]):
        if w in lastw:
            nextb.setdefault(lastw[w], []).append(i); prevb.setdefault(i, []).append(lastw[w])
        lastw[w] = i
cl, ch = args.band if args.band else (m, m)
nb1L = sum(1 for i in left if blocks[i][3] >= cl); nb1R = sum(1 for i in right if blocks[i][3] < ch)
print(f'n={n} units={K} blocks={NB} depth={max(ulay)+1} m={m} left={len(left)} right={len(right)} '
      f'band={cl},{ch} (phase1 {nb1L}+{nb1R}) eps0={args.eps0} eps={args.eps} mode={args.mode} maxb={args.maxb} '
      f'sched={args.sched}', flush=True)

t0 = time.time()
if args.resume:
    st = pickle.load(open(args.resume, 'rb'))
    W = st['W']; iL, iR, step = st['iL'], st['iR'], st['step']; hist = st.get('hist', [])
    left, right = st['left'], st['right']
    print(f'resumed at step {step} iL={iL} iR={iR} {W.stats()}', flush=True)
else:
    W = MPO2(n, eps=args.eps0, mode=args.mode, max_bond=args.maxb)
    iL = iR = step = 0; hist = []
    if args.tno_band and args.band:
        import tno_centre
        Wt = tno_centre.grow_band(n, blocks, left[:nb1L], right[:nb1R], m, cutoff=args.tno_cut, local_cutoff=args.tno_local)
        W = tno_centre.tno_to_mpo(Wt, n, eps=args.eps0, mode=args.mode, max_bond=args.maxb)
        iL, iR = nb1L, nb1R
        print(f'TNO centre block -> MPO: {W.stats()} t={time.time()-t0:.1f}s', flush=True)
    elif args.dense_centre and args.band:
        import centre
        bset = sorted(left[:nb1L] + right[:nb1R], key=lambda i: blocks[i][4])
        groups = centre.components(n, [(blocks[i][0], blocks[i][1]) for i in bset])
        gmax = max(len(g) for g in groups)
        print(f'centre band: {len(bset)} blocks, group sizes {sorted(len(g) for g in groups)}', flush=True)
        if gmax <= args.dense_centre:
            gid = {q: tuple(g) for g in groups for q in g}
            gops = {tuple(g): [] for g in groups}
            for i in bset:
                gops[gid[blocks[i][0]]].append((blocks[i][0], blocks[i][1], blocks[i][2]))
            W = centre.build(n, groups, gops, eps=args.eps_centre if args.eps_centre is not None else args.eps0)
            W.eps = args.eps0; W.mode = args.mode; W.max_bond = args.maxb
            iL, iR = nb1L, nb1R
            print(f'dense centre block: {W.stats()} t={time.time()-t0:.1f}s', flush=True)
        else:
            print('groups too large for the dense centre; growing the band instead', flush=True)
last_ck = time.time(); last_el = 0; last_mb = 0; last_match = -10 ** 9; flip = 0


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 20 if sys.platform == 'darwin' else 2 ** 10)


def local_unswap(i):
    """greedy probes around bond i (window 2) until no change."""
    for _ in range(4):
        g = 0
        for s in range(max(0, i - 2), min(n - 1, i + 3)):
            g += W.unswap_bond(s)
        if g == 0:
            break


def do_unswap():
    global last_el, last_mb, last_match
    st = W.stats()
    if st['max_bond'] <= args.ubond:
        return
    if not (st['elems'] > 1.2 * last_el or st['max_bond'] > last_mb):
        return
    if st['elems'] > args.tau:
        tu = time.time(); e0 = st['elems']; its = 0
        for its in range(1, args.max_its + 1):
            if W.unswap_sweep(thr=2, window=0) < 1:
                break
        print(f"    full unswap at step {step}: {e0} -> {W.elems()} elems, max bond {W.stats()['max_bond']}, {its} sweeps {time.time()-tu:.1f}s", flush=True)
    else:
        for _ in range(3):
            thr = max(2, int(args.hot * W.stats()['max_bond']))
            if W.unswap_sweep(thr=thr) == 0:
                break
    st2 = W.stats()
    if (args.match_lo <= st2['max_bond'] <= args.match_hi and st2['max_bond'] >= last_mb
            and step - last_match >= args.match_every):
        last_match = step
        snap = W.snapshot(); ts = time.time()
        try:
            pi_m, sc_m, nsw = match_unswap(W, stat=args.match_stat)
            if nsw < 0:
                print(f"    match at step {step}: skipped, score median {np.median(sc_m):.3f} {time.time()-ts:.1f}s", flush=True)
                raise StopIteration
            for _ in range(3):
                if W.unswap_sweep(thr=max(2, int(args.hot * W.stats()['max_bond']))) == 0:
                    break
            st3 = W.stats()
            tag = 'kept'
            if st3['elems'] > st2['elems']:
                W.restore(snap); tag = 'reverted'
            print(f"    match at step {step}: {nsw} swaps, score median {np.median(sc_m):.3f} min {sc_m.min():.3f}; "
                  f"{st2['elems']} -> {st3['elems']} ({tag}) {time.time()-ts:.1f}s", flush=True)
        except StopIteration:
            pass
        except Exception as e:
            W.restore(snap); print('    match failed', repr(e), flush=True)
        st2 = W.stats()
    last_el, last_mb = st2['elems'], st2['max_bond']


def pick(q, i0, side):
    """among the queue entries with the same layer as q[i0] (disjoint, commuting blocks) move the one with the
    cheapest routing to position i0."""
    lay0 = blocks[q[i0]][3]; p = W.pu if side == 'up' else W.pd
    best = None; j = i0
    while j < len(q) and blocks[q[j]][3] == lay0:
        a, b = blocks[q[j]][0], blocks[q[j]][1]
        c = W.route_cost(p[a], p[b])
        if best is None or c < best[0]:
            best = (c, j)
        if c == 0:
            break
        j += 1
    j = best[1]
    q[i0], q[j] = q[j], q[i0]


def future_gates(side, i0):
    """upcoming blocks of both queues (lookahead window), as (side, a, b, weight)."""
    if args.lookahead <= 0:
        return None
    out = []
    for q, ii, sd in ((left, iL, 'dn'), (right, iR, 'up')):
        start = ii + (1 if sd == side else 0)
        for r, idx in enumerate(q[start:start + args.lookahead]):
            out.append((sd, blocks[idx][0], blocks[idx][1], 1.0 / (1.0 + r / 4.0)))
    return out


decay = None


def sabre_route(a, b, side):
    """Sabre-style routing (idea: Li, Ding, Xie 2019, as used in Qiskit's SabreSwap): repeatedly apply the adjacent
    site exchange that minimises  d(front) + w * mean d(extended set)  (with a decay on recently moved sites) until
    the front gate's wires are adjacent.  Extended set = the next blocks of both queues."""
    global decay
    if decay is None:
        decay = np.ones(n)
    kind = 'both' if args.route == 'both' else side
    ext = []
    for q, ii, sd in ((left, iL, 'dn'), (right, iR, 'up')):
        start = ii + (1 if sd == side else 0)
        for idx in q[start:start + args.sabre]:
            ext.append((sd, blocks[idx][0], blocks[idx][1]))

    def pos(sd, w):
        return (W.pu if sd == 'up' else W.pd)[w]

    nsw = 0
    while True:
        i, j = pos(side, a), pos(side, b)
        if abs(i - j) <= 1:
            break
        cands = {t for t in (i - 1, i, j - 1, j) if 0 <= t < n - 1}
        best = None
        for t in cands:
            # exchange of sites t, t+1: positions of wires on those sites swap (both legs for 'both', one side otherwise)
            def p2(sd, w):
                p = pos(sd, w)
                if kind != 'both' and sd != kind:
                    return p
                return t + 1 if p == t else (t if p == t + 1 else p)
            hf = abs(p2(side, a) - p2(side, b))
            he = (sum(abs(p2(sd, x) - p2(sd, y)) for sd, x, y in ext) / len(ext)) if ext else 0.0
            h = max(decay[t], decay[t + 1]) * (hf + args.sabre_w * he)
            if best is None or h < best[0] - 1e-12:
                best = (h, t)
        t = best[1]
        W.swap(t, kind); W.nroute += 1; nsw += 1
        decay[t] += 0.001; decay[t + 1] += 0.001
        if nsw > 4 * n:
            raise RuntimeError('sabre routing did not converge')
    if nsw:
        decay[:] = 1.0


import collections
resL = collections.Counter(); resR = collections.Counter(); npair = 0
doneB = set()


def register(i, side):
    """after absorbing block i on `side`, remember the wire pair its mirror partner should have on the other side
    (same two sites, other leg kind)."""
    doneB.add(i)
    if args.sched != 'pair':
        return
    a, b = blocks[i][0], blocks[i][1]
    if side == 'dn':
        s1, s2 = W.pd[a], W.pd[b]; key = frozenset((W.su[s1], W.su[s2]))
        if key in resL_pending(i, 'dn'):
            pass
        resR[key] += 1
    else:
        s1, s2 = W.pu[a], W.pu[b]; key = frozenset((W.sd[s1], W.sd[s2]))
        resL[key] += 1


def resL_pending(i, side):
    return ()


def available(i, side):
    if i in doneB:
        return False
    if side == 'dn':      # all later blocks on its wires that belong to the input side must be absorbed
        return all((j in doneB) or (j not in leftset) for j in nextb.get(i, []))
    return all((j in doneB) or (j not in rightset) for j in prevb.get(i, []))


def pair_choice(canL, canR):
    """absorb an available block whose wire pair matches a registered partner of an absorbed block (mirror cancellation)."""
    best = None
    if canR and resR:
        for j in range(iR, min(len(right), iR + args.pair_window)):
            i = right[j]; key = frozenset((blocks[i][0], blocks[i][1]))
            if resR.get(key, 0) > 0 and available(i, 'up'):
                d = blocks[i][3] - m
                if best is None or d < best[0]:
                    best = (d, 'up', j, key)
                break
    if canL and resL:
        for j in range(iL, min(len(left), iL + args.pair_window)):
            i = left[j]; key = frozenset((blocks[i][0], blocks[i][1]))
            if resL.get(key, 0) > 0 and available(i, 'dn'):
                d = m - blocks[i][3]
                if best is None or d < best[0]:
                    best = (d, 'dn', j, key)
                break
    if best is None:
        return None
    _, side, j, key = best
    if side == 'up':
        resR[key] -= 1
        if resR[key] <= 0: del resR[key]
    else:
        resL[key] -= 1
        if resL[key] <= 0: del resL[key]
    return side, j


def absorb(i, side):
    a, b, M = blocks[i][0], blocks[i][1], blocks[i][2]
    if args.sabre:
        sabre_route(a, b, side)
    W.absorb(a, b, M, side, future_gates(side, i), args.lam)
    p = W.pu if side == 'up' else W.pd
    return min(p[a], p[b])


W.route_kind = args.route
leftset = set(left); rightset = set(right)
doneB.update(left[:iL]); doneB.update(right[:iR])
phase = 1 if (iL < nb1L or iR < nb1R) else 2
if phase == 2:
    W.eps = args.eps
while iL < len(left) or iR < len(right):
    tstep = time.time()
    if phase == 1 and iL >= nb1L and iR >= nb1R:
        phase = 2; W.eps = args.eps
        print(f'== phase 1 done (centre block) at step {step}: {W.stats()} t={time.time()-t0:.1f}s', flush=True)
        for _ in range(6):
            if W.unswap_sweep(thr=2) == 0:
                break
        print(f'   after sweeps {W.stats()} t={time.time()-t0:.1f}s', flush=True)
        last_el = 0; last_mb = 0
    if phase == 2 and 0 < len(left) - iL <= args.state_left and W.A[0].shape[2] == 2:
        ts_ = time.time()
        phi = MPO2.zero_state(n, W.eps, W.mode, W.max_bond); phi.route_kind = 'both'
        for i in reversed(left[iL:]):
            a, b, M = blocks[i][0], blocks[i][1], blocks[i][2]
            phi.absorb(a, b, M, 'up')
            for _ in range(2):
                if phi.unswap_sweep(thr=2) == 0:
                    break
        nrem = len(left) - iL; e0 = W.elems(); sphi = phi.stats()
        W.apply_to_state(phi); iL = len(left)
        print(f'== input side: last {nrem} blocks applied to |0> as an MPS ({sphi}), W -> W|phi>: {e0} -> {W.elems()} elems '
              f'{W.stats()} {time.time()-ts_:.1f}s', flush=True)
        last_el = 0; last_mb = 0
    if phase == 2 and iL >= len(left) and W.A[0].shape[2] == 2 and not args.no_state and not endL:
        e0 = W.elems(); W.to_state()
        print(f'== input side complete at step {step}: W -> W|0> as an MPS, {e0} -> {W.elems()} elems {W.stats()}', flush=True)
        last_el = 0; last_mb = 0
    limL = nb1L if phase == 1 else len(left)
    limR = nb1R if phase == 1 else len(right)
    canL, canR = iL < limL, iR < limR
    if canL and canR and phase == 2 and len(left) - iL <= args.finish_left:
        canR = False
    if canL and canR:
        dl = m - blocks[left[iL]][3]; dr = blocks[right[iR]][3] - m
        if args.sched == 'foot' and phase == 2:
            snap = W.snapshot(); absorb(left[iL], 'dn'); el = W.elems(); sL = W.snapshot(); W.restore(snap)
            absorb(right[iR], 'up'); er = W.elems()
            if el < er or (el == er and flip):
                W.restore(sL); side = 'dn'; iL += 1
            else:
                side = 'up'; iR += 1
            flip ^= 1; bi = None
        else:
            side = 'dn' if (dl < dr or (dl == dr and flip)) else 'up'; flip ^= 1
    else:
        side = 'dn' if canL else 'up'
    if args.sched == 'pair' and phase == 2:
        got = pair_choice(canL, canR)
        if got is not None:
            side, j = got
            if side == 'dn':
                left.insert(iL, left.pop(j)); bi = absorb(left[iL], 'dn'); doneB.add(left[iL]); iL += 1
            else:
                right.insert(iR, right.pop(j)); bi = absorb(right[iR], 'up'); doneB.add(right[iR]); iR += 1
            npair += 1
            side = 'pair-' + side
    if side in ('dn', 'up') and not (args.sched == 'foot' and phase == 2 and canL and canR):
        if side == 'dn':
            pick(left, iL, 'dn'); bi = absorb(left[iL], 'dn'); register(left[iL], 'dn'); iL += 1
        else:
            pick(right, iR, 'up'); bi = absorb(right[iR], 'up'); register(right[iR], 'up'); iR += 1
    step += 1
    ta = time.time(); st_a = W.stats()
    if phase == 1 and bi is not None:
        local_unswap(bi)
    do_unswap()
    st = W.stats(); tb = time.time()
    if tb - tstep > 5:
        print(f'    slow step {step} ({side}): absorb {ta-tstep:.1f}s -> {st_a}, unswap {tb-ta:.1f}s -> {st}', flush=True)
    if step % args.log_every == 0 or (iL >= len(left) and iR >= len(right)):
        lL = blocks[left[iL]][3] if iL < len(left) else -1
        lR = blocks[right[iR]][3] if iR < len(right) else -1
        print(f'[{step}/{NB}] L {iL} (lay {lL}) R {iR} (lay {lR}) {st} rs={W.nroute} pairs={npair} res={sum(resL.values())}/{sum(resR.values())} t={time.time()-t0:.1f}s rss={rss():.0f}MB', flush=True)
        hist.append((step, iL, iR, st['max_bond'], st['elems'], time.time() - t0))
    if args.diag_every and step % args.diag_every == 0:
        from match2 import score_wires
        from scipy.optimize import linear_sum_assignment as lsa
        td = time.time(); Sd = score_wires(W, 'smax'); r_, c_ = lsa(-Sd)
        cur = np.array([Sd[w, W.su[W.pd[w]]] for w in range(n)]); asg = np.array([Sd[w, c_[w]] for w in range(n)])
        ok = sum(W.su[W.pd[w]] == c_[w] for w in range(n))
        dist = [abs(W.pd[w] - W.pu[c_[w]]) for w in range(n) if W.su[W.pd[w]] != c_[w]]
        print(f'    diag step {step}: smax assigned median {np.median(asg):.3f} min {asg.min():.3f} | current pairing median {np.median(cur):.3f}; '
              f'{ok}/{n} already paired; mismatch distances {sorted(dist)} ({time.time()-td:.1f}s) bonds {W.bonds()}', flush=True)
    if st['elems'] > args.stop_elems:
        print('STOP: elems', st, flush=True); break
    if time.time() - t0 > args.max_time:
        print(f'STOP: time at step {step}', st, flush=True); break
    if args.ckpt and time.time() - last_ck > args.ckpt_every:
        pickle.dump(dict(W=W, iL=iL, iR=iR, step=step, hist=hist, args=vars(args), left=left, right=right), open(args.ckpt + '.tmp', 'wb'))
        os.replace(args.ckpt + '.tmp', args.ckpt); last_ck = time.time()
        print(f'    checkpoint at step {step}', flush=True)

done = iL >= len(left) and iR >= len(right)
if args.ckpt:
    pickle.dump(dict(W=W, iL=iL, iR=iR, step=step, hist=hist, args=vars(args), left=left, right=right), open(args.ckpt + '.tmp', 'wb'))
    os.replace(args.ckpt + '.tmp', args.ckpt)
if not done:
    print('not finished', flush=True); sys.exit(2)
for w, T in lone.items():
    W.gate1(w, T, 'up')
if endL or endR:
    import endgame
    for _ in range(4):
        if W.unswap_sweep(thr=2) == 0:
            break
    print('endgame W', W.stats(), flush=True)
    Rg = [(blocks[i][0], blocks[i][1], blocks[i][2]) for i in sorted(endL, key=lambda i: blocks[i][4])]
    Pg = [(blocks[i][0], blocks[i][1], blocks[i][2]) for i in sorted(endR, key=lambda i: blocks[i][4])]
    te = time.time()
    zs, nrm = endgame.marginals(W, Rg, Pg, n, opt=args.end_opt)
    peak = ''.join('0' if z >= 0 else '1' for z in zs)
    prob = lambda bits: abs(endgame.amplitude(W, Rg, Pg, n, bits, opt=args.end_opt)) ** 2 / nrm
    p = prob(peak)
    flips = [prob(peak[:q] + ('1' if peak[q] == '0' else '0') + peak[q + 1:]) for q in range(n)]
    res = dict(qasm=args.qasm, args=vars(args), peak=peak, p=p, max_flip=max(flips), n_flips_higher=int(sum(f > p for f in flips)),
               min_abs_z=float(min(abs(z) for z in zs)), zs=[round(float(z), 4) for z in zs], trunc=W.trunc, final=W.stats(),
               seconds=round(time.time() - t0), endgame_seconds=round(time.time() - te), hist=hist)
    if args.out:
        json.dump(res, open(args.out, 'w'), indent=1)
    print(f"PEAK sha {hashlib.sha256(peak.encode()).hexdigest()[:12]} p={p:.5f} maxflip={max(flips):.3g} "
          f"higher={res['n_flips_higher']} min|Z|={res['min_abs_z']:.3f} trunc={W.trunc:.4f} {res['seconds']}s "
          f"(endgame {res['endgame_seconds']}s)", flush=True)
    sys.exit(0)
for _ in range(4):
    if W.unswap_sweep(thr=2) == 0:
        break
print('final', W.stats(), f't={time.time()-t0:.1f}s', flush=True)
Mps = W.state(); p0, nrm = mps_marginals(Mps)
bits_site = ['0' if x >= 0.5 else '1' for x in p0]
peak = ''.join(bits_site[W.pu[w]] for w in range(n))


def prob(bits_logical):
    sb = [None] * n
    for w in range(n):
        sb[W.pu[w]] = bits_logical[w]
    return float(abs(mps_amp(Mps, sb)) ** 2 / nrm)


p = prob(peak)
flips = [prob(peak[:q] + ('1' if peak[q] == '0' else '0') + peak[q + 1:]) for q in range(n)]
zs = [2 * p0[W.pu[w]] - 1 for w in range(n)]
res = dict(qasm=args.qasm, args=vars(args), peak=peak, p=p, max_flip=max(flips), n_flips_higher=int(sum(f > p for f in flips)),
           min_abs_z=float(min(abs(z) for z in zs)), zs=[round(z, 4) for z in zs], trunc=W.trunc, final=W.stats(),
           seconds=round(time.time() - t0), hist=hist)
if args.out:
    json.dump(res, open(args.out, 'w'), indent=1)
print(f"PEAK sha {hashlib.sha256(peak.encode()).hexdigest()[:12]} p={p:.5f} maxflip={max(flips):.3g} "
      f"higher={res['n_flips_higher']} min|Z|={res['min_abs_z']:.3f} trunc={W.trunc:.4f} {res['seconds']}s", flush=True)
