"""Pair-driven middle-out operator growth on the relabelled circuit (relabel.py), then exact contraction.
usage: pairgrow.py ORIG.qasm PI.npy MATCH.npy CUT_LAYER [cutoff] [budget]"""
import sys, time, json, hashlib, numpy as np, collections
import quimb.tensor as qtn
sys.path.insert(0, '/tmp/peaked-generic')
import gparse as G, tnoq, solve_generic as SG
import relabel as RL

def build(orig, pi, cut):
    n, ops = RL.parse_lines(orig)
    lay = RL.cz_layers(n, ops)
    # block id per cz (same rule as cz_layers / blocks_v2)
    owner = [None]*n; blk = []; bid = {}
    for i, o in enumerate(ops):
        if o[0] != 'cz': continue
        a, b = o[1]; p = tuple(sorted((a, b)))
        if owner[a] is not None and owner[a] == owner[b] and blk[owner[a]] == p: bid[i] = owner[a]
        else: blk.append(p); owner[a] = owner[b] = len(blk)-1; bid[i] = len(blk)-1
    sigma = [int(x) for x in np.argsort(pi)]
    B, sw, A = RL.rewrite(n, ops, cut, sigma)
    # original cz indices in order for B and A parts
    side = []
    nxt = [None]*n; sd = [None]*len(ops)
    for i in range(len(ops)-1, -1, -1):
        o = ops[i]
        if o[0] == 'cz':
            sd[i] = 'B' if lay[i] <= cut else 'A'
            for q in o[1]: nxt[q] = sd[i]
    czB = [i for i in range(len(ops)) if ops[i][0] == 'cz' and sd[i] == 'B']
    czA = [i for i in range(len(ops)) if ops[i][0] == 'cz' and sd[i] == 'A']
    fn = f'/tmp/peaked-p5p6/_relab_{cut}.qasm'
    RL.emit(n, B, sw, A, fn)
    n2, units, tail = G.parse(fn)
    tags = [('L', bid[i], lay[i]) for i in czB] + [('S', -1, -1)]*(3*len(sw)) + [('R', bid[i], lay[i]) for i in czA]
    assert len(tags) == len(units), (len(tags), len(units))
    # sanity: the wires of units must match
    for k, (u, t) in enumerate(zip(units, tags)):
        if t[0] == 'L': assert set(u[:2]) == set(ops[czB[k]][1])
    return n, units, tail, tags, sigma, len(blk)

def grow(n, units, tags, match, cutoff=1e-3, budget=2e6, log=print, middle=(15, 26), allow_unmatched=True):
    W = tnoq.QTNO(n)
    wires = [[] for _ in range(n)]
    for k, u in enumerate(units):
        for q in u[:2]: wires[q].append(k)
    pos = {}  # (k,q) -> index in wires[q]
    for q in range(n):
        for i, k in enumerate(wires[q]): pos[(k, q)] = i
    absorbed = np.zeros(len(units), bool)
    # initial W: all swap units
    for k, t in enumerate(tags):
        if t[0] == 'S':
            W.gate(tnoq.unitG(units[k]), units[k][0], units[k][1], 'after'); absorbed[k] = True
    # pointers: lp[q] = index of next left unit (towards input), rp[q] = next right unit
    firstS = {}
    lp = [None]*n; rp = [None]*n
    for q in range(n):
        seq = wires[q]
        iL = [i for i, k in enumerate(seq) if tags[k][0] == 'L']
        iR = [i for i, k in enumerate(seq) if tags[k][0] == 'R']
        lp[q] = iL[-1] if iL else -1
        rp[q] = iR[0] if iR else len(seq)
    W.compress(cutoff=cutoff)
    # block -> units
    bunits = collections.defaultdict(list)
    for k, t in enumerate(tags):
        if t[0] != 'S': bunits[(t[0], t[1])].append(k)
    def left_ok(b):
        ks = bunits[('L', b)]
        if not ks or absorbed[ks[0]]: return False
        # absorb from last to first; the last unit must be at lp on both wires
        k = ks[-1]; return all(lp[q] == pos[(k, q)] for q in units[k][:2])
    def right_ok(b):
        ks = bunits[('R', b)]
        if not ks or absorbed[ks[0]]: return False
        k = ks[0]; return all(rp[q] == pos[(k, q)] for q in units[k][:2])
    def absorb_left(b):
        for k in reversed(bunits[('L', b)]):
            W.gate(tnoq.unitG(units[k]), units[k][0], units[k][1], 'before'); absorbed[k] = True
            for q in units[k][:2]: lp[q] -= 1
    def absorb_right(b):
        for k in bunits[('R', b)]:
            W.gate(tnoq.unitG(units[k]), units[k][0], units[k][1], 'after'); absorbed[k] = True
            for q in units[k][:2]: rp[q] += 1
    Lb = sorted({t[1] for t in tags if t[0] == 'L'}); Rb = sorted({t[1] for t in tags if t[0] == 'R'})
    blay = {t[1]: t[2] for t in tags if t[0] != 'S'}
    t0 = time.time(); hist = []; rnd = 0
    while True:
        rnd += 1; did = 0
        # phase 1: middle network only (unmatched, middle zone) until exhausted
        if False:
            cand = [('L', b) for b in Lb if left_ok(b) and match[b] < 0 and middle[0] <= blay[b] <= middle[1]] + \
                   [('R', b) for b in Rb if right_ok(b) and match[b] < 0 and middle[0] <= blay[b] <= middle[1]]
            for s_, b in cand:
                (absorb_left if s_ == 'L' else absorb_right)(b); did += 1
            if did == 0:
                grow.net_done = True; log('network phase done')
            else:
                W.compress(cutoff=cutoff); st = W.stats(); st.update(round=rnd, kind='net', did=did, absorbed=int(absorbed.sum())); log(st); continue
        # matched pairs
        for b in Lb:
            j = match[b]
            if j >= 0 and j in Rb and left_ok(b) and right_ok(j):
                absorb_left(b); absorb_right(j); did += 2
        kind = 'pairs'
        if allow_unmatched:
            # unmatched or partner-blocked blocks in the middle zone, closest to the centre first
            cand = [('L', b) for b in Lb if left_ok(b) and (match[b] < 0 or match[b] not in Rb)] + \
                   [('R', b) for b in Rb if right_ok(b) and (match[b] < 0 or match[b] not in Lb)]
            cand = [c for c in cand if middle[0] <= blay[c[1]] <= middle[1]]
            for s, b in cand:
                (absorb_left if s == 'L' else absorb_right)(b); did += 1
            kind = 'unmatched'
        if did == 0: break
        W.compress(cutoff=cutoff)
        st = W.stats(); st.update(round=rnd, kind=kind, did=did, absorbed=int(absorbed.sum()), t=round(time.time()-t0, 1))
        hist.append(st); log(st)
        if st['elems'] > budget: break
    return W, absorbed, hist

def state_net(n, units, tail, W, absorbed, tags):
    tn = W.tn.copy()
    cur_in = {q: f"b{q}" for q in range(n)}; ts = []
    Rk = [k for k in range(len(units)) if not absorbed[k] and tags[k][0] == 'L']
    Pk = [k for k in range(len(units)) if not absorbed[k] and tags[k][0] == 'R']
    for k in reversed(Rk):
        a, b = units[k][:2]; Gt = tnoq.unitG(units[k]).reshape(2, 2, 2, 2)
        na, nb = tnoq._nid("x"), tnoq._nid("x")
        ts.append(qtn.Tensor(Gt, inds=(cur_in[a], cur_in[b], na, nb))); cur_in[a], cur_in[b] = na, nb
    for q in range(n): ts.append(qtn.Tensor(np.array([1, 0], dtype=complex), inds=(cur_in[q],)))
    cur_out = {q: f"k{q}" for q in range(n)}
    for k in Pk:
        a, b = units[k][:2]; Gt = tnoq.unitG(units[k]).reshape(2, 2, 2, 2)
        na, nb = tnoq._nid("y"), tnoq._nid("y")
        ts.append(qtn.Tensor(Gt, inds=(na, nb, cur_out[a], cur_out[b]))); cur_out[a], cur_out[b] = na, nb
    for q in range(n): ts.append(qtn.Tensor(tail[q], inds=(f"s{q}", cur_out[q])))
    return tn | qtn.TensorNetwork(ts), len(Rk), len(Pk)

if __name__ == '__main__':
    orig, pifn, mfn, cut = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    cutoff = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-3
    budget = float(sys.argv[6]) if len(sys.argv) > 6 else 2e6
    pi = np.load(pifn); match = np.load(mfn)
    n, units, tail, tags, sigma, nb = build(orig, pi, cut)
    print('units', len(units), 'blocks', nb, flush=True)
    W, absorbed, hist = grow(n, units, tags, match, cutoff, budget, log=lambda s: print(' ', s, flush=True))
    psi, nR, nP = state_net(n, units, tail, W, absorbed, tags)
    print('unabsorbed: left', nR, 'right', nP, flush=True)
    bits_w, p, zs, nrm, width, cost = SG.contract_peak(n, psi, max_width=27, log=lambda s: print(s, flush=True))
    # map back: x_q = x'_{sigma(q)}
    bits = ''.join(bits_w[sigma[q]] for q in range(n))
    print('width', round(width, 1), 'log2flops', round(cost, 1), 'p', p, 'norm', nrm, 'min|Z|', float(np.min(np.abs(zs))))
    print('peak hash', hashlib.sha256(bits.encode()).hexdigest()[:12])
    tag = sys.argv[7] if len(sys.argv) > 7 else 'run'
    json.dump(dict(peak=bits, p=p, norm=nrm, zs=list(map(float, zs)), sigma=sigma, cut=cut, cutoff=cutoff, absorbed=int(absorbed.sum()),
                   width=width), open(f'/tmp/peaked-p5p6/private/pairgrow_{tag}.json', 'w'))
