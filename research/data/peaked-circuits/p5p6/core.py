"""Core R |> g |> PERM |> P of a mirror circuit: R = units of blocks with layer <= rmax, P = units of blocks
with layer >= pmin, g = the 1q pre-segment of the first middle unit on each wire (seam), perm q -> pi(q).
Exact marginals + peak probability of the core."""
import sys, json, hashlib, numpy as np
sys.path.insert(0,'/tmp/peaked-generic'); sys.path.insert(0,'/tmp/peaked-p5p6')
import quimb.tensor as qtn, gparse as G, tnoq, solve_generic as SG, relabel as RL

def unit_layers(orig):
    n, units, tail = G.parse(orig)
    n_, ops = RL.parse_lines(orig); lay = RL.cz_layers(n_, ops)
    czl = [lay[i] for i in range(len(ops)) if ops[i][0] == 'cz']
    assert len(czl) == len(units)
    return n, units, tail, czl

def core_tn(n, units, tail, czl, pi, rmax, pmin, seam=True, extra_R=(), extra_P=()):
    Rk = [k for k in range(len(units)) if czl[k] <= rmax or k in extra_R]
    Pk = [k for k in range(len(units)) if czl[k] >= pmin or k in extra_P]
    Rs, Ps = set(Rk), set(Pk)
    # seam g: pre-segment of the first non-R unit on each wire
    g = [np.eye(2, dtype=complex) for _ in range(n)]
    done = [False]*n
    for k, u in enumerate(units):
        a, b, Pa, Pb = u
        for q, M in ((a, Pa), (b, Pb)):
            if not done[q] and k not in Rs:
                done[q] = True
                if seam: g[q] = M
    ts = []; cur = {q: f"z{q}" for q in range(n)}
    for q in range(n): ts.append(qtn.Tensor(np.array([1, 0], dtype=complex), inds=(cur[q],)))
    def put(M, q):
        nid = tnoq._nid("w"); ts.append(qtn.Tensor(M, inds=(nid, cur[q]))); cur[q] = nid
    def put2(k):
        a, b = units[k][:2]; Gt = tnoq.unitG(units[k]).reshape(2, 2, 2, 2)
        na, nb = tnoq._nid("w"), tnoq._nid("w")
        ts.append(qtn.Tensor(Gt, inds=(na, nb, cur[a], cur[b]))); cur[a], cur[b] = na, nb
    for k in Rk: put2(k)
    for q in range(n): put(g[q], q)
    cur = {int(pi[q]): cur[q] for q in range(n)}           # content of q moves to wire pi(q)
    for k in Pk: put2(k)
    for q in range(n):
        ts.append(qtn.Tensor(tail[q], inds=(f"s{q}", cur[q])))
    return qtn.TensorNetwork(ts), len(Rk), len(Pk)

if __name__ == '__main__':
    orig, pifn, rmax, pmin = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    seam = (sys.argv[5] != 'noseam') if len(sys.argv) > 5 else True
    tag = sys.argv[6] if len(sys.argv) > 6 else 'core'
    pi = np.load(pifn)
    n, units, tail, czl = unit_layers(orig)
    psi, nR, nP = core_tn(n, units, tail, czl, pi, rmax, pmin, seam)
    print('core units R', nR, 'P', nP, flush=True)
    bits, p, zs, nrm, width, cost = SG.contract_peak(n, psi, max_width=28, log=lambda s: print(s, flush=True))
    print('p', p, 'norm', nrm, 'min|Z|', float(np.min(np.abs(zs))), 'hash', hashlib.sha256(bits.encode()).hexdigest()[:12])
    json.dump(dict(peak=bits, p=p, zs=list(map(float, zs)), rmax=rmax, pmin=pmin, seam=seam), open(f'/tmp/peaked-p5p6/private/{tag}.json', 'w'))
