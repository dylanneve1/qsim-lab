"""Exact(-ish) centre block via the swap-aware arbitrary-geometry TNO (tno.py: our earlier one-tensor-per-wire TNO
with free leg exchange, /tmp/peaked-generic; tno_fast.py: QR-reduced splits, /tmp/mlx-port), then conversion of the
TNO into an MPO2 by ordering its tensors along the bond graph (DFS) and carrying crossing bonds through the sites."""
import numpy as np, time
import tno as TN, tno_fast
from mpou2 import MPO2
tno_fast.install()


def grow_band(n, blocks, selL, selR, m, cutoff=1e-3, local_cutoff=1e-8, log=print, max_elems=5e6):
    """selL / selR: block ids of the band on the input / output side; layer distance from m decides the order."""
    W = TN.TNO(n, cutoff=local_cutoff)
    dist = {}
    for i in selL: dist.setdefault(round(m - blocks[i][3], 6), [[], []])[0].append(i)
    for i in selR: dist.setdefault(round(blocks[i][3] - m, 6), [[], []])[1].append(i)
    t0 = time.time()
    for d in sorted(dist):
        Ls, Rs = dist[d]
        for i in sorted(Rs, key=lambda i: blocks[i][4]):
            W.gate(blocks[i][2], blocks[i][0], blocks[i][1], 'after')
        for i in sorted(Ls, key=lambda i: -blocks[i][4]):
            W.gate(blocks[i][2], blocks[i][0], blocks[i][1], 'before')
        TN.canonical_compress(W, cutoff=cutoff); W.drop_trivial_bonds()
        nus = 0
        while True:
            a = TN.unswap_pass(W); nus += a
            if not a:
                break
            TN.canonical_compress(W, cutoff=cutoff); W.drop_trivial_bonds()
        E = W.bond_graph()
        log(f'  TNO band d={d}: +{len(Ls)}L +{len(Rs)}R, unswaps {nus}, elems {W.size()}, max bond {max(E.values()) if E else 1}, '
            f'nbonds {len(E)}, moved {sum(1 for o, i in W.perm().items() if o != i)}, {time.time()-t0:.1f}s')
        if W.size() > max_elems:
            raise RuntimeError('TNO band too large')
    return W


def tno_to_mpo(T, n, eps=1e-8, mode='rel', max_bond=4096, log=print):
    tids = list(T.T)
    bonds_of = {t: [l for l in T.T[t][1] if l[0] == 'b'] for t in tids}
    owner = {}
    for t in tids:
        for l in bonds_of[t]:
            owner.setdefault(l, []).append(t)
    adj = {t: set() for t in tids}
    for l, ts in owner.items():
        if len(ts) == 2:
            adj[ts[0]].add(ts[1]); adj[ts[1]].add(ts[0])
    # order: DFS per component, starting from a minimum-degree vertex; neighbours by ascending degree
    order = []; seen = set()
    for start in sorted(tids, key=lambda t: (len(adj[t]), t)):
        if start in seen:
            continue
        stack = [start]
        while stack:
            t = stack.pop()
            if t in seen:
                continue
            seen.add(t); order.append(t)
            for u in sorted(adj[t] - seen, key=lambda u: -len(adj[u])):
                stack.append(u)
    pos = {t: k for k, t in enumerate(order)}
    dim = {}
    for t in tids:
        A, legs = T.T[t]
        for d, l in zip(A.shape, legs):
            if l[0] == 'b':
                dim[l] = d
    # crossing bonds after position k
    cross = []
    for k in range(n):
        cross.append(sorted([l for l, ts in owner.items() if len(ts) == 2 and min(pos[ts[0]], pos[ts[1]]) <= k < max(pos[ts[0]], pos[ts[1]])]))
    W = MPO2(n, eps=eps, mode=mode, max_bond=max_bond)
    A_list = []; su = []; sd = []
    for k, t in enumerate(order):
        A, legs = T.T[t]
        Lb = cross[k - 1] if k > 0 else []
        Rb = cross[k] if k < n - 1 else []
        o = [l for l in legs if l[0] == 'o'][0]; i = [l for l in legs if l[0] == 'i'][0]
        attL = [l for l in Lb if l in legs]; attR = [l for l in Rb if l in legs]
        passb = [l for l in Lb if l in Rb and l not in legs]
        # tensor with axes attL + [o, i] + attR
        X = np.transpose(A, [legs.index(l) for l in attL] + [legs.index(o), legs.index(i)] + [legs.index(l) for l in attR])
        axes = attL + ['o', 'i'] + attR
        for l in passb:
            X = np.multiply.outer(X, np.eye(dim[l], dtype=complex)); axes = axes + [('L', l), ('R', l)]
        # final order: Lb (each l: attached l or ('L', l)), o, i, Rb
        fin = [l if l in attL else ('L', l) for l in Lb] + ['o', 'i'] + [l if l in attR else ('R', l) for l in Rb]
        X = np.transpose(X, [axes.index(a) for a in fin])
        Dl = int(np.prod([dim[l] for l in Lb])) if Lb else 1
        Dr = int(np.prod([dim[l] for l in Rb])) if Rb else 1
        A_list.append(np.ascontiguousarray(X).reshape(Dl, 2, 2, Dr).astype(complex))
        su.append(o[1]); sd.append(i[1])
    W.A = A_list; W.su = su; W.sd = sd
    W.pu = [None] * n; W.pd = [None] * n
    for s in range(n):
        W.pu[su[s]] = s; W.pd[sd[s]] = s
    W.c = 0
    log(f'  TNO -> MPO: raw bonds {W.bonds()}')
    W.move(n - 1); W.move(0)
    for s in range(n - 1):
        W.two(s, 'up', None, 'up')
    log(f'  TNO -> MPO: compressed {W.stats()}')
    return W
