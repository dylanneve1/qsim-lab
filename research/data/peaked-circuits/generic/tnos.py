"""Swap-aware middle-out growth: tno.TNO gate absorption (each gate tries both assignments of its two new
legs -> hidden relabellings are removed for free) + canonical arbitrary-geometry compression after every
layer pair (quimb tensor_network_ag_compress)."""
import time
import numpy as np
import tno as TN

CZ = np.diag([1, 1, 1, -1]).astype(complex)


def unitG(u):
    return CZ @ np.kron(u[2], u[3])


def grow(n, units, L, c, cutoff=1e-3, max_elems=5e6, log=False, local_cutoff=1e-8, unswap=False):
    D = max(L) + 1
    bylayer = [[] for _ in range(D)]
    for k in range(len(units)):
        bylayer[L[k]].append(k)
    W = TN.TNO(n, cutoff=local_cutoff)
    lo = hi = int(np.ceil(c)); t0 = time.time(); hist = []
    while lo > 0 or hi < D:
        if hi < D:
            for k in bylayer[hi]:
                W.gate(unitG(units[k]), units[k][0], units[k][1], 'after')
            hi += 1
        if lo > 0:
            for k in reversed(bylayer[lo - 1]):
                W.gate(unitG(units[k]), units[k][0], units[k][1], 'before')
            lo -= 1
        TN.canonical_compress(W, cutoff=cutoff)
        W.drop_trivial_bonds()
        nus = 0
        if unswap:
            while True:
                a = TN.unswap_pass(W)
                nus += a
                if not a:
                    break
                TN.canonical_compress(W, cutoff=cutoff); W.drop_trivial_bonds()
        E = W.bond_graph()
        st = dict(lo=lo, hi=hi, unswaps=nus, elems=W.size(), max_bond=max(E.values()) if E else 1, nbonds=len(E),
                  moved=sum(1 for o, i in W.perm().items() if o != i), t=round(time.time() - t0, 1))
        hist.append(st)
        if log:
            print("  ", st, flush=True)
        if st['elems'] > max_elems:
            break
    return W, lo, hi, hist


def frontier(seq, nxt, prv, units, n):
    aft, bef = [], []
    for q in range(n):
        if nxt[q] < len(seq[q]):
            k = seq[q][nxt[q]]; a, b = units[k][:2]; o = b if a == q else a
            if q == a and nxt[o] < len(seq[o]) and seq[o][nxt[o]] == k:
                aft.append(k)
        if prv[q] >= 0:
            k = seq[q][prv[q]]; a, b = units[k][:2]; o = b if a == q else a
            if q == a and prv[o] >= 0 and seq[o][prv[o]] == k:
                bef.append(k)
    return aft, bef


def grow_greedy(n, units, L, c, cutoff=1e-3, max_elems=5e6, log=False, local_cutoff=1e-6, compress_every=None,
                max_gates=None):
    """Gate-level greedy growth: at every step absorb the frontier gate (either side) whose local split has the
    smallest rank (rank 1 = exact cancellation); ties -> smaller |layer - c|, then lower index.
    Canonical compression every `compress_every` gates (default n // 2)."""
    import bisect
    compress_every = compress_every or max(1, n // 2)
    seq = [[] for _ in range(n)]
    for k, u in enumerate(units):
        seq[u[0]].append(k); seq[u[1]].append(k)
    nxt = [sum(1 for k in seq[q] if L[k] < c) for q in range(n)]
    prv = [x - 1 for x in nxt]
    W = TN.TNO(n, cutoff=local_cutoff)
    inside = []; t0 = time.time(); hist = []
    cache = {}
    while True:
        aft, bef = frontier(seq, nxt, prv, units, n)
        if not aft and not bef:
            break
        best = None
        for side, ks in (('after', aft), ('before', bef)):
            for k in ks:
                a, b = units[k][:2]
                x = (W.out_at if side == 'after' else W.in_at)[a]; y = (W.out_at if side == 'after' else W.in_at)[b]
                key = (side, k)
                sig = (id(W.T[x][0]), id(W.T[y][0]))
                if key not in cache or cache[key][0] != sig:
                    r = W.gate(unitG(units[k]), a, b, side, commit=False)
                    cache[key] = (sig, r)
                r = cache[key][1]
                cand = (r, abs(L[k] - c), k, side)
                if best is None or cand < best:
                    best = cand
        r, _, k, side = best
        a, b = units[k][:2]
        W.gate(unitG(units[k]), a, b, side)
        inside.append(k)
        for q in (a, b):
            if side == 'after': nxt[q] += 1
            else: prv[q] -= 1
        if len(inside) % compress_every == 0:
            TN.canonical_compress(W, cutoff=cutoff); W.drop_trivial_bonds()
            st = dict(gates=len(inside), elems=W.size(), max_bond=W.max_bond_dim(), nbonds=len(W.bond_graph()),
                      lo=min(L[q] for q in inside), hi=max(L[q] for q in inside),
                      moved=sum(1 for o, i in W.perm().items() if o != i), t=round(time.time() - t0, 1))
            hist.append(st)
            if log: print("  ", st, flush=True)
            if st['elems'] > max_elems: break
        if max_gates and len(inside) >= max_gates: break
    TN.canonical_compress(W, cutoff=cutoff); W.drop_trivial_bonds()
    return W, inside, nxt, prv, hist


def grow_adaptive(n, units, L, c, cutoff=1e-3, max_elems=5e6, log=False, local_cutoff=1e-8, unswap=False):
    """Like grow(), but at every step tentatively absorb the next layer on EACH side (with compression) and keep
    the one giving the smaller operator (ties -> 'after'); the other side waits."""
    import copy
    D = max(L) + 1
    bylayer = [[] for _ in range(D)]
    for k in range(len(units)):
        bylayer[L[k]].append(k)
    W = TN.TNO(n, cutoff=local_cutoff)
    lo = hi = int(np.ceil(c)); t0 = time.time(); hist = []

    def absorb(Wx, side, layer):
        ks = bylayer[layer] if side == 'after' else list(reversed(bylayer[layer]))
        for k in ks:
            Wx.gate(unitG(units[k]), units[k][0], units[k][1], side)
        TN.canonical_compress(Wx, cutoff=cutoff); Wx.drop_trivial_bonds()
        if unswap and TN.unswap_pass(Wx, min_bond=4):
            TN.canonical_compress(Wx, cutoff=cutoff); Wx.drop_trivial_bonds()
        return Wx

    while lo > 0 or hi < D:
        opts = []
        if hi < D:
            Wa = absorb(copy.deepcopy(W), 'after', hi); opts.append((Wa.size(), 0, 'after', Wa))
        if lo > 0:
            Wb = absorb(copy.deepcopy(W), 'before', lo - 1); opts.append((Wb.size(), 1, 'before', Wb))
        sz, _, side, W = min(opts, key=lambda x: (x[0], x[1]))
        if side == 'after': hi += 1
        else: lo -= 1
        E = W.bond_graph()
        st = dict(lo=lo, hi=hi, side=side, elems=W.size(), max_bond=max(E.values()) if E else 1, nbonds=len(E),
                  moved=sum(1 for o, i in W.perm().items() if o != i), t=round(time.time() - t0, 1))
        hist.append(st)
        if log:
            print("  ", st, flush=True)
        if st['elems'] > max_elems:
            break
    return W, lo, hi, hist
