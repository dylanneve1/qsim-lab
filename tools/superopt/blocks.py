"""SAT optimality certificates for small arithmetic blocks.

Usage: python blocks.py <block> [args]   (see BLOCKS below)
Prints the optimum circuit and the list of refuted (UNSAT) gate counts.
"""
import itertools
import json
import sys
import time

import synth


def bits(x, n):
    return [(x >> i) & 1 for i in range(n)]


def spec_compare(n, with_c0=True):
    """t ^= [b < a]; wires a0..a{n-1}, b0..b{n-1}, [c0], t. c0 clean."""
    names = [f'a{i}' for i in range(n)] + [f'b{i}' for i in range(n)]
    if with_c0:
        names.append('c0')
    names.append('t')
    nw = len(names)
    rows = []
    for a in range(1 << n):
        for b in range(1 << n):
            for t in (0, 1):
                inp = bits(a, n) + bits(b, n) + ([0] if with_c0 else []) + [t]
                out = bits(a, n) + bits(b, n) + ([0] if with_c0 else []) + [t ^ int(b < a)]
                rows.append((inp, out))
    return nw, rows, names


def spec_compare_const(n, K, ancillas=1):
    """t ^= [b >= K] for classical K; wires b, ancillas (clean), t."""
    names = [f'b{i}' for i in range(n)] + [f'z{i}' for i in range(ancillas)] + ['t']
    nw = len(names)
    rows = []
    for b in range(1 << n):
        for t in (0, 1):
            inp = bits(b, n) + [0] * ancillas + [t]
            out = bits(b, n) + [0] * ancillas + [t ^ int(b >= K)]
            rows.append((inp, out))
    return nw, rows, names


def spec_lookup(w, table, nout, ancillas):
    """out ^= ctrl * T[addr]; wires ctrl, a0..a{w-1}, z (clean anc), out."""
    names = ['c'] + [f'a{i}' for i in range(w)] + [f'z{i}' for i in range(ancillas)] + [f'o{i}' for i in range(nout)]
    nw = len(names)
    rows = []
    for c in (0, 1):
        for v in range(1 << w):
            for o in range(1 << nout):
                inp = [c] + bits(v, w) + [0] * ancillas + bits(o, nout)
                out = [c] + bits(v, w) + [0] * ancillas + bits(o ^ (table[v] if c else 0), nout)
                rows.append((inp, out))
    return nw, rows, names


def spec_add(n, ancillas=1, carry=True):
    """b += a (b has n+1 bits if carry); ancillas clean."""
    nb = n + 1 if carry else n
    names = [f'a{i}' for i in range(n)] + [f'b{i}' for i in range(nb)] + [f'z{i}' for i in range(ancillas)]
    nw = len(names)
    rows = []
    for a in range(1 << n):
        for b in range(1 << n):
            for bt in ((0, 1) if carry else (0,)):
                bb = b | (bt << n)
                s = (bb + a) % (1 << nb)
                rows.append((bits(a, n) + bits(bb, nb) + [0] * ancillas,
                             bits(a, n) + bits(s, nb) + [0] * ancillas))
    return nw, rows, names


def spec_modadd(N, ancillas):
    """b -> (b + L) mod N for b, L < N; L kept; ancillas clean; other inputs don't care."""
    n = max(1, (N - 1).bit_length())
    names = [f'b{i}' for i in range(n)] + [f'L{i}' for i in range(n)] + [f'z{i}' for i in range(ancillas)]
    nw = len(names)
    rows = []
    for b in range(N):
        for L in range(N):
            rows.append((bits(b, n) + bits(L, n) + [0] * ancillas,
                         bits((b + L) % N, n) + bits(L, n) + [0] * ancillas))
    return nw, rows, names


def run(name, nw, rows, names, kmin=0, kmax=40, max_tof=None, **kw):
    t0 = time.time()
    k, g, cert = synth.minimize(nw, rows, kmin=kmin, kmax=kmax, max_tof=max_tof, **kw)
    res = dict(block=name, wires=nw, rows=len(rows), max_tof=max_tof, optimum_gates=k,
               toffolis=None if g is None else sum(1 for x in g if x[0] == 'CCX'),
               circuit=None if g is None else synth.fmt(g, names),
               refuted=[c[0] for c in cert], secs=round(time.time() - t0, 1))
    print(json.dumps(res), flush=True)
    return res


if __name__ == '__main__' and sys.argv[1] != 'tof':
    what = sys.argv[1]
    a = [int(x) for x in sys.argv[2:]]
    if what == 'compare':
        n, c0 = a[0], bool(a[1]) if len(a) > 1 else True
        mt = a[2] if len(a) > 2 else None
        run(f'compare n={n} c0={c0}', *spec_compare(n, c0), max_tof=mt)
    elif what == 'compare_const':
        n, K, anc = a
        run(f'compare_const n={n} K={K} anc={anc}', *spec_compare_const(n, K, anc))
    elif what == 'add':
        n, anc, carry = a
        run(f'add n={n} anc={anc} carry={carry}', *spec_add(n, anc, bool(carry)))
    elif what == 'modadd':
        N, anc = a[0], a[1]
        mt = a[2] if len(a) > 2 else None
        run(f'modadd N={N} anc={anc}', *spec_modadd(N, anc), max_tof=mt)
    elif what == 'lookup':
        w, nout, anc = a[0], a[1], a[2]
        table = a[3:]
        mt = None
        run(f'lookup w={w} T={table} anc={anc}', *spec_lookup(w, table, nout, anc))


def run_tof(name, nw, rows, names, kcap, tmax):
    t0 = time.time()
    t, g, ref = synth.min_toffoli(nw, rows, kcap, tmax)
    res = dict(block=name, wires=nw, rows=len(rows), kcap=kcap, min_toffoli=t,
               gates=None if g is None else len(g),
               circuit=None if g is None else synth.fmt(g, names),
               refuted_toffoli=ref, secs=round(time.time() - t0, 1))
    print(json.dumps(res), flush=True)
    return res


if __name__ == '__main__' and sys.argv[1] == 'tof':
    what = sys.argv[2]
    a = [int(x) for x in sys.argv[3:]]
    kcap, tmax = a[0], a[1]
    rest = a[2:]
    if what == 'compare':
        run_tof(f'compare n={rest[0]} c0={bool(rest[1])}', *spec_compare(rest[0], bool(rest[1])), kcap, tmax)
    elif what == 'lookup':
        w, nout, anc = rest[0], rest[1], rest[2]
        run_tof(f'lookup w={w} T={rest[3:]} anc={anc}', *spec_lookup(w, rest[3:], nout, anc), kcap, tmax)
    elif what == 'modadd':
        run_tof(f'modadd N={rest[0]} anc={rest[1]}', *spec_modadd(rest[0], rest[1]), kcap, tmax)
    elif what == 'add':
        run_tof(f'add n={rest[0]} anc={rest[1]} carry={rest[2]}', *spec_add(rest[0], rest[1], bool(rest[2])), kcap, tmax)
