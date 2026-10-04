"""Exact synthesis of reversible circuits over {X, CNOT, CCX} by SAT.

    exists a circuit of exactly k gates g_1..g_k on `nw` wires such that for
    every specified row (input bits -> output constraint, None = don't care)
    the circuit maps the input to an output matching the constraint?

Iterating k = 0, 1, 2, ... the first satisfiable k is the minimum gate
count; every smaller k was refuted by the solver (UNSAT), which is the
optimality certificate. An optional cap on the number of CCX gates gives
Toffoli-optimal circuits the same way.

Encoding per step s: target one-hot T[s][q]; two control selectors
C1[s][q|none], C2[s][q|none] with idx(C1) < idx(C2) (none = last); a
control may not be the target. X = (none, none), CNOT = (q, none),
CCX = (q1, q2). Per row r: wire values v[r][s][q], control values
a, b, f = a & b, and v[r][s+1][q] = v[r][s][q] ^ (T[s][q] & f).
Solver: CaDiCaL 1.9.5 through PySAT.
"""
import itertools
import sys
import time

from pysat.card import CardEnc, EncType
from pysat.formula import IDPool
from pysat.solvers import Solver


def encode(nw, rows, k, max_tof=None, max_cnot=None, forbid_x=False, allow_nop=False):
    pool = IDPool()
    cls = []
    V = lambda *a: pool.id(a)
    for s in range(k):
        T = [V('T', s, q) for q in range(nw)]
        if allow_nop:
            # NOP = no target; then both controls none; NOPs only trail
            nop = V('NOP', s)
            cls.append(T + [nop])
            for q in range(nw):
                cls.append([-nop, -V('T', s, q)])
            cls.append([-nop, V('C1', s, nw)])
            cls.append([-nop, V('C2', s, nw)])
            if s > 0:
                cls.append([-V('NOP', s - 1), nop])
        else:
            cls.append(T)
        for q1, q2 in itertools.combinations(T, 2):
            cls.append([-q1, -q2])
        for c in ('C1', 'C2'):
            sel = [V(c, s, q) for q in range(nw + 1)]  # nw = none
            cls.append(sel)
            for q1, q2 in itertools.combinations(sel, 2):
                cls.append([-q1, -q2])
            for q in range(nw):
                cls.append([-V(c, s, q), -V('T', s, q)])
        # idx(C1) < idx(C2), or both none
        for i in range(nw + 1):
            for j in range(nw + 1):
                if not (i < j or (i == nw and j == nw)):
                    cls.append([-V('C1', s, i), -V('C2', s, j)])
        if forbid_x:
            cls.append([-V('C1', s, nw)])
        # no two identical adjacent gates (they would cancel)
        if s > 0:
            for q in range(nw):
                for i in range(nw + 1):
                    for j in range(i, nw + 1):
                        if (i < j or i == nw) and i != q and j != q:
                            cls.append([-V('T', s, q), -V('T', s - 1, q),
                                        -V('C1', s, i), -V('C1', s - 1, i),
                                        -V('C2', s, j), -V('C2', s - 1, j)])
    for r, (inp, out) in enumerate(rows):
        for q in range(nw):
            cls.append([V('v', r, 0, q) if inp[q] else -V('v', r, 0, q)])
        for s in range(k):
            for c, x in (('C1', 'a'), ('C2', 'b')):
                xv = V(x, r, s)
                cls.append([-V(c, s, nw), xv])
                for q in range(nw):
                    v = V('v', r, s, q)
                    cls.append([-V(c, s, q), -xv, v])
                    cls.append([-V(c, s, q), xv, -v])
            a, b, f = V('a', r, s), V('b', r, s), V('f', r, s)
            cls += [[-f, a], [-f, b], [f, -a, -b]]
            for q in range(nw):
                t, v, w = V('T', s, q), V('v', r, s, q), V('v', r, s + 1, q)
                cls += [[t, -v, w], [t, v, -w],
                        [-t, -w, v, f], [-t, -w, -v, -f],
                        [-t, w, -v, f], [-t, w, v, -f]]
        for q in range(nw):
            if out[q] is not None:
                v = V('v', r, k, q)
                cls.append([v] if out[q] else [-v])
    if max_tof is not None and k > 0:
        lits = [-V('C2', s, nw) for s in range(k)]
        enc = CardEnc.atmost(lits, bound=max_tof, vpool=pool, encoding=EncType.seqcounter)
        cls += enc.clauses
    return pool, cls


def decode(pool, model, nw, k):
    m = set(l for l in model if l > 0)
    gates = []
    for s in range(k):
        ts = [q for q in range(nw) if pool.id(('T', s, q)) in m]
        if not ts:
            continue
        t = ts[0]
        c1 = next(q for q in range(nw + 1) if pool.id(('C1', s, q)) in m)
        c2 = next(q for q in range(nw + 1) if pool.id(('C2', s, q)) in m)
        if c1 == nw:
            gates.append(('X', t))
        elif c2 == nw:
            gates.append(('CNOT', c1, t))
        else:
            gates.append(('CCX', c1, c2, t))
    return gates


def simulate(gates, bits):
    bits = list(bits)
    for g in gates:
        if g[0] == 'X':
            bits[g[1]] ^= 1
        elif g[0] == 'CNOT':
            bits[g[2]] ^= bits[g[1]]
        else:
            bits[g[3]] ^= bits[g[1]] & bits[g[2]]
    return bits


def check(gates, rows):
    for inp, out in rows:
        o = simulate(gates, inp)
        for q, want in enumerate(out):
            if want is not None and o[q] != want:
                return False
    return True


def solve(nw, rows, k, max_tof=None, solver='cadical195', timeout=None, **kw):
    pool, cls = encode(nw, rows, k, max_tof=max_tof, **kw)
    with Solver(name=solver, bootstrap_with=cls) as s:
        ok = s.solve()
        if not ok:
            return None
        g = decode(pool, s.get_model(), nw, k)
        assert check(g, rows), 'decoded circuit fails the spec'
        return g


def minimize(nw, rows, kmin=0, kmax=40, max_tof=None, log=sys.stderr, **kw):
    """Smallest k with a circuit; returns (k, gates, [(k', 'UNSAT', secs)])."""
    cert = []
    for k in range(kmin, kmax + 1):
        t0 = time.time()
        g = solve(nw, rows, k, max_tof=max_tof, **kw)
        dt = time.time() - t0
        print(f'  k={k} max_tof={max_tof}: {"SAT" if g else "UNSAT"} ({dt:.1f}s)', file=log, flush=True)
        if g is not None:
            return k, g, cert
        cert.append((k, 'UNSAT', round(dt, 2)))
    return None, None, cert


def min_toffoli(nw, rows, kcap, tmax, log=sys.stderr, **kw):
    """Minimum CCX count over all circuits of <= kcap gates (NOP padding):
    t = tmax, tmax-1, ... until UNSAT. Returns (t_min, gates, refuted_t)."""
    best, refuted = None, []
    t = tmax
    while t >= 0:
        t0 = time.time()
        g = solve(nw, rows, kcap, max_tof=t, allow_nop=True, **kw)
        print(f'  kcap={kcap} max_tof={t}: {"SAT" if g else "UNSAT"} ({time.time()-t0:.1f}s)', file=log, flush=True)
        if g is None:
            refuted.append(t)
            break
        best = g
        t = sum(1 for x in g if x[0] == 'CCX') - 1
    return (None if best is None else sum(1 for x in best if x[0] == 'CCX')), best, refuted


def fmt(gates, names=None):
    nm = (lambda q: names[q]) if names else str
    out = []
    for g in gates:
        if g[0] == 'X':
            out.append(f'X({nm(g[1])})')
        elif g[0] == 'CNOT':
            out.append(f'CNOT({nm(g[1])},{nm(g[2])})')
        else:
            out.append(f'CCX({nm(g[1])},{nm(g[2])},{nm(g[3])})')
    return ' '.join(out)
