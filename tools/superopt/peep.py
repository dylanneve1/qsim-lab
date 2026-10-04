"""SAT peephole superoptimisation of a real X/CNOT/CCX gate list.

Slides a window over the circuit: from every start, the longest run of
consecutive gates touching at most Q wires (and at most LMAX gates). A
constant-propagation pass knows which wires hold a known classical value
at the window start (ancillas start at 0 in each controlled-U round); those
rows are dropped from the window's specification (constant-aware). Each
distinct window (canonical relabelling + constant pattern) is sent to the
SAT synthesiser with "<= len-1 gates" (NOP padding) and minimised; every
replacement is re-verified by simulation on all window inputs consistent
with the known constants. Non-overlapping improvements are applied greedily
left to right and the saving is reported.

usage: peep.py gates.txt Q LMAX [max_windows]
"""
import itertools
import json
import sys
import time

import synth


def parse(path):
    gates, nq, anc_from = [], 0, None
    for line in open(path):
        if line.startswith('#'):
            for tok in line.split():
                if tok.startswith('qubits='):
                    nq = int(tok[7:])
                if tok.startswith('ancillas_from='):
                    anc_from = int(tok[14:])
            continue
        p = line.split()
        gates.append(tuple([p[0]] + [int(x) for x in p[1:]]))
    return gates, nq, anc_from


def qubits(g):
    return g[1:]


def const_prop(gates, nq, anc_from):
    """val[q] at the start of each gate: 0/1 or None (unknown)."""
    val = [None] * anc_from + [0] * (nq - anc_from)
    snaps = []
    for g in gates:
        snaps.append(tuple(val))
        if g[0] == 'X':
            if val[g[1]] is not None:
                val[g[1]] ^= 1
        elif g[0] == 'CX':
            c, t = g[1], g[2]
            if val[c] == 0:
                pass
            elif val[c] == 1:
                if val[t] is not None:
                    val[t] ^= 1
            else:
                val[t] = None
        else:
            a, b, t = g[1], g[2], g[3]
            if val[a] == 0 or val[b] == 0:
                pass
            elif val[a] == 1 and val[b] == 1:
                if val[t] is not None:
                    val[t] ^= 1
            else:
                val[t] = None
    return snaps


def window_spec(win, wires, consts):
    idx = {q: i for i, q in enumerate(wires)}
    local = []
    for g in win:
        if g[0] == 'X':
            local.append(('X', idx[g[1]]))
        elif g[0] == 'CX':
            local.append(('CNOT', idx[g[1]], idx[g[2]]))
        else:
            local.append(('CCX', idx[g[1]], idx[g[2]], idx[g[3]]))
    free = [i for i, q in enumerate(wires) if consts[q] is None]
    rows = []
    for bits in itertools.product((0, 1), repeat=len(free)):
        inp = [consts[q] if consts[q] is not None else 0 for q in wires]
        for i, b in zip(free, bits):
            inp[i] = b
        rows.append((inp, synth.simulate(local, inp)))
    key = (len(wires), tuple(local), tuple(consts[q] for q in wires))
    return local, rows, key


def best_replacement(nw, rows, L, log):
    """Smallest circuit with < L gates, or None. Returns (k, gates)."""
    best = None
    k = L - 1
    while k >= 0:
        g = synth.solve(nw, rows, k, allow_nop=True)
        if g is None:
            break
        best = g
        k = len(g) - 1
    return best


def main():
    path, Q, LMAX = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    maxw = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
    gates, nq, anc_from = parse(path)
    snaps = const_prop(gates, nq, anc_from)
    cache = {}
    found = []  # (start, end, saving, new local gates, wires)
    t0 = time.time()
    nwin = 0
    for i in range(len(gates)):
        wires, j = [], i
        while j < len(gates) and j - i < LMAX:
            new = [q for q in qubits(gates[j]) if q not in wires]
            if len(wires) + len(new) > Q:
                break
            wires += new
            j += 1
        L = j - i
        if L < 2:
            continue
        local, rows, key = window_spec(gates[i:j], wires, snaps[i])
        if key not in cache:
            nwin += 1
            if nwin > maxw:
                break
            g = best_replacement(len(wires), rows, L, sys.stderr)
            cache[key] = g
            if g is not None:
                assert synth.check(g, rows)
                print(json.dumps(dict(window=synth.fmt(local), consts=[snaps[i][q] for q in wires],
                                      len=L, better=len(g), circuit=synth.fmt(g))), file=sys.stderr, flush=True)
        g = cache[key]
        if g is not None:
            found.append((i, j, L - len(g), g, wires))
    # greedy non-overlapping, by saving density
    found.sort(key=lambda f: (-f[2], f[0]))
    used = [False] * len(gates)
    saving = 0
    applied = 0
    for i, j, s, g, w in found:
        if any(used[i:j]):
            continue
        for x in range(i, j):
            used[x] = True
        saving += s
        applied += 1
    print(json.dumps(dict(file=path, gates=len(gates), Q=Q, LMAX=LMAX, distinct_windows=len(cache),
                          improvable_distinct=sum(1 for v in cache.values() if v is not None),
                          windows_with_saving=len(found), applied=applied, gate_saving=saving,
                          secs=round(time.time() - t0, 1))), flush=True)


if __name__ == '__main__':
    main()
