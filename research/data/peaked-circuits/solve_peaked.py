#!/usr/bin/env python3
"""Blind classical solver for the BlueQubit HQAP peaked circuits P11 / P12 (98 qubits).

Pipeline (no knowledge of the target bitstring is used anywhere):
  1. Parse the QASM into atomic 2-qubit units  u a; u b; cz a,b; u a; u b.
  2. Fingerprint the single-qubit rotations between consecutive CZs on each wire (SU(2)
     quaternions). Exact inverse pairs that are unique in the circuit ("anchors") reveal the
     obfuscated identity blocks U ▷ U† and, inside each block, a consistent wire map f
     (an involution: a layer of SWAPs absorbed into the seam).
  3. Split the serialised circuit into its generation sections; the two blocks are the sections
     holding the mirror centres; the outer identity (T[U] ▷ ... ▷ U†) is everything between the
     first section (R) and the last section (P, two CZs per 2-qubit gate).
  4. Replace the whole middle by the wire permutation f_A then f_B (a permutation is free in a
     tensor network) and contract the shallow core R ▷ P exactly with quimb/cotengra:
     single-qubit marginals <Z_q>, peak = argmax, peak probability by an exact amplitude.
  5. If some marginals are weak, greedily move frontier gates across the R / middle and
     middle / P boundaries (obfuscation patches can straddle them), accepting a move only if
     the exact peak probability grows by >= 30 %.
Output: the 98-bit peak with qubit 0 as the leftmost character.
"""
import sys, time, itertools, collections, math, re
import numpy as np
import quimb.tensor as qtn


def num(x):
    return float(eval(x.strip(), {"pi": math.pi, "__builtins__": {}}))


def U3(th, ph, la):
    return np.array([[math.cos(th / 2), -np.exp(1j * la) * math.sin(th / 2)],
                     [np.exp(1j * ph) * math.sin(th / 2), np.exp(1j * (ph + la)) * math.cos(th / 2)]])


CZ = np.diag([1, 1, 1, -1]).astype(complex)


def parse(path):
    """-> n, units [(a, b, pre_a, pre_b, post_a, post_b)]"""
    lines = [l.strip() for l in open(path) if l.strip().startswith(("u(", "cz"))]
    units = []
    for i in range(0, len(lines), 5):
        g = lines[i:i + 5]
        def u(l):
            a, b = l.split(")")
            return int(re.search(r"q\[(\d+)\]", b).group(1)), U3(*map(num, a[2:].split(",")))
        (q0, u0), (q1, u1) = u(g[0]), u(g[1])
        a, b = map(int, re.findall(r"q\[(\d+)\]", g[2]))
        (q3, u3), (q4, u4) = u(g[3]), u(g[4])
        pre = {q0: u0, q1: u1}
        post = {q3: u3, q4: u4}
        units.append((a, b, pre[a], pre[b], post[a], post[b]))
    n = 1 + max(max(x[0], x[1]) for x in units)
    return n, units


def unit_matrix(x):
    a, b, pa, pb, qa, qb = x
    return np.kron(qa, qb) @ CZ @ np.kron(pa, pb)


def quat(V):
    V = V / np.sqrt(np.linalg.det(V))
    q = np.array([V[0, 0].real, V[0, 0].imag, V[0, 1].real, V[0, 1].imag])
    for x in q:
        if abs(x) > 1e-9:
            return -q if x < 0 else q
    return q


def anchors(n, units):
    """Unique exact inverse pairs of inter-CZ single-qubit segments: (wire, k_end) <-> (wire', k_start)."""
    last = {}
    segs = []                       # (wire, prev k, next k, V)
    for k, x in enumerate(units):
        a, b = x[0], x[1]
        for q, pre in ((a, x[2]), (b, x[3])):
            if q in last:
                kp, post = last[q]
                segs.append((q, kp, k, pre @ post))
        last[a] = (k, x[4])
        last[b] = (k, x[5])
    Q = [quat(s[3]) for s in segs]
    key = lambda q: tuple(np.round(q, 6))
    cnt = collections.Counter(key(q) for q in Q)
    uniq = {key(q): i for i, q in enumerate(Q) if cnt[key(q)] == 1}
    out = []
    for i, q in enumerate(Q):
        if cnt[key(q)] != 1:
            continue
        qi = q.copy(); qi[1:] *= -1
        qi = quat(np.array([[qi[0] + 1j * qi[1], qi[2] + 1j * qi[3]],
                            [-qi[2] + 1j * qi[3], qi[0] - 1j * qi[1]]]))
        j = uniq.get(key(qi))
        if j is not None and j != i and segs[i][2] <= segs[j][1]:
            out.append((segs[i], segs[j]))
    return out


def sections(units):
    layers, cur, used = [], [], set()
    for k, x in enumerate(units):
        if x[0] in used or x[1] in used:
            layers.append(cur); cur, used = [], set()
        cur.append(k); used |= {x[0], x[1]}
    layers.append(cur)
    secs, cs = [], []
    for i, l in enumerate(layers):
        if cs and len(l) >= 9 and len(layers[i - 1]) <= 3:
            secs.append(cs); cs = []
        cs += l
    secs.append(cs)
    return [(min(s), max(s)) for s in secs]


def block_maps(n, anc, secs):
    """Group anchors by the section containing their mirror centre; one involution per block."""
    by = collections.defaultdict(list)
    for a, b in anc:
        c = (a[2] + b[1]) / 2
        for si, (lo, hi) in enumerate(secs):
            if lo <= c <= hi:
                by[si].append((a[0], b[0]))
    maps = []
    for si in sorted(by):
        if len(by[si]) < 50:
            continue
        f = {}
        for x, y in by[si]:
            f[x] = y
        miss_s = [w for w in range(n) if w not in f]
        miss_d = [w for w in range(n) if w not in set(f.values())]
        for perm in itertools.permutations(miss_d):
            g = dict(f); g.update(zip(miss_s, perm))
            if all(g[g[w]] == w for w in range(n)):
                f = g
                break
        maps.append((si, f))
    return maps


def solve(path, verbose=True):
    t0 = time.time()
    n, units = parse(path)
    M = [unit_matrix(x) for x in units]
    secs = sections(units)
    anc = anchors(n, units)
    maps = block_maps(n, anc, secs)
    L = list(range(n))
    for si, f in maps:
        L = [L[f[w]] for w in range(n)]
    R0 = list(range(secs[0][0], secs[0][1] + 1))
    P0 = list(range(secs[-1][0], secs[-1][1] + 1))
    t1 = time.time()
    if verbose:
        print(f"  structure: {len(units)} gates, {len(secs)} sections, blocks in sections {[s for s, _ in maps]}, "
              f"{len(anc)} anchors, core R={len(R0)} P={len(P0)} gates  ({t1 - t0:.1f}s)", flush=True)
    Z = np.diag([1., -1.])
    seq = collections.defaultdict(list)
    for k, x in enumerate(units):
        seq[x[0]].append(k); seq[x[1]].append(k)

    def run(eR, eP):
        circ = qtn.Circuit(n)
        for k in R0 + sorted(eR):
            circ.apply_gate_raw(M[k], (units[k][0], units[k][1]))
        for k in sorted(eP) + P0:
            circ.apply_gate_raw(M[k], (L[units[k][0]], L[units[k][1]]))
        zs = np.array([float(np.real(circ.local_expectation(Z, (q,), optimize='greedy'))) for q in range(n)])
        s = [0 if z > 0 else 1 for z in zs]
        p = abs(circ.amplitude("".join(map(str, s)), optimize='greedy')) ** 2
        return p, zs, s

    eR, eP = set(), set()
    p, zs, s = run(eR, eP)
    if verbose:
        print(f"  core contraction: peak prob {p:.4f}, mean|Z| {np.mean(np.abs(zs)):.4f} ({time.time() - t1:.1f}s)", flush=True)
    lo1, hi1 = secs[1]
    lo_last, hi_last = secs[-2]
    while True:
        weak = {q for q in range(n) if abs(zs[q]) < 0.6}
        if not weak:
            break
        cands = []
        for w in range(n):
            for k in seq[w]:                      # first unadded gate of section 1 on w
                if lo1 <= k <= hi1 and k not in eR:
                    a, b = units[k][0], units[k][1]; o = b if a == w else a
                    if all(kk in eR for kk in seq[o] if lo1 <= kk < k) and (a in weak or b in weak):
                        cands.append(('R', k))
                    break
            for k in reversed(seq[w]):            # last unadded gate of the section before P
                if lo_last <= k <= hi_last and k not in eP:
                    a, b = units[k][0], units[k][1]; o = b if a == w else a
                    if all(kk in eP for kk in seq[o] if k < kk <= hi_last) and (L[a] in weak or L[b] in weak):
                        cands.append(('P', k))
                    break
        best = (p, None)
        for side, k in set(cands):
            p2, _, _ = run(eR | {k} if side == 'R' else eR, eP | {k} if side == 'P' else eP)
            if p2 > best[0] * 1.3:
                best = (p2, (side, k))
        if best[1] is None:
            break
        side, k = best[1]
        (eR if side == 'R' else eP).add(k)
        p, zs, s = run(eR, eP)
        if verbose:
            print(f"  repair: moved {side}-boundary gate {k} -> peak prob {p:.4f}", flush=True)
    peak = "".join(str(s[L[w]]) for w in range(n))
    return peak, p, time.time() - t0


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(path.split("/")[-1], flush=True)
        peak, p, dt = solve(path)
        print(f"  PEAK (qubit 0 leftmost): {peak}\n  peak probability (de-obfuscated core) {p:.4f}; total {dt:.1f}s", flush=True)
