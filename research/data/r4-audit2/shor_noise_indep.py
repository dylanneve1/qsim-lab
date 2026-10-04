"""Independent noisy semiclassical-Shor simulator (audit §16).
Own bit-sliced big-int evaluator of the dumped windowed oracle blocks, own
location model, own H / phase / measurement / recycling, own success test.
usage: shor_noise_indep.py DUMP N a r kind(depol|phaseflip) k M seed
"""
import sys, random, math, cmath
import numpy as np
dump, N, a, r, kind, K, M, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
lines = open(dump).read().split('\n')
nq, t, m = int(lines[0].split()[1]), int(lines[0].split()[3]), int(lines[0].split()[5])
rounds = []; cur = None
for l in lines[1:]:
    if not l: continue
    p = l.split()
    if p[0] == 'round': cur = []; rounds.append(cur); continue
    g = p[0]; q = tuple(int(x) for x in p[1:])
    cur.append((g, q))
assert len(rounds) == t
pm = 1 if kind == 'depol' else 0
# locations per round: [prep]*pm, h1, slots..., phase, h2, [meas]*pm
slots = [[(gi, s) for gi, (g, q) in enumerate(rd) for s in range(len(q))] for rd in rounds]
per_round = [pm + 1 + len(s) + 2 + pm for s in slots]
L = sum(per_round)
def pauli():
    if kind == 'depol': return random.choice('XYZ')
    if kind == 'phaseflip': return 'Z'
    return 'X'
def sample_faults(k):
    locs = random.sample(range(L), k)
    out = [dict() for _ in range(t)]
    for g in locs:
        i = 0
        while g >= per_round[i]: g -= per_round[i]; i += 1
        d = out[i]
        if pm and g == 0: d['prep'] = True; continue
        g -= pm
        if g == 0: d['h1'] = pauli(); continue
        g -= 1
        if g < len(slots[i]):
            gi, s = slots[i][g]; d.setdefault('gates', []).append((gi, s, pauli())); continue
        g -= len(slots[i])
        if g == 0: d['phase'] = pauli(); continue
        if g == 1: d['h2'] = pauli(); continue
        assert pm and g == 2; d['meas'] = True
    if kind in ('phaseflip',):  # no prep/meas locations for pure Z noise
        pass
    return out
def peak_ok(y):
    s = (y * r + (1 << (t - 1))) >> t
    return abs(y * r - (s << t)) * 2 * r < (1 << t)
def keys_from_cols(cols, S):
    # cols: list of nq ints (bit j = branch j); returns list of int keys
    B = np.zeros((S, nq), dtype=np.uint8)
    nb = (S + 7) // 8
    for q in range(nq):
        B[:, q] = np.unpackbits(np.frombuffer(cols[q].to_bytes(nb, 'little'), dtype=np.uint8), bitorder='little')[:S]
    w = (1 << np.arange(nq, dtype=np.uint64)).astype(np.uint64)
    return [int(x) for x in (B.astype(np.uint64) * w).sum(axis=1)]
def cols_from_keys(keys):
    S = len(keys); arr = np.array(keys, dtype=np.uint64)
    cols = []
    for q in range(nq):
        bits = ((arr >> np.uint64(q)) & np.uint64(1)).astype(np.uint8)
        cols.append(int.from_bytes(np.packbits(bits, bitorder='little').tobytes(), 'little'))
    return cols
CAPSUP = 4096
def trajectory(faults):
    keys = [1 << 1]  # x = 1 (qubit 1 = x bit 0), control bit 0 = 0, ancillas 0
    amps = np.array([1.0 + 0j])
    y = 0; ms = 0
    for i in range(t):
        S = len(keys); ms = max(ms, S)
        if S > CAPSUP: return None, ms
        f = faults[i]
        c0 = 1 if f.get('prep') else 0
        # H on control |c0>: |0> + (-1)^c0 |1>
        k2 = keys + [k | 1 for k in keys]
        a2 = np.concatenate([amps, amps * (-1 if c0 else 1)]) / math.sqrt(2)
        S2 = 2 * S; ALL = (1 << S2) - 1
        cols = cols_from_keys(k2); sign = 0
        def apply_pauli(q, p):
            nonlocal sign
            if p in 'ZY': sign ^= cols[q]
            if p in 'XY': cols[q] ^= ALL
            # Y = i X Z: global phase i (irrelevant)
        if 'h1' in f: apply_pauli(0, f['h1'])
        gf = {}
        for gi, s, p in f.get('gates', []): gf.setdefault(gi, []).append((s, p))
        for gi, (g, q) in enumerate(rounds[i]):
            if g == 'CCX': cols[q[2]] ^= cols[q[0]] & cols[q[1]]
            elif g == 'CX': cols[q[1]] ^= cols[q[0]]
            elif g == 'X': cols[q[0]] ^= ALL
            elif g == 'SWAP': cols[q[0]], cols[q[1]] = cols[q[1]], cols[q[0]]
            else: raise ValueError(g)
            for s, p in gf.get(gi, []): apply_pauli(q[s], p)
        k2 = keys_from_cols(cols, S2)
        sb = np.array([(sign >> j) & 1 for j in range(S2)])
        a2 = a2 * np.where(sb == 1, -1.0, 1.0)
        # phase correction on control=1: Phase(phi), phi = -pi * y / 2^i
        phi = -math.pi * y / (1 << i) if y else 0.0
        e = cmath.exp(1j * phi)
        u = {}  # rest -> [u0, u1]
        for k, am in zip(k2, a2):
            c = k & 1; rest = k >> 1
            v = u.setdefault(rest, [0j, 0j])
            v[c] += am * (e if c else 1)
        def ctrl_pauli(v, p):
            if p in 'ZY': v[1] = -v[1]
            if p in 'XY': v[0], v[1] = v[1], v[0]
        if 'phase' in f:
            for v in u.values(): ctrl_pauli(v, f['phase'])
        o = {}
        for rest, (u0, u1) in u.items():
            v = [(u0 + u1) / math.sqrt(2), (u0 - u1) / math.sqrt(2)]
            if 'h2' in f: ctrl_pauli(v, f['h2'])
            o[rest] = v
        p1 = sum(abs(v[1]) ** 2 for v in o.values())
        bit = 1 if random.random() < p1 else 0
        if bit ^ (1 if f.get('meas') else 0): y |= 1 << i
        nk, na = [], []
        for rest, v in o.items():
            if abs(v[bit]) > 1e-12: nk.append(rest << 1); na.append(v[bit])
        na = np.array(na); na /= math.sqrt(sum(abs(na) ** 2))
        keys, amps = nk, na
    return y, ms
random.seed(seed)
print(f"# N={N} a={a} r={r} t={t} nq={nq} kind={kind} L={L} k={K} M={M}")
res = []
for j in range(M):
    fs = sample_faults(K)
    y, ms = trajectory(fs)
    res.append((y, ms))
    desc = ';'.join(f"{i}:" + ','.join(f"{k}={v}" for k, v in d.items()) for i, d in enumerate(fs) if d)
    print(j, y, ms, int(y is not None and peak_ok(y)), desc.replace(' ', ''), flush=True)
