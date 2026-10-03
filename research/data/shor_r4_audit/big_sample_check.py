"""Spot check of the full-size circuits with the independent interpreter of
perm_check.py: the 31-bit record's controlled-U for a, a^2, a^(2^61)
(N = 1537596787, a = 457167243, w = 4; 132 qubits) on 4000 random valid inputs
x < N per control value; and the 52-bit special N's (216 qubits)."""
import sys, random, numpy as np
from perm_check import dump
EX = sys.argv[1]
random.seed(99)
def run_big(keys, gates):
    # keys: python ints (>64 qubits) -> use object arrays via two uint64 limbs... simpler: list of 4 uint64 limbs
    L = 4
    k = np.zeros((L, len(keys)), dtype=np.uint64)
    for j, v in enumerate(keys):
        for l in range(L): k[l, j] = (v >> (64 * l)) & ((1 << 64) - 1)
    one = np.uint64(1)
    def bit(q): return ((k[q // 64] >> np.uint64(q % 64)) & one).astype(bool)
    for t, c1, c2 in gates:
        m = np.ones(len(keys), dtype=bool)
        if c1 >= 0: m &= bit(c1)
        if c2 >= 0: m &= bit(c2)
        k[t // 64, m] ^= one << np.uint64(t % 64)
    return [sum(int(k[l, j]) << (64 * l) for l in range(L)) for j in range(len(keys))]
for N, a in [(1537596787, 457167243), (3384163410217561, None)]:
    if a is None:
        a = random.randrange(2, N - 1)
    n = N.bit_length(); t = 2 * n
    for k in (0, 1, t - 1):
        m = pow(a, 1 << k, N)
        nq, n2, gates = dump(N, m, 4)
        xs = [random.randrange(N) for _ in range(2000)] + [0, 1, N - 1]
        ins = [c | (x << 1) for c in (0, 1) for x in xs]
        outs = run_big(ins, gates)
        for kin, ko in zip(ins, outs):
            c, x = kin & 1, kin >> 1
            want = c | (((m * x % N) if c else x) << 1)
            assert ko == want, (N, m, c, x, hex(ko), hex(want))
        print(f"OK N={N} ({n} b) mult=a^(2^{k}) qubits={nq} gates={len(gates)} inputs={len(ins)}: out = mult*x mod N (c=1) / x (c=0), ancillas clean", flush=True)
