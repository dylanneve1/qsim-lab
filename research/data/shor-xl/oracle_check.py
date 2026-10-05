#!/usr/bin/env python3
"""Independent check of full-size window blocks (written from scratch for this study; numpy bit
operations on multi-limb basis states, no code shared with the engine).

`ge_shor dump ...` prints one resolved window block of the run (the same outcome stream as the run):
X / CX / CCX gates, Z / CZ phase fix-ups, MX q m (X-basis measurement of a qubit that must be a
deterministic function of the others; outcome m multiplies the branch by (-1)^(m * value) and
resets the qubit), GNEG (a global -1). For random valid inputs |e>|x>|0...> (e < 2^w, x < N) this
script applies every op and checks: e unchanged, x -> g^e x mod N, every other qubit 0, and every
branch carrying the same sign (the program's global sign), i.e. the measurement phases cancel.

usage: oracle_check.py <ge_shor binary> <N> <seed> <we> <wm> <mbu> <var> <window> [inputs]"""
import random
import subprocess
import sys

import numpy as np


def load(argv):
    out = subprocess.run([argv[0], "dump", *argv[1:]], capture_output=True, text=True, check=True).stdout
    lines = out.splitlines()
    h = lines[0].split()
    hd = {h[i]: h[i + 1] for i in range(0, len(h), 2)}
    ops = []
    for l in lines[1:]:
        p = l.split()
        ops.append((p[0], *map(int, p[1:])))
    return hd, ops


def main():
    binary, n_mod, seed, we, wm, mbu, var, window = sys.argv[1:9]
    count = int(sys.argv[9]) if len(sys.argv) > 9 else 2000
    hd, ops = load([binary, n_mod, seed, we, wm, mbu, var, window])
    nq, N, g, w = int(hd["nq"]), int(hd["N"]), int(hd["g"]), int(hd["w"])
    eq = [int(q) for q in hd["e"].split(",")]
    xq = [int(q) for q in hd["x"].split(",")]
    rnd = random.Random(int(seed) * 1000 + int(window))
    ins = [(rnd.randrange(1 << w), rnd.randrange(N)) for _ in range(count)]
    ins += [(e, x) for e in range(1 << w) for x in (0, 1, N - 1)]
    K, limbs = len(ins), (nq + 63) // 64
    st = np.zeros((limbs, K), dtype=np.uint64)
    for j, (e, x) in enumerate(ins):
        v = 0
        for i, q in enumerate(eq):
            v |= ((e >> i) & 1) << q
        for i, q in enumerate(xq):
            v |= ((x >> i) & 1) << q
        for l in range(limbs):
            st[l, j] = (v >> (64 * l)) & (2**64 - 1)
    sign = np.zeros(K, dtype=np.uint64)
    gneg = 0
    one = np.uint64(1)

    def bit(q):
        return (st[q // 64] >> np.uint64(q % 64)) & one

    def flip(t, m):
        st[t // 64] ^= m << np.uint64(t % 64)

    counts = {}
    for op in ops:
        k = op[0]
        counts[k] = counts.get(k, 0) + 1
        if k == "X":
            flip(op[1], np.full(K, 1, dtype=np.uint64))
        elif k == "CX":
            flip(op[2], bit(op[1]))
        elif k == "CCX":
            flip(op[3], bit(op[1]) & bit(op[2]))
        elif k == "SWAP":
            a, b = op[1], op[2]
            d = bit(a) ^ bit(b)
            flip(a, d)
            flip(b, d)
        elif k == "Z":
            sign ^= bit(op[1])
        elif k == "CZ":
            sign ^= bit(op[1]) & bit(op[2])
        elif k == "MX":
            q, m = op[1], op[2]
            if m:
                sign ^= bit(q)
            flip(q, bit(q))
        elif k == "GNEG":
            gneg ^= 1
        else:
            raise SystemExit("unknown op " + k)
    reg_e = set(eq)
    reg_x = set(xq)
    bad = 0
    for j, (e, x) in enumerate(ins):
        v = sum(int(st[l, j]) << (64 * l) for l in range(limbs))
        eo = sum(((v >> q) & 1) << i for i, q in enumerate(eq))
        xo = sum(((v >> q) & 1) << i for i, q in enumerate(xq))
        anc = v
        for q in reg_e | reg_x:
            anc &= ~(1 << q)
        want = pow(g, e, N) * x % N
        if eo != e or xo != want or anc != 0 or int(sign[j]) != gneg:
            bad += 1
            if bad <= 5:
                print(f"MISMATCH e={e} x={x}: e'={eo} x'={xo} want {want} ancillas {anc:#x} sign {int(sign[j])}/{gneg}")
    status = "OK" if bad == 0 else f"FAIL ({bad} bad)"
    print(f"{status} N={N} window={window} w={w} g={g} qubits={nq} ops={len(ops)} "
          f"({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}) inputs={K}: "
          f"e unchanged, x -> g^e x mod N, ancillas 0, uniform sign", flush=True)
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
