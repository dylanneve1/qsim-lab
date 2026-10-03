"""Independent audit: brute-force permutation of the windowed controlled-U_a
circuit, with a minimal gate interpreter written from scratch (numpy bit ops on
basis-state indices). Gate list comes from `examples/audit_shor_r4 dump`.

mode 'all'  : every basis input of the whole register (2^nq, nq <= 25);
              checks (1) bijection on all 2^nq inputs, (2) on the valid domain
              (ancillas 0, x < N): ctrl=1 -> a x mod N, ctrl=0 -> x, ancillas clean.
mode 'valid': only the valid domain (any nq <= 63)."""
import subprocess, sys, numpy as np
EX = sys.argv[1]

def dump(N, a, w):
    out = subprocess.run([EX, "dump", str(N), str(a), str(w)], capture_output=True, text=True, check=True).stdout.split("\n")
    nq, n = int(out[0].split()[1]), int(out[0].split()[3])
    gates = []
    for l in out[1:]:
        if not l: continue
        p = l.split()
        if p[0] == "X": gates.append((int(p[1]), -1, -1))
        elif p[0] == "CX": gates.append((int(p[2]), int(p[1]), -1))
        elif p[0] == "CCX": gates.append((int(p[3]), int(p[1]), int(p[2])))
        else: raise SystemExit("non-permutation gate: " + l)
    return nq, n, gates

def run(keys, gates):
    k = keys.copy()
    one = np.uint64(1)
    for t, c1, c2 in gates:
        m = np.ones_like(k, dtype=bool)
        if c1 >= 0: m &= ((k >> np.uint64(c1)) & one).astype(bool)
        if c2 >= 0: m &= ((k >> np.uint64(c2)) & one).astype(bool)
        k[m] ^= (one << np.uint64(t))
    return k

def check(N, a, w, mode):
    nq, n, gates = dump(N, a, w)
    assert nq == 4 * n + 4 + min(w, n)
    if mode == "all":
        keys = np.arange(1 << nq, dtype=np.uint64)
        out = run(keys, gates)
        assert len(np.unique(out)) == len(out), "not a bijection"
        valid_in = np.array([c | (x << 1) for c in (0, 1) for x in range(N)], dtype=np.uint64)
        vo = out[valid_in.astype(np.int64)]
    else:
        valid_in = np.array([c | (x << 1) for c in (0, 1) for x in range(N)], dtype=np.uint64)
        vo = run(valid_in, gates)
    for idx, kin in enumerate(valid_in.tolist()):
        c, x = kin & 1, kin >> 1
        want = (a * x % N) if c else x
        exp = c | (want << 1)
        if int(vo[idx]) != exp:
            raise SystemExit(f"MISMATCH N={N} a={a} w={w} c={c} x={x}: got {int(vo[idx]):#x} want {exp:#x}")
    return nq, len(gates)

if __name__ == "__main__":
    from math import gcd
    import random
    random.seed(12345)
    only_valid = len(sys.argv) > 2 and sys.argv[2] == "valid"
    cases = [] if only_valid else [(15, a, w, "all") for a in (2, 7, 11, 13) for w in (1, 2, 3, 4)]
    cases += [(21, a, w, "valid") for a in (2, 5, 8, 13) for w in (1, 2, 3, 4, 5)]
    # valid domain, n = 5..8 bits, random bases
    for N in (33, 35, 51, 91, 143, 187, 221, 247, 253):
        for _ in range(2):
            a = random.randrange(2, N - 1)
            while gcd(a, N) != 1: a = random.randrange(2, N - 1)
            for w in (1, 2, 3, 4, 5):
                cases.append((N, a, w, "valid"))
    for N, a, w, mode in cases:
        nq, g = check(N, a, w, mode)
        print(f"OK N={N} a={a} w={w} mode={mode} qubits={nq} gates={g}", flush=True)
    print("ALL OK", len(cases), "cases")
