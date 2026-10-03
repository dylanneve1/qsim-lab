"""Independent audit: exact measured-outcome distribution of the sliced
windowed semiclassical run (Rust, `audit_shor_r4 dist`) vs
 (A) the textbook phase-estimation distribution computed here from scratch:
     P(y) = sum_z | 2^-t sum_{x<2^t, a^x = z} exp(2 pi i x y / 2^t) |^2 ;
 (B) for N = 15, a dense state-vector simulation written here from scratch of
     the WHOLE gate-level register (4n+4+w qubits): H / controlled-U (as the
     permutation obtained by interpreting the dumped X/CNOT/CCX list on every
     basis index) / phase correction / H on qubit 0, measure + reset, walking
     the whole measurement tree. No code shared with the Rust simulator."""
import subprocess, sys, numpy as np
from math import gcd
EX = sys.argv[1]
which = sys.argv[2] if len(sys.argv) > 2 else "A"

def rust_dist(N, a, w):
    out = subprocess.run([EX, "dist", str(N), str(a), str(w)], capture_output=True, text=True, check=True).stdout.split("\n")
    t = int(out[0].split()[1])
    p = np.zeros(1 << t)
    for l in out[1:]:
        if l:
            y, v = l.split(); p[int(y)] = float(v)
    return t, p

def textbook(N, a, t):
    T = 1 << t
    z = np.empty(T, dtype=np.int64); cur = 1
    for i in range(T):
        z[i] = cur; cur = cur * a % N
    P = np.zeros(T)
    for zz in np.unique(z):
        ind = (z == zz).astype(np.complex128)
        P += np.abs(np.fft.fft(ind) / T) ** 2   # |2^-t sum_{x: a^x=z} e^{-2 pi i x y/2^t}|^2
    return P

def perm_of(N, mult, w):
    out = subprocess.run([EX, "dump", str(N), str(mult), str(w)], capture_output=True, text=True, check=True).stdout.split("\n")
    nq = int(out[0].split()[1])
    k = np.arange(1 << nq, dtype=np.uint64); one = np.uint64(1)
    for l in out[1:]:
        if not l: continue
        p = l.split()
        if p[0] == "X": t, cs = int(p[1]), []
        elif p[0] == "CX": t, cs = int(p[2]), [int(p[1])]
        elif p[0] == "CCX": t, cs = int(p[3]), [int(p[1]), int(p[2])]
        else: raise SystemExit("non-permutation gate " + l)
        m = np.ones(k.shape, dtype=bool)
        for c in cs: m &= ((k >> np.uint64(c)) & one).astype(bool)
        k[m] ^= one << np.uint64(t)
    return nq, k.astype(np.int64)   # basis |i> -> |k[i]>

def dense_tree(N, a, w):
    n = N.bit_length(); t = 2 * n
    mults = [pow(a, 1 << k, N) for k in range(t)]
    perms = {}
    for m in set(mults):
        nq, perms[m] = perm_of(N, m, w)
    dim = 1 << nq
    idx = np.arange(dim)
    out = np.zeros(1 << t)
    s = np.zeros(dim, dtype=np.complex128); s[1 << 1] = 1.0   # |ctrl=0>|x=1>|0...>
    def H0(v):
        v0 = v[0::2].copy(); v1 = v[1::2].copy()
        r = np.empty_like(v); r[0::2] = (v0 + v1) / np.sqrt(2); r[1::2] = (v0 - v1) / np.sqrt(2); return r
    def walk(v, i, y, p):
        if i == t:
            out[y] = p; return
        m = mults[t - 1 - i]
        v = H0(v)
        nv = np.zeros_like(v); nv[perms[m]] = v; v = nv           # apply permutation
        phi = -np.pi * y / (1 << i)
        v[1::2] *= np.exp(1j * phi)
        v = H0(v)
        p1 = float(np.sum(np.abs(v[1::2]) ** 2))
        for bit, pb in ((0, 1 - p1), (1, p1)):
            if pb <= 1e-15: continue
            c = np.zeros_like(v)
            # collapse on qubit0 = bit, then reset qubit 0 to |0>
            c[0::2] = v[bit::2] / np.sqrt(pb)
            walk(c, i + 1, y | (bit << i), p * pb)
    walk(s, 0, 0, 1.0)
    return out

if __name__ == "__main__":
    import random
    random.seed(777)
    if which == "A":
        cases = [(15, a, w) for a in (2, 4, 7, 11, 13, 14) for w in (1, 2, 3, 4)]
        cases += [(21, a, w) for a in (2, 5, 8, 13) for w in (2, 4)]
        for N in (35, 39, 51, 55, 77, 91, 95, 119, 143, 187, 209, 221, 247, 253):
            for _ in range(2):
                a = random.randrange(2, N - 1)
                while gcd(a, N) != 1: a = random.randrange(2, N - 1)
                cases.append((N, a, random.choice((1, 2, 3, 4))))
        worst = 0.0
        for N, a, w in cases:
            t, pr = rust_dist(N, a, w)
            pt = textbook(N, a, t)
            d = np.abs(pr - pt).max()
            # also check against bit-reversed y to make sure the comparison is discriminating
            worst = max(worst, d)
            print(f"N={N} a={a} w={w} t={t} sum={pr.sum():.15f} max|dp|={d:.3e} support={np.count_nonzero(pr>1e-14)}", flush=True)
        print("WORST max|dp| vs textbook:", worst)
    else:
        worst = 0.0
        for a, w in [(7, 2), (2, 1), (11, 3), (13, 3), (4, 2)]:
            t, pr = rust_dist(15, a, w)
            pd = dense_tree(15, a, w)
            pt = textbook(15, a, t)
            d1 = np.abs(pr - pd).max(); d2 = np.abs(pd - pt).max()
            worst = max(worst, d1)
            print(f"N=15 a={a} w={w}: sliced vs my dense full-register SV max|dp|={d1:.3e}; dense vs textbook {d2:.3e}", flush=True)
        print("WORST sliced vs dense:", worst)
