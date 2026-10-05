#!/usr/bin/env python3
"""Post-hoc (and memory-safety) facts about the base rule's g = h^(2^n) for generator instances:
ord(g) (odd), from the factorisation of lambda. Uses the factors: write-up / RAM estimates only.
usage: orders.py <ge_shor binary> <seed> [N ...]   (default: the generic N of 30..48 bits and N_W)"""
import subprocess, sys
from gen_generic import generic


def factorize(n):
    f, d = {}, 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def order(g, n, lam, primes):
    r = lam
    for p in primes:
        while r % p == 0 and pow(g, r // p, n) == 1:
            r //= p
    assert pow(g, r, n) == 1
    return r


def info(binary, n, seed):
    out = subprocess.run([binary, "info", str(n), str(seed)], capture_output=True, text=True, check=True).stdout
    kv = dict(t.split("=", 1) for t in out.split() if "=" in t)
    return int(kv["h"]), int(kv["g"].rsplit("=", 1)[1])


if __name__ == "__main__":
    binary, seed = sys.argv[1], int(sys.argv[2])
    lo, hi = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (30, 48)
    rows = [(b, n, p, q) for (b, n, p, q, *_r) in generic(lo, hi)] + [(39, 549755813701, 712321, 771781)]
    for b, n, p, q in rows:
        from math import gcd
        lam = (p - 1) * (q - 1) // gcd(p - 1, q - 1)
        primes = set(factorize(p - 1)) | set(factorize(q - 1))
        h, g = info(binary, n, seed)
        r_h = order(h, n, lam, primes)
        r_g = order(g, n, lam, primes)
        v = (r_h & -r_h).bit_length() - 1
        print(f"bits={b} N={n} seed={seed} h={h} ord(h)={r_h} (nu2={v}) g={g} ord(g)={r_g} "
              f"lambda_odd={lam >> ((lam & -lam).bit_length() - 1)} ord(g)/lambda_odd=1/{(lam >> ((lam & -lam).bit_length() - 1)) // r_g}")
