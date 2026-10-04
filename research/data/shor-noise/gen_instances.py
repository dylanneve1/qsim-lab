#!/usr/bin/env python3
"""Instances for research/shor-noise.md.

N: first balanced semiprime per bit size from random.seed(1) (same generator
as research/data/shor_r4/gen_instances.py, extended down to 10 bits).
a: first base from random.seed(1000 + bits) with gcd(a, N) = 1, r even and
a^(r/2) != -1 mod N ("good" base: the noiseless run factors N whenever it
finds r) and r >= lambda(N)/8 ("typical" order: excludes the rare small-order
bases, whose phase estimate has many spare low bits and for which the
classical multiples search in shor::postprocess alone recovers r).
"""
import math, random, sys
sys.path.insert(0, '../shor_r4')
from gen_instances import isprime, lcm

def order(a, n, lam):
    r = lam
    for p in range(2, 100000):
        if p * p > r and r > 1 and isprime(r):
            if pow(a, r // r, n) == 1 and r != 1:
                pass
        while r % p == 0 and pow(a, r // p, n) == 1:
            r //= p
    # remaining large prime factor of lam
    for q in set(factor(lam)):
        while r % q == 0 and pow(a, r // q, n) == 1:
            r //= q
    return r

def factor(m):
    out, p = [], 2
    while p * p <= m:
        while m % p == 0:
            out.append(p); m //= p
        p += 1
    if m > 1: out.append(m)
    return out

random.seed(1)
rows = []
print('# bits N p q a r nu2(r) log2(r) lambda slack=t-2log2(r)')
for bits in range(10, 25):
    while True:
        h = bits // 2
        p = random.randrange(2 ** (h - 1), 2**h) | 1
        q = random.randrange(2 ** (bits - h - 1), 2 ** (bits - h)) | 1
        if p != q and isprime(p) and isprime(q) and (p * q).bit_length() == bits:
            break
    rows.append((bits, p, q))
for bits, p, q in rows:
    n = p * q
    lam = lcm(p - 1, q - 1)
    rng = random.Random(1000 + bits)
    while True:
        a = rng.randrange(2, n - 1)
        if math.gcd(a, n) != 1: continue
        r = order(a, n, lam)
        assert pow(a, r, n) == 1
        if r % 2 == 0 and pow(a, r // 2, n) != n - 1 and 8 * r >= lam:
            break
    nu = (r & -r).bit_length() - 1
    print(bits, n, p, q, a, r, nu, f"{math.log2(r):.2f}", lam, f"{2*bits-2*math.log2(r):.2f}")
