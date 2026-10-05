#!/usr/bin/env python3
"""The repo's seeded 'generic' instance generator (research/data/shor_r4/gen_instances.py,
generic(): random.seed(1), first balanced semiprime per bit size), with the loop extended
past 32 bits. The RNG stream is sequential, so the 22..32-bit N are unchanged.

Prints: bits N p q lambda lambda_odd nu2(lambda). The factors are printed for the write-up
and for memory-safety estimates; the circuit runs never read them."""
import math, random, sys

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "shor_r4"))
from gen_instances import isprime, lcm  # noqa: E402


def generic(lo=22, hi=64):
    random.seed(1)
    out = []
    for bits in range(22, hi + 1):
        while True:
            h = bits // 2
            p = random.randrange(2 ** (h - 1), 2**h) | 1
            q = random.randrange(2 ** (bits - h - 1), 2 ** (bits - h)) | 1
            if p != q and isprime(p) and isprime(q) and (p * q).bit_length() == bits:
                break
        if bits >= lo:
            lam = lcm(p - 1, q - 1)
            v = (lam & -lam).bit_length() - 1
            out.append((bits, p * q, p, q, lam, lam >> v, v))
    return out


if __name__ == "__main__":
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 22
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    for row in generic(lo, hi):
        print(*row)
