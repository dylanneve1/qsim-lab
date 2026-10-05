#!/usr/bin/env python3
"""Instances used in research/shor/shor.md, round 4.

generic: first balanced semiprime per bit size from random.seed(1)
         (not selected on lambda or on the order of any base).
special: N = p(2p-1) with p, 2p-1 prime (row (c) family), random.seed(11),
         one call per bit size as listed (classically trivial N; cost-law demo).
"""
import math
import random


def isprime(n):
    if n < 2:
        return False
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def lcm(a, b):
    return a * b // math.gcd(a, b)


def generic():
    random.seed(1)
    for bits in range(22, 33):
        while True:
            h = bits // 2
            p = random.randrange(2 ** (h - 1), 2**h) | 1
            q = random.randrange(2 ** (bits - h - 1), 2 ** (bits - h)) | 1
            if p != q and isprime(p) and isprime(q) and (p * q).bit_length() == bits:
                break
        print("generic", bits, p * q, p, q, "lambda", lcm(p - 1, q - 1))


def special(bit_sizes):
    random.seed(11)
    for bits in bit_sizes:
        lo = math.isqrt(2 ** (bits - 1) // 2) + 1
        hi = math.isqrt(2**bits // 2)
        while True:
            p = random.randrange(lo, hi) | 1
            if isprime(p) and isprime(2 * p - 1) and (p * (2 * p - 1)).bit_length() == bits:
                break
        print("special", bits, p * (2 * p - 1), p, 2 * p - 1, "lambda", 2 * (p - 1))


if __name__ == "__main__":
    generic()
    special([46, 48, 50])
    special([52, 54])
