#!/usr/bin/env python3
"""Predicted work of exact branch tracking for the semiclassical gate-level
Shor circuit (research/shor/shor.md, round 4).

Before round i (t = 2n rounds) the work register is supported on
{a^(m 2^(t-i)) : 0 <= m < 2^i}, of size B_i = min(2^i, r / gcd(r, 2^(t-i))).
Each round evaluates the controlled-U circuit (G_i gates) on 2 B_i basis
states (control 0 and 1), so W = sum_i 2 B_i G_i gate applications.

usage: cost_law.py N r [G_total]   (G_total = total_gates printed by qsim;
       used as G_i ~ G_total / t)
"""
import sys


def bounds(n_bits, r):
    t = 2 * n_bits
    nu = (r & -r).bit_length() - 1
    out = []
    for i in range(t):
        g = 1 << min(nu, t - i)
        out.append(min(1 << i, r // g))
    return out


def main():
    N, r = int(sys.argv[1]), int(sys.argv[2])
    n = (N - 1).bit_length()
    b = bounds(n, r)
    s = sum(b)
    nu = (r & -r).bit_length() - 1
    print(f"N={N} n={n} t={2*n} r={r} nu2(r)={nu} r_odd={r >> nu} sum_B={s} peak_B={max(b)}")
    if len(sys.argv) > 3:
        g = int(sys.argv[3]) / (2 * n)
        print(f"G_avg={g:.0f}  W_pred={2 * s * g:.4e}")


if __name__ == "__main__":
    main()
