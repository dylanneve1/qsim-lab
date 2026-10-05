"""Triple-check of a corner case in the paper's prime-set search
(facto/algorithm/prep/_precompute_rns.py): `prune` divides the candidate
L mod N by any chosen prime q dividing it, assuming (L/q) mod N == (L mod N)/q.
That holds iff gcd(q, N) = 1. At toy sizes (n <= 2*ell) a factor of N can be an
ell-bit prime; then the pruned set's true L mod N differs from the pruned
candidate. Uses the paper's own `prune`, `prod_mod` and search order (first
pair, in order); the paper's `_verify_rns_solution` then rejects the set."""
import math

import gidney_env  # noqa: F401
import sympy
from facto.algorithm.prep._precompute_rns import prod_mod, prune

N, g, ell, gap, nw1 = 16016003, 2, 12, 18, 5
max_product_bits = N.bit_length() * nw1
# EH multipliers (a: 16 qubits base g, b: 4 qubits base y^-1), window 4
y = pow(g, (N - 1) // 2, N)
yi = pow(y, -1, N)
mults = []
for reg_len, base in ((16, g), (4, yi)):
    for start in range(0, reg_len, 4):
        b = pow(base, 1 << start, N)
        mults += [pow(b, k, N) for k in range(16)]
avail = list(sympy.primerange(1 << (ell - 1), 1 << ell))
acc = [p for p in avail if all(f % p for f in mults)]
print("factors of N among the candidate primes:", [p for p in acc if N % p == 0])
big, sec = acc.pop(), acc.pop()
fixed = acc[0] * acc[1]
cont = []
while fixed.bit_length() <= max_product_bits:
    p = acc.pop()
    cont.append(p)
    fixed *= p
cont += [big, sec]
shifted = N >> gap
found = 0
for i, p1 in enumerate(acc):
    for p2 in acc[i + 1:]:
        choice = [p1, p2] + cont
        cand = prod_mod(choice, N)
        pc, kept = prune(cand, choice, N)
        if pc < shifted:
            true = math.prod(kept) % N
            dev = min(true, N - true)
            bad = [q for q in choice if q not in kept and N % q == 0]
            print(f"pair ({p1},{p2}): pruned candidate {pc} < N>>gap={shifted}; true L mod N deviation {dev}; pruned factors of N: {bad}")
            found += 1
            if found >= 3:
                raise SystemExit
