"""Place a code [[n, k, d]] against the known weight-6 frontier.

usage: python compare_code.py n k d

Prints: T(n, k) (the best d of any known code or direct sum with n' <= n,
k' >= k), the published / known codes it strictly dominates (n <= n',
k >= k', d >= d', one strict), the best published k d^2/n at n' <= n, and
the known codes at the same n. Uses the same sources as known_codes.py.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv, args = sys.argv[:1], sys.argv[1:]
src = open(os.path.join(HERE, "known_codes.py")).read().replace("\nmain()\n", "\n")
exec(src)
n, k, d = map(int, args)
text = open(os.path.join(CD, "literature.md")).read()
pub = sorted(set(published()))
mine = sorted(set(ours()))
reach = closure(sorted(set(pub + mine)))
t = 0
for n2, v in reach.items():
    if n2 <= n:
        for k2, d2 in v:
            if k2 >= k and d2 > t:
                t = d2
print(f"[[{n},{k},{d}]]: k d^2/n = {k*d*d/n:.2f}; T(n,k) = {t} -> {'NOT dominated' if d > t else 'dominated or tied'}")
dom = [(a, b, c) for a, b, c in pub if n <= a and k >= b and d >= c and (n, k, d) != (a, b, c)]
print("published codes it dominates:", dom[:12], "..." if len(dom) > 12 else "")
best = max(((b * c * c / a, (a, b, c)) for a, b, c in pub if a <= n), default=None)
print("best published k d^2/n at n' <= n:", f"{best[0]:.2f} {best[1]}" if best else None)
print("published at same n:", [x for x in pub if x[0] == n])
print("previous-study codes at same n:", [x for x in mine if x[0] == n])
