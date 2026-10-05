"""Rebuild the [[288,16,16]] code from the presentation alone (no GAP, no group table).

G = Z6 x K, K = <a, b, c | a^3 = b^4 = c^2 = 1, b^-1 a b = a^-1, c a c = a^-1, c b c = b^-1>
(= C3 : D8 = SmallGroup(24,8)), z generates Z6; elements z^i a^j b^k c^l.
A = {1, z^4 c, z^2 a b^3 c},  B = {1, z^4 a b^2, z^5 b};
X-check g: L{g x : x in A}, R{y g : y in B};  Z-check h: L{y^-1 h : y in B}, R{h x^-1 : x in A}.

usage: python code_288_from_presentation.py <out prefix>
Prints n, k (GF(2) ranks) and writes <prefix>_Zlogicals.txt / _Xlogicals.txt for mwlogical.c.
"""
import itertools
import sys

E = list(itertools.product(range(6), range(3), range(4), range(2)))
idx = {e: t for t, e in enumerate(E)}


def mul(x, y):
    i, j, k, l = x
    i2, j2, k2, l2 = y
    s = -1 if (k + l) % 2 else 1      # b^k c^l acts on a by inversion k + l times
    t = -1 if l % 2 else 1            # c^l acts on b by inversion l times
    return ((i + i2) % 6, (j + s * j2) % 3, (k + t * k2) % 4, (l + l2) % 2)


ONE = (0, 0, 0, 0)
inv = {x: next(y for y in E if mul(x, y) == ONE) for x in E}
# relations check
a, b, c = (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)
assert all(mul(mul(x, y), z) == mul(x, mul(y, z)) for x in E[::7] for y in E[::5] for z in E)
assert mul(mul(inv[b], a), b) == inv[a] and mul(mul(c, a), c) == inv[a] and mul(mul(c, b), c) == inv[b]
A = [ONE, (4, 0, 0, 1), (2, 1, 3, 1)]
B = [ONE, (4, 1, 2, 0), (5, 0, 1, 0)]
N = len(E)
xs = [[idx[mul(g, x)] for x in A] + [N + idx[mul(y, g)] for y in B] for g in E]
zs = [[idx[mul(inv[y], h)] for y in B] + [N + idx[mul(h, inv[x])] for x in A] for h in E]


def rank(rows):
    piv, r = {}, 0
    for s in rows:
        v = sum(1 << q for q in s)
        while v:
            h = v.bit_length() - 1
            if h in piv:
                v ^= piv[h]
            else:
                piv[h] = v
                r += 1
                break
    return r


commute = all(len(set(x) & set(z)) % 2 == 0 for x in xs for z in zs)
rx, rz = rank(xs), rank(zs)
print(f"n = {2 * N}, rank H_X = {rx}, rank H_Z = {rz}, k = {2 * N - rx - rz}, commute = {commute}")
pre = sys.argv[1]
for label, chk, oth in (("Z", xs, zs), ("X", zs, xs)):
    with open(f"{pre}_{label}logicals.txt", "w") as f:
        f.write(f"{2 * N} {N} {N}\n")
        for s in chk + oth:
            f.write(f"{len(s)} " + " ".join(map(str, s)) + "\n")
