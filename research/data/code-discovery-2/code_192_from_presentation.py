"""Rebuild the [[192,12,14]] code from its presentation alone (no GAP, no table).

G = C3 : P, P = Q8 : <t>:  a^3 = 1, Q8 = <i, j> (i^2 = j^2 = (ij)^2 = -1), t^4 = 1, <t> meets Q8 trivially,
t^-1 i t = i^-1,  t^-1 j t = i j;  i and t centralise a, j inverts a  (= SmallGroup(96,17)).
A = {1, j t^2, a t},  B = {1, a i, a^2 i^-1 t^3};  checks as in code_288_from_presentation.py.

usage: python code_192_from_presentation.py <out prefix>
"""
import itertools
import sys

# quaternion units as (sign, unit) with unit in 1, i, j, k
QM = {("1", u): (1, u) for u in "1ijk"}
QM.update({(u, "1"): (1, u) for u in "1ijk"})
QM.update({("i", "i"): (-1, "1"), ("j", "j"): (-1, "1"), ("k", "k"): (-1, "1"),
           ("i", "j"): (1, "k"), ("j", "k"): (1, "i"), ("k", "i"): (1, "j"),
           ("j", "i"): (-1, "k"), ("k", "j"): (-1, "i"), ("i", "k"): (-1, "j")})


def qmul(p, q):
    s, u = QM[(p[1], q[1])]
    return (p[0] * q[0] * s, u)


Q8 = [(s, u) for s in (1, -1) for u in "1ijk"]
I, J = (1, "i"), (1, "j")
QINV = {q: next(r for r in Q8 if qmul(q, r) == (1, "1")) for q in Q8}


def phi(q):
    """t q t^-1 (= t^-1 q t, phi has order 2): i -> i^-1, j -> i j."""
    img = {"1": (1, "1"), "i": QINV[I], "j": qmul(I, J)}
    img["k"] = qmul(img["i"], img["j"])
    s, u = q
    return (s * img[u][0], img[u][1])


def chi(q):
    """action of q on a: i centralises, j (and k = ij) inverts."""
    return -1 if q[1] in "jk" else 1


E = [(e, q, h) for e in range(3) for q in Q8 for h in range(4)]
idx = {x: n for n, x in enumerate(E)}


def mul(x, y):
    e, q, h = x
    e2, q2, h2 = y
    q2t = q2
    for _ in range(h % 2):
        q2t = phi(q2t)
    return ((e + chi(q) * e2) % 3, qmul(q, q2t), (h + h2) % 4)


ONE = (0, (1, "1"), 0)
inv = {x: next(y for y in E if mul(x, y) == ONE) for x in E}
a, i, j, t = (1, (1, "1"), 0), (0, I, 0), (0, J, 0), (0, (1, "1"), 1)
assert all(mul(mul(x, y), z) == mul(x, mul(y, z)) for x in E[::5] for y in E[::3] for z in E)
assert mul(mul(inv[t], i), t) == inv[i] and mul(mul(inv[t], j), t) == mul(i, j)
assert mul(mul(inv[j], a), j) == inv[a] and mul(mul(inv[i], a), i) == a and mul(mul(inv[t], a), t) == a
t2, t3 = mul(t, t), mul(mul(t, t), t)
A = [ONE, mul(j, t2), mul(a, t)]
B = [ONE, mul(a, i), mul(mul(mul(a, a), inv[i]), t3)]
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
