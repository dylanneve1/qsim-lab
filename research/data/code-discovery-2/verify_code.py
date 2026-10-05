"""Independent check of a two-block group-algebra code's [[n, k, d]].

usage: python verify_code.py <groups.txt> <gap id> <A> <B> <d> [--witness-z q1,q2,...]
                             [--witness-x q1,q2,...] [--subgroup h1,h2,...] [--no-lower]

Written from scratch (no code shared with qsim_lab): reads the group table
exported by export_groups.g, builds H_X / H_Z in the convention of
src/qec/group_algebra.rs

    X-check g: L{g a : a in A}, R{b g : b in B}
    Z-check h: L{b^-1 h : b in B}, R{h a^-1 : a in A}

With --subgroup H (a coset code, src/qec/group_algebra.rs `CosetCode`) the
qubits and checks are the right cosets Hx (numbered by first appearance over
x = 0, 1, ...) and

    X-check Hg: L{H g a}, R{H b g},   Z-check Hh: L{H b^-1 h}, R{H h a^-1}

(B must normalise H; checked). Then it checks
  1. every X-check meets every Z-check evenly;
  2. k = n - rank H_X - rank H_Z (Gaussian elimination on Python ints);
  3. each witness is a nontrivial logical of weight d (Z-type witness: in
     ker H_X and not in rowspace H_Z; X-type symmetric);
  4. (unless --no-lower) no nontrivial logical of weight < d exists in either
     sector, by an exhaustive search over connected clusters rooted at EVERY
     qubit (earlier roots banned; no symmetry used): a minimum-weight
     nontrivial logical E has no proper non-empty subset in ker H, so from any
     root in E it is reached by repeatedly adding a qubit of an unsatisfied
     check.
Prints one JSON line.
"""
import json
import sys
import time


def load_group(path, gid):
    lines = [l for l in open(path).read().splitlines() if l.strip()]
    i = 0
    while i < len(lines):
        f = lines[i].split()
        n, ident, ngens = int(f[1]), int(f[2]), int(f[5])
        if ident == gid:
            table = [list(map(int, lines[i + 1 + r].split())) for r in range(n)]
            return n, table, " ".join(f[6:])
        i += 1 + n + ngens
    raise SystemExit(f"group id {gid} not in {path}")


def rank(rows):
    rows = [r for r in rows if r]
    r = 0
    piv = {}
    for v in rows:
        while v:
            h = v.bit_length() - 1
            if h in piv:
                v ^= piv[h]
            else:
                piv[h] = v
                r += 1
                break
    return r


def in_span(v, rows):
    basis = {}
    for w in rows:
        while w:
            h = w.bit_length() - 1
            if h in basis:
                w ^= basis[h]
            else:
                basis[h] = w
                break
    while v:
        h = v.bit_length() - 1
        if h not in basis:
            return False
        v ^= basis[h]
    return True


def main():
    a = sys.argv[1:]
    path, gid = a[0], int(a[1])
    A = [int(x) for x in a[2].split(",")]
    B = [int(x) for x in a[3].split(",")]
    d = int(a[4])
    opts = a[5:]
    wz = wx = None
    if "--witness-z" in opts:
        wz = [int(x) for x in opts[opts.index("--witness-z") + 1].split(",")]
    if "--witness-x" in opts:
        wx = [int(x) for x in opts[opts.index("--witness-x") + 1].split(",")]
    M, T, name = load_group(path, gid)
    inv = [next(h for h in range(M) if T[g][h] == 0) for g in range(M)]
    H = [0]
    if "--subgroup" in opts:
        H = [int(x) for x in opts[opts.index("--subgroup") + 1].split(",")]
    Hs = set(H)
    assert 0 in Hs and all(T[x][y] in Hs for x in H for y in H), "H is not a subgroup"
    cos = [-1] * M
    reps = []
    for x in range(M):
        if cos[x] < 0:
            for h in H:
                cos[T[h][x]] = len(reps)
            reps.append(x)
    for y in B:
        assert all(T[T[y][h]][inv[y]] in Hs for h in H), "B does not normalise H"
    N = len(reps)
    n = 2 * N
    xs = [[cos[T[g][x]] for x in A] + [N + cos[T[y][g]] for y in B] for g in reps]
    zs = [[cos[T[inv[y]][h]] for y in B] + [N + cos[T[h][inv[x]]] for x in A] for h in reps]
    for s in xs + zs:
        assert len(set(s)) == len(s), "repeated qubit in a check"
    tomask = lambda s: sum(1 << q for q in s)
    HX = [tomask(s) for s in xs]
    HZ = [tomask(s) for s in zs]
    comm = all(bin(x & z).count("1") % 2 == 0 for x in HX for z in HZ)
    rx, rz = rank(HX), rank(HZ)
    k = n - rx - rz
    out = {"group": name, "subgroup_order": len(H), "n": n, "k": k, "commute": comm, "rank_hx": rx, "rank_hz": rz}

    def logical_ok(w, hcheck, hother):
        v = tomask(w)
        syn0 = all(bin(v & h).count("1") % 2 == 0 for h in hcheck)
        return syn0 and not in_span(v, hother)

    if wz is not None:
        out["witness_z"] = {"weight": len(set(wz)), "nontrivial": logical_ok(wz, HX, HZ)}
    if wx is not None:
        out["witness_x"] = {"weight": len(set(wx)), "nontrivial": logical_ok(wx, HZ, HX)}
    if "--no-lower" not in opts:
        t0 = time.time()
        res = {}
        for label, checks, other in (("z", xs, HZ), ("x", zs, HX)):
            res[label] = no_logical_below(n, checks, other, d)
        out["lower"] = res
        out["lower_s"] = round(time.time() - t0, 1)
    print(json.dumps(out))


def no_logical_below(n, checks, other_rows, d):
    """True iff no nontrivial logical (in ker(checks), not in rowspace(other))
    of weight <= d - 1 exists. Returns {"ok": bool, "nodes": int, "found": [...]}."""
    nc = len(checks)
    qchecks = [[] for _ in range(n)]
    for c, s in enumerate(checks):
        for q in s:
            qchecks[q].append(c)
    qmask = [sum(1 << c for c in qchecks[q]) for q in range(n)]
    maxdeg = max(len(v) for v in qchecks)
    # nontriviality: e is trivial iff it lies in rowspace(other); test lazily
    basis = {}
    for w in other_rows:
        while w:
            h = w.bit_length() - 1
            if h in basis:
                w ^= basis[h]
            else:
                basis[h] = w
                break

    def trivial(v):
        while v:
            h = v.bit_length() - 1
            if h not in basis:
                return False
            v ^= basis[h]
        return True

    W = d - 1
    nodes = 0
    banned = [False] * n
    used = [False] * n
    chosen = []

    def dfs(syn, vec):
        nonlocal nodes
        nodes += 1
        if syn == 0:
            if not trivial(vec):
                return list(chosen)
            return None  # in ker H and trivial: a proper subset of no minimal logical
        unsat = bin(syn).count("1")
        if len(chosen) + (unsat + maxdeg - 1) // maxdeg > W:
            return None
        # check with fewest free qubits
        best, bestc = None, None
        s = syn
        while s:
            c = (s & -s).bit_length() - 1
            s &= s - 1
            free = [q for q in checks[c] if not banned[q] and not used[q]]
            if best is None or len(free) < len(best):
                best, bestc = free, c
                if len(free) <= 1:
                    break
        if not best:
            return None
        newly = []
        hit = None
        for q in best:
            used[q] = True
            chosen.append(q)
            r = dfs(syn ^ qmask[q], vec ^ (1 << q))
            chosen.pop()
            used[q] = False
            banned[q] = True
            newly.append(q)
            if r is not None:
                hit = r
                break
        for q in newly:
            banned[q] = False
        return hit

    if W <= 0:
        return {"ok": True, "nodes": 0}
    for root in range(n):
        for q in range(n):
            banned[q] = q < root
        used[root] = True
        chosen.append(root)
        r = dfs(qmask[root], 1 << root)
        chosen.pop()
        used[root] = False
        if r is not None:
            return {"ok": False, "nodes": nodes, "found": sorted(r)}
    return {"ok": True, "nodes": nodes}


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
