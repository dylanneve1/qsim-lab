#!/usr/bin/env python3
"""Symbolic Z-sector detector error model of the single-auxiliary triangular 6.6.6 colour code
(Kishony-Fowler layout and round structure, src/qec/color.rs), noisy-CNOT model, Z memory.

Fault analysis (one round r = 0..R-1; detector layers 0..R, layer R = final data detectors):

* Z half, CX data q -> anc(p) at step t:
    - X on q afterwards: let E = plaquettes of q that q has met by step t (incl. p). Flips layer-r
      detectors of q's plaquettes NOT in E and layer-(r+1) detectors of those in E.
    - X on anc(p): flips layer r and r+1 of p (time-like).  XX: XOR of the two (= the previous
      prefix of q's meeting order; same family).
* X half, CX anc(p) -> q at step t:
    - X on q: a clean data error (layer r+1, all plaquettes of q).
    - X on anc(p): X on the data qubits of p met AFTER t (a suffix of p's order) -> clean error.
So the Z-sector mechanisms are exactly
    clean(S, layer)      S = {q} (always), or a proper suffix of some plaquette's X-half order;
    partial(q, E, r)     E a proper non-empty prefix of q's Z-half meeting order (and E = {} in
                         round 0, from X⊗X after q's first CNOT: an error 'before' round 0);
    timelike(p, r)       always.
The observable flips iff |S ∩ row0| is odd (resp. q in row 0).  Every Z-sector signature has a pure
X-type version (X⊗I etc.), so Z-sector logicals lift to the full DEM (certified).

`dem(code, order_x, meet_z, R)` builds the merged signature list.  `verify` compares it with the
Rust circuit DEM (`color_search dem`) for random schedules.
"""
import itertools, random, subprocess, sys

CS = "/tmp/cg-target/release/examples/color_search"
KF = [[1, 3, 2, 5, 4, 6], [5, 1, 4, 3, 6, 2], [1, 5, 3, 6, 2, 4]]


class Code:
    def __init__(self, d, cs=CS):
        self.d = d
        out = subprocess.run([cs, "layout", str(d)], capture_output=True, text=True, check=True).stdout
        self.P = []
        for l in out.splitlines():
            v = [int(t) for t in l.split()]
            self.P.append(dict(i=v[0], x=v[1], y=v[2], color=v[3], w=v[4], data=v[5:11]))
        self.np = len(self.P)
        self.nq = (3 * d * d + 1) // 4
        self.qplaq = [[] for _ in range(self.nq)]  # (plaquette, position)
        for p in self.P:
            for k, q in enumerate(p["data"]):
                if q >= 0:
                    self.qplaq[q].append((p["i"], k))
        # data coordinates: recover row 0 from layout (plaquettes at y=0 have data at y=0 offsets
        # (+-4,0)); simpler: ask the generator rule
        self.row0 = self._row0()
        self.pq = [[q for q in p["data"] if q >= 0] for p in self.P]

    def _row0(self):
        d = self.d
        L = 3 * (d - 1) // 2
        idx = 0
        row0 = set()
        for y in range(L + 1):
            pos = [2, 0, 1][y % 3]
            x = 2 * y
            while x <= 4 * L - 2 * y:
                if ((x // 2 - y) // 2) % 3 != pos:
                    if y == 0:
                        row0.add(idx)
                    idx += 1
                x += 4
        assert idx == self.nq
        return row0

    def coords(self):
        d = self.d
        L = 3 * (d - 1) // 2
        out = []
        for y in range(L + 1):
            pos = [2, 0, 1][y % 3]
            x = 2 * y
            while x <= 4 * L - 2 * y:
                if ((x // 2 - y) // 2) % 3 != pos:
                    out.append((x, y))
                x += 4
        return out

    def rotation(self):
        """120-degree rotation of the triangle: (qubit map, plaquette map)."""
        import math
        real = lambda x, y: (x, 2 * math.sqrt(3) * y)
        dq = [real(*c) for c in self.coords()]
        cx = sum(a for a, _ in dq) / len(dq); cy = sum(b for _, b in dq) / len(dq)
        c, s = math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3)
        rot = lambda a, b: (cx + c * (a - cx) - s * (b - cy), cy + s * (a - cx) + c * (b - cy))
        def match(pts, P):
            out = []
            for a, b in P:
                ra, rb = rot(a, b)
                j = min(range(len(pts)), key=lambda k: (pts[k][0] - ra) ** 2 + (pts[k][1] - rb) ** 2)
                assert (pts[j][0] - ra) ** 2 + (pts[j][1] - rb) ** 2 < 1e-6, "not a symmetry"
                out.append(j)
            assert sorted(out) == list(range(len(pts)))
            return out
        qmap = match(dq, dq)
        pp = [real(p["x"], p["y"]) for p in self.P]
        pmap = match(pp, pp)
        for p in self.P:  # incidence preserved
            assert sorted(qmap[q] for q in self.pq[p["i"]]) == sorted(self.pq[pmap[p["i"]]])
        return qmap, pmap

    def kf(self):
        return [list(KF[p["color"]]) for p in self.P]

    def syn(self, S):
        """Plaquettes with odd overlap with qubit set S."""
        S = set(S)
        return tuple(sorted(p["i"] for p in self.P if len(S.intersection(self.pq[p["i"]])) % 2))

    def obs(self, S):
        return len(set(S) & self.row0) % 2 == 1


def orders_from_schedule(code, sched):
    """X-half order of each plaquette (data qubits by time) and Z-half meeting order of each qubit
    (plaquettes by time) for a single schedule used in both halves."""
    ox = []
    for p in code.P:
        pres = [k for k in range(6) if p["data"][k] >= 0]
        ox.append([p["data"][k] for k in sorted(pres, key=lambda k: sched[p["i"]][k])])
    mz = []
    for q in range(code.nq):
        mz.append([pi for pi, k in sorted(code.qplaq[q], key=lambda pk: sched[pk[0]][pk[1]])])
    return ox, mz


def mechanisms(code, ox, mz, R, space_only=False, hook_free=(), extra_hooks=None):
    """dict signature -> list of sources. signature = (tuple of (layer, plaquette)), obs bool).
    Sources: ('single',q,l) ('hook',p,frozenset S,l) ('partial',q,frozenset E,r) ('time',p,r)."""
    M = {}

    def add(dets, ob, src):
        key = (tuple(sorted(dets)), ob)
        if not dets and not ob:
            return
        M.setdefault(key, []).append(src)

    layers = [R] if space_only else range(1, R + 1)  # clean errors land in layers 1..R
    for l in layers:
        for q in range(code.nq):
            add([(l, p) for p in code.syn([q])], q in code.row0, ("single", q, l))
        for p in code.P:
            o = ox[p["i"]]
            if p["i"] in hook_free:
                for S in (extra_hooks or {}).get(p["i"], []):  # residual hooks of a split plaquette
                    add([(l, x) for x in code.syn(S)], code.obs(S), ("hook", p["i"], frozenset(S), l))
                continue  # measured hook-free (two auxiliaries / flag): no multi-qubit hooks
            for k in range(1, len(o)):
                S = o[k:]
                add([(l, x) for x in code.syn(S)], code.obs(S), ("hook", p["i"], frozenset(S), l))
    if not space_only:
        for r in range(R):
            for p in range(code.np):
                add([(r, p), (r + 1, p)], False, ("time", p, r))
            for q in range(code.nq):
                m = mz[q]
                # E = {} in round 0: XX after q's first Z-half CNOT (for r > 0 it equals single(q, r))
                if r == 0:
                    add([(0, x) for x in m], q in code.row0, ("partial", q, frozenset(), 0))
                for k in range(1, len(m)):
                    E = set(m[:k])
                    dets = [(r, x) for x in m if x not in E] + [(r + 1, x) for x in E]
                    add(dets, q in code.row0, ("partial", q, frozenset(E), r))
    return M


def to_problem(code, M, R):
    """Index Z-sector detectors (layer, plaquette) -> layer*np + p."""
    keys = list(M)
    rows = []
    for dets, ob in keys:
        rows.append((ob, [l * code.np + p for (l, p) in dets]))
    return keys, rows, (R + 1) * code.np


def rust_zsector(code, sched, R):
    """Z-sector signature set from the Rust circuit DEM, in the same (layer, plaquette) form."""
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".sched", delete=False) as f:
        for s in sched:
            f.write(" ".join(map(str, s)) + "\n")
        sp = f.name
    out = subprocess.run([CS, "dem", str(code.d), str(R), "cnot", "0.001", sp], capture_output=True,
                         text=True, check=True).stdout
    os.remove(sp)
    np_ = code.np
    # detector index -> (layer, plaquette) for Z type
    zinfo = {}
    idx = 0
    for r in range(R):
        for p in range(np_):
            zinfo[idx] = (r, p); idx += 1
        if r > 0:
            idx += np_  # X detectors
    for p in range(np_):
        zinfo[idx] = (R, p); idx += 1
    S = set()
    for l in out.splitlines():
        if l.startswith("#"):
            continue
        prob, ob, ds = (l.split("\t") + [""])[:3]
        dets = tuple(sorted(zinfo[int(x)] for x in ds.split() if int(x) in zinfo))
        o = int(ob) & 1 == 1
        if dets or o:
            S.add((dets, o))
    return S


def random_schedule(code, rng, tries=10000):
    """Random collision-free per-plaquette schedule (greedy with restarts)."""
    for _ in range(tries):
        used = [set() for _ in range(code.nq)]
        s = []
        ok = True
        for p in code.P:
            pres = [k for k in range(6) if p["data"][k] >= 0]
            for _ in range(200):
                st = rng.sample(range(1, 7), len(pres))
                if all(t not in used[p["data"][k]] for k, t in zip(pres, st)):
                    break
            else:
                ok = False
                break
            row = [0] * 6
            for k, t in zip(pres, st):
                row[k] = t
                used[p["data"][k]].add(t)
            s.append(row)
        if ok:
            return s
    raise RuntimeError("no schedule")


def verify(d, R, n, seed=1):
    code = Code(d)
    rng = random.Random(seed)
    scheds = [("kf", code.kf())] + [(f"rand{i}", random_schedule(code, rng)) for i in range(n)]
    for name, s in scheds:
        ox, mz = orders_from_schedule(code, s)
        mine = set(mechanisms(code, ox, mz, R))
        theirs = rust_zsector(code, s, R)
        print(f"d={d} R={R} {name}: symbolic {len(mine)} rust {len(theirs)} equal={mine == theirs}")
        if mine != theirs:
            print(" only symbolic:", sorted(mine - theirs)[:5])
            print(" only rust:", sorted(theirs - mine)[:5])
            return False
    return True


if __name__ == "__main__":
    ok = True
    for d, R in [(3, 1), (3, 3), (5, 1), (5, 3), (7, 2), (9, 1)]:
        ok &= verify(d, R, int(sys.argv[1]) if len(sys.argv) > 1 else 5)
    print("ALL EQUAL" if ok else "MISMATCH")
