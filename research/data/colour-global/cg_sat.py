#!/usr/bin/env python3
"""Global (exact) search over single-auxiliary colour-code schedules by counterexample-guided SAT.

Question: does a schedule exist whose Z-memory circuit distance (noisy-CNOT, R rounds) is >= D?

* Schedule variables (mode):
    kf    one schedule t_p(q) in 1..T for both halves, collision-free (K-F's design space; T=6);
    sep   independent X-half and Z-half schedules, each collision-free, steps 1..T;
    free  arbitrary per-plaquette X-half orders and per-qubit Z-half meeting orders (any depth).
  The DEM depends on the schedule only through order literals
    bx[p,a,b]  "plaquette p touches data a before data b in the X half"
    bz[q,a,b]  "data q meets plaquette a before plaquette b in the Z half",
  which are tied (<->) to the time variables in kf/sep, and are free total orders in free mode.
* Every potential mechanism (all hooks of all plaquettes, all partial data errors) has an exact
  presence condition (a conjunction of order literals; cg_model.py, verified against the circuit
  DEM). A variable y_sig per signature is forced true by each of its conditions.
* Loop: SAT -> candidate schedule -> exact min-weight logicals of its DEM up to weight D-1
  (dem_distance, branch and bound) -> for every such logical L add the clause OR_{sig in L} -y_sig
  (sound: it removes exactly the schedules that contain L). Stop when a candidate has no logical
  of weight < D (FOUND) or the clauses are UNSAT (no schedule in the space reaches D).
  The clause set is the certificate: re-checking it needs only (i) that each cut is a logical
  (XOR of its signatures is empty with odd observable; checked here) and (ii) UNSAT of the CNF
  (written out as DIMACS; CaDiCaL can emit a DRAT proof).

usage: cg_sat.py <d> <D> <R> <mode> [--T 6] [--space] [--cnf out.cnf] [--log out.jsonl]
       [--max-iter N] [--keep K] [--sym]
"""
import argparse, itertools, json, subprocess, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cg_model import Code, mechanisms, to_problem
from pysat.solvers import Solver
from pysat.card import CardEnc, EncType
from pysat.formula import IDPool

DD = "/tmp/cg-target/release/examples/dem_distance"


class Enc:
    def __init__(self, code, mode, T):
        self.code, self.mode, self.T = code, mode, T
        self.pool = IDPool()
        self.cls = []
        self.v = self.pool.id
        P = code.P
        if mode in ("kf", "sep"):
            halves = ["s"] if mode == "kf" else ["x", "z"]
            for h in halves:
                self._times(h)
            hx = "s" if mode == "kf" else "x"
            hz = "s" if mode == "kf" else "z"
            for p in P:
                pi = p["i"]
                for a, b in itertools.permutations(code.pq[pi], 2):
                    self._before(("bx", pi, a, b), (hx, pi, a), (hx, pi, b))
            for q in range(code.nq):
                ps = [pi for pi, _ in code.qplaq[q]]
                for a, b in itertools.permutations(ps, 2):
                    self._before(("bz", q, a, b), (hz, a, q), (hz, b, q))
        else:  # free total orders
            for p in P:
                self._total_order("bx", p["i"], code.pq[p["i"]])
            for q in range(code.nq):
                self._total_order("bz", q, [pi for pi, _ in code.qplaq[q]])

    # time variables: le[(h,p,q,t)] = step of (p,q) in half h is <= t, t = 1..T
    def _times(self, h):
        code, T = self.code, self.T
        for p in code.P:
            pi = p["i"]
            for q in code.pq[pi]:
                le = [self.v(("le", h, pi, q, t)) for t in range(1, T + 1)]
                self.cls.append([le[-1]])
                for t in range(T - 1):
                    self.cls.append([-le[t], le[t + 1]])
            # distinct steps within a plaquette and per data qubit (collision-free)
            for t in range(1, T + 1):
                xs = [self._eq(h, pi, q, t) for q in code.pq[pi]]
                self.cls += CardEnc.atmost(xs, 1, vpool=self.pool, encoding=EncType.pairwise).clauses
        for q in range(code.nq):
            ps = [pi for pi, _ in code.qplaq[q]]
            for t in range(1, T + 1):
                xs = [self._eq(h, pi, q, t) for pi in ps]
                self.cls += CardEnc.atmost(xs, 1, vpool=self.pool, encoding=EncType.pairwise).clauses

    def _le(self, h, p, q, t):
        if t <= 0:
            return None  # false
        return self.v(("le", h, p, q, min(t, self.T)))

    def _eq(self, h, p, q, t):
        key = ("eq", h, p, q, t)
        if key in self.pool.obj2id:
            return self.pool.obj2id[key]
        e = self.v(key)
        a = self._le(h, p, q, t)
        b = self._le(h, p, q, t - 1)
        # e <-> a & -b
        self.cls.append([-e, a])
        if b is not None:
            self.cls.append([-e, -b])
            self.cls.append([e, -a, b])
        else:
            self.cls.append([e, -a])
        return e

    def _before(self, key, A, B):
        """var(key) <-> step(A) < step(B) (steps are distinct by construction)."""
        hA, pA, qA = A
        hB, pB, qB = B
        x = self.v(key)
        for t in range(1, self.T + 1):
            lb = self._le(hB, pB, qB, t)
            la = self._le(hA, pA, qA, t - 1)
            # x -> (lb -> la)
            self.cls.append([-x, -lb] + ([la] if la else []))
            # -x -> step(B) < step(A): (la' := le_A(t)) -> le_B(t-1)
            la2 = self._le(hA, pA, qA, t)
            lb2 = self._le(hB, pB, qB, t - 1)
            self.cls.append([x, -la2] + ([lb2] if lb2 else []))

    def _total_order(self, tag, owner, items):
        for a, b in itertools.combinations(items, 2):
            x, y = self.v((tag, owner, a, b)), self.v((tag, owner, b, a))
            self.cls += [[x, y], [-x, -y]]
        for a, b, c in itertools.permutations(items, 3):
            self.cls.append([-self.v((tag, owner, a, b)), -self.v((tag, owner, b, c)), self.v((tag, owner, a, c))])

    def bx(self, p, a, b):
        return self.v(("bx", p, a, b))

    def bz(self, q, a, b):
        return self.v(("bz", q, a, b))

    # decode
    def orders(self, model):
        ms = set(l for l in model if l > 0)
        code = self.code
        ox = []
        for p in code.P:
            qs = code.pq[p["i"]]
            ox.append(sorted(qs, key=lambda a: -sum(self.bx(p["i"], a, b) in ms for b in qs if b != a)))
        mz = []
        for q in range(code.nq):
            ps = [pi for pi, _ in code.qplaq[q]]
            mz.append(sorted(ps, key=lambda a: -sum(self.bz(q, a, b) in ms for b in ps if b != a)))
        return ox, mz

    def schedule(self, model, h="s"):
        ms = set(l for l in model if l > 0)
        out = []
        for p in self.code.P:
            row = [0] * 6
            for k, q in enumerate(p["data"]):
                if q >= 0:
                    row[k] = min(t for t in range(1, self.T + 1) if self.v(("le", h, p["i"], q, t)) in ms)
            out.append(row)
        return out


SRC = {}


def sig_of(code, src):
    """Signature of a fault source: ('S', layer, set) clean data error, ('P', r, q, E) partial
    data error, ('T', r, p) time-like."""
    if src[0] == "S":
        _, l, S = src
        return (tuple(sorted((l, x) for x in code.syn(S))), code.obs(S))
    if src[0] == "P":
        _, r, q, E = src
        ps = [pi for pi, _ in code.qplaq[q]]
        return (tuple(sorted([(r, x) for x in ps if x not in E] + [(r + 1, x) for x in E])), q in code.row0)
    _, r, p = src
    return (((r, p), (r + 1, p)), False)


def rotate_src(src, qmap, pmap):
    if src[0] == "S":
        return ("S", src[1], frozenset(qmap[q] for q in src[2]))
    if src[0] == "P":
        return ("P", src[1], qmap[src[2]], frozenset(pmap[p] for p in src[3]))
    return ("T", src[1], pmap[src[2]])


def universe(code, R, space_only):
    """signature -> list of presence conditions (list of (kind, owner, a, b) order literals).
    Also fills SRC[signature] with one fault source (for symmetry images of cuts)."""
    U = {}
    layers = [R] if space_only else range(1, R + 1)

    def add(dets, ob, cond, src=None):
        if not dets and not ob:
            return
        key = (tuple(sorted(dets)), ob)
        U.setdefault(key, []).append(cond)
        if src is not None and key not in SRC:
            assert sig_of(code, src) == key
            SRC[key] = src

    for l in layers:
        for q in range(code.nq):
            add([(l, p) for p in code.syn([q])], q in code.row0, [], ("S", l, frozenset([q])))
        for p in code.P:
            qs = code.pq[p["i"]]
            for k in range(2, len(qs) - 1):
                for S in itertools.combinations(qs, k):
                    rest = [a for a in qs if a not in S]
                    cond = [("bx", p["i"], a, b) for a in rest for b in S]
                    add([(l, x) for x in code.syn(S)], code.obs(S), cond, ("S", l, frozenset(S)))
    if not space_only:
        for r in range(R):
            for p in range(code.np):
                add([(r, p), (r + 1, p)], False, [], ("T", r, p))
            for q in range(code.nq):
                ps = [pi for pi, _ in code.qplaq[q]]
                if r == 0:
                    add([(0, x) for x in ps], q in code.row0, [], ("P", 0, q, frozenset()))
                for k in range(1, len(ps)):
                    for E in itertools.combinations(ps, k):
                        rest = [a for a in ps if a not in E]
                        cond = [("bz", q, a, b) for a in E for b in rest]
                        add([(r, x) for x in rest] + [(r + 1, x) for x in E], q in code.row0, cond,
                            ("P", r, q, frozenset(E)))
    return U


class DistServer:
    def __init__(self):
        self.p = subprocess.Popen([DD], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def query(self, nd, rows, maxw, cap=10**9):
        w = self.p.stdin
        w.write(f"P {nd} {maxw} {cap} {len(rows)}\n")
        w.write("".join(f"{1 if ob else 0} {' '.join(map(str, ds))}\n" for ob, ds in rows))
        w.flush()
        h = self.p.stdout.readline().split()
        weight = None if h[1] == "none" else int(h[1])
        count, nodes, nl = int(h[2]), int(h[3]), int(h[4])
        logs = [[int(x) for x in self.p.stdout.readline().split()[1:]] for _ in range(nl)]
        return weight, count, nodes, logs


def check_logical(keys):
    acc = set()
    ob = False
    for dets, o in keys:
        acc ^= set(dets)
        ob ^= o
    return not acc and ob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("d", type=int)
    ap.add_argument("D", type=int)
    ap.add_argument("R", type=int)
    ap.add_argument("mode", choices=["kf", "sep", "free"])
    ap.add_argument("--T", type=int, default=6)
    ap.add_argument("--space", action="store_true", help="space-only (single layer, hooks + singles)")
    ap.add_argument("--cnf", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--max-iter", type=int, default=10**9)
    ap.add_argument("--keep", type=int, default=2000)
    ap.add_argument("--solver", default="cadical195")
    ap.add_argument("--hookfree", type=int, default=None,
                    help="allow up to K hook-free plaquettes (2 auxiliaries / flag): their hooks vanish")
    ap.add_argument("--split", type=int, default=None, help="allow up to K two-auxiliary plaquettes")
    ap.add_argument("--split-set", default="all", choices=["all", "boundary"])
    ap.add_argument("--hookfree-set", default="all", choices=["all", "corners", "boundary"])
    ap.add_argument("--phase-kf", action="store_true", help="solver phases = K-F schedule (heuristic)")
    ap.add_argument("--sym", action="store_true", help="also cut the 120/240-degree rotated images of each logical")
    ap.add_argument("--warm", action="store_true", help="screen candidates with the space-only DEM first")
    ap.add_argument("--fix-depth", type=int, default=None,
                    help="kf mode: plaquettes at adjacency depth >= k from the boundary keep K-F")
    ap.add_argument("--fix-interior", action="store_true", help="kf mode: plaquettes with no boundary data keep K-F")
    args = ap.parse_args()
    code = Code(args.d)
    t0 = time.time()
    enc = Enc(code, args.mode, args.T)
    U = universe(code, args.R, args.space)
    # y variables
    ysig = {}
    always = set()
    hf = {}
    if args.hookfree is not None:
        cand = [p["i"] for p in code.P]
        if args.hookfree_set == "corners":
            cand = [p["i"] for p in code.P if sum(len(code.qplaq[q]) == 1 for q in code.pq[p["i"]])]
        elif args.hookfree_set == "boundary":
            cand = [p["i"] for p in code.P if set(code.pq[p["i"]]) & boundary_qubits(code)]
        for pi in cand:
            hf[pi] = enc.v(("hf", pi))
        enc.cls += CardEnc.atmost(list(hf.values()), args.hookfree, vpool=enc.pool,
                                  encoding=EncType.seqcounter).clauses
    # two-auxiliary relaxation: plaquette p split into two halves, each measured by its own
    # auxiliary (cat pair; prep faults assumed flagged). Weight 4 -> 2+2: no multi-qubit hook.
    # Weight 6 -> 3+3: residual hooks = the last two qubits of each triple, i.e. any two
    # disjoint pairs (one-hot over 45 options).
    sp = {}
    split_pairs = {}  # (p, frozenset pair) -> list of option vars
    if args.split is not None:
        cand = [p["i"] for p in code.P]
        if args.split_set == "boundary":
            cand = [p["i"] for p in code.P if set(code.pq[p["i"]]) & boundary_qubits(code)]
        for pi in cand:
            sp[pi] = enc.v(("split", pi))
            qs = code.pq[pi]
            if len(qs) == 6:
                opts = []
                for a, b in itertools.combinations(qs, 2):
                    rest = [x for x in qs if x not in (a, b)]
                    for c, e in itertools.combinations(rest, 2):
                        if (a, b) < (c, e):
                            o = enc.v(("splitopt", pi, a, b, c, e))
                            opts.append(o)
                            for pr in ((a, b), (c, e)):
                                split_pairs.setdefault((pi, frozenset(pr)), []).append(o)
                enc.cls.append([-sp[pi]] + opts)
                enc.cls += CardEnc.atmost(opts, 1, vpool=enc.pool, encoding=EncType.seqcounter).clauses
        enc.cls += CardEnc.atmost(list(sp.values()), args.split, vpool=enc.pool,
                                  encoding=EncType.seqcounter).clauses
    for sig, conds in U.items():
        if any(len(c) == 0 for c in conds):
            always.add(sig)
            continue
        y = enc.v(("y", sig))
        ysig[sig] = y
        for c in conds:
            extra = [hf[c[0][1]]] if (c[0][0] == "bx" and c[0][1] in hf) else []
            if c[0][0] == "bx" and c[0][1] in sp:
                extra.append(sp[c[0][1]])
                pi = c[0][1]
                S = frozenset(b for (_, _, a, b) in c)
                comp = frozenset(code.pq[pi]) - S
                for pr in (S, comp):
                    for o in split_pairs.get((pi, pr), []):
                        enc.cls.append([-sp[pi], -o, y])
            enc.cls.append([-enc.v(l) for l in c] + extra + [y])
    if args.fix_interior or args.fix_depth is not None:
        assert args.mode == "kf"
        from cg_model import KF
        depth = plaquette_depth(code)
        kmin = 1 if args.fix_interior else args.fix_depth
        for p in code.P:
            if depth[p["i"]] >= kmin:
                for k, q in enumerate(p["data"]):
                    if q >= 0:
                        t = KF[p["color"]][k]
                        enc.cls.append([enc.v(("le", "s", p["i"], q, t))])
                        if t > 1:
                            enc.cls.append([-enc.v(("le", "s", p["i"], q, t - 1))])
    base_clauses = list(enc.cls)
    s = Solver(name=args.solver, bootstrap_with=enc.cls)
    if args.phase_kf and args.mode in ("kf", "sep"):
        # decision phases = K-F's schedule, so candidates stay close to K-F (heuristic only)
        from cg_model import KF
        ph = []
        for h in (["s"] if args.mode == "kf" else ["x", "z"]):
            for p in code.P:
                for kk, q in enumerate(p["data"]):
                    if q >= 0:
                        tk = KF[p["color"]][kk]
                        for t in range(1, args.T + 1):
                            v = enc.v(("le", h, p["i"], q, t))
                            ph.append(v if t >= tk else -v)
        s.set_phases(ph)
    log = open(args.log, "a") if args.log else None
    L = lambda **kw: (log.write(json.dumps(kw) + "\n"), log.flush()) if log else None
    L(kind="start", d=args.d, D=args.D, R=args.R, mode=args.mode, T=args.T, space=args.space,
      vars=enc.pool.top, clauses=len(base_clauses), signatures=len(U), always=len(always))
    print(f"d={args.d} D={args.D} R={args.R} mode={args.mode} T={args.T} space={args.space}: "
          f"{enc.pool.top} vars, {len(base_clauses)} clauses, {len(U)} signatures ({len(always)} always)", flush=True)
    srv = DistServer()
    qmap, pmap = code.rotation() if args.sym else (None, None)
    cuts = []
    cutset = set()
    it = 0
    result = None
    while it < args.max_iter:
        it += 1
        ts = time.time()
        sat = s.solve()
        tsat = time.time() - ts
        if not sat:
            result = "UNSAT"
            break
        model = s.get_model()
        ox, mz = enc.orders(model)
        msel = set(model)
        hook_free = [pi for pi, v in hf.items() if v in msel]
        split_now = {}
        for pi, v in sp.items():
            if v in msel:
                hook_free.append(pi)  # its single-aux hooks vanish ...
                split_now[pi] = [set(pr) for (pj, pr), os_ in split_pairs.items() if pj == pi and any(o in msel for o in os_)]
        ts = time.time()
        w = None
        if args.warm and not args.space:
            # cheap screen first: the space-only DEM (layer R) is a subset of the full one
            M = mechanisms(code, ox, mz, args.R, space_only=True, hook_free=hook_free, extra_hooks=split_now)
            keys, rows, nd = to_problem(code, M, args.R)
            w, cnt, nodes, logs = srv.query(nd, rows, args.D - 1, args.keep)
        if w is None:
            M = mechanisms(code, ox, mz, args.R, space_only=args.space, hook_free=hook_free, extra_hooks=split_now)
            keys, rows, nd = to_problem(code, M, args.R)
            w, cnt, nodes, logs = srv.query(nd, rows, args.D - 1, args.keep)
        tdist = time.time() - ts
        if w is None:
            result = "FOUND"
            sched = enc.schedule(model, "s" if args.mode == "kf" else "x") if args.mode != "free" else None
            schedz = enc.schedule(model, "z") if args.mode == "sep" else None
            spl = {pi: [sorted(x) for x in v] for pi, v in split_now.items()}
            L(kind="found", iter=it, ox=ox, mz=mz, schedule=sched, schedule_z=schedz, hook_free=hook_free, split=spl)
            print("FOUND", json.dumps(dict(ox=ox, mz=mz, schedule=sched, schedule_z=schedz, hook_free=hook_free, split=spl)))
            break
        new = 0
        cand_logs = [[keys[j] for j in lg] for lg in logs[: args.keep]]
        if args.sym:
            imgs = []
            for ks in cand_logs:
                srcs = [SRC[k] for k in ks]
                for _ in range(2):
                    srcs = [rotate_src(x, qmap, pmap) for x in srcs]
                    rk = [sig_of(code, x) for x in srcs]
                    assert check_logical(rk), "rotated image is not a logical"
                    imgs.append(rk)
            cand_logs += imgs
        for ks in cand_logs:
            assert check_logical(ks)
            # sanity: every signature must be in the universe
            assert all(k in U for k in ks), "signature outside universe"
            cl = tuple(sorted(ysig[k] for k in ks if k not in always))
            if cl in cutset:
                continue
            cutset.add(cl)
            if not cl:
                result = "UNSAT"  # an unavoidable logical of weight < D
                cuts.append(([], [list(map(list, k[0])) for k in ks]))
                break
            s.add_clause([-y for y in cl])
            cuts.append((cl, ks))
            new += 1
        if result:
            break
        if it % 10 == 0 or it < 5:
            print(f"iter {it}: weight {w} count {cnt}, +{new} cuts (total {len(cuts)}), sat {tsat:.2f}s dist {tdist:.2f}s, "
                  f"elapsed {time.time() - t0:.0f}s", flush=True)
        L(kind="iter", iter=it, weight=w, count=cnt, new_cuts=new, cuts=len(cuts), tsat=tsat, tdist=tdist)
        assert new > 0, "no new cut: presence conditions incomplete"
    el = time.time() - t0
    print(f"RESULT {result} after {it} iterations, {len(cuts)} cuts, {el:.1f}s", flush=True)
    L(kind="result", result=result, iterations=it, cuts=len(cuts), seconds=el)
    if args.cnf and result == "UNSAT":
        with open(args.cnf, "w") as f:
            allc = base_clauses + [[-y for y in cl] for cl, _ in cuts if cl]
            f.write(f"c colour-code schedule space d={args.d} D={args.D} R={args.R} mode={args.mode} T={args.T} space={args.space}\n")
            f.write(f"p cnf {enc.pool.top} {len(allc)}\n")
            for c in allc:
                f.write(" ".join(map(str, c)) + " 0\n")
        # store the logicals (signature lists) for independent re-checking
        with open(args.cnf + ".logicals.jsonl", "w") as f:
            for cl, ks in cuts:
                f.write(json.dumps([[[list(x) for x in d], o] for d, o in ks]) + "\n")
    srv.p.kill()


def plaquette_depth(code):
    """0 for plaquettes touching a boundary qubit, else 1 + min over adjacent plaquettes."""
    bnd = boundary_qubits(code)
    depth = {p["i"]: (0 if set(code.pq[p["i"]]) & bnd else None) for p in code.P}
    adj = {p["i"]: {o for q in code.pq[p["i"]] for o, _ in code.qplaq[q] if o != p["i"]} for p in code.P}
    lvl = 0
    while any(v is None for v in depth.values()):
        for i in [i for i, v in depth.items() if v is None and any(depth[o] == lvl for o in adj[i])]:
            depth[i] = lvl + 1
        lvl += 1
    return depth


def boundary_qubits(code):
    return {q for q in range(code.nq) if len(code.qplaq[q]) < 3}


if __name__ == "__main__":
    main()
