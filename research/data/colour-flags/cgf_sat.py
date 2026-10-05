#!/usr/bin/env python3
"""Flag-aware counterexample-guided SAT schedule search (extends ../colour-global/cg_sat.py).

colour-global's `--hookfree` relaxation treats a flagged plaquette's hooks as *absent*. In a real
flag circuit (src/qec/color.rs `memory_flagged`) they are present and also flip that plaquette's
flag detector, and a flag-only fault exists (an X on the flag from a flag CNOT). So two flagged
hooks of one plaquette in one round cost 2 faults and leave a *segment* of its X-half order, and a
hook plus a flag-only fault leaves a suffix: the absent-hook model is optimistic (it certified
schedules that are d - 1 as circuits, see research/qec/colour-flags.md).

Exact flag model used here (Z memory, noisy-CNOT, flag window from `ColorCode::flag_slots`): for a
flagged plaquette p and clean-error layer l, the multi-qubit unflagged hooks are removed and
replaced by
    flagged suffix:  syn(S)@l + F(p,l)   for every suffix S of p's X-half order, 1 <= |S| <= w-1
    flag only:       F(p,l)              (always present)
Single-qubit suffixes also exist unflagged (they equal single data errors). Everything else is the
cg_model DEM. `verify` checks the model's signature set against the Rust circuit DEM.

usage: cgf_sat.py <d> <D> <R> <kf|sep|free> [--T 6] [--flags boundary] [--warm] [--sym]
                  [--phase-kf] [--log f] [--out f.sched]
       cgf_sat.py verify <d> <R> <schedule-file-with-F-markers>
"""
import argparse, itertools, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "colour-global"))
from cg_model import Code, mechanisms, orders_from_schedule, KF
import cg_sat
from cg_sat import Enc, DistServer, check_logical, boundary_qubits
from pysat.solvers import Solver

FOFF = 100000
CS = os.environ.get("CS", "/tmp/cf-target/release/examples/color_search")


def mechanisms_f(code, ox, mz, R, flagged, space_only=False):
    M = mechanisms(code, ox, mz, R, space_only=space_only, hook_free=flagged)

    def add(dets, ob, src):
        M.setdefault((tuple(sorted(dets)), ob), []).append(src)

    for l in ([R] if space_only else range(1, R + 1)):
        for p in flagged:
            add([(l, FOFF + p)], False, ("flagonly", p, l))
            o = ox[p]
            for k in range(1, len(o)):
                S = o[k:]
                add([(l, x) for x in code.syn(S)] + [(l, FOFF + p)], code.obs(S), ("fhook", p, frozenset(S), l))
    return M


def to_problem_f(code, M, R):
    keys = list(M)
    rows = []
    for dets, ob in keys:
        rows.append((ob, [l * code.np + p if p < FOFF else (R + 1 + l - 1) * code.np + (p - FOFF) for (l, p) in dets]))
    return keys, rows, (2 * R + 1) * code.np


def sig_of_f(code, src):
    if src[0] == "F":
        _, l, p, S = src
        return (tuple(sorted([(l, x) for x in code.syn(S)] + [(l, FOFF + p)])), code.obs(S))
    if src[0] == "O":
        return (((src[1], FOFF + src[2]),), False)
    return cg_sat.sig_of(code, src)


def rotate_src_f(src, qmap, pmap):
    if src[0] == "F":
        return ("F", src[1], pmap[src[2]], frozenset(qmap[q] for q in src[3]))
    if src[0] == "O":
        return ("O", src[1], pmap[src[2]])
    return cg_sat.rotate_src(src, qmap, pmap)


def universe_f(code, R, space_only, flagged):
    cg_sat.SRC.clear()
    U0 = cg_sat.universe(code, R, space_only)
    fl = set(flagged)
    U, SRC = {}, {}
    for key, conds in U0.items():
        keep = [c for c in conds if not (c and c[0][0] == "bx" and c[0][1] in fl)]
        if keep:
            U[key] = keep
            SRC[key] = cg_sat.SRC[key]
    for l in ([R] if space_only else range(1, R + 1)):
        for p in flagged:
            key = (((l, FOFF + p),), False)
            U.setdefault(key, []).append([])
            SRC[key] = ("O", l, p)
            qs = code.pq[p]
            for k in range(1, len(qs)):
                for S in itertools.combinations(qs, k):
                    rest = [a for a in qs if a not in S]
                    cond = [("bx", p, a, b) for a in rest for b in S]
                    src = ("F", l, p, frozenset(S))
                    key = sig_of_f(code, src)
                    U.setdefault(key, []).append(cond)
                    SRC.setdefault(key, src)
    # a key whose source was an unflagged hook of a flagged plaquette may survive through other
    # sources; re-point SRC at a source that still exists
    for key in U:
        if sig_of_f(code, SRC[key]) != key:
            raise AssertionError("SRC mismatch")
    return U, SRC


def flag_set(code, which):
    if which == "boundary":
        bq = boundary_qubits(code)
        return [p["i"] for p in code.P if set(code.pq[p["i"]]) & bq]
    if which == "all":
        return [p["i"] for p in code.P]
    if which == "none":
        return []
    return [int(x) for x in which.split(",")]


def rust_zsector_f(code, sched_path, R):
    """Z-sector signature set of the Rust *flagged* circuit DEM, in model form."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".dem", delete=False) as f:
        dem = f.name
    subprocess.run([CS, "dem", str(code.d), str(R), "cnot", "0.001", sched_path, dem], check=True)
    st = os.path.join(os.path.dirname(dem), "x.stim")
    subprocess.run([CS, "export", str(code.d), str(R), "cnot", "0.001", sched_path, st], check=True)
    info = [tuple(map(int, l.split())) for l in open(st + ".info")]
    lines = open(dem).read().splitlines()
    os.remove(dem)
    out = set()
    for ln in lines[2:]:
        p, ob, *ds = ln.split("\t")
        ds = [int(t) for t in (ds[0].split() if ds else [])]
        z = []
        for t in ds:
            pi, isx, r, fl = info[t]
            if isx:
                continue
            z.append((r + 1, FOFF + pi) if fl else (r, pi))
        if z or ob == "1":
            out.add((tuple(sorted(z)), ob == "1"))
    return out


def verify(d, R, path):
    code = Code(d, CS)
    rows = [l.split() for l in open(path) if l.strip()]
    sched = [[int(x) for x in r[:6]] for r in rows]
    flagged = [i for i, r in enumerate(rows) if len(r) > 6 and r[6] == "F"]
    ox, mz = orders_from_schedule(code, sched)
    model = set(mechanisms_f(code, ox, mz, R, flagged))
    circ = rust_zsector_f(code, path, R)
    print(f"d={d} R={R} {os.path.basename(path)}: model {len(model)} sigs, circuit {len(circ)} sigs, "
          f"model-only {len(model - circ)}, circuit-only {len(circ - model)}", flush=True)
    for k in list(circ - model)[:5]:
        print("  circuit-only", k)
    for k in list(model - circ)[:5]:
        print("  model-only", k)
    return model == circ


def main():
    if sys.argv[1] == "verify":
        ok = verify(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
        print("EQUAL" if ok else "DIFFERENT")
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("d", type=int)
    ap.add_argument("D", type=int)
    ap.add_argument("R", type=int)
    ap.add_argument("mode", choices=["kf", "sep", "free"])
    ap.add_argument("--T", type=int, default=6)
    ap.add_argument("--flags", default="boundary")
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--sym", action="store_true")
    ap.add_argument("--phase-kf", action="store_true")
    ap.add_argument("--keep", type=int, default=2000)
    ap.add_argument("--solver", default="cadical195")
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cnf", default=None)
    ap.add_argument("--max-iter", type=int, default=10**9)
    a = ap.parse_args()
    code = Code(a.d, CS)
    flagged = flag_set(code, a.flags)
    t0 = time.time()
    enc = Enc(code, a.mode, a.T)
    U, SRC = universe_f(code, a.R, False, flagged)
    ysig, always = {}, set()
    for sig, conds in U.items():
        if any(len(c) == 0 for c in conds):
            always.add(sig)
            continue
        y = enc.v(("y", sig))
        ysig[sig] = y
        for c in conds:
            enc.cls.append([-enc.v(l) for l in c] + [y])
    base = list(enc.cls)
    s = Solver(name=a.solver, bootstrap_with=enc.cls)
    if a.phase_kf and a.mode in ("kf", "sep"):
        ph = []
        for h in (["s"] if a.mode == "kf" else ["x", "z"]):
            for p in code.P:
                for kk, q in enumerate(p["data"]):
                    if q >= 0:
                        tk = KF[p["color"]][kk]
                        for t in range(1, a.T + 1):
                            v = enc.v(("le", h, p["i"], q, t))
                            ph.append(v if t >= tk else -v)
        s.set_phases(ph)
    log = open(a.log, "a") if a.log else None
    L = lambda **kw: (log.write(json.dumps(kw) + "\n"), log.flush()) if log else None
    print(f"d={a.d} D={a.D} R={a.R} mode={a.mode} T={a.T} flags={a.flags} ({len(flagged)}): {enc.pool.top} vars, "
          f"{len(base)} clauses, {len(U)} signatures ({len(always)} always)", flush=True)
    srv = DistServer()
    qmap, pmap = code.rotation() if a.sym else (None, None)
    if a.sym:
        assert sorted(pmap[p] for p in flagged) == sorted(flagged), "flag set not rotation invariant"
    cuts, cutset, it, result = [], set(), 0, None
    while it < a.max_iter:
        it += 1
        ts = time.time()
        if not s.solve():
            result = "UNSAT"
            break
        tsat = time.time() - ts
        model = s.get_model()
        ox, mz = enc.orders(model)
        ts = time.time()
        w = None
        if a.warm:
            M = mechanisms_f(code, ox, mz, a.R, flagged, space_only=True)
            keys, rows, nd = to_problem_f(code, M, a.R)
            w, cnt, nodes, logs = srv.query(nd, rows, a.D - 1, a.keep)
        if w is None:
            M = mechanisms_f(code, ox, mz, a.R, flagged)
            keys, rows, nd = to_problem_f(code, M, a.R)
            w, cnt, nodes, logs = srv.query(nd, rows, a.D - 1, a.keep)
        tdist = time.time() - ts
        if w is None:
            result = "FOUND"
            sched = enc.schedule(model, "s" if a.mode == "kf" else "x") if a.mode != "free" else None
            L(kind="found", iter=it, ox=ox, mz=mz, schedule=sched, flagged=flagged)
            print("FOUND", json.dumps(dict(ox=ox, mz=mz, schedule=sched, flagged=flagged)), flush=True)
            if a.out and sched:
                with open(a.out, "w") as f:
                    for i, r in enumerate(sched):
                        f.write(" ".join(map(str, r)) + (" F" if i in flagged else "") + "\n")
            break
        cand = [[keys[j] for j in lg] for lg in logs[: a.keep]]
        if a.sym:
            imgs = []
            for ks in cand:
                srcs = [SRC[k] for k in ks]
                for _ in range(2):
                    srcs = [rotate_src_f(x, qmap, pmap) for x in srcs]
                    rk = [sig_of_f(code, x) for x in srcs]
                    assert check_logical(rk), "rotated image is not a logical"
                    imgs.append(rk)
            cand += imgs
        new = 0
        for ks in cand:
            assert check_logical(ks)
            assert all(k in U for k in ks), "signature outside universe"
            cl = tuple(sorted(ysig[k] for k in ks if k not in always))
            if cl in cutset:
                continue
            cutset.add(cl)
            if not cl:
                result = "UNSAT"
                break
            s.add_clause([-y for y in cl])
            cuts.append(cl)
            new += 1
        if result:
            break
        if it % 10 == 0 or it < 5:
            print(f"iter {it}: weight {w} count {cnt}, +{new} cuts (total {len(cuts)}), sat {tsat:.2f}s "
                  f"dist {tdist:.2f}s, elapsed {time.time() - t0:.0f}s", flush=True)
        L(kind="iter", iter=it, weight=w, count=cnt, new_cuts=new, cuts=len(cuts))
        assert new > 0, "no new cut"
    if a.cnf and result == "UNSAT":
        allc = base + [[-y for y in cl] for cl in cuts]
        with open(a.cnf, "w") as f:
            f.write(f"c flag-aware schedule space d={a.d} D={a.D} R={a.R} mode={a.mode} T={a.T} flags={a.flags}\n")
            f.write(f"p cnf {enc.pool.top} {len(allc)}\n")
            for c in allc:
                f.write(" ".join(map(str, c)) + " 0\n")
    print(f"RESULT {result} after {it} iterations, {len(cuts)} cuts, {time.time() - t0:.1f}s", flush=True)
    L(kind="result", result=result, iterations=it, cuts=len(cuts), seconds=time.time() - t0)


if __name__ == "__main__":
    main()
