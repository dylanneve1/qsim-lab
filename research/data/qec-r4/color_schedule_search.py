#!/usr/bin/env python3
"""Search per-plaquette CNOT schedules of the single-auxiliary triangular colour code.

Space: every plaquette keeps one auxiliary qubit, the same 6-step schedule is used for the Z
and X halves of a round (Kishony-Fowler's parallel construction), and the whole circuit must be
collision-free (no data qubit in two CNOTs at one step). Bulk plaquettes far from the boundary
keep the Kishony-Fowler (KF) colour schedule; plaquettes that touch a boundary data qubit
("free" plaquettes) may take ANY collision-free step assignment of their present positions.

Objective (lexicographic): circuit distance of the Z-memory circuit under the noisy-CNOT
model (DEPOLARIZE2 after every CNOT), then fewer minimum-weight logical fault sets (counted
up to a cap by MaxSAT model enumeration).

usage: color_schedule_search.py <color_search bin> <d> <rounds> <iters> <seed> <out.jsonl>
       [--uniform-csv kf_zero_collision.csv]   (instead: evaluate every colour-uniform schedule)
"""
import sys, os, json, random, subprocess, itertools, time, tempfile
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from circuit_distance import read_dem, x_detectors

KF = [[1, 3, 2, 5, 4, 6], [5, 1, 4, 3, 6, 2], [1, 5, 3, 6, 2, 4]]


def layout(cs, d):
    out = subprocess.run([cs, "layout", str(d)], capture_output=True, text=True, check=True).stdout
    P = []
    for l in out.splitlines():
        v = [int(t) for t in l.split()]
        P.append(dict(i=v[0], x=v[1], y=v[2], color=v[3], w=v[4], data=v[5:11]))
    return P


def zsector_wcnf(path):
    nd, ents = read_dem(path)
    xs = x_detectors(path)
    zmap = {i: k for k, i in enumerate(i for i in range(nd) if i not in xs)}
    red, pure = {}, set()
    for ds, ob, p in ents:
        zs = tuple(zmap[i] for i in ds if i in zmap)
        key = (zs, ob & 1)
        if not zs and not key[1]:
            continue
        red[key] = p
        if all(i in zmap for i in ds):
            pure.add(key)
    keys = list(red)
    nz = len(zmap)
    w = WCNF()
    nv = len(keys)
    rows = [[] for _ in range(nz + 1)]
    for j, (zs, ob) in enumerate(keys):
        for i in zs:
            rows[i].append(j + 1)
        if ob:
            rows[nz].append(j + 1)
    for i, r in enumerate(rows):
        tgt = i == nz
        if not r:
            continue
        acc = r[0]
        for v in r[1:]:
            nv += 1
            t = nv
            w.append([-t, acc, v]); w.append([-t, -acc, -v]); w.append([t, -acc, v]); w.append([t, acc, -v])
            acc = t
        w.append([acc] if tgt else [-acc])
    for j in range(len(keys)):
        w.append([-(j + 1)], weight=1)
    inv = {}
    for i, k in zmap.items():
        inv[k] = i
    return w, keys, pure, inv


def wait_for_bench_lock():
    # be polite on the shared Mac: never compute while a peer's timing run holds the lock
    while os.path.isdir("/tmp/qsim-mac-bench.lock"):
        time.sleep(5)


def evaluate(cs, d, rounds, sched, cap=30, workdir="/tmp"):
    wait_for_bench_lock()
    with tempfile.NamedTemporaryFile("w", dir=workdir, suffix=".sched", delete=False) as f:
        for s in sched:
            f.write(" ".join(map(str, s)) + "\n")
        sp = f.name
    dp = sp + ".dem"
    try:
        subprocess.run([cs, "dem", str(d), str(rounds), "cnot", "0.001", sp, dp], check=True)
        w, keys, pure, zinv = zsector_wcnf(dp)
    finally:
        os.remove(sp)
        if os.path.exists(dp):
            os.remove(dp)
    t = time.time()
    with RC2(w) as rc2:
        first = None
        count = 0
        certified = True
        involved = set()
        for model in rc2.enumerate():
            cost = rc2.cost
            if first is None:
                first = cost
            if cost > first or count >= cap:
                break
            count += 1
            chosen = [keys[j] for j in range(len(keys)) if model[j] > 0]
            certified &= all(k in pure for k in chosen)
            for zs, _ in chosen:
                involved.update(zinv[z] for z in zs)
    return dict(distance=first, n_min=count, capped=count >= cap, certified=certified,
                involved_z_detectors=sorted(involved), sat_s=round(time.time() - t, 2))


def main():
    cs, d, rounds, iters, seed, outp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    P = layout(cs, d)
    np_ = len(P)
    out = open(outp, "a")
    if "--uniform-csv" in sys.argv:
        import csv, ast
        rows = list(csv.DictReader(open(sys.argv[sys.argv.index("--uniform-csv") + 1])))
        lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else len(rows)
        for r in rows[:lim]:
            by = [ast.literal_eval(r["r_schedule"]), ast.literal_eval(r["g_schedule"]), ast.literal_eval(r["b_schedule"])]
            sched = [by[p["color"]] for p in P]
            res = evaluate(cs, d, rounds, sched, cap=int(os.environ.get("CAP", "30")))
            res.update(index=r["index"], schedule=by, d=d, rounds=rounds)
            res.pop("involved_z_detectors")
            out.write(json.dumps(res) + "\n"); out.flush()
        return
    rng = random.Random(seed)
    # data -> plaquettes
    dq_pl = {}
    for p in P:
        for k, q in enumerate(p["data"]):
            if q >= 0:
                dq_pl.setdefault(q, []).append((p["i"], k))
    boundary_q = {q for q, l in dq_pl.items() if len(l) < 3}
    free = [p["i"] for p in P if any(q in boundary_q for q in p["data"] if q >= 0)]
    sched = [list(KF[p["color"]]) for p in P]
    # options per free plaquette: assignments of steps to present positions, as full 6-lists
    def options(pi):
        p = P[pi]
        pres = [k for k in range(6) if p["data"][k] >= 0]
        opts = []
        for steps in itertools.permutations(range(1, 7), len(pres)):
            s = [0] * 6
            for k, t in zip(pres, steps):
                s[k] = t
            opts.append(s)
        return opts
    opts = {pi: options(pi) for pi in free}

    def ok(pi, s):
        p = P[pi]
        for k in range(6):
            q = p["data"][k]
            if q < 0:
                continue
            for (o, ko) in dq_pl[q]:
                if o != pi and sched[o][ko] == s[k]:
                    return False
        return True

    det_pl = None
    cur = evaluate(cs, d, rounds, sched)
    best = (cur["distance"], -cur["n_min"])
    out.write(json.dumps(dict(it=0, kind="start-KF", d=d, rounds=rounds, free=free, **cur)) + "\n"); out.flush()
    # detector index -> plaquette: Z detectors are listed per round as np_ plaquettes, X ones from
    # round 1 on (see ColorCode::memory); recover from structure
    def det_plaquette(i):
        # round 0: Z (np_); round r>=1: Z (np_) then X (np_); final: Z (np_)
        if i < np_:
            return i
        j = i - np_
        return (j % (2 * np_)) % np_ if j < 2 * np_ * (rounds - 1) else (j - 2 * np_ * (rounds - 1)) % np_
    best_sched = [s[:] for s in sched]
    for it in range(1, iters + 1):
        inv = {det_plaquette(i) for i in cur["involved_z_detectors"]}
        near = [pi for pi in free if pi in inv or any(
            q in [qq for o in inv for qq in P[o]["data"] if qq >= 0] for q in P[pi]["data"] if q >= 0)]
        pool = near if near and rng.random() < 0.8 else free
        pi = rng.choice(pool)
        cand = [s for s in opts[pi] if ok(pi, s) and s != sched[pi]]
        if not cand:
            continue
        old = sched[pi]
        sched[pi] = rng.choice(cand)
        res = evaluate(cs, d, rounds, sched)
        score = (res["distance"], -res["n_min"])
        cur_score = (cur["distance"], -cur["n_min"])
        accept = score >= cur_score or rng.random() < 0.05
        rec = dict(it=it, plaquette=pi, new=sched[pi], old=old, accept=accept, **{k: v for k, v in res.items() if k != "involved_z_detectors"})
        if accept:
            cur = res
            if score > best:
                best = score
                best_sched = [s[:] for s in sched]
                rec["best"] = True
                rec["schedule"] = best_sched
        else:
            sched[pi] = old
        out.write(json.dumps(rec) + "\n"); out.flush()
    out.write(json.dumps(dict(final_best=best, schedule=best_sched)) + "\n")


if __name__ == "__main__":
    main()
