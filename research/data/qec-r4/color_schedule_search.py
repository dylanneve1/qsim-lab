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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

KF = [[1, 3, 2, 5, 4, 6], [5, 1, 4, 3, 6, 2], [1, 5, 3, 6, 2, 4]]


def layout(cs, d):
    out = subprocess.run([cs, "layout", str(d)], capture_output=True, text=True, check=True).stdout
    P = []
    for l in out.splitlines():
        v = [int(t) for t in l.split()]
        P.append(dict(i=v[0], x=v[1], y=v[2], color=v[3], w=v[4], data=v[5:11]))
    return P


def wait_for_bench_lock():
    # be polite on the shared Mac: never compute while a peer's timing run holds the lock
    while os.path.isdir("/tmp/qsim-mac-bench.lock"):
        time.sleep(5)


def evaluate(cs, d, rounds, sched, cap=100000, workdir="/tmp"):
    """Exact Z-memory circuit distance + exact number of minimum-weight logicals (Rust
    branch-and-bound, `color_search distance`)."""
    wait_for_bench_lock()
    with tempfile.NamedTemporaryFile("w", dir=workdir, suffix=".sched", delete=False) as f:
        for s in sched:
            f.write(" ".join(map(str, s)) + "\n")
        sp = f.name
    try:
        out = subprocess.run([cs, "distance", str(d), str(rounds), sp, str(cap)], capture_output=True,
                             text=True, check=True).stdout
    finally:
        os.remove(sp)
    r = json.loads(out)
    inv = set()
    for tok in r["example"].replace("[", " ").replace("]", " ").split():
        if "@" in tok:
            inv.add(int(tok.split("@")[0]))
    return dict(distance=r["distance"], n_min=r["count"], capped=r["count_capped"],
                certified=r["certified"], involved_plaquettes=sorted(inv), seconds=r["seconds"])


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
            res.pop("involved_plaquettes")
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
    best_sched = [s[:] for s in sched]
    for it in range(1, iters + 1):
        inv = set(cur["involved_plaquettes"])
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
        # never accept a lower distance; occasionally accept more minimum-weight logicals
        accept = score >= cur_score or (score[0] == cur_score[0] and rng.random() < 0.1)
        rec = dict(it=it, plaquette=pi, new=sched[pi], old=old, accept=accept, **{k: v for k, v in res.items() if k != "involved_plaquettes"})
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
