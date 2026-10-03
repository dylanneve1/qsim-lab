#!/usr/bin/env python3
"""Large-neighbourhood search for single-auxiliary colour-code schedules.

Start: Kishony-Fowler's colour schedule (or --base file). Repeat:
  * compute the exact circuit distance, the exact number of minimum-weight logicals and the set
    of plaquettes those logicals touch ("involved");
  * for each involved plaquette (then each adjacent involved pair, most frequent first), enumerate
    EVERY DEM-equivalence class of its collision-free schedules (others fixed) and evaluate each
    exactly; move to the best one if it improves (distance, -count) lexicographically.
Stops at a schedule that no single- or pair-plaquette change improves (a certified local optimum
for those neighbourhoods) or after --max-moves.

usage: color_lns.py <color_search> <d> <rounds> <workers> <out.jsonl> [--pairs N] [--base f] [--max-moves M]
"""
import sys, os, json, subprocess, time
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from color_exhaustive import KF, layout, enumerate_classes, wait_lock
import tempfile


def full_eval(cs, d, rounds, sched, cap=1_000_000):
    wait_lock()
    with tempfile.NamedTemporaryFile("w", suffix=".sched", delete=False) as f:
        for s in sched:
            f.write(" ".join(map(str, s)) + "\n")
        sp = f.name
    try:
        out = subprocess.run([cs, "distance", str(d), str(rounds), sp, str(cap)], capture_output=True,
                             text=True, check=True, env=dict(os.environ, DUMP_ALL="1")).stdout
    finally:
        os.remove(sp)
    return json.loads(out)


def _task(args):
    cs, d, rounds, sched, cap = args
    r = full_eval(cs, d, rounds, sched, cap)
    return r["distance"] or 0, r["count"], r["count_capped"]


def main():
    cs, d, rounds, workers, outp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    npairs = int(sys.argv[sys.argv.index("--pairs") + 1]) if "--pairs" in sys.argv else 30
    max_moves = int(sys.argv[sys.argv.index("--max-moves") + 1]) if "--max-moves" in sys.argv else 100
    P = layout(cs, d)
    if "--base" in sys.argv:
        sched = [[int(t) for t in l.split()] for l in open(sys.argv[sys.argv.index("--base") + 1]) if l.strip()]
    else:
        sched = [list(KF[p["color"]]) for p in P]
    dq = {}
    for p in P:
        for k, q in enumerate(p["data"]):
            if q >= 0:
                dq.setdefault(q, []).append((p["i"], k))
    out = open(outp, "a")
    log = lambda **kw: (out.write(json.dumps(kw) + "\n"), out.flush())
    cur = full_eval(cs, d, rounds, sched)
    score = (cur["distance"], -cur["count"])
    log(kind="start", d=d, rounds=rounds, distance=cur["distance"], count=cur["count"], involved=cur["involved"], schedule=sched)
    pool = mp.get_context("fork").Pool(workers)
    moves = 0
    while moves < max_moves:
        inv = cur["involved"]
        groups = [[pi] for pi in inv]
        adj = []
        for a in inv:
            for b in inv:
                if a < b and set(q for q in P[a]["data"] if q >= 0) & set(q for q in P[b]["data"] if q >= 0):
                    adj.append([a, b])
        groups += adj[:npairs]
        improved = False
        for g in groups:
            t = time.time()
            reps, raw = enumerate_classes(P, sched, dq, g)
            assigns = list(reps.values())
            tasks = []
            for asg in assigns:
                s = [list(x) for x in sched]
                for pi, v in asg.items():
                    s[pi] = v
                tasks.append((cs, d, rounds, s, cur["count"]))
            res = pool.map(_task, tasks, chunksize=1)
            best_i, best_sc = None, score
            for i, (dist, cnt, capped) in enumerate(res):
                sc = (dist, -cnt)
                if dist > score[0] or (dist == score[0] and not capped and cnt < -score[1]):
                    if best_i is None or sc > best_sc:
                        best_i, best_sc = i, sc
            hist = {}
            for dist, cnt, capped in res:
                hist[dist] = hist.get(dist, 0) + 1
            log(kind="group", group=g, classes=len(assigns), raw=raw, seconds=round(time.time() - t, 1),
                distance_hist=hist, improved=best_i is not None)
            if best_i is not None:
                for pi, v in assigns[best_i].items():
                    sched[pi] = v
                cur = full_eval(cs, d, rounds, sched)
                score = (cur["distance"], -cur["count"])
                moves += 1
                log(kind="move", group=g, new={pi: sched[pi] for pi in g}, distance=cur["distance"],
                    count=cur["count"], involved=cur["involved"], schedule=sched)
                improved = True
                break
        if not improved:
            log(kind="local_optimum", distance=cur["distance"], count=cur["count"], schedule=sched,
                neighbourhoods=dict(singles=len(inv), pairs=len(adj[:npairs])))
            break
    pool.close()


if __name__ == "__main__":
    main()
