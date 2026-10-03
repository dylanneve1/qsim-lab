#!/usr/bin/env python3
"""Exhaustive search over the schedules of a GROUP of plaquettes (all others fixed), up to
DEM-equivalence, of the single-auxiliary triangular colour code.

Equivalence (exact for the Z-memory noisy-CNOT DEM, see research/qec-r4.md §5.2): the Z-sector
DEM depends on a plaquette's schedule only through (a) the order of its own CNOTs (hooks from
the X half) and (b) for each of its data qubits, the order in which that qubit meets its 2-3
plaquettes during the Z half. Options with equal (a, b) keys give identical DEMs, so one
representative per key is evaluated.

usage: color_exhaustive.py <color_search> <d> <rounds> <p1,p2,..> <workers> <out.jsonl> [cap] [--base file]
"""
import sys, os, json, itertools, subprocess, tempfile, time
import multiprocessing as mp

KF = [[1, 3, 2, 5, 4, 6], [5, 1, 4, 3, 6, 2], [1, 5, 3, 6, 2, 4]]
CS = None


def layout(cs, d):
    P = []
    for l in subprocess.run([cs, "layout", str(d)], capture_output=True, text=True, check=True).stdout.splitlines():
        v = [int(t) for t in l.split()]
        P.append(dict(i=v[0], x=v[1], y=v[2], color=v[3], w=v[4], data=v[5:11]))
    return P


def wait_lock():
    while os.path.isdir("/tmp/qsim-mac-bench.lock"):
        time.sleep(5)


def evaluate(args):
    cs, d, rounds, sched, cap = args
    wait_lock()
    with tempfile.NamedTemporaryFile("w", suffix=".sched", delete=False) as f:
        for s in sched:
            f.write(" ".join(map(str, s)) + "\n")
        sp = f.name
    try:
        out = subprocess.run([cs, "distance", str(d), str(rounds), sp, str(cap)], capture_output=True,
                             text=True, check=True).stdout
    finally:
        os.remove(sp)
    r = json.loads(out)
    return dict(distance=r["distance"], count=r["count"], capped=r["count_capped"],
                certified=r["certified"], seconds=r["seconds"])


def enumerate_classes(P, sched, dq, group):
    """One representative assignment {plaquette: steps} per DEM-equivalence class of the
    collision-free joint schedules of `group` (others fixed at `sched`). Returns (reps, raw)."""
    gset = set(group)

    def raw_options(pi, assigned):
        p = P[pi]
        pres = [k for k in range(6) if p["data"][k] >= 0]
        for st in itertools.permutations(range(1, 7), len(pres)):
            s = [0] * 6
            for k, t in zip(pres, st):
                s[k] = t
            ok = True
            for k in pres:
                for o, ko in dq[p["data"][k]]:
                    if o == pi:
                        continue
                    if (o not in gset or o in assigned) and (assigned.get(o, sched[o]))[ko] == s[k]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                yield s

    def joint_key(assign):
        full = lambda o: assign.get(o, sched[o])
        key = []
        for pi in group:
            p = P[pi]
            s = full(pi)
            pres = [k for k in range(6) if p["data"][k] >= 0]
            order = tuple(sorted(pres, key=lambda k: s[k]))
            rel = tuple(tuple(sorted((full(o)[ko] < s[k], o) for o, ko in dq[p["data"][k]] if o != pi)) for k in pres)
            key.append((order, rel))
        return tuple(key)

    reps = {}
    n_raw = 0

    def rec(i, assigned):
        nonlocal n_raw
        if i == len(group):
            n_raw += 1
            k = joint_key(assigned)
            if k not in reps:
                reps[k] = dict(assigned)
            return
        pi = group[i]
        for s in raw_options(pi, assigned):
            assigned[pi] = s
            rec(i + 1, assigned)
            del assigned[pi]

    rec(0, {})
    return reps, n_raw


def main():
    cs, d, rounds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    group = [int(x) for x in sys.argv[4].split(",")]
    workers, outp = int(sys.argv[5]), sys.argv[6]
    cap = int(sys.argv[7]) if len(sys.argv) > 7 and not sys.argv[7].startswith("--") else 1_000_000
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
    reps, n_raw = enumerate_classes(P, sched, dq, group)
    keys = list(reps)
    if os.environ.get("COUNT_ONLY"):
        print(json.dumps(dict(group=group, raw=n_raw, classes=len(keys))))
        return
    out = open(outp, "a")
    out.write(json.dumps(dict(kind="header", d=d, rounds=rounds, group=group, raw=n_raw, classes=len(keys), cap=cap)) + "\n")
    out.flush()
    tasks = []
    for k in keys:
        s = [list(x) for x in sched]
        for pi, v in reps[k].items():
            s[pi] = v
        tasks.append((cs, d, rounds, s, cap))
    best = None
    with mp.get_context("fork").Pool(workers) as pool:
        for k, res in zip(keys, pool.imap(evaluate, tasks, chunksize=1)):
            rec_ = dict(group_schedules={pi: reps[k][pi] for pi in group}, **res)
            sc = (res["distance"] or 0, -res["count"])
            if best is None or sc > best[0]:
                best = (sc, rec_)
            out.write(json.dumps(rec_) + "\n")
            out.flush()
    out.write(json.dumps(dict(kind="best", score=best[0], **best[1])) + "\n")


if __name__ == "__main__":
    main()
