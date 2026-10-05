#!/usr/bin/env python3
"""Planner v2 data (research/simulability/planner-v2.md §1): one process per
(instance, engine), `planner_v2 req ENGINE SPEC SEED`, which evolves once and
times every read-out (e, a1, a1k, prep, s1, s1k, s100k).

Resumable: rows already in OUT are skipped. Stops after --budget seconds
(the Mac bench lock is held by the caller). Pairs that the original sweep
measured as censored or slower than --skip-over seconds are recorded as
`skipped` with the prior time (they count as censored).

    collect_req.py --bin B --instances FILE --prior DIR... --out OUT.jsonl
                   [--workers 2] [--budget 150] [--skip-over 4]
    collect_req.py --bin B --plan v1:e,v2nc:e,rule:s1k,... --instances FILE --out OUT.jsonl
"""
import argparse, csv, glob, json, os, random, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ENGINES = ["sv", "sparse", "mps", "hsf", "cstate", "tableau"]


def load_prior(paths):
    prior = {}
    for p in paths:
        for f in glob.glob(os.path.join(p, "*.csv")):
            if ".mid" in f:
                continue
            for r in csv.DictReader(open(f)):
                k = (r["spec"], r["seed"], r["engine"])
                t = float(r["secs"]) if r["status"] == "ok" else None
                prior[k] = (r["status"], t)
    return prior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--instances", required=True)
    ap.add_argument("--prior", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--budget", type=float, default=150)
    ap.add_argument("--skip-over", type=float, default=4.0)
    ap.add_argument("--timeout", type=float, default=40)
    ap.add_argument("--jobs", default=None, help="file of 'SPEC SEED VARIANT:REQ' lines (end to end)")
    ap.add_argument("--engines", default=None, help="comma list (default: all state engines); hsfamp = HSF amplitudes only")
    ap.add_argument("--only-skipped", default=None,
                    help="with --engines hsfamp: only instances whose ENGINE row in this file was skipped")
    ap.add_argument("--plan", default=None,
                    help="VARIANT:REQ,... : run `plan VARIANT REQ` instead (end to end)")
    a = ap.parse_args()
    inst = [l.split() for l in open(a.instances) if l.strip()]
    if a.jobs:
        # explicit jobs, one "SPEC SEED VARIANT:REQ" per line, in file order
        a.plan = a.plan or "file"
    prior = load_prior(a.prior)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l)
                done.add((r["spec"], r["seed"], r["job"]))
            except Exception:
                pass
    jobs = []
    for spec, seed in ([] if a.jobs else inst):
        if a.plan:
            # comma-separated VARIANT:REQ pairs, run back to back per instance
            for vr in a.plan.split(","):
                jobs.append((spec, seed, vr))
            continue
        for e in (a.engines.split(",") if a.engines else ENGINES):
            if e == "tableau" and not (spec.startswith("ct:") and ",t=0," in spec):
                continue
            jobs.append((spec, seed, e))
    if a.only_skipped:
        sk = set()
        for l in open(a.only_skipped):
            r = json.loads(l)
            if r["job"] == "hsf" and r.get("skipped"):
                sk.add((r["spec"], r["seed"]))
        jobs = [j for j in jobs if (j[0], j[1]) in sk]
    if a.jobs:
        jobs = [tuple(l.split()) for l in open(a.jobs) if l.strip()]
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    if a.plan and not a.jobs:
        # instances in random order, an instance's variants back to back
        order = {k: i for i, k in enumerate(random.Random(7).sample(sorted({(j[0], j[1]) for j in jobs}), len({(j[0], j[1]) for j in jobs})))}
        jobs.sort(key=lambda j: order[(j[0], j[1])])
    else:
        random.Random(7).shuffle(jobs)
    t_start = time.time()
    env = dict(os.environ, RAYON_NUM_THREADS="1")
    out = open(a.out, "a")

    def run(job):
        spec, seed, e = job
        if time.time() - t_start > a.budget:
            return None
        row = dict(spec=spec, seed=seed, job=e)
        if not a.plan:
            pr = prior.get((spec, seed, e))
            if e != "hsfamp" and pr is not None and (pr[0] != "ok" or pr[1] > a.skip_over):
                row.update(ok=False, skipped=True, prior_status=pr[0], prior_secs=pr[1])
                return row
            cmd = ["nice", "-n", "10", a.bin, "req", e, spec, seed]
        else:
            var, rq = e.split(":")
            cmd = ["nice", "-n", "10", a.bin, "plan", var, rq, spec, seed]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout, env=env)
            wall = time.time() - t0
            line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
            try:
                row.update(json.loads(line))
            except Exception:
                row.update(ok=False, error=f"rc={p.returncode} {p.stderr[-300:]}")
            row["wall"] = wall
        except subprocess.TimeoutExpired:
            row.update(ok=False, error="timeout", wall=a.timeout)
        return row

    n = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for row in ex.map(run, jobs):
            if row is None:
                continue
            out.write(json.dumps(row) + "\n")
            out.flush()
            n += 1
    left = len(jobs) - n
    print(f"wrote {n}, left {left}, {time.time() - t_start:.0f}s", flush=True)
    sys.exit(0 if left == 0 else 3)


if __name__ == "__main__":
    main()
