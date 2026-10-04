#!/usr/bin/env python3
"""Simulability harness: features + every exact engine per instance.

Runs `simulability features` once per instance and `simulability run ENGINE`
in a fresh subprocess per engine (wall timeout, peak RSS from wait4).
Appends rows to a CSV and resumes where it left off. Stdlib only, so it runs
on the Mac as well as the VPS.

    driver.py --bin target/release/examples/simulability --grid ct24 \
        --out ct24.csv --timeout 5 --budget 150
"""
import argparse, csv, json, os, signal, subprocess, sys, time, itertools

ENGINES = ["sv", "sparse", "mps", "hsf", "tableau", "cstate", "frame", "dense", "auto"]
MONOTONE = {"ct": ["n", "L", "t"], "brick": ["n", "D"], "arith": ["bits", "h", "reps"],
            "qaoa": ["n", "p", "deg"]}
# Monotone in difficulty for *every* engine? Not quite (t for mps, n for
# frame), so dominance skipping is per engine.
ENGINE_MONO = {
    "sv": {"ct": ["n"], "brick": ["n"], "arith": ["bits"], "qaoa": ["n"]},
    "sparse": {"ct": ["n", "L"], "brick": ["n", "D"], "arith": ["bits", "h"], "qaoa": ["n"]},
    "mps": {"ct": ["n", "L", "t"], "brick": ["n", "D"], "arith": ["bits", "h", "reps"], "qaoa": ["n", "p", "deg"]},
    "hsf": {"ct": ["n", "L"], "brick": ["n", "D"], "arith": ["bits", "reps"], "qaoa": ["n", "p", "deg"]},
    "frame": {"ct": ["t"], "brick": ["n", "D"], "arith": ["h", "reps"], "qaoa": ["n", "p"]},
    "cstate": {"ct": ["t"], "brick": ["n", "D"], "arith": ["h", "reps"], "qaoa": ["n", "p"]},
    "dense": {"ct": ["t"], "brick": ["n", "D"], "arith": ["h", "reps"], "qaoa": ["n", "p"]},
    "auto": {"ct": ["t"], "brick": ["n", "D"], "arith": ["h", "reps"], "qaoa": ["n", "p"]},
    "tableau": {},
}
for _e in ENGINE_MONO:
    if _e != "tableau":
        ENGINE_MONO[_e]["hea"] = ["n", "D"] if _e != "sv" else ["n"]
        ENGINE_MONO[_e]["qft"] = ["n"]


def grid(name):
    """Named instance grids: list of (spec, seed). `file:PATH` reads lines
    `spec seed [engine,engine,...]` (a re-timing / confirmation pass)."""
    out = []
    if name.startswith("file:"):
        for line in open(name[5:]):
            parts = line.split()
            if len(parts) >= 2:
                out.append((parts[0], int(parts[1]), parts[2].split(",") if len(parts) > 2 else None))
        return out
    def add(fam, seeds=(1,), **ranges):
        keys = list(ranges)
        for vals in itertools.product(*[ranges[k] for k in keys]):
            spec = fam + ":" + ",".join(f"{k}={v}" for k, v in zip(keys, vals))
            for s in seeds:
                out.append((spec, s, None))
    if name == "smoke":
        add("ct", n=[10], L=[2, 4], t=[0, 6], nn=[1])
        add("brick", n=[10], D=[2, 6], nn=[1])
        add("arith", bits=[4], h=[1, 4], reps=[1, 2])
        add("qaoa", n=[10], p=[1, 2], deg=[3], nn=[0, 1])
    elif name == "ct24":
        add("ct", n=[24], L=[1, 2, 4, 8, 16, 32], t=[0, 2, 4, 8, 12, 16, 24, 32, 48, 64], nn=[1])
    elif name == "ct32":
        add("ct", n=[32], L=[1, 2, 4, 8, 16, 32], t=[0, 4, 8, 16, 24, 32, 48, 64], nn=[1])
    elif name == "ctnn0":
        add("ct", n=[20, 24], L=[1, 2, 4, 8], t=[0, 4, 8, 16, 24, 32], nn=[0])
    elif name == "brick":
        add("brick", n=[12, 16, 20, 24, 26], D=[1, 2, 3, 4, 6, 8, 12, 16], nn=[1])
        add("brick", n=[16, 24], D=[1, 2, 3, 4, 6], nn=[0])
    elif name == "arith":
        add("arith", bits=[6, 8, 10, 12], h=[0, 1, 2, 4, 6], reps=[1, 2, 4])
    elif name == "qaoa":
        add("qaoa", n=[12, 16, 20, 24], p=[1, 2, 3], deg=[2, 3], nn=[0, 1])
    elif name == "hea":  # held-out family for the planner study
        add("hea", n=[12, 16, 20, 24], D=[1, 2, 3, 4, 6, 8])
    elif name == "qft":  # held-out family for the planner study
        add("qft", n=[10, 14, 18, 22], h=[0, 2, 5])
    elif name == "small":  # dev grid for the VPS (n <= 16)
        add("ct", n=[12, 16], L=[1, 4, 16], t=[0, 4, 12, 24], nn=[1])
        add("brick", n=[12, 16], D=[1, 3, 6, 12], nn=[1])
        add("arith", bits=[5, 7], h=[0, 2, 5], reps=[1, 3])
        add("qaoa", n=[12, 16], p=[1, 3], deg=[3], nn=[0, 1])
    else:
        raise SystemExit(f"unknown grid {name}")
    return out


def parse_spec(spec):
    fam, rest = spec.split(":", 1)
    return fam, {k: float(v) for k, v in (kv.split("=") for kv in rest.split(","))}


def run(cmd, timeout, threads):
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                         start_new_session=True)
    deadline = t0 + timeout
    while True:
        pid, status, ru = os.wait4(p.pid, os.WNOHANG)
        if pid:
            break
        if time.perf_counter() > deadline:
            os.killpg(p.pid, signal.SIGKILL)
            pid, status, ru = os.wait4(p.pid, 0)
            wall = time.perf_counter() - t0
            return None, wall, ru.ru_maxrss, "timeout"
        time.sleep(0.002)
    wall = time.perf_counter() - t0
    out = p.stdout.read().decode()
    err = p.stderr.read().decode()
    rss = ru.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    try:
        return json.loads(out.strip().splitlines()[-1]), wall, rss, ""
    except Exception:
        return None, wall, rss, "crash: " + (err.strip().splitlines() or ["?"])[-1][:200]


FIELDS = ["family", "spec", "seed", "params", "engine", "status", "secs", "wall", "rss",
          "value", "size", "ref", "abs_err", "note", "features"]


def dominated(fam, params, prev_fail, engine):
    keys = ENGINE_MONO.get(engine, {}).get(fam)
    if not keys:
        return False
    for fp in prev_fail:
        if all((params[k] >= fp[k]) if k in keys else (params[k] == fp[k]) for k in params):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--budget", type=float, default=1e9, help="stop after this many seconds")
    ap.add_argument("--mem", type=int, default=1 << 30)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--engines", default=",".join(ENGINES))
    ap.add_argument("--reps", type=int, default=1, help="timing repeats (min taken)")
    ap.add_argument("--obs", default="all", help="observable: all | mid2 | mid4")
    a = ap.parse_args()
    engines = a.engines.split(",")
    done = {}
    fails = {}
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done[(r["spec"], r["seed"], r["engine"])] = r
            if r["status"] in ("timeout", "skipped", "toolarge"):
                fam, params = parse_spec(r["spec"])
                fails.setdefault((fam, r["engine"]), []).append(params)
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    start = time.perf_counter()
    for spec, seed, only in grid(a.grid):
        if time.perf_counter() - start > a.budget:
            print("budget exhausted", flush=True)
            return 3
        todo = [e for e in (only or engines) if (spec, str(seed), e) not in done]
        if not todo:
            continue
        fam, params = parse_spec(spec)
        feats, _, _, ferr = run([a.bin, "features", spec, str(seed), a.obs], 120, a.threads)
        if feats is None:
            print("features failed", spec, ferr, flush=True)
            continue
        rows = []
        for e in todo:
            row = dict(family=fam, spec=spec, seed=seed, params=json.dumps(params), engine=e,
                       features=json.dumps(feats), note="", value="", size="", secs="", ref="",
                       abs_err="")
            if dominated(fam, params, fails.get((fam, e), []), e):
                row.update(status="skipped", wall="", rss="")
                rows.append(row)
                continue
            best = None
            for rep in range(a.reps):
                res, wall, rss, err = run([a.bin, "run", e, spec, str(seed), str(a.mem), a.obs],
                                          a.timeout, a.threads)
                if res is None or not res.get("ok"):
                    break
                if best is None or res["secs"] < best[0]["secs"]:
                    best = (res, wall, rss)
            if best is None:
                if err == "timeout":
                    status = "timeout"
                elif res is not None and "non-Clifford" in res.get("error", ""):
                    status = "na"
                elif res is not None and "TooLarge" in res.get("error", "") or \
                        (res is not None and "TooManyTerms" in res.get("error", "")):
                    status = "toolarge"
                elif res is not None and "non-Clifford" in res.get("error", ""):
                    status = "na"
                else:
                    status = "error"
                row.update(status=status, wall=f"{wall:.4f}", rss=rss,
                           note=(err or (res or {}).get("error", ""))[:300])
                if status in ("timeout", "toolarge"):
                    fails.setdefault((fam, e), []).append(params)
            else:
                res, wall, rss = best
                row.update(status="ok", secs=res["secs"], wall=f"{wall:.4f}", rss=rss,
                           value=res["value"], size=res["size"], note=res.get("note", ""))
            rows.append(row)
        oks = {r["engine"]: float(r["value"]) for r in rows if r["status"] == "ok"}
        for r in done.values():
            if r["spec"] == spec and r["seed"] == str(seed) and r["status"] == "ok":
                oks.setdefault(r["engine"], float(r["value"]))
        if oks:
            ref = oks.get("sv")
            if ref is None:
                vals = sorted(oks.values())
                ref = vals[len(vals) // 2]
            for r in rows:
                r["ref"] = ref
                if r["status"] == "ok":
                    r["abs_err"] = abs(float(r["value"]) - ref)
        for r in rows:
            w.writerow(r)
            done[(spec, str(seed), r["engine"])] = r
        f.flush()
        summ = " ".join(f"{r['engine']}={r['status'] if r['status']!='ok' else '%.3g' % float(r['secs'])}"
                        for r in rows)
        bad = [r["engine"] for r in rows if r["abs_err"] != "" and float(r["abs_err"]) > 1e-6]
        print(f"{spec} s{seed} {summ}" + (f" MISMATCH {bad}" if bad else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
