#!/usr/bin/env python3
"""autoimprove: propose -> verify -> benchmark -> keep, for qsim-lab kernels.

Runs on the benchmark host (the M1 Pro, see README.md). Standard library only.

Pipeline for one candidate (a unified diff against the base commit):

  1. build    reset the eval tree to the base commit, `git apply` the patch,
              drop in the harness drivers (rust/ai_bench.rs -> examples/,
              rust/ai_gate.rs -> tests/), `cargo build --release --example
              ai_bench` (-j 2).
  2. verify   `cargo test --release --test ai_gate` (differential fuzz vs the
              independent reference SV of tests/audit_common: every Gate
              variant, adversarial block configs, f32+f64, thousands of
              circuits) plus the module's own test files. Any failure or
              mismatch -> reject.
  3. bench    interleaved A/B (A = base binary, B = candidate binary), one
              process per measurement (min of up to 9 in-process timings),
              ABBA order, 4 processes per side; metric = single-thread
              process CPU time (see METRIC). A cheap screen suite (n = 20,
              22) first; survivors run the full suite (QFT / brickwork /
              random Clifford+T, n = 20, 22, 24, f32 + f64). Cases that look
              slower are re-measured once (pooled). Every A/B pair also
              compares a fingerprint of the final state (reject on
              mismatch). Timings run under the Mac bench lock in chunks of
              <= 140 s with >= 65 s gaps and at most 19 locked minutes per
              rolling hour. `confirm` re-times a winner multi-threaded
              (wall clock, n = 20, 23, 26).
  4. decide   accept iff the geo-mean speedup is >= 1.03 by both the
              ratio of mins and the median paired ratio, and no case is
              below 0.97x by both.
  5. log      append one JSON record (diff, build/test logs, every timing,
              load averages, verdict) to the ledger.

Nothing here pushes anywhere: accepted patches are copied to
<work>/accepted/ for a human (or the parent agent) to turn into branches.

Subcommands (see README.md):
  setup  --base REF                 create the eval worktree + base binary
  eval   PATCH [PATCH ...]          run candidates through the pipeline
  queue                             process <work>/queue/*.patch forever
  aa                                A/A noise calibration (base vs base)
  knobs  SPEC.json                  runtime-knob search on the base binary
  consts SPEC.json                  compile-time-constant search (patches)
  summary                           ledger summary table
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("AI_WORK", os.path.expanduser("~/qsim-ai"))
REPO = os.environ.get("AI_REPO", os.path.expanduser("~/qsim-lab"))
CARGO = os.environ.get("AI_CARGO", shutil.which("cargo") or "/opt/homebrew/bin/cargo")
TREE = os.path.join(WORK, "tree")
TARGET = os.path.join(WORK, "target")
BIN = os.path.join(WORK, "bin")
LEDGER = os.environ.get("AI_LEDGER", os.path.join(WORK, "ledger.jsonl"))
LOCK = "/tmp/qsim-mac-bench.lock"
LOCKLOG = os.path.join(WORK, "lock-intervals.json")

JOBS = os.environ.get("AI_JOBS", "2")  # MAC CORE BUDGET: 2 workers
GATE_THREADS = os.environ.get("AI_GATE_THREADS", "2")
BENCH_THREADS = os.environ.get("AI_BENCH_THREADS", "1")
# "cpu": process CPU time (default; with 1 thread it ignores descheduling,
# the dominant noise on the shared Mac); "wall": wall-clock seconds.
METRIC = os.environ.get("AI_METRIC", "cpu")
GATE_CASES = os.environ.get("AI_GATE_CASES", "4000")
CHUNK_S = float(os.environ.get("AI_CHUNK_S", "140"))  # <= 150 s per locked chunk
GAP_S = float(os.environ.get("AI_GAP_S", "65"))  # >= 60 s between chunks
HOUR_BUDGET_S = float(os.environ.get("AI_HOUR_BUDGET_S", "1140"))  # < 20 min/h
REPS = int(os.environ.get("AI_REPS", "4"))  # processes per side, ABBA order
INNER = os.environ.get("AI_INNER", "9")  # max timings per process (stops at 0.4 s)

ACCEPT_GEO = 1.03
ACCEPT_MIN = 0.97
SCREEN_GEO = 1.01
SCREEN_MIN = 0.94

WLS = ("qft", "brick", "clifft")
PRECS = ("f32", "f64")
SCREEN = [(w, n, p) for n in (20, 22) for w in WLS for p in PRECS]
# single-thread timing: 20-24 qubits keeps a full run near 2-3 locked
# minutes; the multi-thread wall-clock confirmation (`confirm`) goes to 26
FULL = [(w, n, p) for n in (20, 22, 24) for w in WLS for p in PRECS]
CONFIRM = [(w, n, p) for n in (20, 23, 26) for w in WLS for p in PRECS]

# Module tests run in addition to the ai_gate fuzz (whichever exist).
# (graph / ooc / pipeline drive the blocked executor's planner and prepared
# stages from outside: the graph compiler maps ops to stage positions)
MODULE_TESTS = ["blocked", "l1_tiling", "simd", "dense_fusion", "differential_fuzz", "graph", "ooc", "pipeline"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sh(cmd, cwd=None, env=None, timeout=None, check=False):
    e = dict(os.environ)
    if env:
        e.update(env)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, env=e, timeout=timeout, capture_output=True, text=True,
                       shell=isinstance(cmd, str))
    dt = time.time() - t0
    if check and p.returncode != 0:
        raise RuntimeError(f"{cmd} failed ({p.returncode}):\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    return p, dt


def git(*a, cwd=TREE, check=True):
    p, _ = sh(["git", *a], cwd=cwd, check=check)
    return p.stdout.strip()


def loadavg():
    try:
        out = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout
        return [float(x) for x in out.strip("{} \n").split()[:3]]
    except Exception:
        return os.getloadavg()


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def append_ledger(rec):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# tree / build
# ---------------------------------------------------------------------------

def state():
    p = os.path.join(WORK, "state.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def save_state(s):
    json.dump(s, open(os.path.join(WORK, "state.json"), "w"), indent=1)


def install_drivers(tree):
    """Copy the drivers into the tree; strip `// @dense` lines when the tree
    has no dense fusion."""
    has_dense = "dense_fusion" in open(os.path.join(tree, "src/engines/blocked.rs")).read()
    for src, dst in (("ai_bench.rs", "examples/ai_bench.rs"), ("ai_gate.rs", "tests/ai_gate.rs")):
        text = open(os.path.join(HERE, "rust", src)).read()
        if not has_dense:
            text = "\n".join(l for l in text.split("\n") if "// @dense" not in l)
        open(os.path.join(tree, dst), "w").write(text)
    return has_dense


def reset_tree(base):
    git("reset", "-q", "--hard", base)
    git("clean", "-qfd", "-e", "target")


def apply_patch(path):
    p, _ = sh(["git", "apply", "--whitespace=nowarn", path], cwd=TREE)
    if p.returncode != 0:
        return False, p.stderr[-2000:]
    return True, ""


def cargo(args, env=None, timeout=3600):
    e = {"CARGO_TARGET_DIR": TARGET}
    if env:
        e.update(env)
    return sh(["nice", "-n", "10", CARGO, *args], cwd=TREE, env=e, timeout=timeout)


def build_bench(out_name):
    p, dt = cargo(["build", "--release", "-j", JOBS, "--example", "ai_bench"])
    ok = p.returncode == 0
    if ok:
        os.makedirs(BIN, exist_ok=True)
        shutil.copy2(os.path.join(TARGET, "release/examples/ai_bench"), os.path.join(BIN, out_name))
    return ok, dt, (p.stderr[-3000:] if not ok else "")


def run_gate(seed):
    """ai_gate fuzz (before any timing). Returns (ok, info)."""
    info = {"tests": []}
    env = {"RAYON_NUM_THREADS": GATE_THREADS, "AI_GATE_CASES": GATE_CASES, "AI_GATE_SEED": str(seed),
           "PROPTEST_CASES": "64"}
    p, dt = cargo(["test", "--release", "-j", JOBS, "--test", "ai_gate", "--", "--nocapture"], env=env)
    m = re.search(r"AI_GATE cases=(\d+) f64=(\d+) f32=(\d+) worst_f64=(\S+) worst_f32=(\S+)", p.stdout)
    info["ai_gate"] = {"ok": p.returncode == 0, "secs": round(dt, 1)}
    if m:
        info["ai_gate"].update(cases=int(m[1]), f64=int(m[2]), f32=int(m[3]),
                               worst_f64=float(m[4]), worst_f32=float(m[5]))
    if p.returncode != 0:
        info["ai_gate"]["log"] = (p.stdout + p.stderr)[-4000:]
        return False, info
    return True, info


def run_module_tests(info):
    """The module's own test files (slow to build in release: run only for
    candidates that pass the benchmark)."""
    env = {"RAYON_NUM_THREADS": GATE_THREADS, "PROPTEST_CASES": "64"}
    have = [t for t in MODULE_TESTS if os.path.exists(os.path.join(TREE, "tests", t + ".rs"))]
    args = ["test", "--release", "-j", JOBS]
    for t in have:
        args += ["--test", t]
    p, dt = cargo(args, env=env)
    passed = sum(int(x) for x in re.findall(r"test result: ok\. (\d+) passed", p.stdout))
    info["tests"] = {"files": have, "ok": p.returncode == 0, "passed": passed, "secs": round(dt, 1)}
    if p.returncode != 0:
        info["tests"]["log"] = (p.stdout + p.stderr)[-4000:]
        return False, info
    return True, info


def ensure_base(base_ref):
    st = state()
    base = git("rev-parse", base_ref)
    drv = sha(open(os.path.join(HERE, "rust/ai_bench.rs")).read())[:8]
    name = f"base-{base[:10]}-{drv}"
    if st.get("base_bin") == name and os.path.exists(os.path.join(BIN, name)):
        return base, name
    log("building base", base[:10])
    reset_tree(base)
    install_drivers(TREE)
    ok, dt, err = build_bench(name)
    if not ok:
        raise RuntimeError("base build failed:\n" + err)
    st.update(base=base, base_bin=name)
    save_state(st)
    return base, name


# ---------------------------------------------------------------------------
# timing under the bench lock
# ---------------------------------------------------------------------------

def lock_intervals():
    try:
        return json.load(open(LOCKLOG))
    except Exception:
        return []


def locked_in_last_hour():
    now = time.time()
    return sum(max(0.0, e - max(s, now - 3600)) for s, e in lock_intervals() if e > now - 3600)


def acquire(est):
    iv = lock_intervals()
    if iv:  # >= GAP_S since our last release, also across processes
        gap = iv[-1][1] + GAP_S - time.time()
        if gap > 0:
            time.sleep(gap)
    while locked_in_last_hour() + est > HOUR_BUDGET_S:
        log(f"hourly lock budget: {locked_in_last_hour():.0f}s used, waiting")
        time.sleep(60)
    waited = 0
    while True:
        try:
            os.mkdir(LOCK)
            return waited
        except FileExistsError:
            time.sleep(5)
            waited += 5


def release(t0):
    try:
        os.rmdir(LOCK)
    except FileNotFoundError:
        pass
    iv = [x for x in lock_intervals() if x[1] > time.time() - 7200]
    iv.append([t0, time.time()])
    json.dump(iv, open(LOCKLOG, "w"))


EST = {}  # (bin-agnostic) case -> seconds per process, learned


def est(case):
    w, n, p = case
    if case in EST:
        return EST[case]
    base = {"qft": 0.02, "brick": 0.25, "clifft": 0.15}[w] * (2 ** (n - 22)) * (1.8 if p == "f64" else 1.0)
    base *= {1: 3.5, 2: 1.9}.get(int(BENCH_THREADS), 1.0)
    return max(base, 0.4) + 0.15 + (2 ** n) * (16 if p == "f64" else 8) / 3e9


def run_one(binary, case, knobs):
    w, n, p = case
    cmd = [os.path.join(BIN, binary), w, str(n), p, INNER, *knobs]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True,
                         env={**os.environ, "RAYON_NUM_THREADS": BENCH_THREADS}, timeout=600)
    EST[case] = time.time() - t0
    if out.returncode != 0:
        raise RuntimeError(f"bench failed: {cmd}\n{out.stderr[-2000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def fp_ok(ra, rb, prec):
    tol = 1e-3 if prec == "f32" else 1e-8
    d = math.hypot(ra["fp"][0] - rb["fp"][0], ra["fp"][1] - rb["fp"][1])
    return d <= tol and abs(ra["norm"] - rb["norm"]) <= tol, d


def ab_bench(suite, a_bin, b_bin, a_knobs=(), b_knobs=(), reps=REPS, res=None, loads=None):
    """Interleaved A/B timing of `suite` under the lock. Returns the summary;
    `res`/`loads` (from an earlier call) are extended, so suspicious cases
    can be re-measured and pooled."""
    jobs = []
    for r in range(reps):
        for case in suite:
            pair = [("A", a_bin, a_knobs), ("B", b_bin, b_knobs)]
            if r % 2:
                pair.reverse()
            jobs.append((case, pair))
    res = {} if res is None else res
    for c in suite:
        res.setdefault(c, {"A": [], "B": [], "A_all": [], "B_all": [], "A_wall": [], "B_wall": [], "fp": []})
    loads = [] if loads is None else loads
    i = 0
    while i < len(jobs):
        t0 = time.time()
        acquire(min(CHUNK_S, sum(2 * est(c) for c, _ in jobs[i:])))
        t_lock = time.time()
        loads.append(loadavg())
        try:
            while i < len(jobs):
                case, pair = jobs[i]
                if time.time() - t_lock + 2.2 * est(case) > CHUNK_S and time.time() > t_lock + 1:
                    break
                outs = {}
                for tag, b, k in pair:
                    outs[tag] = run_one(b, case, list(k))
                    ts = outs[tag]["cpu" if METRIC == "cpu" else "times"]
                    res[case][tag].append(min(ts))
                    res[case][tag + "_all"].append(ts)
                    res[case][tag + "_wall"].append(min(outs[tag]["times"]))
                    res[case][tag + "_plan"] = [outs[tag].get("stages"), outs[tag].get("passes")]
                ok, d = fp_ok(outs["A"], outs["B"], case[2])
                res[case]["fp"].append(d)
                if not ok:
                    raise FingerprintMismatch(f"{case}: fingerprint differs by {d:e}")
                i += 1
        finally:
            release(t_lock)
        log(f"bench chunk: {i}/{len(jobs)} pairs, locked {time.time() - t_lock:.0f}s "
            f"(waited {t_lock - t0:.0f}s) load {loads[-1]}")
    return summarize(res, loads)


def case_ok(c, floor):
    """A case only counts as slower than `floor` when both statistics (ratio
    of mins, median of paired per-rep ratios) say so."""
    return max(c["speedup"], c["paired_median"]) >= floor


def measure(suite, a_bin, b_bin, floor, a_knobs=(), b_knobs=()):
    """ab_bench, then one re-measurement round (REPS more reps, pooled) for
    every case that looks slower than `floor`: on the shared Mac a single
    rep can land on an efficiency core or a frequency dip, and A/A runs show
    occasional 10% single-case outliers."""
    res, loads = {}, []
    s = ab_bench(suite, a_bin, b_bin, a_knobs, b_knobs, res=res, loads=loads)
    bad = [(c["wl"], c["n"], c["prec"]) for c in s["cases"] if not case_ok(c, floor)]
    if bad:
        log(f"re-measuring {len(bad)} suspicious case(s): {bad}")
        s = ab_bench(bad, a_bin, b_bin, a_knobs, b_knobs, res=res, loads=loads)
        s["remeasured"] = [list(b) for b in bad]
    return s


def verdict_of(s, geo_floor, case_floor):
    """(passes, reason): geo-mean over cases of both statistics >= geo_floor
    and no case below case_floor (see case_ok)."""
    if min(s["geo"], s["geo_paired_median"]) < geo_floor:
        return False, "geo"
    if not all(case_ok(c, case_floor) for c in s["cases"]):
        return False, "case"
    return True, ""


class FingerprintMismatch(Exception):
    pass


def summarize(res, loads):
    cases = []
    for (w, n, p), r in res.items():
        a, b = min(r["A"]), min(r["B"])
        paired = sorted(x / y for x, y in zip(r["A"], r["B"]))
        med = paired[len(paired) // 2] if len(paired) % 2 else 0.5 * (paired[len(paired) // 2 - 1] + paired[len(paired) // 2])
        cases.append({"wl": w, "n": n, "prec": p, "a": r["A"], "b": r["B"], "a_all": r["A_all"], "b_all": r["B_all"],
                      "a_wall": r["A_wall"], "b_wall": r["B_wall"],
                      "a_plan": r.get("A_plan"), "b_plan": r.get("B_plan"),
                      "a_min": a, "b_min": b, "speedup": a / b, "paired_median": med, "fp_max": max(r["fp"])})
    sp = [c["speedup"] for c in cases]
    geo = math.exp(sum(math.log(x) for x in sp) / len(sp))
    pm = [c["paired_median"] for c in cases]
    worst = min(max(c["speedup"], c["paired_median"]) for c in cases)
    return {"cases": cases, "geo": geo, "geo_paired_median": math.exp(sum(math.log(x) for x in pm) / len(pm)),
            "min": min(sp), "max": max(sp), "worst_case": worst, "load": loads,
            "threads": int(BENCH_THREADS), "reps": REPS, "metric": METRIC}


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------

def patch_meta(path):
    meta = {"slug": os.path.splitext(os.path.basename(path))[0], "idea": "", "family": ""}
    for line in open(path):
        m = re.match(r"#\s*(slug|idea|family|parent):\s*(.*)", line)
        if m:
            meta[m[1]] = m[2].strip()
        elif line.startswith("diff --git"):
            break
    return meta


def evaluate(path, base_ref="origin/main", screen_only=False, bench_knobs=()):
    base, base_bin = ensure_base(base_ref)
    meta = patch_meta(path)
    diff = open(path).read()
    rid = f"{time.strftime('%Y%m%dT%H%M%S')}-{meta['slug']}"
    rec = {"id": rid, "ts": time.time(), "kind": "patch", **meta, "base": base,
           "diff_sha": sha(diff)[:16], "diff": diff, "host": os.environ.get("AI_HOST", "m1-pro"),
           "bench_threads": int(BENCH_THREADS)}
    log("==", rid)

    def done(verdict, **kw):
        rec.update(kw, verdict=verdict)
        append_ledger(rec)
        log("verdict", meta["slug"], verdict)
        if verdict == "accepted":
            os.makedirs(os.path.join(WORK, "accepted"), exist_ok=True)
            shutil.copy2(path, os.path.join(WORK, "accepted", os.path.basename(path)))
        return rec

    reset_tree(base)
    ok, err = apply_patch(path)
    if not ok:
        return done("rejected:apply", apply_err=err)
    install_drivers(TREE)
    cand_bin = f"cand-{meta['slug']}"
    ok, dt, err = build_bench(cand_bin)
    rec["build"] = {"ok": ok, "secs": round(dt, 1)}
    if not ok:
        return done("rejected:build", build_err=err)
    ok, info = run_gate(int(sha(diff)[:8], 16))
    rec["gate"] = info
    if not ok:
        return done("rejected:gate")
    if os.environ.get("AI_TESTS_FIRST"):
        ok, info = run_module_tests(info)
        if not ok:
            return done("rejected:tests")
    try:
        scr = measure(SCREEN, base_bin, cand_bin, SCREEN_MIN, bench_knobs, bench_knobs)
    except FingerprintMismatch as e:
        return done("rejected:fingerprint", fp_err=str(e))
    rec["screen"] = scr
    log(f"screen geo {scr['geo']:.3f}/{scr['geo_paired_median']:.3f} worst {scr['worst_case']:.3f}")
    ok, why = verdict_of(scr, SCREEN_GEO, SCREEN_MIN)
    if not ok:
        return done("rejected:screen-" + why)
    if screen_only:
        return done("screened")
    try:
        full = measure(FULL, base_bin, cand_bin, ACCEPT_MIN, bench_knobs, bench_knobs)
    except FingerprintMismatch as e:
        return done("rejected:fingerprint", fp_err=str(e))
    rec["full"] = full
    log(f"full geo {full['geo']:.3f}/{full['geo_paired_median']:.3f} worst {full['worst_case']:.3f}")
    ok, why = verdict_of(full, ACCEPT_GEO, ACCEPT_MIN)
    if not ok:
        return done("rejected:full-" + why)
    if not os.environ.get("AI_TESTS_FIRST"):
        ok, info = run_module_tests(rec["gate"])
        if not ok:
            return done("rejected:tests")
    return done("accepted")


def cmd_eval(a):
    for p in a.patches:
        try:
            evaluate(p, a.base, a.screen_only)
        except Exception as e:  # keep going; record the crash
            append_ledger({"id": time.strftime("%Y%m%dT%H%M%S"), "ts": time.time(), "slug": os.path.basename(p),
                           "verdict": "error", "error": str(e)[-3000:]})
            log("error", p, e)


def cmd_queue(a):
    q = os.path.join(WORK, "queue")
    os.makedirs(os.path.join(q, "done"), exist_ok=True)
    while True:
        if os.path.exists(os.path.join(WORK, "STOP")):
            log("STOP file present, exiting")
            return
        pending = sorted(f for f in os.listdir(q) if f.endswith(".patch") or f.endswith(".json"))
        if not pending:
            time.sleep(20)
            continue
        p = os.path.join(q, pending[0])
        try:  # a broken base build must not consume the queue
            ensure_base(a.base)
        except Exception as e:
            log("base build failed; retrying in 300 s:", str(e)[-500:])
            time.sleep(300)
            continue
        try:
            if p.endswith(".json"):  # a knob / constant sweep spec, or a confirm request
                a.spec = p
                spec = json.load(open(p))
                if "fetch" in spec:  # roll the base forward (next item rebuilds it)
                    git("fetch", "-q", "origin")
                    log("fetched; origin/main is now", git("rev-parse", "--short", "origin/main"))
                elif "ci" in spec:
                    a.slug, a.patch = spec["ci"], os.path.join(WORK, spec["patch"])
                    a.light = spec.get("light", False)
                    cmd_ci(a)
                elif "confirm" in spec:
                    a.slug = spec["confirm"]
                    cmd_confirm(a)
                elif "aa" in spec:
                    a.runs, a.full = spec["aa"].get("runs", 1), spec["aa"].get("full", False)
                    a.metric, a.threads = spec["aa"].get("metric"), spec["aa"].get("threads")
                    cmd_aa(a)
                else:
                    (cmd_consts if "consts" in spec else cmd_knobs)(a)
            else:
                a.patches = [p]
                cmd_eval(a)
        except Exception as e:
            log("queue item failed", p, e)
        shutil.move(p, os.path.join(q, "done", pending[0]))
        if a.once:
            return


def cmd_setup(a):
    os.makedirs(WORK, exist_ok=True)
    sh(["git", "fetch", "-q", "origin"], cwd=REPO, check=True)
    if not os.path.exists(TREE):
        sh(["git", "worktree", "add", "--detach", TREE, a.base], cwd=REPO, check=True)
    git("fetch", "-q", "origin")
    print(ensure_base(a.base))


def cmd_aa(a):
    global METRIC, BENCH_THREADS
    METRIC = getattr(a, "metric", None) or METRIC
    BENCH_THREADS = getattr(a, "threads", None) or BENCH_THREADS
    base, base_bin = ensure_base(a.base)
    copy = base_bin + "-copy"
    shutil.copy2(os.path.join(BIN, base_bin), os.path.join(BIN, copy))
    for k in range(a.runs):
        suite = FULL if a.full else SCREEN
        r = ab_bench(suite, base_bin, copy)
        rec = {"id": f"{time.strftime('%Y%m%dT%H%M%S')}-aa{k}", "ts": time.time(), "kind": "aa",
               "slug": f"aa-{METRIC}-t{BENCH_THREADS}", "base": base, "suite": "full" if a.full else "screen", "result": r,
               "verdict": "calibration"}
        append_ledger(rec)
        log(f"A/A {k}: geo {r['geo']:.3f} min {r['min']:.3f} max {r['max']:.3f}")


def cmd_knobs(a):
    """Runtime knobs (BlockConfig fields): no rebuild. A = base binary with
    default config, B = base binary with the knob setting."""
    spec = json.load(open(a.spec))
    base, base_bin = ensure_base(a.base)
    if "grid" in spec:
        keys = list(spec["grid"].keys())
        combos = [dict(zip(keys, v)) for v in itertools.product(*spec["grid"].values())]
    else:  # explicit list, e.g. one-knob-at-a-time around the default
        combos = spec["configs"]
    suite = [tuple(x) for x in spec.get("suite", [])] or SCREEN
    # "binary": sweep on top of an evaluated candidate (e.g. "cand-<slug>")
    binary = spec.get("binary", base_bin)
    for c in combos:
        knobs = [f"{k}={v}" for k, v in c.items()]
        r = ab_bench(suite, binary, binary, (), knobs)
        append_ledger({"id": f"{time.strftime('%Y%m%dT%H%M%S')}-knobs", "ts": time.time(), "kind": "knobs",
                       "slug": "knobs:" + ",".join(knobs), "base": base, "binary": binary, "knobs": c, "result": r,
                       "verdict": "measured"})
        log("knobs", knobs, f"geo {r['geo']:.3f} min {r['min']:.3f}")


def cmd_consts(a):
    """Compile-time constants: each value becomes a one-line patch made by a
    regex substitution on the base source, then goes through `evaluate`."""
    spec = json.load(open(a.spec))
    base, _ = ensure_base(a.base)
    outdir = os.path.join(WORK, "gen")
    os.makedirs(outdir, exist_ok=True)
    for item in spec["consts"]:
        for v in item["values"]:
            reset_tree(base)
            if item.get("parent"):  # search on top of an earlier winner
                ok, err = apply_patch(os.path.join(WORK, item["parent"]))
                if not ok:
                    log("parent patch does not apply", item["parent"], err)
                    break
            fp = os.path.join(TREE, item["file"])
            src = open(fp).read()
            new, k = re.subn(item["pattern"], item["replace"].replace("{v}", str(v)), src, count=1)
            if k != 1 or new == src:
                log("skip (no match / unchanged)", item["name"], v)
                continue
            open(fp, "w").write(new)
            d = git("diff")
            slug = f"const-{item['name']}-{v}".replace(" ", "")
            p = os.path.join(outdir, slug + ".patch")
            par = f"# parent: {item['parent']}\n" if item.get("parent") else ""
            open(p, "w").write(f"# slug: {slug}\n# family: const:{item['name']}\n{par}# idea: {item['name']} = {v}\n{d}\n")
            try:
                evaluate(p, a.base, a.screen_only)
            except Exception as e:
                append_ledger({"id": time.strftime("%Y%m%dT%H%M%S"), "ts": time.time(), "slug": slug,
                               "verdict": "error", "error": str(e)[-3000:]})


def cmd_confirm(a):
    """Multi-thread wall-clock confirmation of an evaluated candidate (its
    binary must exist): CONFIRM suite (n = 20, 23, 26), AI_CONFIRM_THREADS
    threads (default 4)."""
    global METRIC, BENCH_THREADS
    base, base_bin = ensure_base(a.base)
    METRIC, BENCH_THREADS = "wall", os.environ.get("AI_CONFIRM_THREADS", "4")
    try:
        r = ab_bench(CONFIRM, base_bin, f"cand-{a.slug}")
    except FingerprintMismatch as e:
        r = {"error": str(e)}
    append_ledger({"id": f"{time.strftime('%Y%m%dT%H%M%S')}-confirm-{a.slug}", "ts": time.time(),
                   "kind": "confirm", "slug": a.slug, "base": base, "result": r,
                   "verdict": "measured" if "geo" in r else "rejected:fingerprint"})
    if "geo" in r:
        log(f"confirm {a.slug}: geo {r['geo']:.3f} min {r['min']:.3f}")


def cmd_ci(a):
    """CI-gate check of a patch on the base (in its own worktree and target
    dir, sequential with the loop so the 2-worker budget holds): cargo fmt
    --check, clippy --all-targets -D warnings, every example and test target
    built, and the blocked-executor tests run (CI runs the full suite)."""
    base, _ = ensure_base(a.base)
    ci_tree, ci_target = os.path.join(WORK, "ci"), os.path.join(WORK, "ci-target")
    if not os.path.exists(ci_tree):
        sh(["git", "worktree", "add", "--detach", ci_tree, base], cwd=TREE, check=True)
    git("reset", "-q", "--hard", base, cwd=ci_tree)
    git("clean", "-qfd", cwd=ci_tree)
    p, _ = sh(["git", "apply", a.patch], cwd=ci_tree)
    rec = {"id": f"{time.strftime('%Y%m%dT%H%M%S')}-ci-{a.slug}", "ts": time.time(), "kind": "ci",
           "slug": a.slug, "base": base, "steps": {}}
    env = {**os.environ, "CARGO_TARGET_DIR": ci_target, "RAYON_NUM_THREADS": GATE_THREADS}
    steps = [("apply", None)] if p.returncode else []
    if not steps:
        for name, cmd in [
            ("fmt", [CARGO, "fmt", "--all", "--", "--check"]),
            ("clippy", [CARGO, "clippy", "--all-targets", "-j", JOBS, "--", "-D", "warnings"]),
            ("examples", [CARGO, "build", "--release", "-j", JOBS, "--examples"]),
            ("tests-build", [CARGO, "test", "--release", "-j", JOBS, "--no-run"]),
            ("tests-blocked", [CARGO, "test", "--release", "-j", JOBS] + sum(
                (["--test", t] for t in MODULE_TESTS + ["properties", "cross_check"]
                 if os.path.exists(os.path.join(ci_tree, "tests", t + ".rs"))), [])),
        ]:
            if a.light and name not in ("fmt", "clippy"):
                continue
            q = subprocess.run(["nice", "-n", "10", *cmd], cwd=ci_tree, env=env, capture_output=True, text=True)
            rec["steps"][name] = {"ok": q.returncode == 0, "tail": (q.stdout + q.stderr)[-1500:] if q.returncode else ""}
            log(f"ci {a.slug} {name}: {'ok' if q.returncode == 0 else 'FAIL'}")
            if q.returncode:
                break
    rec["verdict"] = "ci-green" if rec["steps"] and all(v["ok"] for v in rec["steps"].values()) else "ci-red"
    append_ledger(rec)


def cmd_summary(a):
    rows = [json.loads(l) for l in open(a.ledger or LEDGER)]
    print("| id | kind | slug | verdict | screen geo/min | full geo/min |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        s = r.get("screen") or r.get("result") or {}
        f = r.get("full") or {}
        g = lambda x: f"{x['geo']:.3f}/{x['min']:.3f}" if x else ""
        print(f"| {r.get('id','')} | {r.get('kind','')} | {r.get('slug','')} | {r.get('verdict','')} | {g(s)} | {g(f)} |")


def main():
    # SIGTERM/SIGHUP -> SystemExit, so `finally` releases the bench lock
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: sys.exit(143))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # the base is pinned by <work>/BASE when present (the Mac clone's
    # origin/main moves whenever anyone fetches)
    pin = os.path.join(WORK, "BASE")
    ap.add_argument("--base", default=open(pin).read().strip() if os.path.exists(pin) else "origin/main")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    e = sub.add_parser("eval")
    e.add_argument("patches", nargs="+")
    e.add_argument("--screen-only", action="store_true")
    q = sub.add_parser("queue")
    q.add_argument("--screen-only", action="store_true")
    q.add_argument("--once", action="store_true", help="exit after one item (run.sh loops, picking up harness edits)")
    aa = sub.add_parser("aa")
    aa.add_argument("--runs", type=int, default=2)
    aa.add_argument("--full", action="store_true")
    k = sub.add_parser("knobs")
    k.add_argument("spec")
    c = sub.add_parser("consts")
    c.add_argument("spec")
    c.add_argument("--screen-only", action="store_true")
    cf = sub.add_parser("confirm")
    cf.add_argument("slug")
    s = sub.add_parser("summary")
    s.add_argument("--ledger")
    a = ap.parse_args()
    {"setup": cmd_setup, "eval": cmd_eval, "queue": cmd_queue, "aa": cmd_aa, "knobs": cmd_knobs,
     "consts": cmd_consts, "confirm": cmd_confirm, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
