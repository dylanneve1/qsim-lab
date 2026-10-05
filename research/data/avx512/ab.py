#!/usr/bin/env python3
"""Interleaved A/B timing of qsim-lab builds/configurations on gate-list files.

Each cell (workload, n, precision) is timed under the swarm's bench lock
(/dev/shm/qsim/bench.lock, held per cell only): `rounds` rounds, each running
every configuration once in a rotating order, one process per run
(examples/sv_file_bench.rs, `reps` in-process repetitions, min taken). The
1-min load average and MemAvailable are recorded before and after each cell.

usage: ab.py --out results.csv --threads 8 --prec f32 --n 24,26 \
             --wl qft,brick_cz --rounds 3 [--reps 1] [--circuits DIR] \
             --cfg NAME=BINARY[@ENV=VAL,...][:key=val,...] ...
e.g.   --cfg main=/dev/shm/qsim/avx512/bin/sv_file_bench-main \
       --cfg avx2=/dev/shm/qsim/avx512/bin/new@QSIM_NO_AVX512=1 \
       --cfg avx512=/dev/shm/qsim/avx512/bin/new:block_kib=1024
Threads: RAYON_NUM_THREADS; 8 threads are pinned one per physical core
(taskset -c 0,2,..,14; vCPU siblings are (0,1), (2,3), ...), 16 to 0-15.
"""
import argparse
import csv
import fcntl
import os
import subprocess
import sys
import time

LOCK = "/dev/shm/qsim/bench.lock"


def parse_cfg(spec):
    name, rest = spec.split("=", 1)
    keys = []
    if ":" in rest:
        rest, kv = rest.split(":", 1)
        keys = [k for k in kv.split(",") if k]
    env = {}
    if "@" in rest:
        rest, ev = rest.split("@", 1)
        for e in ev.split(","):
            k, v = e.split("=", 1)
            env[k] = v
    return name, rest, env, keys


def load1():
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def mem_avail_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    return -1.0


def pin(threads):
    if threads == 8:
        return ["taskset", "-c", "0,2,4,6,8,10,12,14"]
    if threads == 16:
        return ["taskset", "-c", "0-15"]
    return []


def run_one(binary, env, keys, path, prec, reps, threads):
    e = dict(os.environ)
    e.update(env)
    e["RAYON_NUM_THREADS"] = str(threads)
    cmd = pin(threads) + [binary, path, prec, str(reps)] + keys
    out = subprocess.run(cmd, env=e, capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("|"):
            f = [x.strip() for x in line.strip("|").split("|")]
            return float(f[5]), f[8]
    raise RuntimeError(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--prec", default="f32")
    ap.add_argument("--n", default="24")
    ap.add_argument("--wl", default="qft")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--circuits", default="/dev/shm/qsim/avx512/circuits")
    ap.add_argument("--cfg", action="append", required=True)
    ap.add_argument("--max-load", type=float, default=1e9,
                    help="skip (record as skipped) a cell whose pre-cell load exceeds this")
    a = ap.parse_args()
    cfgs = [parse_cfg(c) for c in a.cfg]
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fo:
        w = csv.writer(fo)
        if new:
            w.writerow(["cfg", "workload", "n", "prec", "threads", "round", "seconds",
                        "norm", "load1_before", "load1_after", "memavail_gb_before", "t_unix"])
        for n in [int(x) for x in a.n.split(",")]:
            for wl in a.wl.split(","):
                path = os.path.join(a.circuits, f"{wl}_{n}.txt")
                with open(LOCK, "w") as lk:
                    fcntl.flock(lk, fcntl.LOCK_EX)
                    l0, m0 = load1(), mem_avail_gb()
                    if l0 > a.max_load:
                        print(f"skip {wl} n={n}: load {l0}", file=sys.stderr)
                        continue
                    rows = []
                    for r in range(a.rounds):
                        order = cfgs[r % len(cfgs):] + cfgs[:r % len(cfgs)]
                        for name, binary, env, keys in order:
                            t, norm = run_one(binary, env, keys, path, a.prec, a.reps, a.threads)
                            rows.append([name, wl, n, a.prec, a.threads, r, f"{t:.6f}", norm])
                    l1 = load1()
                for row in rows:
                    w.writerow(row + [l0, l1, f"{m0:.1f}", int(time.time())])
                fo.flush()
                best = {}
                for row in rows:
                    best[row[0]] = min(best.get(row[0], 1e9), float(row[6]))
                base = best[cfgs[0][0]]
                summary = "  ".join(f"{k}={v:.4f}s ({base / v:.2f}x)" for k, v in best.items())
                print(f"{wl} n={n} {a.prec} t={a.threads} load {l0:.1f}->{l1:.1f}: {summary}",
                      flush=True)


if __name__ == "__main__":
    main()
