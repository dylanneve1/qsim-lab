#!/usr/bin/env python3
"""Collect (contraction cost, measured time) pairs for the planner's TN model.

    python3 calib.py --bin <target>/release/examples/tn_bench --out calib.jsonl [--threads 1]

Runs `tn_bench calib FAMILY N DEPTH SEED --run --reps 3` for a grid of random
instances (four families), one process each, with RAYON_NUM_THREADS fixed, and
appends the JSON lines (path stage + run stages) to --out. Instances whose
predicted cost is above --max-log10 are searched but not contracted.
Wrap the whole call in `flock /dev/shm/qsim/bench.lock` for timing runs.
"""
import argparse
import json
import os
import subprocess
import sys

GRID = {
    "brick": [(n, d) for n in (16, 24, 32, 40) for d in (4, 8, 12, 16, 20)],
    "sycpat": [(n, d) for n in (18, 24, 30, 36, 42, 48) for d in (4, 6, 8, 10, 12)],
    "qaoa": [(n, d) for n in (16, 20, 24, 28, 32) for d in (1, 2, 3)],
    "rqc": [(n, d) for n in (16, 20, 24, 28) for d in (2, 4, 6, 8)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", default="1")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--max-log10", type=float, default=10.0)
    ap.add_argument("--families", default=",".join(GRID))
    ap.add_argument("--timeout", type=float, default=120.0)
    a = ap.parse_args()
    env = dict(os.environ, RAYON_NUM_THREADS=a.threads)
    with open(a.out, "a") as out:
        for fam in a.families.split(","):
            for (n, d) in GRID[fam]:
                for seed in range(a.seeds):
                    base = [a.bin, "calib", fam, str(n), str(d), str(seed),
                            "--trials", "16", "--secs", "20", "--mem-gb", "1"]
                    # search only first, to skip contractions that are too large
                    try:
                        r = subprocess.run(base, env=env, capture_output=True, text=True,
                                           timeout=a.timeout)
                    except subprocess.TimeoutExpired:
                        print(f"timeout {fam} {n} {d} {seed}", file=sys.stderr)
                        continue
                    lines = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
                    if not lines:
                        print(r.stderr, file=sys.stderr)
                        continue
                    if lines[0]["log10_sliced_flops"] > a.max_log10:
                        out.write(json.dumps(lines[0]) + "\n")
                        continue
                    try:
                        r = subprocess.run(base + ["--run", "--reps", "3"], env=env,
                                           capture_output=True, text=True, timeout=a.timeout)
                    except subprocess.TimeoutExpired:
                        print(f"run timeout {fam} {n} {d} {seed}", file=sys.stderr)
                        continue
                    for l in r.stdout.splitlines():
                        if l.startswith("{"):
                            out.write(l + "\n")
                    out.flush()
                    print(fam, n, d, seed, file=sys.stderr)


if __name__ == "__main__":
    main()
