#!/usr/bin/env python3
"""Run the Rust driver over prepared programs with a small worker pool (Mac core budget: 2).

  python run.py BIN DATA_DIR OUT.jsonl TASKS...     TASK = profile:GLOB | energy:GLOB:SWEEPS | check:GLOB

Each task line becomes one JSON line in OUT.jsonl (appended), with "task" and "wall" fields.
Skips (task, file) pairs already present in OUT.jsonl.
"""
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

WORKERS = int(os.environ.get("WORKERS", "2"))


def fcidump_for(prog):
    base = os.path.basename(prog).split(".")[0]
    return os.path.join(os.path.dirname(prog), f"{base}.fcidump")


def main():
    binp, data, out = sys.argv[1:4]
    jobs = []
    for t in sys.argv[4:]:
        kind, pat, *rest = t.split(":")
        for p in sorted(glob.glob(os.path.join(data, pat))):
            if kind == "profile":
                cmd = [binp, "profile", p, rest[0] if rest else "1024"]
            elif kind == "energy":
                cmd = [binp, "energy", fcidump_for(p), p, rest[0] if rest else "0", "26"]
            elif kind == "check":
                cmd = [binp, "check", fcidump_for(p), p]
            else:
                raise SystemExit(f"bad task {t}")
            jobs.append((kind, p, cmd))
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line)
                done.add((r["task"], r["file"]))
            except Exception:
                pass
    jobs = [j for j in jobs if (j[0], j[1]) not in done]
    print(f"{len(jobs)} jobs", flush=True)

    def run(job):
        kind, p, cmd = job
        t0 = time.time()
        env = dict(os.environ, RAYON_NUM_THREADS="1")
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=7200)
        wall = time.time() - t0
        try:
            rec = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            rec = {"error": (r.stderr or r.stdout)[-400:]}
        rec.update({"task": kind, "file": p, "wall": round(wall, 3)})
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(kind, os.path.basename(p), f"{wall:.1f}s", "ERR" if "error" in rec else "", flush=True)

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(run, jobs))


if __name__ == "__main__":
    main()
