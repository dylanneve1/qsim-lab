#!/usr/bin/env python3
"""Deterministic MPS cost data for every simulability instance (no timing
claims): replayed cost per bound (`mpscost`), the real exact MPS run's
operation counts + bond trace check, and capped-probe predictions
(`mpstrace`). Usage: collect.py BIN OUT.jsonl [WORKERS] [CAPS]"""
import csv, glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "simulability", "raw")


def instances():
    seen = {}
    for p in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
        b = os.path.basename(p)
        if "mid2" in b or "confirm" in b:
            continue
        for r in csv.DictReader(open(p)):
            seen.setdefault((r["spec"], r["seed"]), r["family"])
    return [(f, s, sd) for (s, sd), f in seen.items()]


def run(binp, args, timeout):
    try:
        out = subprocess.run([binp] + args, capture_output=True, text=True, timeout=timeout)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa
        return {"error": repr(e)[:200]}


def main():
    binp, outp = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    caps = sys.argv[4] if len(sys.argv) > 4 else "8,16,32"
    done = set()
    if os.path.exists(outp):
        for line in open(outp):
            d = json.loads(line)
            done.add((d["spec"], d["seed"]))
    todo = [x for x in instances() if (x[1], x[2]) not in done]
    print(f"{len(todo)} instances to do", flush=True)

    def job(x):
        fam, spec, seed = x
        d = {"family": fam, "spec": spec, "seed": seed}
        d["cost"] = run(binp, ["mpscost", spec, seed], 120)
        d["trace"] = run(binp, ["mpstrace", spec, seed, caps], 300)
        return d

    with ThreadPoolExecutor(workers) as ex, open(outp, "a") as fh:
        for i, d in enumerate(ex.map(job, todo)):
            fh.write(json.dumps(d) + "\n")
            fh.flush()
            if i % 20 == 0:
                print(i, d["spec"], d["trace"].get("secs"), flush=True)


if __name__ == "__main__":
    main()
