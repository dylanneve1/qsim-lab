#!/usr/bin/env python3
"""Split the qsim-lab test targets into N balanced CI shards.

    tools/ci_shard.py SHARD N        # prints the `cargo test` target flags for shard SHARD (1-based)
    tools/ci_shard.py --table N      # prints the whole assignment with estimated seconds

Targets come from `cargo metadata`, so new tests/*.rs files are picked up
automatically (with DEFAULT_SECS until they are added to WEIGHTS). Shard 1
also runs the lib/bin unit tests and the doctests. Assignment is greedy
longest-first onto the least-loaded shard, deterministic for a given target
list, and every test target lands in exactly one shard.

WEIGHTS are run times in seconds measured on the GitHub ubuntu-latest runner
(CI run 37256235532, dev profile, opt-level 1); refresh them from a CI log
when the balance drifts.
"""
import json
import subprocess
import sys

WEIGHTS = {
    "spd": 136, "ge_shor": 108, "theory_shor_mbu": 92, "theory_shor_opt": 88,
    "shor_scale": 54, "noise_oracles": 33, "ooc": 24, "shor_r4_audit": 17,
    "mbu_shor": 16, "theory_rank": 12, "superopt": 12, "symphase": 7,
    "dense_fusion": 3.4, "hsf": 3.2, "planner_v2": 3.1, "colour_global": 3.0,
    "pauli_frame": 2.9, "stabrank_lower": 2.5, "audit_repeat": 2.1, "l1_tiling": 1.9,
}
DEFAULT_SECS = 2.0
UNIT_AND_DOC_SECS = 15.3 + 3.1  # lib unit tests + doctests, always shard 1


def test_targets():
    meta = json.loads(subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        capture_output=True, check=True, text=True).stdout)
    root = next(p for p in meta["packages"] if p["name"] == "qsim-lab")
    return sorted(t["name"] for t in root["targets"] if "test" in t["kind"])


def assign(n):
    shards = [[] for _ in range(n)]
    load = [0.0] * n
    load[0] += UNIT_AND_DOC_SECS
    for t in sorted(test_targets(), key=lambda t: (-WEIGHTS.get(t, DEFAULT_SECS), t)):
        i = min(range(n), key=lambda k: (load[k], k))
        shards[i].append(t)
        load[i] += WEIGHTS.get(t, DEFAULT_SECS)
    return shards, load


def main(argv):
    if len(argv) == 2 and argv[0] == "--table":
        shards, load = assign(int(argv[1]))
        for i, (s, l) in enumerate(zip(shards, load), 1):
            print(f"shard {i}: ~{l:.0f} s  {' '.join(s)}")
        return
    if len(argv) != 2:
        sys.exit(__doc__)
    k, n = int(argv[0]), int(argv[1])
    shards, _ = assign(n)
    flags = ["--lib", "--bins"] if k == 1 else []
    for t in shards[k - 1]:
        flags += ["--test", t]
    print(" ".join(flags))


if __name__ == "__main__":
    main(sys.argv[1:])
