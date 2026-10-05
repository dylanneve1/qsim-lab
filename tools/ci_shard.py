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
(CI run 37261204680: dev profile, opt-level 1, no debuginfo); refresh them from a CI log
when the balance drifts.
"""
import json
import subprocess
import sys

WEIGHTS = {
    "shor_noise": 274, "spd": 180, "ft_shor": 113, "theory_shor_mbu": 93, "theory_shor": 86,
    "shor_scale": 82, "theory_shor_opt": 68, "ooc": 27, "noise_oracles": 25, "theory_rank": 22,
    "mbu_shor": 20, "shor_r4_audit": 16, "superopt": 8.3, "theory_coset": 8, "symphase": 6.3,
    "ge_shor": 6, "hsf": 4, "colour_global": 3.1, "pauli_frame": 2.6, "audit_repeat": 2.4,
    "stabrank_lower": 2.3, "planner": 2.2, "l1_tiling": 2, "dense_fusion": 2, "simulability": 1.6,
}
DEFAULT_SECS = 1.0
UNIT_AND_DOC_SECS = 12.4 + 4.0  # lib unit tests + doctests, always shard 1


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
