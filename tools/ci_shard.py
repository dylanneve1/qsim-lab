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
(CI run 37269611586 on main: dev profile, opt-level 1, no debuginfo); refresh them from a CI log
when the balance drifts.
"""
import json
import subprocess
import sys

WEIGHTS = {
    "shor_noise": 353, "spd": 155, "ft_shor": 108, "theory_shor_opt": 97, "theory_shor": 91,
    "theory_shor_mbu": 75, "shor_scale": 68, "ooc": 40, "noise_oracles": 32, "theory_rank": 22,
    "shor_r4_audit": 18.5, "mbu_shor": 14, "superopt": 13.7, "theory_coset": 13.3, "symphase": 7,
    "identities": 6.9, "hsf": 3.8, "colour_global": 3.1, "stabrank_lower": 3.1, "planner": 3,
    "pauli_frame": 2.8, "audit_repeat": 2.5, "dense_fusion": 2.3, "l1_tiling": 1.7, "simulability": 1.7,
}
DEFAULT_SECS = 1.0
# Each extra test binary also costs ~17 s to compile and link on the runner
# (run 37262442574: 4 binaries built in 3m15s, 19 in 7m31s), so balance that too.
PER_TARGET_SECS = 17.0
UNIT_AND_DOC_SECS = 15.6 + 4.0  # lib unit tests + doctests, always shard 1


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
        load[i] += WEIGHTS.get(t, DEFAULT_SECS) + PER_TARGET_SECS
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
