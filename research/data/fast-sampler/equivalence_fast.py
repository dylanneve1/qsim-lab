#!/usr/bin/env python3
"""10^6-shot equivalence of the Poisson-hit FastSampler with Stim 1.16 on identical circuits.

Re-uses research/data/qec-r4/stim_equivalence.py unchanged (tests T0-T3, both directions,
1% family-wise error, Bonferroni), but our side samples with `stim_compare sample-fast` /
`sample-native-fast` (FastSampler, Xoshiro256++, 256-shot batches) instead of the old path.

usage: equivalence_fast.py <shots> <d,d,...> <p> <out.jsonl> [directions=AB]
env: QSIM_STIM_COMPARE (binary), WORK (scratch dir), QSIM_FAST_RNG = wy (default) | xo
"""
import sys, os, json, time, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qec-r4"))
import stim_equivalence as E

_run = subprocess.run
def run_fast(args, *a, **k):
    args = list(args)
    if len(args) > 1 and args[1] in ("sample", "sample-native"):
        args[1] += "-fast"
        args.append(RNG)
    return _run(args, *a, **k)
E.subprocess.run = run_fast
RNG = os.environ.get("QSIM_FAST_RNG", "wy")

shots = int(sys.argv[1])
ds = [int(x) for x in sys.argv[2].split(",")]
E.P = float(sys.argv[3])
out = sys.argv[4]
dirs = sys.argv[5] if len(sys.argv) > 5 else "AB"
for direction in dirs:
    for d in ds:
        t = time.time()
        r = E.run_cell(direction, d, shots, seed=300 + d)
        r["p"] = E.P
        r["sampler"] = "FastSampler (Poisson hits, %s)" % {"wy": "wyrand", "xo": "Xoshiro256++"}[RNG]
        r["wall_s"] = round(time.time() - t, 1)
        print(json.dumps(r), flush=True)
        open(out, "a").write(json.dumps(r) + "\n")
