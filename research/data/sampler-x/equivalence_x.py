#!/usr/bin/env python3
"""10^6-shot two-sample equivalence of the sampler-x pipeline with Stim on identical circuits.

Re-uses research/data/qec-r4/stim_equivalence.py unchanged (T0 DEM support, T1 marginals, T2 DEM-correlated
pairs, T3 mean/variance of events per shot; Bonferroni at 1% family-wise error per cell, both directions),
but our side runs `stim_compare sample-x` (fast parser, backward detector compiler, per-batch wyrand
streams) with the given thread count and table mode, and T0 reads the new compiler's hit tables
(`dem-support-x`). Direction A samples the exported .stim file through the parser (the old script sampled
the in-memory circuit).

usage: equivalence_x.py <shots> <d,d,...> <p> <threads> <tables:auto|on|off> <out.jsonl> [directions=AB]
env: QSIM_STIM_COMPARE (binary), WORK (scratch dir)
"""
import sys, os, json, time, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qec-r4"))
import stim_equivalence as E

shots = int(float(sys.argv[1]))
ds = [int(x) for x in sys.argv[2].split(",")]
E.P = float(sys.argv[3])
THREADS, TABLES, out = sys.argv[4], sys.argv[5], sys.argv[6]
dirs = sys.argv[7] if len(sys.argv) > 7 else "AB"

_run = subprocess.run


def run_x(args, *a, **k):
    args = [str(x) for x in args]
    if len(args) > 1 and args[1] == "sample":
        # sample <file> <shots> <out> <seed>
        args = [args[0], "sample-x", args[2], args[3], args[4], args[5], THREADS, TABLES]
    elif len(args) > 1 and args[1] == "sample-native":
        # sample-native <d> <p> <shots> <out> <seed>: the file export-surface just wrote
        args = [args[0], "sample-x", f"{E.WORK}/ours_d{args[2]}.stim", args[4], args[5], args[6], THREADS,
                TABLES]
    elif len(args) > 1 and args[1] == "dem-support":
        args = [args[0], "dem-support-x", args[2]]
    return _run(args, *a, **k)


E.subprocess.run = run_x
for direction in dirs:
    for d in ds:
        t = time.time()
        r = E.run_cell(direction, d, shots, seed=500 + d)
        r.update(p=E.P, sampler="sampler-x", threads=int(THREADS), tables=TABLES,
                 wall_s=round(time.time() - t, 1))
        print(json.dumps(r), flush=True)
        open(out, "a").write(json.dumps(r) + "\n")
