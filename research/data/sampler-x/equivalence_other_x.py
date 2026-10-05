#!/usr/bin/env python3
"""The audit's other-circuit equivalence (research/data/fast-sampler-audit/equivalence_other.py: colour,
repetition, unrotated and rotated-X surface codes; T0-T4 incl. joint 4-detector histograms; 1% FWER per
cell) run unchanged against the sampler-x pipeline: `sample-fast` -> `sample-x <threads> <tables>`,
`dem-support-fast` -> `dem-support-x`.
usage: equivalence_other_x.py <stim_compare> <work> <shots> <out.jsonl> <threads> <tables> [perturb]"""
import os, sys, subprocess, runpy
here = os.path.dirname(os.path.abspath(__file__))
B, WORK, SHOTS, OUT, THREADS, TABLES = sys.argv[1:7]
rest = sys.argv[7:]
_run = subprocess.run


def run_x(args, *a, **k):
    args = [str(x) for x in args]
    if len(args) > 1 and args[1] == "sample-fast":
        # sample-fast <file> <shots> <out> <seed> wy
        args = [args[0], "sample-x", args[2], args[3], args[4], args[5], THREADS, TABLES]
    elif len(args) > 1 and args[1] == "dem-support-fast":
        args = [args[0], "dem-support-x", args[2]]
    return _run(args, *a, **k)


subprocess.run = run_x
sys.argv = [os.path.join(here, "..", "fast-sampler-audit", "equivalence_other.py"), B, WORK, SHOTS, OUT] + rest
runpy.run_path(sys.argv[0], run_name="__main__")
