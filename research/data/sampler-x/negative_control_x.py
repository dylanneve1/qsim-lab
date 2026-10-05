#!/usr/bin/env python3
"""Power check for equivalence_x.py: our side (`sample-x`, given threads/tables) samples a deliberately
wrong copy of Stim's d = 7 rotated_memory_z (p = 0.003): every DEPOLARIZE2 at 0.0033 (+10% on one
channel), and separately the reset X_ERRORs removed. The T1 + T2 tests must reject. Same construction as
research/data/qec-r4/stim_equivalence_negative_control.py, which this runs with `sample` patched.
usage: negative_control_x.py <threads> <tables>   (env QSIM_STIM_COMPARE, WORK)"""
import os, sys, subprocess, runpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "qec-r4"))
THREADS, TABLES = sys.argv[1], sys.argv[2]
_run = subprocess.run


def run_x(args, *a, **k):
    args = [str(x) for x in args]
    if len(args) > 1 and args[1] == "sample":
        args = [args[0], "sample-x", args[2], args[3], args[4], args[5], THREADS, TABLES]
    return _run(args, *a, **k)


subprocess.run = run_x
runpy.run_path(os.path.join(here, "..", "qec-r4", "stim_equivalence_negative_control.py"), run_name="__main__")
