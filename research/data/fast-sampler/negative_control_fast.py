#!/usr/bin/env python3
"""Power check for the FastSampler equivalence: our side samples a deliberately wrong circuit
(Stim's d=7 rotated_memory_z, p=0.003, every DEPOLARIZE2 at 0.0033, i.e. +10% on one channel;
and separately with the reset X_ERRORs removed) with `sample-fast`; the T1+T2 tests must reject.
Same construction as research/data/qec-r4/stim_equivalence_negative_control.py."""
import sys, os, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qec-r4"))
import stim_equivalence as E
_run = subprocess.run
def run_fast(args, *a, **k):
    args = list(args)
    if len(args) > 1 and args[1] == "sample":
        args[1] = "sample-fast"
    return _run(args, *a, **k)
subprocess.run = run_fast
import runpy
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qec-r4",
                            "stim_equivalence_negative_control.py"), run_name="__main__")
