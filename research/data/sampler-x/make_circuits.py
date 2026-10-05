#!/usr/bin/env python3
"""Writes Stim's rotated_memory_z (rounds = d, all four noise knobs = p) as <dir>/B_d{d}_p{p}.stim.
usage: make_circuits.py <dir> [d list] [p list]"""
import os, sys
import stim
out = sys.argv[1]
ds = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "3,5,7,11,15,21,25").split(",")]
ps = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0.001,0.003").split(",")]
os.makedirs(out, exist_ok=True)
for p in ps:
    for d in ds:
        c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                                   after_clifford_depolarization=p, before_round_data_depolarization=p,
                                   before_measure_flip_probability=p, after_reset_flip_probability=p)
        open(f"{out}/B_d{d}_p{p}.stim", "w").write(str(c))
