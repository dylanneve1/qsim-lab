#!/usr/bin/env python3
"""Exact check (T0 on the new code): the distinct non-empty row sets of FastSampler's hit tables
(`stim_compare dem-support-fast`) equal the error-target sets of Stim's DEM, for Stim's generated
rotated_memory_z (direction B) and qsim-lab's exported circuit (direction A), d = 3, 7, 11, 15.
usage: support_fast.py <p> >> support_fast.jsonl"""
import sys, os, subprocess, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qec-r4"))
import stim_equivalence as E
import stim
p = float(sys.argv[1])
for direction in "AB":
    for d in [3, 7, 11, 15]:
        if direction == "A":
            path = f"{E.WORK}/ours_d{d}_p{p}.stim"
            subprocess.run([E.B, "export-surface", str(d), str(p), path], check=True)
        else:
            path = f"{E.WORK}/stimgen_d{d}_p{p}.stim"
            open(path, "w").write(str(stim.Circuit.generated(
                "surface_code:rotated_memory_z", distance=d, rounds=d, after_clifford_depolarization=p,
                before_round_data_depolarization=p, before_measure_flip_probability=p,
                after_reset_flip_probability=p)))
        sigs, _ = E.stim_support(stim.Circuit.from_file(path))
        out = subprocess.run([E.B, "dem-support-fast", path], capture_output=True, text=True, check=True).stdout
        ours = {tuple(int(x) for x in l.split()) for l in out.splitlines() if l.strip()}
        print(json.dumps(dict(direction=direction, d=d, p=p, equal=ours == sigs, ours=len(ours), stim=len(sigs))))
