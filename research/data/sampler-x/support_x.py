#!/usr/bin/env python3
"""T0 (exact): the distinct non-empty row sets of the new compiler's hit tables (`stim_compare
dem-support-x`) equal the error-target sets of Stim's DEM (decompose_errors=False), for Stim's generated
rotated_memory_z (rounds = d, all four knobs = p).
usage: support_x.py <stim_compare> <circuit dir> <d list> <p list>   (circuits from make_circuits.py)"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qec-r4"))
import stim_equivalence as E
import stim
B, CD = sys.argv[1], sys.argv[2]
for p in [float(x) for x in sys.argv[4].split(",")]:
    for d in [int(x) for x in sys.argv[3].split(",")]:
        path = f"{CD}/B_d{d}_p{p}.stim"
        sigs, _ = E.stim_support(stim.Circuit.from_file(path))
        out = subprocess.run([B, "dem-support-x", path], capture_output=True, text=True, check=True).stdout
        ours = {tuple(int(x) for x in l.split()) for l in out.splitlines() if l.strip()}
        print(json.dumps(dict(d=d, p=p, equal=ours == sigs, ours=len(ours), stim=len(sigs))), flush=True)
