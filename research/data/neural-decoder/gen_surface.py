#!/usr/bin/env python3
"""Stim's rotated surface-code Z memory (rounds = d) -> <prefix>.stim/.dem/.meta for nd_tool and the
neural decoder.

usage: gen_surface.py <d> <p> <prefix> [noise=uniform|si1000]
  uniform: Stim's four knobs all = p (as research/qec/qec-r4.md, direction B).
  si1000 : SI1000-style (Gidney-Newman-McEwen): CZ-free approximation on Stim's CX circuit:
           2q depol p, 1q depol p/10, data idle depol p/10 per round (before_round), reset flip 2p,
           measure flip 5p.
.dem is the tab format of nd_tool (p \t obs-mask \t dets) from Stim's flattened, undecomposed DEM.
.meta: index x y t is_x colour(=0)."""
import sys
import stim

d, p, pre = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
noise = sys.argv[4] if len(sys.argv) > 4 else "uniform"
if noise == "uniform":
    kw = dict(after_clifford_depolarization=p, before_round_data_depolarization=p,
              before_measure_flip_probability=p, after_reset_flip_probability=p)
elif noise == "si1000":
    kw = dict(after_clifford_depolarization=p, before_round_data_depolarization=p / 10,
              before_measure_flip_probability=5 * p, after_reset_flip_probability=2 * p)
else:
    raise SystemExit(noise)
c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d, **kw)
open(pre + ".stim", "w").write(str(c.flattened()))
dem = c.detector_error_model(decompose_errors=False, flatten_loops=True)
coords = c.get_detector_coordinates()
nd = c.num_detectors
z_sites = {(int(v[0]), int(v[1])) for v in coords.values() if v[2] == 0}
with open(pre + ".meta", "w") as f:
    for k in range(nd):
        x, y, t = (int(round(v)) for v in coords[k][:3])
        f.write(f"{k} {x} {y} {t} {int((x, y) not in z_sites)} 0\n")
xs = [k for k in range(nd) if (int(coords[k][0]), int(coords[k][1])) not in z_sites]
with open(pre + ".dem", "w") as f:
    f.write("#x " + " ".join(map(str, xs)) + "\n")
    f.write(f"# detectors {nd}\n")
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        ds, ob = [], 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                ds.append(t.val)
            elif t.is_logical_observable_id():
                ob ^= 1 << t.val
        f.write(f"{inst.args_copy()[0]:e}\t{ob}\t{' '.join(map(str, ds))}\n")
print(f"d={d} p={p} noise={noise} detectors={nd} mechanisms={dem.num_errors}")
