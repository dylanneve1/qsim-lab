#!/usr/bin/env python3
"""Exact lower bound on circuit distance from STIM's detector error model of an exported circuit
(independent DEM builder; our exact branch and bound as solver). Projects the DEM onto the
memory-basis detectors (a lower bound on the full-DEM distance) and asks dem_distance for logicals
of weight <= maxw. 'none' => circuit distance >= maxw + 1.
Detector order of `color_search export` (src/qec/color.rs): round 0: np own-type; round r >= 1:
np own-type then np other-type; final: np own-type.
usage: stim_dem_distance.py circuit.stim <np> <rounds> <maxw>"""
import sys, os, time, stim
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cg_sat import DistServer
path, np_, R, maxw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
own = {}
idx = 0
for r in range(R):
    for p in range(np_):
        own[idx] = len(own); idx += 1
    if r > 0:
        idx += np_
for p in range(np_):
    own[idx] = len(own); idx += 1
c = stim.Circuit.from_file(path)
assert c.num_detectors == idx, (c.num_detectors, idx)
dem = c.detector_error_model(decompose_errors=False)
sigs = set()
for inst in dem.flattened():
    if inst.type != "error":
        continue
    ds, ob = [], False
    for t in inst.targets_copy():
        if t.is_relative_detector_id() and t.val in own:
            ds.append(own[t.val])
        elif t.is_logical_observable_id():
            ob ^= True
    if ds or ob:
        sigs.add((tuple(sorted(ds)), ob))
rows = [(ob, list(ds)) for ds, ob in sigs]
t = time.time()
w, cnt, nodes, logs = DistServer().query(len(own), rows, maxw, 1)
print(f"{os.path.basename(path)}: stim DEM, {len(rows)} sector mechanisms, logical of weight <= {maxw}: "
      f"{'none -> distance >= %d' % (maxw + 1) if w is None else 'found weight %d' % w} ({time.time()-t:.0f}s, {nodes} nodes)", flush=True)
