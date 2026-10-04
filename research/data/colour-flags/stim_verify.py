#!/usr/bin/env python3
"""Independent distance checks on an exported (possibly flagged) colour-code circuit.

(1) Exact lower bound from STIM's own DEM: project onto the memory-basis sector (own-type
    plaquette detectors + flag detectors of that sector, read from the `.info` sidecar that
    `color_search export` writes) and ask our branch and bound (examples/dem_distance) for a
    logical of weight <= maxw.  'none' => circuit distance >= maxw + 1 (sector projection can
    only lower the distance).
(2) Upper bound: stim's search_for_undetectable_logical_errors (heuristic, finds *a* logical).

usage: stim_verify.py circuit.stim <basis z|x> <maxw> [ev] [deg] [--no-search] [--no-bb]
"""
import sys, time, subprocess, stim
path, basis, maxw = sys.argv[1], sys.argv[2], int(sys.argv[3])
pos = [a for a in sys.argv[4:] if not a.startswith("--")]
ev = int(pos[0]) if len(pos) > 0 else 4
deg = int(pos[1]) if len(pos) > 1 else 10
DD = "/tmp/cf-target/release/examples/dem_distance"
xb = basis == "x"
info = [tuple(map(int, l.split())) for l in open(path + ".info")]
own = {}
for i, (pi, isx, r, fl) in enumerate(info):
    if isx == xb:
        own[i] = len(own)
c = stim.Circuit.from_file(path)
assert c.num_detectors == len(info), (c.num_detectors, len(info))
if "--no-bb" not in sys.argv:
    dem = c.detector_error_model(decompose_errors=False)
    sigs = set()
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        ds, ob = set(), False
        for t in inst.targets_copy():
            if t.is_relative_detector_id() and t.val in own:
                ds ^= {own[t.val]}
            elif t.is_logical_observable_id():
                ob ^= True
        if ds or ob:
            sigs.add((tuple(sorted(ds)), ob))
    rows = [(ob, ds) for ds, ob in sigs]
    t = time.time()
    inp = f"P {len(own)} {maxw} 1 {len(rows)}\n" + "".join(f"{int(o)} {' '.join(map(str, d))}\n" for o, d in rows)
    out = subprocess.run([DD], input=inp, capture_output=True, text=True, check=True).stdout.split("\n")[0].split()
    w = out[1]
    print(f"{path}: stim DEM {basis}-sector, {len(rows)} mechanisms, {len(own)} dets: logical of weight <= {maxw}: "
          f"{'none -> d_circ >= %d' % (maxw + 1) if w == 'none' else 'FOUND weight ' + w} ({time.time()-t:.0f}s)", flush=True)
if "--no-search" not in sys.argv:
    t = time.time()
    err = c.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=ev,
        dont_explore_edges_with_degree_above=deg,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=True)
    print(f"{path}: stim upper bound {len(err)} ({time.time()-t:.1f}s, ev<={ev}, deg<={deg})", flush=True)
