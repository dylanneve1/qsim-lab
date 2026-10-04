"""Independent colour-code circuit-distance + min-weight-logical count (audit §16).
Stim builds the DEM from the exported .stim; Stim's own search gives the full-DEM
distance; my own DFS counts minimum-weight logicals in the Z-sector merged DEM
(Z-type detectors = those built only from M (not MX) records)."""
import stim, sys, itertools, collections, time
path = sys.argv[1]; W = int(sys.argv[2])
c = stim.Circuit.from_file(path)
# measurement types
mtype = []
dets = []
last = {}
for inst in c.flattened():
    nm = inst.name
    if nm not in ('DETECTOR', 'OBSERVABLE_INCLUDE', 'TICK', 'QUBIT_COORDS', 'SHIFT_COORDS') and not nm.startswith('DEPOL') and not nm.endswith('ERROR'):
        if nm not in ('M', 'MZ', 'MR'):
            for t in inst.targets_copy(): last[t.value] = nm
    if nm in ('M', 'MZ', 'MR'):
        # the export writes MX as H; M; H -> a measurement right after H is X-type
        mtype += ['X' if last.get(t.value) == 'H' else 'Z' for t in inst.targets_copy()]
    elif nm in ('MX', 'MRX'): mtype += ['X'] * len(inst.targets_copy())
    elif nm.startswith('M'): raise ValueError(nm)
    elif nm == 'DETECTOR':
        cur = len(mtype)
        dets.append(set(mtype[cur + t.value] for t in inst.targets_copy()))
    elif nm == 'OBSERVABLE_INCLUDE':
        cur = len(mtype)
        assert all(mtype[cur + t.value] == 'Z' for t in inst.targets_copy())
zdet = {i for i, s in enumerate(dets) if s == {'Z'}}
assert all(s in ({'Z'}, {'X'}) for s in dets)
dem = c.detector_error_model(decompose_errors=False)
t0 = time.time()
err = [] if len(sys.argv)>3 else c.search_for_undetectable_logical_errors(dont_explore_detection_event_sets_with_size_above=6,
        dont_explore_edges_with_degree_above=6, dont_explore_edges_increasing_symptom_degree=False)
print(path, 'stim full-DEM distance', len(err), '(%.1fs)' % (time.time() - t0))
# Z-sector merged mechanisms
mech = set()
for inst in dem.flattened():
    if inst.type != 'error': continue
    ds = frozenset(t.val for t in inst.targets_copy() if t.is_relative_detector_id() and t.val in zdet)
    ob = sum(1 for t in inst.targets_copy() if t.is_logical_observable_id()) & 1
    if ds or ob: mech.add((ds, ob))
mech = sorted(mech, key=lambda m: (sorted(m[0]), m[1]))
print('Z-sector distinct mechanisms', len(mech), 'Z dets', len(zdet))
by_det = collections.defaultdict(list)
for i, (ds, ob) in enumerate(mech):
    for d in ds: by_det[d].append(i)
maxdeg = max(len(ds) for ds, _ in mech)
found = set()
def dfs(chosen, syn, ob, depth):
    if not syn:
        if ob: found.add(frozenset(chosen))
        if depth == 0 or not ob: pass
        # an empty-syndrome set with ob=0 can't be extended minimally; stop
        return
    if depth == 0 or -(-len(syn) // maxdeg) > depth: return
    d = min(syn)
    for i in by_det[d]:
        if i in chosen: continue
        ds, o = mech[i]
        dfs(chosen | {i}, syn ^ ds, ob ^ o, depth - 1)
# every logical contains >=1 observable-flipping mechanism; start from each
for w in range(1, W + 1):
    found.clear()
    for i, (ds, o) in enumerate(mech):
        if o: dfs(frozenset([i]), frozenset(ds), 1, w - 1)
    # keep only sets of exactly size w (smaller would have been found earlier)
    fw = {s for s in found if len(s) == w}
    print(f'weight {w}: {len(fw)} Z-sector logicals', flush=True)
    if fw: break
