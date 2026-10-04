#!/usr/bin/env python3
"""Independent check that a schedule's exported circuit is inside Kishony-Fowler's design space,
by diffing it against the exported K-F circuit (Stim text, same generator):
  * every line other than CX / DEPOLARIZE2 is identical, in order (resets, H, measurements,
    detectors, observable, noise placement, round structure, layout and qubit indices);
  * CX layers sit at the same positions; each DEPOLARIZE2 follows its CX with identical targets;
  * no qubit appears twice in a CX layer; every half-round has the same number of CX layers;
  * per half-round, the multiset of CNOT pairs is identical (only the ORDER differs);
  * in every round the X half uses the same step for each (auxiliary, data) pair as the Z half;
  * noiselessly, every detector and the observable are deterministic (Stim builds the DEM, which
    rejects non-deterministic detectors, and 2000 noiseless shots are all zero).
usage: design_space_diff.py kf.stim new.stim"""
import sys, collections, stim

def lines(p):
    return [l.strip() for l in open(p) if l.strip()]

def pairs(l):
    t = list(map(int, l.split()[1:]))
    return list(zip(t[0::2], t[1::2]))

A, B = lines(sys.argv[1]), lines(sys.argv[2])
ok = True
def fail(msg):
    global ok
    ok = False
    print("FAIL:", msg)

if len(A) != len(B):
    fail(f"line counts differ {len(A)} vs {len(B)}")
halves_a, halves_b, cur_a, cur_b = [], [], [], []
layers = 0
for i, (a, b) in enumerate(zip(A, B)):
    ka, kb = a.split()[0], b.split()[0]
    if ka != kb:
        fail(f"line {i}: instruction {ka} vs {kb}"); break
    if ka == "CX":
        layers += 1
        for name, l, cur in (("kf", a, cur_a), ("new", b, cur_b)):
            pr = pairs(l)
            qs = [q for p in pr for q in p]
            if len(qs) != len(set(qs)):
                fail(f"{name} line {i}: a qubit is used twice in one CX layer")
            cur.append(pr)
    elif ka.startswith("DEPOLARIZE2"):
        if a.split()[0] != b.split()[0]:
            fail(f"line {i}: noise strength differs")
        if a.split()[1:] != A[i - 1].split()[1:] or b.split()[1:] != B[i - 1].split()[1:]:
            fail(f"line {i}: DEPOLARIZE2 targets differ from the preceding CX")
    else:
        if a != b:
            fail(f"line {i}: '{a[:60]}' vs '{b[:60]}'")
        if cur_a:
            halves_a.append(cur_a); halves_b.append(cur_b); cur_a, cur_b = [], []
nl = collections.Counter(len(h) for h in halves_a)
# K-F: the X half (CX anc->data) repeats the Z half's (CX data->anc) step for every CNOT
for h in range(0, len(halves_b) - 1, 2):
    for name, H in (("kf", halves_a), ("new", halves_b)):
        z, x = H[h], H[h + 1]
        if [sorted(l) for l in z] != [sorted((t, c) for c, t in l) for l in x]:
            fail(f"{name} half-rounds {h},{h+1}: X half does not repeat the Z-half schedule")
for h, (ha, hb) in enumerate(zip(halves_a, halves_b)):
    if len(ha) != len(hb):
        fail(f"half {h}: {len(ha)} vs {len(hb)} CX layers")
    ma = collections.Counter(p for layer in ha for p in layer)
    mb = collections.Counter(p for layer in hb for p in layer)
    if ma != mb:
        fail(f"half {h}: CNOT pair multisets differ")
for name, path in (("kf", sys.argv[1]), ("new", sys.argv[2])):
    c = stim.Circuit.from_file(path)
    c0 = c.without_noise()
    c0.detector_error_model()  # raises on non-deterministic detectors / observables
    det, obs = c0.compile_detector_sampler().sample(2000, separate_observables=True)
    if det.any() or obs.any():
        fail(f"{name}: noiseless detector/observable flips")
    print(f"{name}: {c.num_qubits} qubits, {c.num_detectors} detectors, {c.num_observables} observable, "
          f"{sum(1 for l in lines(path) if l.startswith('CX'))} CX layers, noiseless deterministic")
print(f"{len(A)} lines compared; {layers} CX layers in {len(halves_a)} half-rounds "
      f"(layers per half: {dict(nl)}); CNOT multiset identical per half-round; X half = Z-half schedule")
print("INSIDE K-F DESIGN SPACE: same circuit except CNOT order" if ok else "NOT IDENTICAL")
