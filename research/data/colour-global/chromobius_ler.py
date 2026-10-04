#!/usr/bin/env python3
"""Second-decoder LER check: Chromobius (Gidney, colour-code mobius decoder) on the exported
circuit, with detectors annotated DETECTOR(x, y, t, colour+3*is_Z) as Chromobius requires.
Sampling and DEM by Stim (equivalent to our sampler, research/qec-r4.md Part 1).
usage: chromobius_ler.py <color_search> <d> <rounds> <p> <schedule|kf> <shots> <seed> [x]"""
import sys, os, json, math, subprocess, tempfile, time
import numpy as np, stim, chromobius
CS, d, R, p, sched, shots, seed = sys.argv[1:8]
d, R, shots, seed = int(d), int(R), int(shots), int(seed)
xb = len(sys.argv) > 8 and sys.argv[8] == "x"
lay = [list(map(int, l.split())) for l in subprocess.run([CS, "layout", str(d)], capture_output=True, text=True, check=True).stdout.splitlines()]
np_ = len(lay)
with tempfile.TemporaryDirectory() as td:
    f = os.path.join(td, "c.stim")
    subprocess.run([CS, "export", str(d), str(R), "cnot", p, sched, f] + (["x"] if xb else []), check=True)
    text = open(f).read().splitlines()
# detector order: round 0 own; r >= 1 own then other; final own (own = Z for Z memory)
info = []
for r in range(R):
    info += [(pi, not xb, r) for pi in range(np_)]
    if r > 0:
        info += [(pi, xb, r) for pi in range(np_)]
info += [(pi, not xb, R) for pi in range(np_)]
out, k = [], 0
for l in text:
    if l.startswith("DETECTOR"):
        pi, isz, t = info[k]; k += 1
        x, y, col = lay[pi][1], lay[pi][2], lay[pi][3]
        l = l.replace("DETECTOR", f"DETECTOR({x}, {y}, {t}, {col + (3 if isz else 0)})", 1)
    out.append(l)
assert k == len(info)
c = stim.Circuit("\n".join(out))
dem = c.detector_error_model(decompose_errors=False)
dec = chromobius.compile_decoder_for_dem(dem)
t0 = time.time()
dets, obs = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True, bit_packed=True)
pred = dec.predict_obs_flips_from_dets_bit_packed(dets)
fails = int(np.count_nonzero((pred ^ obs).any(axis=1)))
pl = fails / shots
pr = lambda P: (1 - (1 - 2 * P) ** (1 / R)) / 2
print(json.dumps(dict(decoder="chromobius", d=d, rounds=R, p=float(p), schedule=os.path.basename(sched), basis="x" if xb else "z",
                      shots=shots, fails=fails, p_L=pl, p_L_round=pr(pl), seconds=round(time.time() - t0, 1))), flush=True)
