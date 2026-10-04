#!/usr/bin/env python3
"""Chromobius (Gidney's colour-code Mobius decoder) on an exported (possibly flagged) circuit.
Detectors are annotated DETECTOR(x, y, t, colour + 3*is_Z) from the `.info` sidecar; flag
detectors get 4th coordinate -1 (Chromobius: 'ignore this detector'). Sampling and DEM by Stim.
usage: chromobius_ler.py <d> <rounds> <p> <spec> <shots> <seed> [x]"""
import sys, os, json, subprocess, tempfile, time
import numpy as np, stim, chromobius
CS = os.environ.get("CS", "/tmp/cf-target/release/examples/color_search")
d, R, p, spec, shots, seed = sys.argv[1:7]
d, R, shots, seed = int(d), int(R), int(shots), int(seed)
xb = len(sys.argv) > 7 and sys.argv[7] == "x"
lay = [list(map(int, l.split())) for l in subprocess.run([CS, "layout", str(d)], capture_output=True, text=True, check=True).stdout.splitlines()]
with tempfile.TemporaryDirectory() as td:
    f = os.path.join(td, "c.stim")
    subprocess.run([CS, "export", str(d), str(R), "cnot", p, spec, f] + (["x"] if xb else []), check=True)
    text = open(f).read().splitlines()
    info = [tuple(map(int, l.split())) for l in open(f + ".info")]
out, k = [], 0
for l in text:
    if l.startswith("DETECTOR"):
        pi, isx, t, fl = info[k]; k += 1
        x, y, col = lay[pi][1], lay[pi][2], lay[pi][3]
        c4 = -1 if fl else col + (0 if isx else 3)
        l = l.replace("DETECTOR", f"DETECTOR({x}, {y}, {t}, {c4})", 1)
    out.append(l)
assert k == len(info)
c = stim.Circuit("\n".join(out))
dem = c.detector_error_model(decompose_errors=False)
try:
    dec = chromobius.compile_decoder_for_dem(dem)
except Exception as e:
    print(json.dumps(dict(decoder="chromobius", d=d, rounds=R, p=float(p), schedule=os.path.basename(spec),
                          error=str(e)[:300])))
    sys.exit(0)
t0 = time.time()
dets, obs = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True, bit_packed=True)
try:
    pred = dec.predict_obs_flips_from_dets_bit_packed(dets)
except Exception as e:
    print(json.dumps(dict(decoder="chromobius", d=d, rounds=R, p=float(p), schedule=os.path.basename(spec),
                          error="decode: " + str(e)[:300])))
    sys.exit(0)
fails = int(np.count_nonzero((pred ^ obs).any(axis=1)))
pl = fails / shots
pr = lambda P: (1 - (1 - 2 * P) ** (1 / R)) / 2
print(json.dumps(dict(decoder="chromobius", d=d, rounds=R, p=float(p), schedule=os.path.basename(spec), basis="x" if xb else "z",
                      shots=shots, fails=fails, p_L=pl, p_L_round=pr(pl), seconds=round(time.time() - t0, 1))), flush=True)
