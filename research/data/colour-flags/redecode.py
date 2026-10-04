#!/usr/bin/env python3
"""Re-decode one ler_tess.py chunk: sample with its seed, decode with the light setting, then
re-decode the failing shots with heavier settings (and report Hamming weight of their syndromes).
usage: redecode.py <d> <R> <p> <spec> <shots> <seed> [basis]"""
import sys, os, subprocess, tempfile, numpy as np, stim
from tesseract_decoder import tesseract, utils as tu
CS = os.environ.get("CS", "/tmp/cf-target/release/examples/color_search")
d, R, p, spec, shots, seed = sys.argv[1:7]
basis = sys.argv[7] if len(sys.argv) > 7 else "z"
with tempfile.TemporaryDirectory() as td:
    f = os.path.join(td, "c.stim")
    subprocess.run([CS, "export", d, R, "cnot", p, spec, f] + (["x"] if basis == "x" else []), check=True)
    c = stim.Circuit.from_file(f)
dem = c.detector_error_model(decompose_errors=False)
dets, obs = c.compile_detector_sampler(seed=int(seed)).sample(int(shots), separate_observables=True)
obs = obs[:, 0]
def dec(orders, beam):
    cfg = tesseract.TesseractConfig(dem=dem, pqlimit=200_000, det_beam=beam, beam_climbing=True,
        det_orders=tu.build_det_orders(dem=dem, num_det_orders=orders, method=tu.DetOrder.DetIndex), no_revisit_dets=True)
    return tesseract.TesseractDecoder(cfg)
bo, bb = map(int, os.environ.get("BASE", "1,5").split(","))
light = dec(bo, bb)
pred = light.decode_batch(dets)[:, 0]
bad = np.nonzero(pred != obs)[0]
print(f"base ({bo},{bb}) fails:", len(bad), "syndrome weights:", [int(dets[i].sum()) for i in bad], flush=True)
for o, b in [(o, b) for o, b in [(4, 8), (16, 15), (32, 30)] if (o, b) > (bo, bb)]:
    D = dec(o, b)
    pr = D.decode_batch(dets[bad])[:, 0]
    print(f"orders {o} beam {b}: still failing {int((pr != obs[bad]).sum())} of {len(bad)}", flush=True)
