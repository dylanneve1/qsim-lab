#!/usr/bin/env python3
"""Build AlphaQubit-lite training data for one distance and test fold.

For every Sycamore experiment (distance d, rounds R >= 3, both bases, every area):
  sim/<exp>.s<scale>.npy : `--shots` FastSampler samples (qsim-lab nd_tool stream, wyrand) of the
                           pij DEM fitted to the *training* half, as packed per-shot bit rows
                           (np.packbits over nd+1 columns: detectors then observable)
  real/<exp>.npz         : the experiment's real shots (same packing), plus 'idx' = shot indices
Test fold 'odd' (paper default): DEM pij_from_even_for_odd (fitted on even shots), fine-tune on even
shots, test on odd shots. Fold 'even' swaps the roles.

Willow (root containing google_105Q_surface_code_d3_d5_d7/): no per-fold fitted DEMs are released;
--source si1000 samples the shipped circuit_noisy_si1000.stim (sweep-controlled ops removed: sweep bits
only choose the initial data-qubit pattern, which does not change detectors) and --source rl the
shipped RL-optimized prior DEM (decoding_results/*_with_rl_optimized_prior/error_model.dem).

usage: aq_gen.py <root> <out> --d 3 [--fold odd] [--shots 102400] [--scales 1.0] [--seed 1]
       [--source pij|si1000|rl] [--rounds 10,13,30]
"""
import argparse, os, subprocess, sys
import numpy as np
from aq_data import *

ND_TOOL = os.environ.get("ND_TOOL", os.path.expanduser("~/qsim-aq-data/bin/nd_tool"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "neural-decoder"))
from nd_common import unpack_ptb64

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("out")
ap.add_argument("--d", type=int, default=3)
ap.add_argument("--fold", default="odd")
ap.add_argument("--shots", type=int, default=102400)
ap.add_argument("--scales", default="1.0")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--no-real", action="store_true")
ap.add_argument("--source", default="pij")
ap.add_argument("--rounds", default=None)
a = ap.parse_args()
for sub in ("sim", "real", "circ"):
    os.makedirs(os.path.join(a.out, sub), exist_ok=True)
dem_name = "pij_from_even_for_odd.dem" if a.fold == "odd" else "pij_from_odd_for_even.dem"
exps = [e for e in experiments(a.root) if e["d"] == a.d and e["R"] >= 3]
if a.rounds:
    exps = [e for e in exps if e["R"] in [int(x) for x in a.rounds.split(",")]]


def strip_sweeps(txt):
    out = []
    for line in txt.splitlines():
        if line.startswith("CX") and "sweep[" in line:
            t = line.split()[1:]
            keep = [f"{t[i]} {t[i + 1]}" for i in range(0, len(t), 2) if not t[i].startswith("sweep[")]
            if keep:
                out.append("CX " + " ".join(keep))
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def source_circuit(e, sc):
    if a.source == "pij":
        return dem_to_circuit(open(os.path.join(e["path"], dem_name)).read(), float(sc))
    if a.source == "rl":
        p = os.path.join(e["path"], "decoding_results", "correlated_matching_decoder_with_rl_optimized_prior", "error_model.dem")
        return dem_to_circuit(open(p).read(), float(sc))
    if a.source == "si1000":
        assert float(sc) == 1.0
        import stim
        return str(stim.Circuit(strip_sweeps(open(os.path.join(e["path"], "circuit_noisy_si1000.stim")).read())).flattened())
    raise SystemExit(a.source)
for k, e in enumerate(exps):
    import stim
    txt = open(os.path.join(e["path"], "circuit_ideal.stim")).read()
    nd = stim.Circuit(txt).num_detectors
    if not a.no_real:
        fn = os.path.join(a.out, "real", e["name"] + ".npz")
        if not os.path.exists(fn):
            dets, obs, _ = load(e, nd)
            rows = np.packbits(np.concatenate([dets, obs[:, None]], 1), axis=1)
            np.savez(fn, rows=rows, nd=nd)
    for sc in a.scales.split(","):
        fn = os.path.join(a.out, "sim", f"{e['name']}.{a.source}.s{sc}.npy")
        if os.path.exists(fn):
            continue
        cpath = os.path.join(a.out, "circ", f"{e['name']}.{a.fold}.{a.source}.s{sc}.stim")
        open(cpath, "w").write(source_circuit(e, sc))
        raw = subprocess.run([ND_TOOL, "stream", cpath, str(a.seed * 1000 + k), str(a.shots)],
                             capture_output=True, check=True).stdout
        bits = unpack_ptb64(np.frombuffer(raw, dtype="<u8"), nd + 1)[:a.shots]
        np.save(fn, np.packbits(bits, axis=1))
        print(f"{e['name']} s{sc}: {len(bits)} shots, det density {bits[:, :nd].mean():.4f}, "
              f"obs rate {bits[:, nd].mean():.4f}", flush=True)
