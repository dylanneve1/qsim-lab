#!/usr/bin/env python3
"""Power check for stim_equivalence.py: sample a slightly WRONG circuit on our side
(Stim's generated d=7 circuit with every DEPOLARIZE2 at 0.0033 instead of 0.003, i.e. +10% on
one channel; and separately with the reset X_ERRORs removed) and confirm the same tests reject."""
import os, subprocess, json
import numpy as np, stim
import stim_equivalence as E
from statistics import NormalDist

d, shots, P = 7, 1_000_000, 0.003
g = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                           after_clifford_depolarization=P, before_round_data_depolarization=P,
                           before_measure_flip_probability=P, after_reset_flip_probability=P)
txt = str(g)
lines = txt.splitlines()
variants = {
    "dep2_plus10pct": "\n".join(l.replace("DEPOLARIZE2(0.003)", "DEPOLARIZE2(0.0033)") for l in lines),
    "no_reset_errors": "\n".join(l for i, l in enumerate(lines)
                                 if not (l.strip().startswith("X_ERROR") and i + 1 < len(lines)
                                         and lines[i - 1].strip().startswith(("R ", "MR "))
                                         and not lines[i + 1].strip().startswith(("M", "MR")))),
}
nd = g.num_detectors
_, pairs = E.stim_support(g)
fs = f"{E.WORK}/s.ptb64"
g.compile_detector_sampler(seed=5).sample_write(shots, filepath=fs, format="ptb64", append_observables=True)
as_ = E.load_ptb64(fs, nd + 1, shots)
cs, ps = E.popcounts(as_), E.pair_counts(as_, pairs)
for name, t in variants.items():
    path = f"{E.WORK}/neg_{name}.stim"
    open(path, "w").write(t)
    fo = f"{E.WORK}/o.ptb64"
    subprocess.run([E.B, "sample", path, str(shots), fo, "9"], check=True)
    ao = E.load_ptb64(fo, nd + 1, shots)
    z1 = E.z2(E.popcounts(ao), cs, shots)
    zp = E.z2(E.pair_counts(ao, pairs), ps, shots)
    n = nd + 1 + len(pairs) + 2
    zc = NormalDist().inv_cdf(1 - 0.01 / (2 * n))
    allz = np.concatenate([z1, zp])
    print(json.dumps(dict(variant=name, max_abs_z=float(np.abs(allz).max()),
                          n_reject=int((np.abs(allz) > zc).sum()), zcrit=zc,
                          verdict="FAIL(as expected)" if (np.abs(allz) > zc).any() else "PASS(test has no power!)")))
    os.remove(fo)
os.remove(fs)
