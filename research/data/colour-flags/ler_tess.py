#!/usr/bin/env python3
"""Logical error rate with the Tesseract decoder on an exported (possibly flagged) colour-code
memory circuit. The circuit is `color_search export` of exactly the circuit our tools analyse;
sampling and the DEM (decompose_errors=False, all detectors incl. flags) come from Stim.
Default Tesseract setting = Kishony-Fowler's (det_beam=15, beam_climbing, 16 det orders,
pqlimit=2e5); TESS_ORDERS / TESS_BEAM override (lighter settings are labelled in the output).
TESS_SECTOR=1 decodes only the memory-basis sector (own-type plaquette detectors + that sector's
flag detectors; mechanisms merged by sector signature), like our BP+OSD: ~70x faster at d = 9.

usage: ler_tess.py <d> <rounds> <cnot|uniform> <p> <schedule-spec> <shots> <seed> <workers> [z|x]
"""
import sys, os, json, time, subprocess, math, tempfile
import multiprocessing as mp
import numpy as np
import stim
from tesseract_decoder import tesseract, utils as tu

d, rounds, noise, p, sched, shots, seed, workers = sys.argv[1:9]
basis = sys.argv[9] if len(sys.argv) > 9 else "z"
d, rounds, shots, seed, workers = int(d), int(rounds), int(shots), int(seed), int(workers)
CS = os.environ.get("CS", "/tmp/cf-target/release/examples/color_search")
ORDERS = int(os.environ.get("TESS_ORDERS", "16"))
BEAM = int(os.environ.get("TESS_BEAM", "15"))
with tempfile.TemporaryDirectory() as td:
    f = os.path.join(td, "c.stim")
    subprocess.run([CS, "export", str(d), str(rounds), noise, p, sched, f] + (["x"] if basis == "x" else []), check=True)
    circ = stim.Circuit.from_file(f)
    SECTOR_INFO = [tuple(map(int, l.split())) for l in open(f + ".info")]
dem = circ.detector_error_model(decompose_errors=False)
t0 = time.time()
dets, obs = circ.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
t_sample = time.time() - t0
obs = obs[:, 0]
SECTOR = os.environ.get("TESS_SECTOR", "0") == "1"
if SECTOR:
    # memory-basis sector only (own-type plaquette detectors + that sector's flag detectors), mechanisms
    # merged by sector signature: the CSS-decoupled problem BP+OSD also solves. ~70x faster at d = 9.
    info = SECTOR_INFO
    keep = [i for i, x in enumerate(info) if x[1] == (1 if basis == "x" else 0)]
    idx = {k: i for i, k in enumerate(keep)}
    merged = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        ds, ob = set(), False
        for t in inst.targets_copy():
            if t.is_relative_detector_id() and t.val in idx:
                ds ^= {idx[t.val]}
            elif t.is_logical_observable_id():
                ob ^= True
        if not ds and not ob:
            continue
        key = (tuple(sorted(ds)), ob)
        p0, p1 = merged.get(key, 0.0), inst.args_copy()[0]
        merged[key] = p0 * (1 - p1) + p1 * (1 - p0)
    lines = [f"error({pp}) " + " ".join(f"D{x}" for x in ds) + (" L0" if ob else "") for (ds, ob), pp in merged.items()]
    lines.append(f"detector D{len(keep) - 1}")
    dem = stim.DetectorErrorModel("\n".join(lines))
    dets = np.ascontiguousarray(dets[:, keep])


def make():
    cfg = tesseract.TesseractConfig(dem=dem, pqlimit=200_000, det_beam=BEAM, beam_climbing=True,
                                    det_orders=tu.build_det_orders(dem=dem, num_det_orders=ORDERS,
                                                                   method=tu.DetOrder.DetIndex),
                                    no_revisit_dets=True)
    return tesseract.TesseractDecoder(cfg)


def work_chunk(idx):
    while os.environ.get("OWN_LOCK") != "1" and os.path.isdir("/tmp/qsim-mac-bench.lock"):
        # never decode during a peer's timing run (campaign.py holds the lock itself: OWN_LOCK=1)
        time.sleep(5)
    dec = make()
    sl = slice(idx[0], idx[1])
    pred = dec.decode_batch(dets[sl])
    return int((pred[:, 0] != obs[sl]).sum())


t1 = time.time()
step = int(os.environ.get("TESS_CHUNK", max(1, shots // (workers * 4))))
chunks = [(i, min(shots, i + step)) for i in range(0, shots, step)]
with mp.get_context("fork").Pool(workers) as pool:
    fails = sum(pool.map(work_chunk, chunks))
n = shots
pl = fails / n
z = 1.96
den = 1 + z * z / n
c = (pl + z * z / (2 * n)) / den
h = z * math.sqrt(pl * (1 - pl) / n + z * z / (4 * n * n)) / den
pr = lambda x: (1 - max(0.0, 1 - 2 * x) ** (1 / rounds)) / 2
print(json.dumps(dict(decoder="tesseract", sector=SECTOR, det_orders=ORDERS, det_beam=BEAM, d=d, rounds=rounds, noise=noise,
                      p=float(p), schedule=os.path.basename(sched), basis=basis, shots=n, fails=fails, p_L=pl,
                      ci95=[c - h, c + h], p_L_round=pr(pl), ci95_round=[pr(c - h), pr(c + h)],
                      sample_s=round(t_sample, 2), decode_s=round(time.time() - t1, 1), workers=workers,
                      seed=seed, host=os.uname().nodename)), flush=True)
