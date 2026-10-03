#!/usr/bin/env python3
"""LER with OUR sampler + OUR circuit-derived DEM + the Tesseract decoder (Kishony-Fowler's
decoder and settings: det_beam=15, beam_climbing, 16 det orders, pqlimit=2e5).

usage: tesseract_ler.py <color_search> <stim_compare> <d> <rounds> <noise> <p> <schedule> <shots> <seed> <workers>
"""
import sys, os, json, time, subprocess, math
import multiprocessing as mp
import numpy as np
import stim
from tesseract_decoder import tesseract, utils as tu

CS, SC, d, rounds, noise, p, sched, shots, seed, workers = sys.argv[1:11]
d, rounds, shots, seed, workers = int(d), int(rounds), int(shots), int(seed), int(workers)
work = os.environ.get("WORK", "/tmp/qsim-wt/qec-data")
tag = f"{os.path.basename(sched)}_d{d}_r{rounds}_{noise}_{p}_{seed}"
stimf, demf, samp = f"{work}/{tag}.stim", f"{work}/{tag}.dem", f"{work}/{tag}.ptb64"
subprocess.run([CS, "export", str(d), str(rounds), noise, p, sched, stimf], check=True)
subprocess.run([CS, "dem", str(d), str(rounds), noise, p, sched, demf], check=True)
lines = []
nd = None
for line in open(demf):
    if line.startswith("#x"):
        continue
    if line.startswith("#"):
        nd = int(line.split()[-1]); continue
    pp, ob, ds = line.rstrip("\n").split("\t")
    t = " ".join(f"D{x}" for x in ds.split()) + (" L0" if int(ob) & 1 else "")
    lines.append(f"error({pp}) {t}")
dem_text = "\n".join(lines)
shots = (shots // 64) * 64
ORDERS = int(os.environ.get("TESS_ORDERS", "16"))
BEAM = int(os.environ.get("TESS_BEAM", "15"))
t0 = time.time()
subprocess.run([SC, "sample", stimf, str(shots), samp, str(seed)], check=True)
a = np.fromfile(samp, dtype="<u8").reshape(-1, nd + 1)
os.remove(samp)
bits = np.unpackbits(a.view(np.uint8).reshape(a.shape[0], nd + 1, 8), axis=2, bitorder="little")
bits = bits.transpose(0, 2, 1).reshape(-1, nd + 1).astype(bool)
dets, obs = bits[:, :nd], bits[:, nd]
t_sample = time.time() - t0


def make():
    dem = stim.DetectorErrorModel(dem_text)
    cfg = tesseract.TesseractConfig(dem=dem, pqlimit=200_000, det_beam=BEAM, beam_climbing=True,
                                    det_orders=tu.build_det_orders(dem=dem, num_det_orders=ORDERS,
                                                                   method=tu.DetOrder.DetIndex),
                                    no_revisit_dets=True)
    return tesseract.TesseractDecoder(cfg)


def work_chunk(idx):
    dec = make()
    sl = slice(idx[0], idx[1])
    pred = dec.decode_batch(dets[sl])
    return int((pred[:, 0] != obs[sl]).sum())


t1 = time.time()
step = max(1, shots // (workers * 8))
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
print(json.dumps(dict(decoder="tesseract", det_orders=ORDERS, det_beam=BEAM, d=d, rounds=rounds, noise=noise, p=float(p), schedule=sched,
                      shots=n, fails=fails, p_L=pl, ci95=[c - h, c + h], p_L_round=pr(pl),
                      ci95_round=[pr(c - h), pr(c + h)], sample_s=round(t_sample, 2),
                      decode_s=round(time.time() - t1, 1), workers=workers)))
for f in (stimf, demf):
    os.remove(f)
