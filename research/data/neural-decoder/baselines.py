#!/usr/bin/env python3
"""Decode a held-out ptb64 file with the baseline decoders; saves per-shot fail vectors
(<out>.<decoder>.fails.npy, bool) and appends a JSON line per decoder to <out>.jsonl.

usage: baselines.py <prefix> <test.ptb64> <rounds> <out> [pm] [bposd[:order[:zonly]]] [tess:shots[:beam:orders]]
  pm    : PyMatching 2 on Stim's decomposed DEM (surface code only; needs <prefix>.stim from Stim)
  bposd : our BP+OSD-CS via nd_tool (2 threads)
  tess  : Tesseract on the first <shots> shots, our .dem, 2 worker processes."""
import sys, os, json, time, subprocess
import multiprocessing as mp
import numpy as np
from nd_common import *

pre, test, rounds, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
nd = num_dets(pre)
bits = read_ptb64(test, nd + 1)
dets, obs = bits[:, :nd].astype(bool), bits[:, nd].astype(bool)
N = len(obs)


def report(name, fails, n, secs, extra=None):
    f = int(fails.sum())
    lo, hi = wilson(f, n)
    r = dict(decoder=name, prefix=os.path.basename(pre), shots=n, fails=f, p_L=f / n, ci95=[lo, hi],
             p_L_round=per_round(f / n, rounds), ci95_round=[per_round(lo, rounds), per_round(hi, rounds)],
             decode_s=round(secs, 1), **(extra or {}))
    np.save(f"{out}.{name}.fails.npy", fails)
    print(json.dumps(r), flush=True)
    open(out + ".jsonl", "a").write(json.dumps(r) + "\n")


def dem_text():
    lines = []
    for line in open(pre + ".dem"):
        if line.startswith("#"):
            continue
        pp, ob, ds = line.rstrip("\n").split("\t")
        lines.append(f"error({pp}) " + " ".join(f"D{x}" for x in ds.split()) + (" L0" if int(ob) & 1 else ""))
    return "\n".join(lines)


TESS = {}


def tess_chunk(args):
    lo, hi, beam, orders = args
    from tesseract_decoder import tesseract, utils as tu
    import stim
    wait_lock()
    if "dec" not in TESS:
        dem = stim.DetectorErrorModel(TESS["text"])
        cfg = tesseract.TesseractConfig(dem=dem, pqlimit=200_000, det_beam=beam, beam_climbing=True,
                                        det_orders=tu.build_det_orders(dem=dem, num_det_orders=orders,
                                                                       method=tu.DetOrder.DetIndex),
                                        no_revisit_dets=True)
        TESS["dec"] = tesseract.TesseractDecoder(cfg)
    pred = TESS["dec"].decode_batch(dets[lo:hi])
    return pred[:, 0] != obs[lo:hi]


for arg in sys.argv[5:]:
    kind = arg.split(":")
    t0 = time.time()
    if kind[0] == "pm":
        import stim, pymatching
        c = stim.Circuit(open(pre + ".stim").read())
        m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
        pred = m.decode_batch(dets)
        report("pymatching", pred[:, 0].astype(bool) != obs, N, time.time() - t0)
    elif kind[0] == "bposd":
        order = kind[1] if len(kind) > 1 else "10"
        z = kind[2] if len(kind) > 2 else "all"
        r = subprocess.run([ND_TOOL, "bposd", pre + ".dem", test, out + ".bposd.pred", "2", order, z],
                           check=True, capture_output=True, text=True)
        js = json.loads(r.stdout)
        pred = np.fromfile(out + ".bposd.pred", dtype=np.uint8)[:N].astype(bool)
        os.remove(out + ".bposd.pred")
        name = f"bposd{order}" + ("z" if z == "zonly" else "")
        report(name, pred != obs, N, time.time() - t0, dict(osd_order=int(order), zonly=z == "zonly"))
    elif kind[0] == "tess":
        n = min(N, int(kind[1]))
        beam = int(kind[2]) if len(kind) > 2 else 15
        orders = int(kind[3]) if len(kind) > 3 else 16
        TESS["text"] = dem_text()
        step = 2048
        chunks = [(i, min(n, i + step), beam, orders) for i in range(0, n, step)]
        with mp.get_context("fork").Pool(2) as pool:
            fails = np.concatenate(pool.map(tess_chunk, chunks))
        report(f"tess_b{beam}o{orders}", fails, n, time.time() - t0, dict(det_beam=beam, det_orders=orders))
