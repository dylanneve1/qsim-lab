#!/usr/bin/env python3
"""Baselines on the real Sycamore 2022 data, paper conventions (Bausch et al. 2024, Methods 'Metrics').

For every (distance, area, basis, fold) the per-round LER eps is fitted over round counts 3, 5, ..., 25
(log-fidelity linear fit, n = 1 excluded), with a 499-resample bootstrap; the reported number is the
mean over the 16 (d = 3) or 4 (d = 5) datasets, errors combined in quadrature / count (paper: 'Gaussian
error propagation' of the per-dataset bootstrap errors for the mean).

Fold 'odd'  = shots with odd index (decoded with pij_from_even_for_odd.dem)
Fold 'even' = shots with even index (decoded with pij_from_odd_for_even.dem)

Decoders
  shipped:<name>   per-shot predictions shipped with the dataset (pymatching, correlated_matching,
                   belief_matching, tensor_network_contraction)
  pm               PyMatching 2 (uncorrelated) on the fold's pij DEM
  pmcorr           PyMatching 2 correlated matching (enable_correlations=True) on the pij DEM
  tess[:beam]      Tesseract on the pij DEM (first --tess-shots shots of each fold)
Per-shot fail vectors are cached in <out>/fails/<exp>.<fold>.<decoder>.npy so neural-decoder results can
be compared on exactly the same shots (paired).

usage: baselines_real.py <syc-root> <out-dir> [--d 3,5] [--folds odd,even] [--decoders ...] [--rounds 3,...,25]
"""
import argparse, json, os, sys, time
import numpy as np
from aq_data import *

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("out")
ap.add_argument("--d", default="3,5")
ap.add_argument("--folds", default="odd,even")
ap.add_argument("--decoders", default="shipped:pymatching,shipped:correlated_matching,shipped:belief_matching,"
                                      "shipped:tensor_network_contraction,pm,pmcorr")
ap.add_argument("--rounds", default=",".join(str(r) for r in range(3, 26, 2)))
ap.add_argument("--tess-shots", type=int, default=5000)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--prior", default="si1000", help="Willow: DEM prior for pm/pmcorr/tess (si1000 | rl_optimized)")
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "fails"), exist_ok=True)
ds = [int(x) for x in a.d.split(",")]
rounds = [int(x) for x in a.rounds.split(",")]
decs = a.decoders.split(",")
exps = [e for e in experiments(a.root) if e["d"] in ds and e["R"] in rounds]


def fold_idx(n, fold):
    return np.arange(1 if fold == "odd" else 0, n, 2)


def dem_for(e, fold):
    if e.get("dataset") == "willow":
        return open(os.path.join(e["path"], "decoding_results", f"correlated_matching_decoder_with_{a.prior}_prior",
                                 "error_model.dem")).read()
    return open(os.path.join(e["path"], "pij_from_even_for_odd.dem" if fold == "odd" else "pij_from_odd_for_even.dem")).read()


def decode(e, fold, dec, dets, obs):
    import stim
    if dec.startswith("shipped:"):
        name = dec.split(":")[1]
        if e.get("dataset") == "willow":
            if name not in e["shipped"]:
                return None
            pred = read_b8(e["shipped"][name], 1)[:, 0]
        else:
            pred = read_01(os.path.join(e["path"], f"obs_flips_predicted_by_{name}.01"))
        idx = fold_idx(len(pred), fold)
        return pred[idx] != obs
    dem = stim.DetectorErrorModel(dem_for(e, fold))
    if dec in ("pm", "pmcorr"):
        import pymatching
        m = pymatching.Matching.from_detector_error_model(dem, enable_correlations=(dec == "pmcorr"))
        pred = m.decode_batch(dets.astype(bool), enable_correlations=(dec == "pmcorr"))
        return pred[:, 0].astype(bool) != obs.astype(bool)
    if dec.startswith("tess"):
        from tesseract_decoder import tesseract, utils as tu
        beam = int(dec.split(":")[1]) if ":" in dec else 15
        n = min(len(obs), a.tess_shots)
        cfg = tesseract.TesseractConfig(dem=dem, pqlimit=200_000, det_beam=beam, beam_climbing=True,
                                        det_orders=tu.build_det_orders(dem=dem, num_det_orders=16,
                                                                       method=tu.DetOrder.DetIndex),
                                        no_revisit_dets=True)
        dc = tesseract.TesseractDecoder(cfg)
        pred = dc.decode_batch(dets[:n].astype(bool))
        return pred[:, 0].astype(bool) != obs[:n].astype(bool)
    raise SystemExit(dec)


res = {}
for e in exps:
    dets, obs, _ = load(e)
    for fold in a.folds.split(","):
        idx = fold_idx(len(obs), fold)
        for dec in decs:
            fn = os.path.join(a.out, "fails", f"{e['name']}.{fold}.{dec.replace(':', '_')}.npy")
            if os.path.exists(fn):
                f = np.load(fn)
            else:
                t0 = time.time()
                f = decode(e, fold, dec, dets[idx], obs[idx])
                if f is None:
                    continue
                np.save(fn, f)
                print(f"{e['name']} {fold} {dec}: {f.mean():.5f} ({time.time() - t0:.1f}s)", flush=True)
            res.setdefault((e["d"], e["area"], e["basis"], fold, dec), {})[e["R"]] = f

summary = []
for d in ds:
    for dec in decs:
        keys = [k for k in res if k[0] == d and k[4] == dec and len(res[k]) == len(rounds)]
        if not keys:
            continue
        eps, err, per = [], [], []
        for k in sorted(keys):
            rs = sorted(res[k])
            e_, s_, F0, r2 = fit_ler_boot(rs, [res[k][r] for r in rs])
            eps.append(e_); err.append(s_)
            per.append(dict(area=k[1], basis=k[2], fold=k[3], eps=e_, err=s_, F0=F0, R2=r2,
                            shots=int(sum(len(res[k][r]) for r in rs))))
        m = float(np.mean(eps)); s = float(np.sqrt(np.sum(np.square(err))) / len(err))
        # fixed-round inversion per round count, mean over datasets (Willow / sim-to-real convention)
        byR = {r: float(np.mean([eps_from_E(res[k][r].mean(), r) for k in keys])) for r in rounds}
        rec = dict(d=d, decoder=dec, datasets=len(keys), ler=m, ler_err=s, eps_by_round=byR, per_dataset=per)
        summary.append(rec)
        print(f"d={d} {dec:42s} LER = {100 * m:.3f} +- {100 * s:.3f} %  ({len(keys)} datasets)  per-round inv: "
              + " ".join(f"r{r}:{100 * v:.3f}" for r, v in byR.items()), flush=True)
json.dump(summary, open(os.path.join(a.out, f"summary_{a.decoders.replace(':', '_').replace(',', '+')[:80]}.json"), "w"), indent=1)
