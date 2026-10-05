#!/usr/bin/env python3
"""One table from cached per-shot fail vectors (<dir>/fails/<exp>.<fold>.<decoder>.npy, as written by
baselines_real.py and aq_train.py --mode eval), restricted to the shots every listed decoder has
(Tesseract / BP+OSD may cover only the first N shots of an experiment).

For each decoder: per (area, basis) per-round LER fitted over the round counts (paper protocol), mean
over datasets, plus the fixed-round inversion per R. Ratios vs a reference decoder use a paired
bootstrap (resample shots jointly within every experiment, refit both, 400 resamples) -> 95% CI.

usage: compare.py <dataset: syc|willow> <d> <fold> <ref-decoder> <dec1,dec2,...> <dir1> [<dir2> ...] [--rounds 3,...]
"""
import glob, json, os, re, sys
import numpy as np
from aq_data import fit_ler, eps_from_E

ds, d, fold, ref, decs = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5].split(",")
dirs = []
_skip = False
for x in sys.argv[6:]:
    if _skip:
        _skip = False
        continue
    if x.startswith("--"):
        _skip = True
        continue
    dirs.append(x)
mean_rounds = None  # also report mean over these R of the fixed-R per-round inversion (sim-to-real convention)
if "--mean-rounds" in sys.argv:
    mean_rounds = [int(x) for x in sys.argv[sys.argv.index("--mean-rounds") + 1].split(",")]
rounds = None
if "--rounds" in sys.argv:
    rounds = [int(x) for x in sys.argv[sys.argv.index("--rounds") + 1].split(",")]
pat = (re.compile(rf"surface_code_b(.)_d{d}_r(\d+)_center_(\d_\d)\.{fold}\.(.+)\.npy") if ds == "syc"
       else re.compile(rf"willow_b(.)_d{d}_r(\d+)_(q\d+_\d+)\.{fold}\.(.+)\.npy"))
F = {}
for dd in dirs:
    for f in glob.glob(os.path.join(dd, "fails", "*.npy")):
        m = pat.fullmatch(os.path.basename(f))
        if not m:
            continue
        b, r, area, dec = m.groups()
        if dec in decs and (rounds is None or int(r) in rounds):
            F.setdefault(dec, {})[(area, b, int(r))] = np.load(f)
missing = [x for x in decs if x not in F]
if missing:
    raise SystemExit(f"no fails for {missing}")
keys = sorted(set.intersection(*[set(F[x]) for x in decs]))
n = {k: min(len(F[x][k]) for x in decs) for k in keys}
groups = sorted({(a, b) for a, b, _ in keys})
rs = sorted({r for _, _, r in keys})


def ler(fails_by_key):
    e = []
    for g in groups:
        rr = [r for r in rs if (g[0], g[1], r) in fails_by_key]
        e.append(fit_ler(rr, [fails_by_key[(g[0], g[1], r)].sum() for r in rr],
                         [len(fails_by_key[(g[0], g[1], r)]) for r in rr])[0])
    return float(np.mean(e))


rng = np.random.default_rng(0)
# paired bootstrap via the joint fail pattern of all decoders on each shot (multinomial over 2^m cells)
m = len(decs)
cells = {}
for k in keys:
    pat_ = np.zeros(n[k], np.int64)
    for j, x in enumerate(decs):
        pat_ |= F[x][k][:n[k]].astype(np.int64) << j
    cells[k] = np.bincount(pat_, minlength=1 << m)
bits = (np.arange(1 << m)[:, None] >> np.arange(m)[None, :]) & 1      # (cells, m)


def ler_counts(fails, shots):
    e = []
    for g in groups:
        rr = [r for r in rs if (g[0], g[1], r) in fails]
        e.append(fit_ler(rr, [fails[(g[0], g[1], r)] for r in rr], [shots[(g[0], g[1], r)] for r in rr])[0])
    return float(np.mean(e))


def mean_fixed(fails, shots):
    return float(np.mean([np.mean([eps_from_E(fails[k] / shots[k], k[2]) for k in keys if k[2] == r]) for r in mean_rounds]))


shots = {k: n[k] for k in keys}
point = [ler_counts({k: int(cells[k] @ bits[:, j]) for k in keys}, shots) for j in range(m)]
mpoint = [mean_fixed({k: int(cells[k] @ bits[:, j]) for k in keys}, shots) for j in range(m)] if mean_rounds else None
boot = np.zeros((400, m)); mboot = np.zeros((400, m))
for t in range(400):
    fk = {k: rng.multinomial(n[k], cells[k] / n[k]) @ bits for k in keys}
    for j in range(m):
        boot[t, j] = ler_counts({k: int(fk[k][j]) for k in keys}, shots)
        if mean_rounds:
            mboot[t, j] = mean_fixed({k: int(fk[k][j]) for k in keys}, shots)
jr = decs.index(ref)
out = []
for j, x in enumerate(decs):
    l = point[j]
    rat = boot[:, j] / boot[:, jr]
    byR = {r: float(np.mean([eps_from_E(F[x][k][:n[k]].mean(), r) for k in keys if k[2] == r])) for r in rs}
    rec = dict(decoder=x, ler=l, ler_ci=[float(np.quantile(boot[:, j], 0.025)), float(np.quantile(boot[:, j], 0.975))],
               ratio_vs_ref=l / point[jr], ratio_ci=[float(np.quantile(rat, 0.025)), float(np.quantile(rat, 0.975))],
               eps_by_round=byR)
    if mean_rounds:
        mr = mboot[:, j] / mboot[:, jr]
        rec.update(mean_fixed=mpoint[j], mean_fixed_ci=[float(np.quantile(mboot[:, j], 0.025)), float(np.quantile(mboot[:, j], 0.975))],
                   mean_fixed_ratio=mpoint[j] / mpoint[jr], mean_fixed_ratio_ci=[float(np.quantile(mr, 0.025)), float(np.quantile(mr, 0.975))])
        print(f"   mean eps over R={mean_rounds}: {100 * mpoint[j]:.3f}% [{100 * rec['mean_fixed_ci'][0]:.3f}, {100 * rec['mean_fixed_ci'][1]:.3f}]"
              f"  ratio {rec['mean_fixed_ratio']:.3f} [{rec['mean_fixed_ratio_ci'][0]:.3f}, {rec['mean_fixed_ratio_ci'][1]:.3f}]")
    out.append(rec)
    print(f"{x:52s} LER {100 * l:.3f}% [{100 * rec['ler_ci'][0]:.3f}, {100 * rec['ler_ci'][1]:.3f}]  "
          f"vs {ref}: {rec['ratio_vs_ref']:.3f} [{rec['ratio_ci'][0]:.3f}, {rec['ratio_ci'][1]:.3f}]  "
          + " ".join(f"r{r}:{100 * v:.3f}" for r, v in byR.items()))
print(f"datasets {len(groups)}, rounds {rs}, shots {sum(n.values())}")
json.dump(dict(dataset=ds, d=d, fold=fold, ref=ref, rows=out, shots=int(sum(n.values())), rounds=rs),
          open(f"compare_{ds}_d{d}_{fold}_{ref}.json", "w"), indent=1)
