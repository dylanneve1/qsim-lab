#!/usr/bin/env python3
"""Audit: FastSampler vs Stim 1.16 on Stim-generated circuits the author did not test.

Circuits: stim.Circuit.generated(<task>, d, rounds=d, all four noise knobs = p); colour code
goes through .decomposed() (C_XYZ is not in qsim-lab's parser; decomposed() keeps the noise).
Tests per cell (Bonferroni over all tests, 1% family-wise):
  T0 hit-table signatures (dem-support-fast) == Stim DEM error signatures
  T1 marginals, T2 DEM-correlated pairs, T3 mean/var of events per shot (as stim_equivalence.py)
  T4 joint 16-cell histogram of 4-detector neighbourhoods (a detector + 3 detectors it shares DEM
     errors with), two-sample chi^2 -> z (Wilson-Hilferty); catches higher-order correlation errors
usage: equivalence_other.py <stim_compare> <work> <shots> <out.jsonl> [perturb]
"""
import sys, os, json, time, subprocess, random
import numpy as np
import stim
from statistics import NormalDist
from scipy.stats import chi2 as CHI2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qec-r4"))
import stim_equivalence as E

B, WORK, SHOTS, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
PERTURB = len(sys.argv) > 5
os.makedirs(WORK, exist_ok=True)

CELLS = [
    ("color_code:memory_xyz", 3, 0.003), ("color_code:memory_xyz", 5, 0.003), ("color_code:memory_xyz", 7, 0.001),
    ("repetition_code:memory", 3, 0.01), ("repetition_code:memory", 9, 0.003),
    ("surface_code:unrotated_memory_x", 3, 0.003), ("surface_code:unrotated_memory_x", 5, 0.003),
    ("surface_code:unrotated_memory_z", 7, 0.001), ("surface_code:rotated_memory_x", 5, 0.003),
]
if PERTURB:
    CELLS = [("color_code:memory_xyz", 5, 0.003), ("surface_code:unrotated_memory_x", 5, 0.003)]


def our_support(path):
    out = subprocess.run([B, "dem-support-fast", path], capture_output=True, text=True, check=True).stdout
    return {tuple(int(x) for x in l.split()) for l in out.splitlines() if l.strip()}


def neighbourhoods(c, nd, k=200, seed=0):
    dem = c.detector_error_model(decompose_errors=False, flatten_loops=True)
    nb = {i: set() for i in range(nd)}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        ds = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
        for a in ds:
            nb[a].update(x for x in ds if x != a)
    rng = random.Random(seed)
    sets = []
    cand = [i for i in range(nd) if len(nb[i]) >= 3]
    for _ in range(min(k, len(cand))):
        i = rng.choice(cand)
        sets.append([i] + rng.sample(sorted(nb[i]), 3))
    return sets


def joint16(a, s):
    w = [a[:, j] for j in s]
    cnt = np.zeros(16, dtype=np.int64)
    for v in range(16):
        m = np.full(a.shape[0], ~np.uint64(0), dtype=np.uint64)
        for b in range(4):
            m &= w[b] if (v >> b) & 1 else ~w[b]
        cnt[v] = int(np.bitwise_count(m).sum(dtype=np.int64))
    return cnt


def two_sample_chi2_z(c1, c2):
    keep = (c1 + c2) > 0
    c1, c2 = c1[keep], c2[keep]
    n1, n2 = c1.sum(), c2.sum()
    e1 = (c1 + c2) * n1 / (n1 + n2)
    e2 = (c1 + c2) * n2 / (n1 + n2)
    x2 = (((c1 - e1) ** 2) / e1 + ((c2 - e2) ** 2) / e2).sum()
    k = len(c1) - 1
    if k <= 0:
        return 0.0
    return float((((x2 / k) ** (1 / 3)) - (1 - 2 / (9 * k))) / np.sqrt(2 / (9 * k)))


for task, d, p in CELLS:
    t0 = time.time()
    g = stim.Circuit.generated(task, distance=d, rounds=d, after_clifford_depolarization=p,
                               before_round_data_depolarization=p, before_measure_flip_probability=p,
                               after_reset_flip_probability=p)
    if task.startswith("color"):
        g = g.decomposed()
    path = f"{WORK}/{task.replace(':', '_')}_d{d}.stim"
    ours_path = path
    open(path, "w").write(str(g))
    if PERTURB:  # our side samples every DEPOLARIZE2 at +5%
        ours_path = path.replace(".stim", "_pert.stim")
        open(ours_path, "w").write(str(g).replace(f"DEPOLARIZE2({p})", f"DEPOLARIZE2({p * 1.05!r})"))
    c = stim.Circuit.from_file(path)
    nd, no = c.num_detectors, c.num_observables
    nbits = nd + no
    s_sig, pairs = E.stim_support(c)
    try:
        o_sig = our_support(path)
    except subprocess.CalledProcessError as e:
        print(json.dumps(dict(task=task, d=d, p=p, error=e.stderr[-300:])), flush=True)
        continue
    fo, fs = f"{WORK}/o.ptb64", f"{WORK}/s.ptb64"
    subprocess.run([B, "sample-fast", ours_path, str(SHOTS), fo, str(900 + d), "wy"], check=True)
    c.compile_detector_sampler(seed=77 + d).sample_write(SHOTS, filepath=fs, format="ptb64", append_observables=True)
    ao, as_ = E.load_ptb64(fo, nbits, SHOTS), E.load_ptb64(fs, nbits, SHOTS)
    n = SHOTS
    z1 = E.z2(E.popcounts(ao), E.popcounts(as_), n)
    zp = E.z2(E.pair_counts(ao, pairs), E.pair_counts(as_, pairs), n)
    to, ts = E.per_shot_counts(ao, nd), E.per_shot_counts(as_, nd)
    zm = (to.mean() - ts.mean()) / np.sqrt(to.var() / n + ts.var() / n)
    def var_se(x):
        m = x.mean(); v = x.var(); m4 = ((x - m) ** 4).mean()
        return (m4 - v * v) / n
    zv = (to.var() - ts.var()) / np.sqrt(var_se(to.astype(float)) + var_se(ts.astype(float)))
    nbh = neighbourhoods(c, nd)
    z4 = np.array([two_sample_chi2_z(joint16(ao, s), joint16(as_, s)) for s in nbh])
    ntests = nbits + len(pairs) + 2 + len(z4)
    zcrit = NormalDist().inv_cdf(1 - 0.01 / (2 * ntests))
    allz = np.concatenate([z1, zp, [zm, zv]])
    rej = int((np.abs(allz) > zcrit).sum()) + int((z4 > NormalDist().inv_cdf(1 - 0.01 / ntests)).sum())
    res = dict(task=task, d=d, p=p, shots=n, perturbed_ours=PERTURB, detectors=nd, observables=no,
               support_equal=s_sig == o_sig, support_size=len(o_sig), stim_support_size=len(s_sig),
               max_abs_z_marg=float(np.abs(z1).max()), max_abs_z_pair=float(np.abs(zp).max()) if len(zp) else 0,
               z_mean_events=float(zm), z_var_events=float(zv), n_joint4=len(z4),
               max_z_joint4=float(z4.max()) if len(z4) else 0,
               obs_rate_ours=float(E.popcounts(ao)[nd] / n), obs_rate_stim=float(E.popcounts(as_)[nd] / n),
               zcrit=float(zcrit), n_tests=int(ntests), n_reject=rej,
               verdict="PASS" if (s_sig == o_sig and rej == 0) else "FAIL", wall_s=round(time.time() - t0, 1))
    print(json.dumps(res), flush=True)
    open(OUT, "a").write(json.dumps(res) + "\n")
    os.remove(fo); os.remove(fs)
