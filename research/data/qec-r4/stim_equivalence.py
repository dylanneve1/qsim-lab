#!/usr/bin/env python3
"""Statistical equivalence of qsim-lab SymPhase and Stim 1.16 on IDENTICAL circuits.

Directions
  A  ours -> Stim : qsim-lab's native SurfaceCode::new(d,d) circuit, serialised op by op
                    (stim_io::to_stim). qsim-lab samples its native circuit object,
                    Stim samples the .stim file.
  B  Stim -> ours : stim.Circuit.generated("surface_code:rotated_memory_z", d, rounds=d,
                    all four noise knobs = p). Stim samples it, qsim-lab parses the same
                    file (stim_io::parse_stim) and samples that.

Tests (per direction, per d), family-wise error rate 1% (Bonferroni over all tests in the cell):
  T0 support : set of distinct single-fault signatures (detectors+observable) of our compiled
               sampler == set of error-target sets in Stim's DEM (exact, not statistical).
  T1 marginals: every detector and the observable: two-proportion z.
  T2 pairs    : P(D_i and D_j) for every pair that co-occurs in some DEM error: two-proportion z.
  T3 counts   : mean and variance of #detection events per shot (Welch z / large-sample z).
"""
import sys, os, subprocess, time, json
from statistics import NormalDist
import numpy as np
import stim

B = os.environ.get("QSIM_STIM_COMPARE", "/tmp/qsim-wt/qec-r4-target/release/examples/stim_compare")
WORK = os.environ.get("WORK", "/tmp/qsim-wt/qec-data")
P = 0.003
os.makedirs(WORK, exist_ok=True)


def load_ptb64(path, nbits, shots):
    a = np.fromfile(path, dtype="<u8")
    assert a.size % nbits == 0
    a = a.reshape(-1, nbits)
    assert a.shape[0] * 64 == shots, (a.shape, shots)
    return a


def popcounts(a):
    return np.bitwise_count(a).sum(axis=0, dtype=np.int64)


def pair_counts(a, pairs):
    out = np.empty(len(pairs), dtype=np.int64)
    for k, (i, j) in enumerate(pairs):
        out[k] = int(np.bitwise_count(a[:, i] & a[:, j]).sum(dtype=np.int64))
    return out


def per_shot_counts(a, ndet):
    tot = np.zeros(a.shape[0] * 64, dtype=np.int32)
    step = max(1, 20_000_000 // (ndet * 64))
    for s in range(0, a.shape[0], step):
        blk = a[s:s + step, :ndet]
        bits = np.unpackbits(blk.view(np.uint8).reshape(blk.shape[0], ndet, 8), axis=2, bitorder="little")
        tot[s * 64:(s + blk.shape[0]) * 64] = bits.sum(axis=1, dtype=np.int32).reshape(-1)
    return tot


def z2(c1, c2, n):
    p1, p2 = c1 / n, c2 / n
    pp = (c1 + c2) / (2 * n)
    se = np.sqrt(np.maximum(pp * (1 - pp) * 2 / n, 1e-300))
    z = (p1 - p2) / se
    return np.where((c1 + c2) == 0, 0.0, z)


def stim_support(c):
    dem = c.detector_error_model(decompose_errors=False, flatten_loops=True, allow_gauge_detectors=False)
    nd = c.num_detectors
    sigs = set()
    pairs = set()
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        ts = []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                ts.append(t.val)
            elif t.is_logical_observable_id():
                ts.append(nd + t.val)
        ts = tuple(sorted(ts))
        if ts:
            sigs.add(ts)
        dets = [x for x in ts if x < nd]
        for x in range(len(dets)):
            for y in range(x + 1, len(dets)):
                pairs.add((dets[x], dets[y]))
    return sigs, sorted(pairs)


def our_support(path):
    out = subprocess.run([B, "dem-support", path], capture_output=True, text=True, check=True).stdout
    return {tuple(int(x) for x in l.split()) for l in out.splitlines() if l.strip()}


def run_cell(direction, d, shots, seed):
    if direction == "A":
        path = f"{WORK}/ours_d{d}.stim"
        subprocess.run([B, "export-surface", str(d), str(P), path], check=True)
    else:
        path = f"{WORK}/stimgen_d{d}.stim"
        g = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                                   after_clifford_depolarization=P, before_round_data_depolarization=P,
                                   before_measure_flip_probability=P, after_reset_flip_probability=P)
        open(path, "w").write(str(g))
    c = stim.Circuit.from_file(path)
    nd, no = c.num_detectors, c.num_observables
    nbits = nd + no
    # T0
    s_sig, pairs = stim_support(c)
    o_sig = our_support(path)
    t0 = (s_sig == o_sig)
    # samples
    f_ours, f_stim = f"{WORK}/o.ptb64", f"{WORK}/s.ptb64"
    if direction == "A":
        subprocess.run([B, "sample-native", str(d), str(P), str(shots), f_ours, str(seed)], check=True)
    else:
        subprocess.run([B, "sample", path, str(shots), f_ours, str(seed)], check=True)
    c.compile_detector_sampler(seed=seed + 7).sample_write(
        shots, filepath=f_stim, format="ptb64", append_observables=True)
    ao, as_ = load_ptb64(f_ours, nbits, shots), load_ptb64(f_stim, nbits, shots)
    n = shots
    co, cs = popcounts(ao), popcounts(as_)
    z1 = z2(co, cs, n)
    po, ps = pair_counts(ao, pairs), pair_counts(as_, pairs)
    zp = z2(po, ps, n)
    to, ts = per_shot_counts(ao, nd), per_shot_counts(as_, nd)
    zm = (to.mean() - ts.mean()) / np.sqrt(to.var() / n + ts.var() / n)
    # variance: z using 4th central moments
    def var_se(x):
        m = x.mean(); v = x.var(); m4 = ((x - m) ** 4).mean()
        return (m4 - v * v) / n
    zv = (to.var() - ts.var()) / np.sqrt(var_se(to.astype(np.float64)) + var_se(ts.astype(np.float64)))
    ntests = nbits + len(pairs) + 2
    zcrit = NormalDist().inv_cdf(1 - 0.01 / (2 * ntests))
    allz = np.concatenate([z1, zp, [zm, zv]])
    res = dict(direction=direction, d=d, shots=n, detectors=nd, pairs=len(pairs),
               support_equal=bool(t0), support_size=len(o_sig), stim_support_size=len(s_sig),
               max_abs_z_marg=float(np.abs(z1).max()), mean_z2_marg=float((z1 ** 2).mean()),
               max_abs_z_pair=float(np.abs(zp).max()) if len(zp) else 0.0,
               mean_z2_pair=float((zp ** 2).mean()) if len(zp) else 0.0,
               z_mean_events=float(zm), z_var_events=float(zv),
               events_per_shot_ours=float(to.mean()), events_per_shot_stim=float(ts.mean()),
               obs_rate_ours=float(co[nd] / n), obs_rate_stim=float(cs[nd] / n), z_obs=float(z1[nd]),
               zcrit=float(zcrit), n_tests=int(ntests), n_reject=int((np.abs(allz) > zcrit).sum()),
               verdict="PASS" if (t0 and (np.abs(allz) <= zcrit).all()) else "FAIL")
    os.remove(f_ours); os.remove(f_stim)
    return res


if __name__ == "__main__":
    shots = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    ds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [3, 7, 11, 15]
    out = []
    for direction in ["A", "B"]:
        for d in ds:
            t = time.time()
            r = run_cell(direction, d, shots, seed=100 + d)
            r["wall_s"] = round(time.time() - t, 1)
            print(json.dumps(r), flush=True)
            out.append(r)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stim_equivalence.json"), "w"), indent=1)
