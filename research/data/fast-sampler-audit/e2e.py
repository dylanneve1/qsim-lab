#!/usr/bin/env python3
"""Audit timing: whole-process wall time at realistic shot counts, best Stim mode vs ours.

usage: e2e.py <stim_compare> <stim CLI> <work> <d> <p> [reps=3] [shots=10240,102400,1024000]
Per shot count, interleaved (order rotates per rep), min over reps, single thread, ptb64 to
/dev/null, detectors + observables:
  stim_detect   : stim detect --in circuit --out_format ptb64 --append_observables
  stim_dem_total: stim analyze_errors (circuit -> DEM)  +  stim sample_dem (DEM -> ptb64, obs to
                  a second ptb64 stream): Stim's DEM-sampler route including its precompute
  stim_sample_dem: the sample_dem process alone (DEM given for free)
  ours          : stim_compare sample-fast (parse, compile, sample, write; wyrand)
"""
import sys, os, subprocess, time, json, platform
import stim

B, S, work, d, p = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
reps = int(sys.argv[6]) if len(sys.argv) > 6 else 3
shots_list = [int(x) for x in (sys.argv[7] if len(sys.argv) > 7 else "10240,102400,1024000").split(",")]
os.makedirs(work, exist_ok=True)
path = f"{work}/B_d{d}_p{p}.stim"
circ = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                              after_clifford_depolarization=p, before_round_data_depolarization=p,
                              before_measure_flip_probability=p, after_reset_flip_probability=p)
open(path, "w").write(str(circ))
dem = f"{work}/B_d{d}_p{p}.dem"
env = dict(os.environ, RAYON_NUM_THREADS="1")
DN = os.devnull


def wall(cmd):
    t = time.perf_counter()
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)
    return time.perf_counter() - t


def c_detect(n):
    return wall([S, "detect", "--shots", str(n), "--in", path, "--out", DN, "--out_format", "ptb64",
                 "--append_observables"])


def c_analyze():
    return wall([S, "analyze_errors", "--in", path, "--out", dem])


def c_sample_dem(n):
    return wall([S, "sample_dem", "--shots", str(n), "--in", dem, "--out", DN, "--out_format", "ptb64",
                 "--obs_out", DN, "--obs_out_format", "ptb64"])


def c_ours(n):
    return wall([B, "sample-fast", path, str(n), DN, "1", "wy"])


c_analyze()
for n in shots_list:
    rs = {}
    order = ["stim_detect", "stim_dem", "ours"]
    for r in range(reps):
        for who in order[r % 3:] + order[:r % 3]:
            if who == "stim_detect":
                rs.setdefault("stim_detect", []).append(c_detect(n))
            elif who == "stim_dem":
                a = c_analyze(); s = c_sample_dem(n)
                rs.setdefault("stim_analyze_errors", []).append(a)
                rs.setdefault("stim_sample_dem", []).append(s)
                rs.setdefault("stim_dem_total", []).append(a + s)
            else:
                rs.setdefault("ours", []).append(c_ours(n))
    m = {k: min(v) for k, v in rs.items()}
    best = min(m["stim_detect"], m["stim_dem_total"])
    print(json.dumps(dict(d=d, p=p, shots=n, machine=platform.machine(), node=platform.node(),
                          **{k + "_s": v for k, v in m.items()},
                          ratio_best_stim_over_ours=best / m["ours"],
                          ratio_stim_detect_over_ours=m["stim_detect"] / m["ours"],
                          ratio_stim_sample_dem_only_over_ours=m["stim_sample_dem"] / m["ours"],
                          load1=os.getloadavg()[0])), flush=True)
