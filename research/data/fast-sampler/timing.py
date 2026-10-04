#!/usr/bin/env python3
"""Single-thread throughput: qsim-lab FastSampler vs Stim 1.16 on Stim's own circuit.

usage: timing.py <stim_compare binary> <workdir> <d> <p> <shots> [reps=3] [old=1]

Circuit: stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d, all four
noise knobs = p), written once to a .stim file that every contender reads.
Contenders (interleaved, alternating order per rep, min over reps):
  stim_write      pip Stim: compile_detector_sampler() once, then
                  sample_write(shots, /dev/null, "ptb64", append_observables=True)
  stim_cli        natively built Stim CLI (env STIM_CLI), `stim detect --out_format ptb64
                  --append_observables` to /dev/null: wall time of the whole process
  stim_cli_net    stim_cli minus the wall time of the same command with 64 shots (start-up,
                  parse, compile removed: this favours Stim)
  ours_*          `stim_compare bench-fast <file> <shots> 1 16`: FastSampler sampling only
                  (internal timer, ptb64 bytes written to /dev/null), all code-path variants
  ours_e2e        wall time of `stim_compare sample-fast <file> <shots> /dev/null` (parse,
                  compile, sample, write; wyrand), comparable to stim_cli
  stim_dem_write  pip Stim's DEM sampler: circuit.detector_error_model() once (time reported
                  separately), compile_sampler(), sample_write(ptb64 detectors and observables
                  to /dev/null)
  stim_dem_cli_net  native `stim sample_dem` on the same DEM, minus its 64-shot wall time
  old_*           (old=1) the previous SymPhase paths: `stim_compare bench` dense/StdRng,
                  sparse/StdRng, sparse/SmallRng
"""
import sys, os, subprocess, time, json, platform
import stim

binary, work, d, p, shots = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
reps = int(sys.argv[6]) if len(sys.argv) > 6 else 3
old = int(sys.argv[7]) if len(sys.argv) > 7 else 1
shots = (shots // 64) * 64
os.makedirs(work, exist_ok=True)
path = f"{work}/B_d{d}_p{p}.stim"
circ = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                              after_clifford_depolarization=p, before_round_data_depolarization=p,
                              before_measure_flip_probability=p, after_reset_flip_probability=p)
open(path, "w").write(str(circ))
env = dict(os.environ, RAYON_NUM_THREADS="1")
cli = os.environ.get("STIM_CLI")


def kv(out):
    return {k: v for k, v in (x.split("=", 1) for x in out.split() if "=" in x)}


def run_ours():
    out = subprocess.run([binary, "bench-fast", path, str(shots), "1", "16"], capture_output=True,
                         text=True, check=True, env=env).stdout
    m = kv(out)
    r = {f"ours_{k}": float(v) for k, v in m.items()
         if k.split("_")[0] in ("blocked", "blocked32", "unblocked", "column")}
    r["_compile"] = float(m["compile"]) + float(m["parse"])
    r["_hits"] = float(m["hits_per_shot"])
    t = time.perf_counter()
    subprocess.run([binary, "sample-fast", path, str(shots), os.devnull, "1", "wy"], check=True, env=env)
    r["ours_e2e"] = time.perf_counter() - t
    if old:
        m = kv(subprocess.run([binary, "bench", path, str(shots), "1"], capture_output=True, text=True,
                              check=True, env=env).stdout)
        r["old_dense_stdrng"] = float(m["sample_min"])
        r["old_sparse_stdrng"] = float(m["sparse_min"])
        r["old_sparse_smallrng"] = float(m["sparse_smallrng_min"])
    return r


def cli_time(n):
    t = time.perf_counter()
    subprocess.run([cli, "detect", "--shots", str(n), "--in", path, "--out", os.devnull,
                    "--out_format", "ptb64", "--append_observables"], check=True)
    return time.perf_counter() - t


t = time.perf_counter()
dem = circ.detector_error_model()
extra_dem_s = time.perf_counter() - t
dem_path = path.replace(".stim", ".dem")
open(dem_path, "w").write(str(dem))
ds = dem.compile_sampler(seed=2)
ds.sample_write(1024, det_out_file=os.devnull, det_out_format="ptb64", obs_out_file=os.devnull,
                obs_out_format="ptb64")


def dem_cli_time(n):
    t = time.perf_counter()
    subprocess.run([cli, "sample_dem", "--shots", str(n), "--in", dem_path, "--out", os.devnull,
                    "--out_format", "ptb64", "--obs_out", os.devnull, "--obs_out_format", "ptb64"],
                   check=True)
    return time.perf_counter() - t


s = circ.compile_detector_sampler(seed=1)
s.sample_write(1024, filepath=os.devnull, format="ptb64", append_observables=True)  # warm
rs = {}
extra = {}
for r in range(reps):
    for who in (["stim", "ours"] if r % 2 == 0 else ["ours", "stim"]):
        if who == "stim":
            t = time.perf_counter()
            s.sample_write(shots, filepath=os.devnull, format="ptb64", append_observables=True)
            rs.setdefault("stim_write", []).append(time.perf_counter() - t)
            if cli:
                full, tiny = cli_time(shots), cli_time(64)
                rs.setdefault("stim_cli", []).append(full)
                rs.setdefault("stim_cli_net", []).append(full - tiny)
            t = time.perf_counter()
            ds.sample_write(shots, det_out_file=os.devnull, det_out_format="ptb64",
                            obs_out_file=os.devnull, obs_out_format="ptb64")
            rs.setdefault("stim_dem_write", []).append(time.perf_counter() - t)
            if cli:
                rs.setdefault("stim_dem_cli_net", []).append(dem_cli_time(shots) - dem_cli_time(64))
        else:
            o = run_ours()
            extra["ours_compile_s"] = o.pop("_compile")
            extra["hits_per_shot"] = o.pop("_hits")
            for k, v in o.items():
                rs.setdefault(k, []).append(v)
extra["stim_dem_extract_s"] = extra_dem_s
res = dict(circuit="stim rotated_memory_z", d=d, p=p, shots=shots, detectors=circ.num_detectors,
           machine=platform.machine(), node=platform.node(), stim=stim.__version__, **extra,
           **{k + "_min_s": min(v) for k, v in rs.items()},
           **{k + "_all_s": v for k, v in rs.items()})
best = res["ours_blocked_wyrand_min_s"]
res["mshots_ours"] = shots / best / 1e6
res["mshots_stim_write"] = shots / res["stim_write_min_s"] / 1e6
res["ratio_vs_stim_write"] = res["stim_write_min_s"] / best
if cli:
    res["mshots_stim_cli_net"] = shots / res["stim_cli_net_min_s"] / 1e6
    res["ratio_vs_stim_cli_net"] = res["stim_cli_net_min_s"] / best
    res["ratio_e2e_vs_stim_cli"] = res["stim_cli_min_s"] / res["ours_e2e_min_s"]
res["load1"] = os.getloadavg()[0]
print(json.dumps(res), flush=True)
