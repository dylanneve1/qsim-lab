#!/usr/bin/env python3
"""Single-thread sampling throughput, qsim-lab SymPhase vs Stim, on identical .stim files.

usage: stim_timing.py <stim_compare binary> <workdir> <d> <shots> [reps=3]

For each circuit (A = qsim-lab's native surface code exported op-by-op; B = Stim's generated
rotated_memory_z), interleaved reps, alternating which simulator goes first:
  stim_write : compile_detector_sampler() once, then sample_write(shots, /dev/null, 'ptb64',
               append_observables=True)  [same output layout as ours]
  stim_mem   : sample(shots, append_observables=True, bit_packed=True)  [shot-major numpy]
  ours       : stim_compare bench <file> <shots> 1  (parse + SymPhase compile once, then
               sampling with ptb64 bytes streamed to /dev/null), internal timer
Reports min over reps for each, plus compile times.
"""
import sys, os, subprocess, time, json, platform
import stim

binary, work, d, shots = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
reps = int(sys.argv[5]) if len(sys.argv) > 5 else 3
P = 0.003
shots = (shots // 64) * 64
os.makedirs(work, exist_ok=True)
pa = f"{work}/ours_d{d}.stim"
subprocess.run([binary, "export-surface", str(d), str(P), pa], check=True)
pb = f"{work}/stimgen_d{d}.stim"
open(pb, "w").write(str(stim.Circuit.generated(
    "surface_code:rotated_memory_z", distance=d, rounds=d,
    after_clifford_depolarization=P, before_round_data_depolarization=P,
    before_measure_flip_probability=P, after_reset_flip_probability=P)))
env = dict(os.environ, RAYON_NUM_THREADS="1")

def ours(path):
    out = subprocess.run([binary, "bench", path, str(shots), "1"], capture_output=True, text=True,
                         check=True, env=env).stdout
    kv = dict(x.split("=") for x in out.split())
    return float(kv["sample_min"]), float(kv["compile"]) + float(kv["parse"])

for name, path in [("A_ours_circuit", pa), ("B_stim_circuit", pb)]:
    c = stim.Circuit.from_file(path)
    t = time.perf_counter(); s = c.compile_detector_sampler(seed=1); tc = time.perf_counter() - t
    s.sample_write(1024, filepath=os.devnull, format="ptb64", append_observables=True)  # warm
    rs = {"stim_write": [], "stim_mem": [], "ours": []}
    cli = os.environ.get("STIM_CLI")  # optional natively compiled stim (e.g. -DSIMD_WIDTH=256)
    if cli:
        rs["stim_cli_native"] = []
    oc = None
    for r in range(reps):
        order = ["stim", "ours"] if r % 2 == 0 else ["ours", "stim"]
        for who in order:
            if who == "stim":
                t = time.perf_counter()
                s.sample_write(shots, filepath=os.devnull, format="ptb64", append_observables=True)
                rs["stim_write"].append(time.perf_counter() - t)
                t = time.perf_counter()
                _ = s.sample(shots, append_observables=True, bit_packed=True)
                rs["stim_mem"].append(time.perf_counter() - t)
                del _
                if cli:
                    t = time.perf_counter()
                    subprocess.run([cli, "detect", "--shots", str(shots), "--in", path, "--out", os.devnull,
                                    "--out_format", "ptb64", "--append_observables"], check=True)
                    rs["stim_cli_native"].append(time.perf_counter() - t)
            else:
                ts, oc = ours(path)
                rs["ours"].append(ts)
    res = dict(circuit=name, d=d, shots=shots, detectors=c.num_detectors, qubits=c.num_qubits,
               machine=platform.machine(), stim=stim.__version__,
               stim_compile_s=tc, ours_compile_s=oc,
               **{k + "_min_s": min(v) for k, v in rs.items()},
               **{k + "_all_s": v for k, v in rs.items()})
    res["ratio_write"] = res["stim_write_min_s"] / res["ours_min_s"]
    res["ratio_mem"] = res["stim_mem_min_s"] / res["ours_min_s"]
    if cli:
        res["ratio_cli_native"] = res["stim_cli_native_min_s"] / res["ours_min_s"]
    res["load1"] = os.getloadavg()[0]
    print(json.dumps(res), flush=True)
