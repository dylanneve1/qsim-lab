#!/usr/bin/env python3
"""
Stim vs qsim-lab SymPhase identical-circuit audit.

Validates:
1. Exact circuit export matches qsim-lab's SurfaceCode memory circuit.
2. Per-detector firing rates and observable rate match between Stim and SymPhase.
3. Interleaved single-threaded bit-packed benchmark on the same circuit for d = 3, 5, 7, 11, 15.
"""

import sys
import os
import subprocess
import time
import numpy as np

try:
    import stim
except ImportError:
    sys.exit("stim not found; use /tmp/fw/bin/python")

QSIM_DIR = "/mnt/HC_Volume_106989832/dylan/qsim-swarm/audit"
BINARY = os.path.join(QSIM_DIR, "target/release/examples/stim_export")
DATA_DIR = os.path.join(QSIM_DIR, "research/data/audit")
os.makedirs(DATA_DIR, exist_ok=True)

def two_proportion_z(p1, n1, p2, n2):
    if n1 == 0 or n2 == 0:
        return 0.0
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    if p_pool <= 0 or p_pool >= 1:
        return 0.0
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0:
        return 0.0
    return (p1 - p2) / se

def export_circuit(d: int, p: float = 0.003) -> str:
    path = os.path.join(DATA_DIR, f"d{d}_p{p:.3f}.stim".replace("0.", "0"))
    with open(path, "w") as f:
        subprocess.run([BINARY, str(d), str(p), "1", "export"], stdout=f, check=True)
    return path

def get_sym_rates(d: int, p: float = 0.003, shots: int = 100_000):
    proc = subprocess.run([BINARY, str(d), str(p), str(shots), "rates"],
                          capture_output=True, text=True, check=True)
    rates = {}
    obs_rate = None
    actual_shots = shots
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "shots=" in line:
            parts = line.split("shots=")
            actual_shots = int(parts[1].split(")")[0])
        elif line.startswith("D"):
            idx, val = line.split(":")
            rates[int(idx[1:])] = float(val)
        elif line.startswith("OBS:"):
            obs_rate = float(line.split(":")[1])
    return rates, obs_rate, actual_shots

def get_stim_rates(circuit_path: str, shots: int = 100_000):
    c = stim.Circuit.from_file(circuit_path)
    s = c.compile_detector_sampler()
    # sample with append_observables=True
    data = s.sample(shots, append_observables=True, bit_packed=False)
    num_dets = c.num_detectors
    det_rates = data[:, :num_dets].mean(axis=0)
    obs_rate = data[:, num_dets:].mean()
    return det_rates, obs_rate

def run_validation():
    print("=" * 70)
    print("TASK 1 VALIDATION: Per-detector and Observable Firing Rates")
    print("=" * 70)
    
    threshold = 4.9  # Bonferroni bound for ~1000 hypotheses
    val_configs = [
        (3, 0.003, 100_000),
        (5, 0.003, 100_000),
        (7, 0.003, 100_000),
        (11, 0.003, 50_000),
    ]
    
    for d, p, shots in val_configs:
        stim_path = export_circuit(d, p)
        sym_rates, sym_obs, sym_shots = get_sym_rates(d, p, shots)
        stim_rates, stim_obs = get_stim_rates(stim_path, sym_shots)
        
        num_dets = len(stim_rates)
        max_z = 0.0
        disagreements = 0
        
        for i in range(num_dets):
            p_sym = sym_rates[i]
            p_stim = stim_rates[i]
            z = two_proportion_z(p_sym, sym_shots, p_stim, sym_shots)
            if abs(z) > max_z:
                max_z = abs(z)
            if abs(z) > threshold:
                disagreements += 1
                print(f"  d={d} D{i}: sym={p_sym:.6f} stim={p_stim:.6f} z={z:.2f} (DISAGREE)")
        
        obs_z = two_proportion_z(sym_obs, sym_shots, stim_obs, sym_shots)
        if abs(obs_z) > max_z:
            max_z = abs(obs_z)
        if abs(obs_z) > threshold:
            disagreements += 1
            print(f"  d={d} OBS: sym={sym_obs:.6f} stim={stim_obs:.6f} z={obs_z:.2f} (DISAGREE)")
            
        status = "PASS" if disagreements == 0 else "FAIL"
        print(f"d={d:2d} (p={p}, {sym_shots} shots, {num_dets} dets + 1 obs): "
              f"max |z| = {max_z:.2f} | failures = {disagreements}/{num_dets+1} | {status} "
              f"(OBS: sym={sym_obs:.5f}, stim={stim_obs:.5f}, z={obs_z:+.2f})")
        assert disagreements == 0, f"Validation failed for d={d}"

def run_bench():
    print("\n" + "=" * 70)
    print("TASK 1 BENCHMARK: Identical Circuit Single-Threaded Bit-Packed")
    print("=" * 70)
    
    # Distances required: d = 3, 5, 7, 11, 15
    configs = [
        (3, 0.003, 1_000_000),
        (5, 0.003, 500_000),
        (7, 0.003, 200_000),
        (11, 0.003, 50_000),
        (15, 0.003, 20_000),
    ]
    reps = 5
    
    results = []
    
    for d, p, shots in configs:
        stim_path = export_circuit(d, p)
        circuit = stim.Circuit.from_file(stim_path)
        stim_sampler = circuit.compile_detector_sampler()
        
        # Warmup both
        stim_sampler.sample(1000, append_observables=True, bit_packed=True)
        subprocess.run([BINARY, str(d), str(p), "1000", "bench"],
                       capture_output=True, text=True, check=True)
        
        sym_times = []
        stim_times = []
        actual_shots = (shots // 64) * 64
        
        print(f"\n--- Benchmarking d={d} ({circuit.num_qubits} qubits, {circuit.num_detectors} dets, {actual_shots} shots, {reps} reps) ---")
        
        for rep in range(reps):
            # 1. SymPhase
            proc = subprocess.run([BINARY, str(d), str(p), str(shots), "bench"],
                                  capture_output=True, text=True, check=True)
            # Parse output: d=... shots=... time=... rate=...
            t_sym = None
            for line in proc.stdout.splitlines():
                if "time=" in line:
                    parts = line.split()
                    for pt in parts:
                        if pt.startswith("time="):
                            t_sym = float(pt.split("=")[1])
            assert t_sym is not None, f"Failed to parse SymPhase time: {proc.stdout}"
            sym_times.append(t_sym)
            
            # 2. Stim
            t0 = time.perf_counter()
            stim_sampler.sample(actual_shots, append_observables=True, bit_packed=True)
            t_stim = time.perf_counter() - t0
            stim_times.append(t_stim)
            
            rate_sym = actual_shots / t_sym
            rate_stim = actual_shots / t_stim
            print(f"  Rep {rep+1}: SymPhase {t_sym:.5f}s ({rate_sym:.0f} shots/s) | "
                  f"Stim {t_stim:.5f}s ({rate_stim:.0f} shots/s) | "
                  f"ratio: {rate_sym / rate_stim:.2f}x")
        
        min_sym = min(sym_times)
        min_stim = min(stim_times)
        rate_sym_best = actual_shots / min_sym
        rate_stim_best = actual_shots / min_stim
        ratio = rate_sym_best / rate_stim_best
        
        results.append({
            "d": d,
            "qubits": circuit.num_qubits,
            "detectors": circuit.num_detectors,
            "shots": actual_shots,
            "min_sym_s": min_sym,
            "min_stim_s": min_stim,
            "rate_sym": rate_sym_best,
            "rate_stim": rate_stim_best,
            "ratio": ratio,
        })
        
    output_lines = []
    output_lines.append("\n" + "=" * 70)
    output_lines.append("FINAL RESULTS TABLE (Identical circuit, min-of-5, bit-packed)")
    output_lines.append("=" * 70)
    output_lines.append("| distance | qubits | detectors | qsim-lab (shots/s) | Stim (shots/s) | ratio (qsim/Stim) |")
    output_lines.append("|---|---|---|---|---|---|")
    for r in results:
        line = (f"| {r['d']:2d} | {r['qubits']:3d} | {r['detectors']:4d} | "
                f"**{r['rate_sym']:.2e}** ({r['min_sym_s']:.5f}s) | "
                f"{r['rate_stim']:.2e} ({r['min_stim_s']:.5f}s) | "
                f"**{r['ratio']:.2f}x** |")
        output_lines.append(line)
        print(line)

    results_file = os.path.join(DATA_DIR, "stim_benchmark_results.txt")
    with open(results_file, "w") as f:
        f.write("\n".join(output_lines) + "\n\n")
        f.write("Raw reps:\n")
        for r in results:
            f.write(f"d={r['d']}: sym_min={r['min_sym_s']:.5f}s stim_min={r['min_stim_s']:.5f}s ratio={r['ratio']:.2f}x\n")
    print(f"\nWrote benchmark results to {results_file}")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("validate", "all"):
        run_validation()
    if mode in ("bench", "all"):
        run_bench()
