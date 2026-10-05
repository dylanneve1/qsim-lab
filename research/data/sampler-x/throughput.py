#!/usr/bin/env python3
"""Sampling-only throughput (Mshot/s), single thread, Stim vs qsim-lab (research/qec/sampler-x.md).

usage: throughput.py <stim_compare> <stim CLI> <wtime> <circuit.stim> <d> <p> <shots> <reps>

Interleaved (order rotates per rep), min over reps, ptb64 with observables appended to /dev/null:
  stim_pip    pip Stim in this process: compile_detector_sampler() once, then
              sample_write(shots, os.devnull, "ptb64", append_observables=True)
  stim_net    native `stim detect` wall time at N shots minus at 64 shots (start-up, parse and
              compile removed; favours Stim, as in fast-sampler.md)
  stim_b8_net the same with --out_format b8 (faster than Stim's ptb64 writer at large d)
  x_net       `stim_compare sample-x F N /dev/null 1 1 on` minus the same at 64 shots
  x_int       `stim_compare bench-x F N 1 1`: internal timer of write_ptb64 with hit tables
  fast_net    `stim_compare sample-fast` (previous pipeline) at N minus at 64 shots; with env QSIM_OLD_BIN set,
              that binary (e.g. stim_compare built at main 6b21728)
"""
import json, os, platform, subprocess, sys, time
import stim

B, S, WT, F, d, p, shots, reps = sys.argv[1:9]
d, p, shots, reps = int(d), float(p), int(float(shots)), int(reps)
shots = (shots // 64) * 64
DN = os.devnull
circ = stim.Circuit.from_file(F)
smp = circ.compile_detector_sampler(seed=1)
smp.sample_write(1024, filepath=DN, format="ptb64", append_observables=True)


def wall(c):
    r = subprocess.run([WT, "1"] + c, check=True, capture_output=True, text=True)
    return float(r.stderr.split()[0])


def stim_cli(n):
    return wall([S, "detect", "--shots", str(n), "--in", F, "--out", DN, "--out_format", "ptb64",
                 "--append_observables"])


def stim_cli_b8(n):
    return wall([S, "detect", "--shots", str(n), "--in", F, "--out", DN, "--out_format", "b8",
                 "--append_observables"])


def x(n):
    return wall([B, "sample-x", F, str(n), DN, "1", "1", "on"])


def fast(n):
    return wall([os.environ.get("QSIM_OLD_BIN", B), "sample-fast", F, str(n), DN, "1", "wy"])


def kv(out):
    return {k: v for k, v in (t.split("=", 1) for t in out.split() if "=" in t)}


rs = {}
info = {}
who = ["stim_pip", "stim_net", "stim_b8_net", "x_net", "x_int", "fast_net"]
load0 = os.getloadavg()[0]
for r in range(reps):
    for w in who[r % len(who):] + who[:r % len(who)]:
        if w == "stim_pip":
            t = time.perf_counter()
            smp.sample_write(shots, filepath=DN, format="ptb64", append_observables=True)
            v = time.perf_counter() - t
        elif w == "stim_net":
            v = stim_cli(shots) - stim_cli(64)
        elif w == "stim_b8_net":
            v = stim_cli_b8(shots) - stim_cli_b8(64)
        elif w == "x_net":
            v = x(shots) - x(64)
        elif w == "fast_net":
            v = fast(shots) - fast(64)
        else:
            m = kv(subprocess.run([B, "bench-x", F, str(shots), "1", "1"], check=True,
                                  capture_output=True, text=True).stdout)
            v = float(m["sample_tables"])
            info = dict(compile_s=float(m["compile"]), parse_s=float(m["parse"]),
                        tables_s=float(m["tables"]), hits_per_shot=float(m["hits_per_shot"]),
                        table_bytes=int(m["table_bytes"]), rows=int(m["rows"]))
        rs.setdefault(w, []).append(v)
m = {w: min(v) for w, v in rs.items()}
best_stim = min(m["stim_pip"], m["stim_net"], m["stim_b8_net"])
res = dict(d=d, p=p, shots=shots, node=platform.node(), stim_pip_version=stim.__version__,
           load1_before=load0, load1_after=os.getloadavg()[0], **info,
           **{f"{w}_min_s": m[w] for w in who}, **{f"{w}_all_s": rs[w] for w in who},
           **{f"mshots_{w}": shots / m[w] / 1e6 for w in who},
           x_over_best_stim=best_stim / m["x_int"], x_net_over_best_stim=best_stim / m["x_net"],
           x_over_fast=m["fast_net"] / m["x_net"])
print(json.dumps(res), flush=True)
