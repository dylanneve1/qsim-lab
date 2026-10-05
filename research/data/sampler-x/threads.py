#!/usr/bin/env python3
"""Multi-threaded sampling throughput of the sampler-x pipeline (research/qec/sampler-x.md §3).

usage: threads.py <stim_compare> <circuit.stim> <d> <p> <shots> <reps> <threads,...>
For every thread count T (interleaved, min over reps): `stim_compare bench-x F shots reps T` run under
`taskset`: T <= 8 -> one logical CPU per physical core (0,2,4,...), T = 16 -> all 16 logical CPUs.
Times the sampler only (hit tables built, compile excluded), ptb64 detection events + observables
written to /dev/null in shot order (per-batch wyrand streams: the output is identical for every T).
Reports shots/s and detector-shots/s (detectors x shots per second, observables not counted).
"""
import json, os, subprocess, sys

B, F, d, p, shots, reps = sys.argv[1:7]
ts = [int(x) for x in sys.argv[7].split(",")]
shots = int(float(shots))


def kv(out):
    return {k: v for k, v in (t.split("=", 1) for t in out.split() if "=" in t)}


def cpus(t):
    return ",".join(str(2 * i) for i in range(t)) if t <= 8 else ",".join(str(i) for i in range(t))


res = {}
load0 = os.getloadavg()[0]
for r in range(int(reps)):
    for t in ts[r % len(ts):] + ts[:r % len(ts)]:
        m = kv(subprocess.run(["taskset", "-c", cpus(t), B, "bench-x", F, str(shots), "1", str(t)], check=True,
                              capture_output=True, text=True).stdout)
        res.setdefault(t, []).append(float(m["sample_tables"]))
        rows = int(m["rows"])
for t, v in res.items():
    dets = rows - 1
    print(json.dumps(dict(d=int(d), p=float(p), shots=shots, threads=t, cpus=cpus(t), min_s=min(v), all_s=v,
                          shots_per_s=shots / min(v), detector_shots_per_s=shots * dets / min(v),
                          detectors=dets, load1_before=load0, load1_after=os.getloadavg()[0])), flush=True)
