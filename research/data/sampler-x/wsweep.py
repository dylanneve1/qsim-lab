#!/usr/bin/env python3
"""Batch-width sweep of the sampler-x sampling loop (research/qec/sampler-x.md §3).

usage: wsweep.py <stim_compare> <circuit.stim> <d> <p> <shots> <reps> <words,...>
For each batch width W (64-shot words per batch; interleaved, order rotated per rep, min over reps),
`stim_compare bench-x F shots 1 1 W`: single thread, internal timer of write_ptb64 to /dev/null, with
hit tables (sample_tables), without (sample_notables), and with the AVX-512 kernel (sample_simd).
"""
import json, os, subprocess, sys

B, F, d, p, shots, reps = sys.argv[1:7]
ws = [int(x) for x in sys.argv[7].split(",")]
shots = int(float(shots))


def kv(out):
    return {k: v for k, v in (t.split("=", 1) for t in out.split() if "=" in t)}


res = {}
load0 = os.getloadavg()[0]
for r in range(int(reps)):
    for w in ws[r % len(ws):] + ws[:r % len(ws)]:
        m = kv(subprocess.run([B, "bench-x", F, str(shots), "1", "1", str(w)], check=True, capture_output=True,
                              text=True).stdout)
        for k in ("sample_tables", "sample_notables", "sample_simd"):
            res.setdefault((w, k), []).append(float(m[k]))
for w in ws:
    out = dict(d=int(d), p=float(p), shots=shots, words=w, load1_before=load0, load1_after=os.getloadavg()[0])
    for k in ("sample_tables", "sample_notables", "sample_simd"):
        v = res[(w, k)]
        out[k + "_min_s"] = min(v)
        out["mshots_" + k] = shots / min(v) / 1e6
    print(json.dumps(out), flush=True)
