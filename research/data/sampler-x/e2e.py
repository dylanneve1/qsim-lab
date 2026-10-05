#!/usr/bin/env python3
"""Whole-process wall time, Stim vs qsim-lab, at every shot count (research/qec/sampler-x.md).

usage: e2e.py <stim_compare> <stim CLI> <wtime> <circuit.stim> <d> <p> <reps> <shots,shots,...> [contenders]

Per shot count (rounded up to a multiple of 64, which ptb64 needs), interleaved (the contender order rotates every rep), min over reps
(at least 11 below 10^6 shots: the minimum then filters scheduling delays of a loaded machine), single thread,
ptb64 (detection events with observables appended) to /dev/null, the same .stim file for everyone:
  stim       native `stim detect --shots N --in F --out /dev/null --out_format ptb64 --append_observables`
  stim_b8    the same with --out_format b8 (shot-major bytes; Stim writes it faster than ptb64 at large d)
  x          `stim_compare sample-x F N /dev/null 1 1 auto` (this branch: fast parse, backward compiler,
             hit tables only when they pay off)
  x_tables   same with mode on (always compile and build the hit tables)
  x_notables same with mode off (compile, no hit tables)
  x_frames   same with mode frames (Pauli-frame simulation, no compile)
  fast       `stim_compare sample-fast F N /dev/null 1 wy` (the previous FastSampler pipeline; with env
             QSIM_OLD_BIN set, that binary, e.g. stim_compare built at main 6b21728)
contenders: comma list, default stim,x,fast. Times come from wtime (posix_spawn + wait4, CLOCK_MONOTONIC).
Prints one JSON line per shot count with the 1-min load before and after the block.
"""
import json, os, platform, subprocess, sys

B, S, WT, F, d, p, reps = sys.argv[1:8]
d, p, reps = int(d), float(p), int(reps)
# ptb64 needs a multiple of 64 shots (Stim refuses others): round up (1e2 -> 128, 1e3 -> 1024, ...)
shots_list = [-(-int(float(x)) // 64) * 64 for x in sys.argv[8].split(",")]
who = (sys.argv[9] if len(sys.argv) > 9 else "stim,stim_b8,x,fast").split(",")
DN = os.devnull


def cmd(name, n):
    if name == "stim":
        return [S, "detect", "--shots", str(n), "--in", F, "--out", DN, "--out_format", "ptb64",
                "--append_observables"]
    if name == "stim_b8":
        # Stim's b8 writer is faster than its ptb64 writer at large d (§1): the best Stim mode
        return [S, "detect", "--shots", str(n), "--in", F, "--out", DN, "--out_format", "b8",
                "--append_observables"]
    if name == "x":
        return [B, "sample-x", F, str(n), DN, "1", "1", "auto"]
    if name == "x_tables":
        return [B, "sample-x", F, str(n), DN, "1", "1", "on"]
    if name == "x_notables":
        return [B, "sample-x", F, str(n), DN, "1", "1", "off"]
    if name == "x_frames":
        return [B, "sample-x", F, str(n), DN, "1", "1", "frames"]
    if name == "fast":
        return [os.environ.get("QSIM_OLD_BIN", B), "sample-fast", F, str(n), DN, "1", "wy"]
    raise SystemExit(f"unknown contender {name}")


def wall(c):
    r = subprocess.run([WT, "1"] + c, check=True, capture_output=True, text=True)
    return float(r.stderr.split()[0])


for n in shots_list:
    load0 = os.getloadavg()[0]
    rs = {w: [] for w in who}
    # short runs: more repetitions, so that the minimum filters out scheduling delays on a loaded box
    reps_n = reps if n >= 1_000_000 else max(reps, 11)
    for r in range(reps_n):
        order = who[r % len(who):] + who[:r % len(who)]
        for w in order:
            rs[w].append(wall(cmd(w, n)))
    m = {w: min(v) for w, v in rs.items()}
    out = dict(d=d, p=p, shots=n, reps=reps_n, node=platform.node(), load1_before=load0,
               load1_after=os.getloadavg()[0],
               **{f"{w}_s": m[w] for w in who}, **{f"{w}_all_s": rs[w] for w in who})
    if "stim" in m:
        for w in who:
            if not w.startswith("stim"):
                out[f"stim_over_{w}"] = m["stim"] / m[w]
    if "stim" in m and "stim_b8" in m:
        best = min(m["stim"], m["stim_b8"])
        out["best_stim_s"] = best
        for w in who:
            if not w.startswith("stim"):
                out[f"best_stim_over_{w}"] = best / m[w]
    print(json.dumps(out), flush=True)
