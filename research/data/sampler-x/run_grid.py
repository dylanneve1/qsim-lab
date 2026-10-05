#!/usr/bin/env python3
"""Runs the timing grid of research/qec/sampler-x.md, one bench-lock hold per cell (< 10 min each).

usage: run_grid.py <what: e2e|throughput|threads> <stim_compare> <stim CLI> <wtime> <circuit dir> <out.jsonl>
                   [d list] [p list] [shots list] [reps] [max load]
Before every cell it waits until the 1-min load is below `max load` (default 16 = vCPUs; env GRID_DEADLINE,
epoch seconds, ends the wait so a cell runs, loaded, rather than not at all), then runs the
cell under `flock /dev/shm/qsim/bench.lock`, and records `uptime` and `free -g` immediately before and after
the timed block in a JSON note line. Circuits: <circuit dir>/B_d{d}_p{p}.stim (Stim's rotated_memory_z,
rounds = d, all four noise knobs = p; written by make_circuits.py).
"""
import json, os, subprocess, sys, time

what, B, S, WT, CD, OUT = sys.argv[1:7]
ds = [int(x) for x in (sys.argv[7] if len(sys.argv) > 7 else "3,5,7,11,15").split(",")]
ps = [float(x) for x in (sys.argv[8] if len(sys.argv) > 8 else "0.001,0.003").split(",")]
shots = sys.argv[9] if len(sys.argv) > 9 else "1e2,1e3,1e4,1e5,1e6,1e7"
reps = sys.argv[10] if len(sys.argv) > 10 else "3"
maxload = float(sys.argv[11]) if len(sys.argv) > 11 else 16.0
here = os.path.dirname(os.path.abspath(__file__))
py = sys.executable


def state():
    return dict(uptime=subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(),
                free_g=subprocess.run(["free", "-g"], capture_output=True, text=True).stdout.split("\n")[1])


def note(**k):
    with open(OUT, "a") as f:
        f.write(json.dumps(dict(note=True, time=time.strftime("%H:%M:%S"), **k)) + "\n")


for p in ps:
    for d in ds:
        F = f"{CD}/B_d{d}_p{p}.stim"
        # wait for a quiet window, but not past $GRID_DEADLINE (epoch seconds): after that the cell
        # runs anyway and its load is in the record
        deadline = float(os.environ.get("GRID_DEADLINE", "inf"))
        while os.getloadavg()[0] >= maxload and time.time() < deadline:
            time.sleep(30)
        if what == "e2e":
            cell = [py, f"{here}/e2e.py", B, S, WT, F, str(d), str(p), reps, shots]
        elif what == "throughput":
            n = {3: 4_000_000, 5: 2_000_000, 7: 1_000_000, 11: 256_000, 15: 128_000, 21: 64_000,
                 25: 32_000}[d] if shots == "auto" else int(float(shots))
            cell = [py, f"{here}/throughput.py", B, S, WT, F, str(d), str(p), str(n), reps]
        else:
            raise SystemExit("unknown grid " + what)
        inner = ("uptime; free -g | sed -n 2p; " + " ".join(cell) + " >> " + OUT + "; uptime; free -g | sed -n 2p")
        t = time.time()
        r = subprocess.run(["flock", "/dev/shm/qsim/bench.lock", "bash", "-c", inner], capture_output=True,
                           text=True)
        note(cell=f"{what} d={d} p={p}", wall_s=round(time.time() - t, 1), lock_log=r.stdout.strip(),
             err=r.stderr.strip()[-500:])
