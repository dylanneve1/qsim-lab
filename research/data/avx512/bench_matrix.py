#!/usr/bin/env python3
"""Interleaved, locked timing driver for baselines.py.

usage: bench_matrix.py <plan.json> <out.csv> [--max-hold 480] [--reps 2] [--rounds 2]

plan.json: list of cells {"workload", "n", "prec", "threads", "engines": [[fw, file, {knobs}], ...]}
(`file` is a path). Each cell runs `rounds` rounds; in every round each engine runs once
(a fresh process doing `reps` in-process repetitions), engine order reversed in odd rounds
(ABBA), so every engine gets rounds x reps timings interleaved with the others.

Locking: engine runs are packed into holds of the global bench lock (flock on
/dev/shm/qsim/bench.lock, same lock as flock(1)) of at most --max-hold seconds (default
420), using the last measured duration of the same engine run as the estimate; a cell may
span several holds; the lock is released for >= 20 s between holds. `uptime` and
`free -g` are logged before and after every hold (holds.log next to out.csv). Per engine
run we record load1 before/after and the CPU used by other processes during the run
("others_cores" = (busy CPU time of the whole machine - CPU time of our child processes)
/ wall time) and in a 1 s idle sample just before it ("others_idle").

Pinning: 8 threads -> taskset -c 0,2,...,14 (one vCPU per physical core); 16 -> 0-15.
"""
import csv
import fcntl
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = "/dev/shm/qsim/bench.lock"
CPUS = {8: "0,2,4,6,8,10,12,14", 16: "0-15", 4: "0,2,4,6", 1: "0"}
FIELDS = ["framework", "workload", "file", "n", "prec", "threads", "config", "round", "rep",
          "seconds", "timer_kind", "load1_before", "load1_after", "others_cores", "others_idle",
          "extra"]


def load1():
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def busy_ticks():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    # user nice system idle iowait irq softirq steal
    return v[0] + v[1] + v[2] + v[5] + v[6] + v[7]


def log_state(path, tag):
    up = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    fr = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout.strip()
    with open(path, "a") as f:
        f.write(f"== {time.strftime('%Y-%m-%dT%H:%M:%S')} {tag}\n{up}\n{fr}\n")


def mem_available():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) << 10
    return 0


def state_bytes(path, prec):
    with open(path) as f:
        for line in f:
            if line.startswith("n "):
                return (1 << int(line.split()[1])) * (8 if prec == "c64" else 16)
    return 0


def wait_for_memory(need, tag):
    """Coordinator rule: re-check free memory right before every run > 2 GB and pause
    rather than push the machine into swap (keep >= 6 GiB available after the run)."""
    while mem_available() < need + (6 << 30):
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {tag}: MemAvailable "
                         f"{mem_available() >> 20} MiB < {(need + (6 << 30)) >> 20} MiB, pausing\n")
        time.sleep(30)


def run_engine(fw, path, prec, threads, knobs, reps):
    env = dict(os.environ, THREADS=str(threads), OMP_NUM_THREADS=str(threads),
               RAYON_NUM_THREADS=str(threads), OMP_PROC_BIND="false")
    sb = state_bytes(path, prec)
    # peak RSS estimate: one state (two for Aer's result path, to be safe) + 1 GiB
    need = (2 if fw == "aer" else 1) * sb + (1 << 30)
    pre = []
    if sb >= (1 << 30):
        wait_for_memory(need, f"{fw} {os.path.basename(path)} {prec}")
        # address-space cap (virtual: thread stacks/arenas count, hence the slack)
        pre = ["prlimit", f"--as={2 * sb + (8 << 30)}"]
    cmd = pre + ["taskset", "-c", CPUS[threads], sys.executable, os.path.join(HERE, "baselines.py"),
                 "run", fw, path, prec, str(reps)] + [f"{k}={v}" for k, v in knobs.items()]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    rows = []
    for line in r.stdout.splitlines():
        if line.startswith("{"):
            rows.append(json.loads(line))
    if r.returncode != 0 or not rows:
        sys.stderr.write(f"ERROR {fw} {path} {prec} {knobs}: rc={r.returncode} {r.stderr[-600:]}\n")
    return rows


class Lock:
    """The global bench lock, held in holds of at most `max_hold` seconds: a hold is
    released before an engine run that would not fit (cells may span several holds; the
    lock is released for >= 20 s between holds). Before each hold the 1-min load must be
    <= max_load (until `gate_until`); `uptime` / `free -g` are logged per hold."""

    def __init__(self, max_hold, max_load, gate_until, stop_at, log):
        self.max_hold, self.max_load = max_hold, max_load
        self.gate_until, self.stop_at, self.log = gate_until, stop_at, log
        self.f, self.t0, self.last_msg = None, 0.0, 0.0

    def stopped(self):
        return bool(self.stop_at) and time.strftime("%H:%M") >= self.stop_at

    def release(self):
        if self.f:
            log_state(self.log, f"hold end ({time.time() - self.t0:.0f}s)")
            fcntl.flock(self.f, fcntl.LOCK_UN)
            self.f.close()
            self.f = None
            time.sleep(20)

    def ensure(self, need):
        """Hold the lock with at least `need` seconds left in the hold; False = stop time."""
        if self.f and time.time() - self.t0 + need <= self.max_hold:
            return True
        self.release()
        while load1() > self.max_load and time.strftime("%H:%M") < self.gate_until:
            if self.stopped():
                return False
            if time.time() - self.last_msg > 300:
                print(f"[{time.strftime('%H:%M:%S')}] load {load1():.1f} > {self.max_load}: waiting",
                      flush=True)
                self.last_msg = time.time()
            time.sleep(15)
        if self.stopped():
            return False
        self.f = open(LOCK, "w")
        tw = time.time()
        fcntl.flock(self.f, fcntl.LOCK_EX)
        self.t0 = time.time()
        log_state(self.log, f"hold start (waited {self.t0 - tw:.0f}s for the lock)")
        return True


def main():
    plan = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    args = sys.argv[3:]
    opt = lambda k, d: type(d)(args[args.index(k) + 1]) if k in args else d
    max_hold, reps, rounds = opt("--max-hold", 420.0), opt("--reps", 2), opt("--rounds", 2)
    max_load, stop_at = opt("--max-load", 16.0), opt("--stop-at", "")
    # after --gate-until (HH:MM) holds start even above --max-load; every row carries its
    # load1 / others_cores columns and such rows are reported as loaded, never as clean
    gate_until = opt("--gate-until", "99:99")
    holds_log = os.path.join(os.path.dirname(os.path.abspath(out)), "holds.log")
    new = not os.path.exists(out) or os.path.getsize(out) == 0
    fout = open(out, "a", newline="")
    w = csv.DictWriter(fout, fieldnames=FIELDS)
    if new:
        w.writeheader()
    lock = Lock(max_hold, max_load, gate_until, stop_at, holds_log)
    est = {}  # (fw, file, prec, threads) -> seconds of the last run of that engine
    tick = os.sysconf("SC_CLK_TCK")
    for i, c in enumerate(plan):
        t_cell = time.time()
        loads, others_all = [], []
        stop = False
        for rd in range(rounds):
            engines = c["engines"] if rd % 2 == 0 else list(reversed(c["engines"]))
            for fw, path, knobs in engines:
                key = (fw, path, c["prec"], c["threads"])
                guess = est.get(key, 5.0 * reps * 4.0 ** max(0, (c["n"] - 24) / 2))
                if not lock.ensure(guess + 5):
                    stop = True
                    break
                b_idle, t_idle = busy_ticks(), time.time()
                time.sleep(1.0)
                others_idle = (busy_ticks() - b_idle) / tick / (time.time() - t_idle)
                l_before, b0, ch0, t0 = load1(), busy_ticks(), os.times(), time.time()
                rows = run_engine(fw, path, c["prec"], c["threads"], knobs, reps)
                wall = time.time() - t0
                ch1 = os.times()
                ours = (ch1.children_user - ch0.children_user) + (ch1.children_system - ch0.children_system)
                others = max(0.0, ((busy_ticks() - b0) / tick - ours) / wall)
                l_after = load1()
                est[key] = wall
                loads.append(l_before)
                others_all.append(others)
                for r in rows:
                    extra = {k: v for k, v in r.items() if k not in (
                        "fw", "file", "n", "prec", "threads", "config", "rep", "seconds", "timer")}
                    w.writerow(dict(framework=r["fw"], workload=c["workload"], file=r["file"],
                                    n=r["n"], prec=r["prec"], threads=r["threads"],
                                    config=r["config"], round=rd, rep=r["rep"],
                                    seconds=f"{r['seconds']:.6f}", timer_kind=r["timer"],
                                    load1_before=f"{l_before:.2f}", load1_after=f"{l_after:.2f}",
                                    others_cores=f"{others:.2f}", others_idle=f"{others_idle:.2f}",
                                    extra=json.dumps(extra, separators=(",", ":"))))
                fout.flush()
            if stop:
                break
        if loads:
            print(f"[{time.strftime('%H:%M:%S')}] cell {i + 1}/{len(plan)} {c['workload']} n={c['n']} "
                  f"{c['prec']} t={c['threads']}: {time.time() - t_cell:.0f}s load1 "
                  f"{min(loads):.1f}-{max(loads):.1f} others {max(others_all):.1f} cores max", flush=True)
        if stop:
            print(f"stop time {stop_at} reached at cell {i + 1}/{len(plan)}", flush=True)
            break
    lock.release()
    fout.close()


if __name__ == "__main__":
    main()
