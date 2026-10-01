#!/usr/bin/env python3
"""T-count of the circuits in circuits/*.qasm under several PyZX pipelines.

  basic      zx.optimize.basic_optimization (gate-level cancel, no ZX)
  phase_blk  zx.optimize.phase_block_optimize (phase-polynomial based)
  teleport   full_reduce + phase teleportation, circuit kept (zx.teleport_reduce)
  full_red   full_reduce + circuit extraction + basic_optimization
Usage: /tmp/fw/bin/python pyzx_compare.py [dir]
"""
import sys, glob, os, time, csv, signal

class TO(Exception): pass
def _h(*a): raise TO()
signal.signal(signal.SIGALRM, _h)
LIMIT = 90  # seconds per method
import pyzx as zx

d = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
rows = []
def nc(c):
    # non-Clifford count: gates whose phase is not a multiple of pi/2
    from fractions import Fraction
    n = 0
    for g in c.gates:
        ph = getattr(g, "phase", None)
        if ph is not None and Fraction(ph) % Fraction(1, 2) != 0:
            n += 1
        elif g.name in ("CCZ", "Toffoli", "CCX"):
            n += 1
    return n
for f in sorted(glob.glob(os.path.join(d, "circuits", "*.qasm"))):
    name = os.path.basename(f)[:-5]
    c = zx.Circuit.from_qasm_file(f)
    row = {"circuit": name, "nc_in": nc(c), "t_in": c.tcount()}
    def run(label, fn):
        t0 = time.time()
        signal.alarm(LIMIT)
        try:
            out = fn()
            signal.alarm(0)
            row["nc_" + label] = nc(out)
            row["t_" + label] = out.tcount()
            row["s_" + label] = round(time.time() - t0, 2)
        except BaseException as e:
            signal.alarm(0)
            row["nc_" + label] = "ERR"
            row["t_" + label] = "ERR"
            row["s_" + label] = type(e).__name__
    run("basic", lambda: zx.optimize.basic_optimization(c.to_basic_gates()))
    run("phase_blk", lambda: zx.optimize.phase_block_optimize(c.to_basic_gates()))
    def tele():
        g = c.to_graph()
        g2 = zx.teleport_reduce(g)
        return zx.Circuit.from_graph(g2.copy()) if False else g2
    def tele_count():
        g = c.to_graph(); zx.simplify.to_gh(g) if False else None
        g = zx.teleport_reduce(g)
        class W:  # graph T-count
            def tcount(s): return zx.tcount(g)
            gates = []
        return W()
    run("teleport", tele_count)
    def fullred():
        g = c.to_graph(); zx.simplify.full_reduce(g)
        return zx.optimize.basic_optimization(zx.extract_circuit(g.copy()).to_basic_gates())
    run("full_red", fullred)
    rows.append(row)
    print(row, flush=True)
keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "circuit", k))
with open(os.path.join(d, "pyzx_counts.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
