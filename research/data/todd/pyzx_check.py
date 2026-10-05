#!/usr/bin/env python3
"""PyZX baseline and independent equality check for the T-count study.

Usage:
  pyzx_check.py baseline CIRCUIT.qc...          # PyZX T-counts (teleport_reduce, full_reduce)
  pyzx_check.py verify ORIG_DIR OUT_DIR NAME... # ZX equality of ORIG_DIR/NAME.qc and OUT_DIR/NAME.qc

`verify` composes the original with the adjoint of the optimised circuit and
runs PyZX's full_reduce (Circuit.verify_equality). A True answer is a proof
of equality up to global phase that shares no code with qsim-lab; False
only means PyZX could not reduce the composition to the identity.

Needs pyzx (tested with 0.10.0) in the active environment.
"""
import sys
import time

import pyzx as zx
from pyzx.circuit.qcparser import parse_qc


def load(path):
    """Loads a .qc file; `Zd` (inverse controlled Z) with controls is the
    same gate as `Z` and is renamed for PyZX's parser."""
    lines = []
    for line in open(path):
        t = line.split()
        if len(t) >= 3 and t[0] == "Zd":
            line = "Z " + " ".join(t[1:]) + "\n"
        lines.append(line)
    return parse_qc("".join(lines)).to_basic_gates()


def tcount(c):
    return zx.tcount(c)


def baseline(paths):
    print("circuit,t_orig,pyzx_teleport,pyzx_full,seconds")
    for p in paths:
        name = p.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        c = load(p)
        t0 = time.time()
        g = c.to_graph()
        tele = zx.tcount(zx.simplify.teleport_reduce(g))
        g = c.to_graph()
        zx.simplify.full_reduce(g)
        full = tcount(zx.optimize.basic_optimization(
            zx.extract_circuit(g.copy()).to_basic_gates()))
        print(f"{name},{tcount(c)},{tele},{full},{time.time() - t0:.1f}", flush=True)


def verify(orig_dir, out_dir, names):
    print("circuit,t_orig,t_out,zx_equal,seconds")
    for name in names:
        a = load(f"{orig_dir}/{name}.qc")
        b = load(f"{out_dir}/{name}.qc")
        t0 = time.time()
        ok = a.verify_equality(b)
        print(f"{name},{tcount(a)},{tcount(b)},{ok},{time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("baseline", "verify"):
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "baseline":
        baseline(sys.argv[2:])
    else:
        verify(sys.argv[2], sys.argv[3], sys.argv[4:])
