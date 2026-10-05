#!/usr/bin/env python3
"""Cross-validation: every framework's final state (qsim-lab bit layout) against the
numpy reference of gen_circuits.py, on small instances of every workload.

usage: [ONLY=fw,...] validate.py <circuit dir> <n[,n...]> [out.csv]
Pass: fidelity |<ref|psi>|^2 >= 1 - 1e-5 (complex64) / 1 - 1e-10 (complex128) and
max |psi - e^{i phi} ref| reported (phase-aligned).
"""
import csv
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen_circuits import WORKLOADS, reference_state  # noqa: E402

CASES = [("qsim", "c64"), ("qulacs", "c128"), ("qulacs_src", "c128"), ("aer", "c64"),
         ("aer", "c128"), ("lightning", "c64"), ("lightning", "c128"), ("qsimlab", "c64"),
         ("qsimlab", "c128")]
if os.environ.get("ONLY"):  # e.g. ONLY=qulacs_src
    CASES = [c for c in CASES if c[0] in os.environ["ONLY"].split(",")]


def main():
    cdir, ns = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "validation.csv")
    rows = []
    tmp = "/dev/shm/qsim/avx512/scratch/val.npy"
    env = dict(os.environ, THREADS="4", OMP_NUM_THREADS="4", RAYON_NUM_THREADS="4")
    ok_all = True
    for n in ns:
        for wl in WORKLOADS:
            files = [f"{wl}_{n}.txt"]
            if os.path.exists(os.path.join(cdir, f"{wl}_{n}.dense.txt")):
                files.append(f"{wl}_{n}.dense.txt")
            ref = reference_state(os.path.join(cdir, files[0]))
            for fname in files:
                for fw, prec in CASES:
                    if fw == "qsimlab" and fname.endswith(".dense.txt"):
                        continue
                    if os.path.exists(tmp):
                        os.remove(tmp)
                    cmd = [sys.executable, os.path.join(HERE, "baselines.py"), "run", fw,
                           os.path.join(cdir, fname), prec, "1", f"dump={tmp}"]
                    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
                    if r.returncode != 0:
                        print(f"FAIL {fw} {fname} {prec}: {r.stderr[-400:]}")
                        rows.append(dict(n=n, workload=wl, file=fname, fw=fw, prec=prec,
                                         fidelity="error", max_abs_diff="", passed=False))
                        ok_all = False
                        continue
                    psi = np.load(tmp)
                    ov = np.vdot(ref, psi)
                    fid = abs(ov) ** 2 / (np.vdot(psi, psi).real * np.vdot(ref, ref).real)
                    ph = ov / abs(ov)
                    dmax = float(np.abs(psi - ph * ref).max())
                    tol = 1e-5 if prec == "c64" else 1e-10
                    passed = bool(fid >= 1 - tol)
                    ok_all &= passed
                    rows.append(dict(n=n, workload=wl, file=fname, fw=fw, prec=prec,
                                     fidelity=f"{fid:.15f}", max_abs_diff=f"{dmax:.2e}",
                                     passed=passed))
                    print(f"{'ok  ' if passed else 'FAIL'} n={n} {fname:22s} {fw:9s} {prec}: "
                          f"1-F = {1 - fid:.2e}, max|d| = {dmax:.2e}", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("ALL PASS" if ok_all else "SOME FAILED", "->", out)


if __name__ == "__main__":
    main()
