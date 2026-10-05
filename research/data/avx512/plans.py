#!/usr/bin/env python3
"""Writes the timing plans (JSON cell lists) for bench_matrix.py.

usage: plans.py sweep <circuit dir> <out.json>               knob sweep, n = 24, 8 threads
       plans.py matrix <circuit dir> <best.json> <out.json> <n,...> <c64|c128|both> <threads>
best.json: {"<fw>|<workload>|<prec>": [[file_suffix, {knobs}], ...]} from pick_best.py
"""
import json
import os
import sys

WL = ["qft", "brick_cz", "brick_su4", "qv", "qaoa"]
SU4 = {"brick_su4", "qv"}


def files(cdir, wl, n):
    fs = [os.path.join(cdir, f"{wl}_{n}.txt")]
    if wl in SU4:
        fs.append(os.path.join(cdir, f"{wl}_{n}.dense.txt"))
    return fs


def sweep(cdir, n=24, threads=8):
    cells = []
    for wl in WL:
        fs = files(cdir, wl, n)
        eng = []
        for f in fs:
            eng += [["qsim", f, {"f": k}] for k in (2, 3, 4, 5)]
            eng += [["aer", f, {"fusion": 0}]] + [["aer", f, {"fusion": 1, "fmax": k}] for k in (3, 4, 5)]
            eng += [["lightning", f, {}]]
        eng += [["qsimlab", fs[0], {}]]
        cells.append(dict(workload=wl, n=n, prec="c64", threads=threads, engines=eng))
        eng = []
        for f in fs:
            eng += [["aer", f, {"fusion": 0}]] + [["aer", f, {"fusion": 1, "fmax": k}] for k in (3, 4, 5)]
            eng += [["qulacs_src", f, {"opt": o}] for o in ("none", "light", "block2", "block3")]
            eng += [["qulacs", f, {"opt": o}] for o in ("none", "block2")]
            eng += [["lightning", f, {}]]
        eng += [["qsimlab", fs[0], {}]]
        cells.append(dict(workload=wl, n=n, prec="c128", threads=threads, engines=eng))
    return cells


def matrix(cdir, best, ns, precs, threads):
    cells = []
    for n in ns:
        for prec in precs:
            for wl in WL:
                eng = []
                for fw in ("qsim", "qulacs", "qulacs_src", "aer", "lightning"):
                    for suffix, kn in best.get(f"{fw}|{wl}|{prec}", []):
                        eng.append([fw, os.path.join(cdir, f"{wl}_{n}{suffix}"), kn])
                eng.append(["qsimlab", os.path.join(cdir, f"{wl}_{n}.txt"), {}])
                cells.append(dict(workload=wl, n=n, prec=prec, threads=threads, engines=eng))
    return cells


def sweep_qulacs_src(cdir, n=24, threads=8):
    cells = []
    for wl in WL:
        eng = [["qulacs_src", f, {"opt": o}] for f in files(cdir, wl, n)
               for o in ("none", "light", "block2", "block3")]
        eng += [["qulacs", files(cdir, wl, n)[0], {"opt": "none"}]]
        cells.append(dict(workload=wl, n=n, prec="c128", threads=threads, engines=eng))
    return cells


if __name__ == "__main__":
    if sys.argv[1] == "sweep_qulacs_src":
        json.dump(sweep_qulacs_src(sys.argv[2]), open(sys.argv[3], "w"), indent=1)
    elif sys.argv[1] == "sweep":
        json.dump(sweep(sys.argv[2]), open(sys.argv[3], "w"), indent=1)
    elif sys.argv[1] == "matrix":
        best = json.load(open(sys.argv[3]))
        ns = [int(x) for x in sys.argv[5].split(",")]
        precs = ["c64", "c128"] if sys.argv[6] == "both" else [sys.argv[6]]
        json.dump(matrix(sys.argv[2], best, ns, precs, int(sys.argv[7])), open(sys.argv[4], "w"), indent=1)
    else:
        raise SystemExit(__doc__)
