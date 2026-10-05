#!/usr/bin/env python3
"""Cell lists for the timing matrix (engine sets fixed from the n = 24 knob sweep).

usage: make_matrix.py <circuit dir> <out.json> <n[,n...]> <c64|c128> <threads>
                      [--variants variants.json] [--part all|fast|slow] [--skip fw,...]

Per framework (sweep: sweep_n24_raw.csv / sweep_summary.md):
  qsim        max_fused_gate_size 3 and 4 (5 too for qft); SU(4) workloads on both the
              dense (u4) and the decomposed file, at every n
  aer         fusion on, fusion_max_qubit 5 (the default; best or within noise in the sweep)
  lightning   default kernels
  qulacs_src  (c128) source build; opt=block2 on decomposed files, none on dense files
  qsimlab     examples/sv_file_bench.rs built from main 6b21728, default BlockConfig
SU(4) workloads for aer / lightning / qulacs_src: both files at n = 24; at n >= 26 only the
file variant that was faster at n = 24 (variants.json: {"<fw>|<wl>|<prec>": ".dense.txt"|".txt"}).
Cells at n >= 28 are split per framework group so every hold of the bench lock stays short:
part "fast" = qsim / qulacs_src + qsim-lab, part "slow" = one cell for Aer + qsim-lab and
one for lightning + qsim-lab (qsim-lab in every cell as the interleaved reference).
"""
import json
import os
import sys

WL = ["qft", "brick_cz", "brick_su4", "qv", "qaoa"]
SU4 = {"brick_su4", "qv"}


def cell(cdir, wl, n, prec, threads, variants):
    base = os.path.join(cdir, f"{wl}_{n}.txt")
    dense = os.path.join(cdir, f"{wl}_{n}.dense.txt")

    def files_for(fw):
        if wl not in SU4:
            return [base]
        if n <= 24:
            return [base, dense]
        v = variants.get(f"{fw}|{wl}|{prec}", ".dense.txt")
        return [os.path.join(cdir, f"{wl}_{n}{v}")]

    eng = []
    if prec == "c64":
        for f in ([base, dense] if wl in SU4 else [base]):
            for k in ((3, 4, 5) if wl == "qft" else (3, 4)):
                eng.append(["qsim", f, {"f": k}])
    for f in files_for("aer"):
        eng.append(["aer", f, {"fusion": 1, "fmax": 5}])
    for f in files_for("lightning"):
        eng.append(["lightning", f, {}])
    if prec == "c128":
        for f in files_for("qulacs_src"):
            eng.append(["qulacs_src", f, {"opt": "none" if f.endswith(".dense.txt") else "block2"}])
    eng.append(["qsimlab", base, {}])
    return dict(workload=wl, n=n, prec=prec, threads=threads, engines=eng)


def main():
    a = sys.argv[1:]
    cdir, out = a[0], a[1]
    ns = [int(x) for x in a[2].split(",")]
    prec, threads = a[3], int(a[4])
    variants = json.load(open(a[a.index("--variants") + 1])) if "--variants" in a else {}
    part = a[a.index("--part") + 1] if "--part" in a else "all"
    skip = set(a[a.index("--skip") + 1].split(",")) if "--skip" in a else set()
    cells = []
    for n in ns:
        for wl in WL:
            c = cell(cdir, wl, n, prec, threads, variants)
            c["engines"] = [e for e in c["engines"] if e[0] not in skip]
            if n >= 28:
                groups = {"fast": [("qsim", "qulacs_src")], "slow": [("aer",), ("lightning",)]}
                parts = ["fast", "slow"] if part == "all" else [part]
                for p in parts:
                    for g in groups[p]:
                        eng = [e for e in c["engines"] if e[0] in g]
                        if eng:
                            cells.append(dict(c, engines=eng + [e for e in c["engines"] if e[0] == "qsimlab"]))
            else:
                cells.append(c)
    json.dump(cells, open(out, "w"), indent=1)
    print([(c["workload"], c["n"], [e[0] for e in c["engines"]]) for c in cells])


if __name__ == "__main__":
    main()
