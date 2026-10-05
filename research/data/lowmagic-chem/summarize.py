#!/usr/bin/env python3
"""Tables from the run outputs: python summarize.py DATA_DIR [OUT_DIR] -> markdown on stdout."""
import glob
import json
import os
import sys

data = sys.argv[1]
outd = sys.argv[2] if len(sys.argv) > 2 else os.path.join(data, "out")
refs = {}
for p in glob.glob(os.path.join(data, "*.refs.json")):
    r = json.load(open(p))
    refs[r["name"]] = r
for p in glob.glob(os.path.join(data, "*.dmrg.json")):
    r = json.load(open(p))
    refs.setdefault(r["name"], {})["e_dmrg"] = r["e_dmrg"]
    refs[r["name"]]["dmrg_m"] = r["maxm"]


def best(r):
    for k in ("e_fci", "e_dmrg"):
        if r.get(k) is not None:
            return r[k], k[2:]
    return r.get("e_ccsd_t"), "ccsd_t"


def load(pat):
    rows = []
    for p in glob.glob(os.path.join(outd, pat)):
        for line in open(p):
            r = json.loads(line)
            if "error" not in r:
                rows.append(r)
    return rows


def meta(prog):
    m = {}
    for line in open(prog):
        if line.startswith("# "):
            k, *v = line[2:].split()
            m[k] = " ".join(v)
        elif not line.startswith(("n ", "param")):
            break
    return m


mode = sys.argv[3] if len(sys.argv) > 3 else "energy"
if mode == "energy":
    print("| molecule | qubits | ansatz | K gen. | d | nnz | E_init err (mEh) | E_opt err (mEh) | % corr | ref | CCSD err | CCSD(T) err | eval s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    seen = {}
    for r in load("energy*.jsonl"):
        seen[r["file"]] = r
    for r in sorted(seen.values(), key=lambda r: (r["n"], os.path.basename(r["file"]))):
        name = os.path.basename(r["file"]).split(".")[0]
        ans = os.path.basename(r["file"]).split(".")[1]
        R = refs.get(name)
        if not R:
            continue
        eref, kind = best(R)
        if eref is None:
            continue
        ehf = R["e_hf"]
        f = lambda e: f"{1000 * (e - eref):+.2f}"
        pc = 100 * (ehf - r["e_opt"]) / (ehf - eref) if ehf != eref else 0
        cc = f(R["e_ccsd"]) if R.get("e_ccsd") is not None else "-"
        cct = f(R["e_ccsd_t"]) if R.get("e_ccsd_t") is not None else "-"
        print(f"| {name} | {r['n']} | {ans} | {r['params']} | {r['d']} | {r['nnz']} | {f(r['e_init'])} | {f(r['e_opt'])} | {pc:.1f} | {kind} | {cc} | {cct} | {r['eval_secs']:.3g} |")
