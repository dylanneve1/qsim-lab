#!/usr/bin/env python3
"""Condensed beyond-SV table for lowmagic-chem.md: python headline.py DATA_DIR."""
import glob
import json
import math
import os
import sys

R = sys.argv[1]
refs = {}
for p in glob.glob(os.path.join(R, "refs", "*.refs.json")):
    r = json.load(open(p))
    refs[r["name"]] = r
for p in glob.glob(os.path.join(R, "refs", "*.dmrg.json")):
    r = json.load(open(p))
    refs[r["name"]].update(e_dmrg=r["e_dmrg"], dmrg_m=r["maxm"])
E = {}
for p in glob.glob(os.path.join(R, "runs", "energy*.jsonl")):
    for line in open(p):
        r = json.loads(line)
        if "error" not in r:
            E[os.path.basename(r["file"])] = r
print("| system | n | ref | CCSD | CCSD(T) | D=8 E_reg | D=16 E_opt / E_reg | D=20 E_opt / E_reg | D=24 E (CCSD angles) | % corr, best register |")
print("|---|---|---|---|---|---|---|---|---|---|")
for m in ["h4x4", "h4x4_s", "h20", "h4x5_s", "h2o_dz", "n2_dz", "n2_dz_s", "h30", "h50", "h50_s"]:
    R0 = refs[m]
    if R0.get("e_dmrg") is not None:
        e0, kind = R0["e_dmrg"], f"DMRG {R0['dmrg_m']}"
    else:
        e0, kind = R0["e_ccsd_t"], "CCSD(T)*"
    f = lambda e: "—" if e is None or (isinstance(e, float) and math.isnan(e)) else f"{1000 * (e - e0):+.0f}"
    g = lambda d, k: E.get(f"{m}.span{d}.jw.prog", {}).get(k)
    best = min(x for x in [g(8, "e_reg"), g(16, "e_reg"), g(20, "e_reg"), g(24, "e_init")] if x is not None and not math.isnan(x))
    pc = 100 * (R0["e_hf"] - best) / (R0["e_hf"] - e0)
    cc = f(R0["e_ccsd"]) + ("†" if not R0.get("ccsd_converged", True) else "")
    print(f"| {m} | {R0['qubits']} | {kind} | {cc} | {f(R0['e_ccsd_t'])} | {f(g(8, 'e_reg'))} | {f(g(16, 'e_opt'))} / {f(g(16, 'e_reg'))} | "
          f"{f(g(20, 'e_opt'))} / {f(g(20, 'e_reg'))} | {f(g(24, 'e_init'))} | {pc:.0f} % |")
