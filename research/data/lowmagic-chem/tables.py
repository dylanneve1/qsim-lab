#!/usr/bin/env python3
"""All tables of research/simulability/lowmagic-chem.md from the collected data.

  python tables.py RESULTS_DIR > tables.md

RESULTS_DIR holds refs/*.refs.json, refs/*.dmrg.json and the run outputs *.jsonl
(profile*.jsonl, energy*.jsonl, check*.jsonl).
"""
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
    refs.setdefault(r["name"], {}).update({"e_dmrg": r["e_dmrg"], "dmrg_m": r["maxm"], "dmrg_secs": r["secs"]})


def rows(pat):
    out = {}
    for p in sorted(glob.glob(os.path.join(R, "runs", pat))):
        for line in open(p):
            r = json.loads(line)
            if "error" not in r:
                out[(r["task"], os.path.basename(r["file"]))] = r
    return out


def parts(f):
    b = f.replace(".prog", "").split(".")
    return b[0], ".".join(b[1:-1]), b[-1]


def ref(m):
    r = refs[m]
    if r.get("e_fci") is not None:
        return r["e_fci"], "FCI"
    if r.get("e_dmrg") is not None:
        return r["e_dmrg"], f"DMRG M={r['dmrg_m']}"
    return r.get("e_ccsd_t"), "CCSD(T)"


def mev(e, e0):
    return "—" if e is None or e != e or e0 is None else f"{1000 * (e - e0):+.1f}"


prof = rows("profile*.jsonl")
eng = rows("energy*.jsonl")
chk = rows("check*.jsonl")

# ---------------------------------------------------------------- 1. structure map
print("## Structure map: d, saturation point, f, branching rank\n")
print("`sat` = rotation index at which d reaches its final value (of `rot`); `r` = branching rank "
      "(`>1024` = overflow of the 1024-term cap); `k_sym = n - d`.\n")
print("| system | n | ansatz / circuit | enc | rot | d | n−d | sat | f | r |")
print("|---|---|---|---|---|---|---|---|---|---|")
order = {"trot0.01": 0, "trot0.1": 1, "trot0.5": 2, "uccsd": 3, "uccd": 4, "upccgsd1": 5, "upccgsd2": 6, "pucc": 7, "qpe3": 8, "qpe5": 9}
items = []
for (t, f), r in prof.items():
    m, a, e = parts(f)
    if a in order or a.startswith("trot"):
        items.append((m, r["n"], order.get(a, 10), a, e, r))
for m, n, _, a, e, r in sorted(items, key=lambda x: (x[1], x[0], x[2], x[4])):
    rk = r.get("rank")
    rs = "—" if not rk else (str(rk["r_max"]) if rk["ok"] else f">{rk['r_max']}")
    print(f"| {m} | {n} | {a} | {e} | {r['rotations']} | {r['d']} | {n - r['d']} | {r['sat_rot']} | {r['f']} | {rs} |")

# ---------------------------------------------------------------- 2. analytic d at every size
print("\n## d of full ansätze at every size (GF(2) rank of the excitation x-vectors)\n")
print("| system | n | electrons | d(UCCSD) | d(UCCD) | d(1-UpCCGSD) | d(pUCCD) | n − d(UCCSD) |")
print("|---|---|---|---|---|---|---|---|")
for m, r in sorted(refs.items(), key=lambda x: (x[1].get("qubits", 0), x[0])):
    if "d_uccsd" not in r:
        continue
    print(f"| {m} | {r['qubits']} | {r['nelec']} | {r['d_uccsd']} | {r['d_uccd']} | {r['d_upccgsd']} | {r['d_pair']} | {r['qubits'] - r['d_uccsd']} |")

# ---------------------------------------------------------------- 3. energies
print("\n## Energies (errors in mEh against the reference; + = above)\n")
print("`E_ccsd-angles`: UCC with the CCSD amplitudes as angles; `E_opt`: after Rotosolve "
      "(sweeps/params as in the data); `E_reg`: lowest energy of H projected on the same 2^d register "
      "(Lanczos; a bound no circuit in this register can beat); `nnz`: non-zero register amplitudes.\n")
print("| system | n | ref | HF | CCSD | CCSD(T) | ansatz | K | d | nnz/2^d | E_ccsd-angles | E_opt | E_reg | % corr (opt) | s/eval |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def akey(a):
    import re

    m = re.match(r"([a-z]+)(\d*)", a)
    return (m.group(1), int(m.group(2) or 0))


items = []
for (t, f), r in eng.items():
    m, a, e = parts(f)
    if m not in refs:
        continue
    items.append((r["n"], m, akey(a), a, r))
for n, m, _, a, r in sorted(items):
    e0, kind = ref(m)
    if e0 is None:
        continue
    R0 = refs[m]
    corr = R0["e_hf"] - e0
    pc = 100 * (R0["e_hf"] - r["e_opt"]) / corr if corr else 0
    er = r.get("e_reg")
    er = None if er is None or (isinstance(er, float) and math.isnan(er)) else er
    print(f"| {m} | {n} | {kind} | {mev(R0['e_hf'], e0)} | {mev(R0['e_ccsd'], e0)}{'' if R0.get('ccsd_converged', True) else '†'} | "
          f"{mev(R0.get('e_ccsd_t'), e0)} | {a} | {r['params']} | {r['d']} | {r['nnz']}/{2 ** r['d']} | "
          f"{mev(r['e_init'], e0)} | {mev(r['e_opt'], e0)} | {mev(er, e0)} | {pc:.1f} | {r['eval_secs']:.3g} |")

# ---------------------------------------------------------------- 4. checks
if chk:
    print("\n## Engine vs state vector (check runs)\n")
    print("| program | n | d | |E_cs − E_sv| | infidelity |")
    print("|---|---|---|---|---|")
    for (t, f), r in sorted(chk.items()):
        print(f"| {f} | {r['n']} | {r['d']} | {abs(r['e_cs'] - r['e_sv']):.1e} | {max(r['infidelity'], 0):.1e} |")

# ---------------------------------------------------------------- 5. references
print("\n## Classical references\n")
print("| system | basis | n | e⁻ | HF | MP2 | CCSD | CCSD(T) | FCI | DMRG (M) | FCI dim |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for m, r in sorted(refs.items(), key=lambda x: (x[1].get("qubits", 0), x[0])):
    if "e_hf" not in r:
        continue
    f = lambda v: "—" if v is None else f"{v:.6f}"
    dm = "—" if r.get("e_dmrg") is None else f"{r['e_dmrg']:.6f} ({r['dmrg_m']})"
    cc = f(r["e_ccsd"]) + ("†" if not r.get("ccsd_converged", True) else "")
    print(f"| {m} | {r['basis']} | {r['qubits']} | {r['nelec']} | {f(r['e_hf'])} | {f(r['e_mp2'])} | {cc} | {f(r.get('e_ccsd_t'))} | {f(r.get('e_fci'))} | {dm} | {r['fci_dim']:.3g} |")
print("\n† CCSD not converged (300 iterations); value is the last iterate.")
