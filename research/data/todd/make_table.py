#!/usr/bin/env python3
"""Merges the T-count runs into research/data/todd/results.csv and prints the
notebook's results table (markdown).

Inputs (paths relative to this script):
  slot.csv, pauli.csv    -- `todd_bench --csv` output of the two modes
  pyzx_baseline.csv      -- `pyzx_check.py baseline` output
  published_best.csv     -- literature table (best ancilla-free and gadget counts)

Usage: make_table.py [slot.csv] [pauli.csv]
"""
import csv
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))


def read(path):
    p = os.path.join(here, path)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return {r["circuit"]: r for r in csv.DictReader(f)}


slot = read(sys.argv[1] if len(sys.argv) > 1 else "slot.csv")
pauli = read(sys.argv[2] if len(sys.argv) > 2 else "pauli.csv")
pyzx = read("pyzx_baseline.csv")
pub = read("published_best.csv")

order = [
    "tof_3", "tof_4", "tof_5", "tof_10", "barenco_tof_3", "barenco_tof_4", "barenco_tof_5",
    "barenco_tof_10", "mod5_4", "vbe_adder_3", "mod_mult_55", "mod_red_21", "rc_adder_6",
    "csla_mux_3", "csum_mux_9", "qcla_com_7", "qcla_adder_10", "qcla_mod_7", "adder_8",
    "qft_4", "grover_5", "hwb6", "fprenorm", "ham15-low", "ham15-med", "ham15-high",
    "mod_adder_1024", "cycle_17_3", "gf2^4_mult", "gf2^5_mult", "gf2^6_mult", "gf2^7_mult",
    "gf2^8_mult", "gf2^9_mult", "gf2^10_mult", "hwb8",
]
pub_alias = {}

rows = []
for name in order:
    s, p = slot.get(name), pauli.get(name)
    if not s and not p:
        continue
    cand = []
    if s and s["verified"] == "true":
        cand.append((int(s["t_todd"]), "slot", s))
    if p and p["verified"] == "true":
        cand.append((int(p["t_todd"]), "pauli", p))
    best = min(cand, key=lambda c: c[0]) if cand else None
    b = pub.get(name, {})
    z = pyzx.get(name, {})
    any_row = s or p
    rows.append({
        "circuit": name,
        "qubits": any_row["qubits"],
        "t_orig": any_row["t_orig"],
        "repo_phase_fold": any_row["t_repo_phasefold"],
        "pyzx_full_reduce": z.get("pyzx_full", ""),
        "slot_folded": s["t_slotfold"] if s else "",
        "slot_todd": s["t_todd"] if s else "",
        "pauli_merged": p["t_slotfold"] if p else "",
        "pauli_todd": p["t_todd"] if p else "",
        "ours_best": best[0] if best else "",
        "ours_mode": best[1] if best else "",
        "published_ancilla_free": b.get("best_no_gadget", ""),
        "published_ancilla_free_source": b.get("source_no_gadget", ""),
        "published_gadget": b.get("best_gadget", ""),
    })

out = os.path.join(here, "results.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("| circuit | n | T orig | repo phase_fold | PyZX full_reduce | slot+TODD | Pauli+TODD "
      "| **ours (best)** | best published ancilla-free | published with H-gadgets |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    ours = r["ours_best"]
    pubv = r["published_ancilla_free"]
    mark = ""
    if ours != "" and pubv not in ("", None):
        try:
            if int(ours) < int(pubv):
                mark = " (below)"
            elif int(ours) == int(pubv):
                mark = " (=)"
        except ValueError:
            pass
    print(f"| {r['circuit']} | {r['qubits']} | {r['t_orig']} | {r['repo_phase_fold']} | "
          f"{r['pyzx_full_reduce']} | {r['slot_todd']} | {r['pauli_todd']} | "
          f"**{ours}**{mark} | {pubv} | {r['published_gadget']} |")
