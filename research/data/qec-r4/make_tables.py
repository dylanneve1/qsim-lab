#!/usr/bin/env python3
"""Markdown tables for research/qec/qec-r4.md from the raw JSON in this directory."""
import json, sys, os
H = os.path.dirname(os.path.abspath(__file__))


def equiv():
    rows = json.load(open(f"{H}/stim_equivalence.json"))
    print("| direction | d | detectors | DEM support equal | max abs z (marginals) | max abs z (pairs, #pairs) | z (events/shot mean, var) | observable rate ours / Stim | Bonferroni z* | rejections | verdict |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        dr = "A: ours→Stim" if r["direction"] == "A" else "B: Stim→ours"
        print(f"| {dr} | {r['d']} | {r['detectors']} | {'yes' if r['support_equal'] else 'NO'} ({r['support_size']}) | {r['max_abs_z_marg']:.2f} | {r['max_abs_z_pair']:.2f} ({r['pairs']}) | {r['z_mean_events']:+.2f}, {r['z_var_events']:+.2f} | {r['obs_rate_ours']:.4f} / {r['obs_rate_stim']:.4f} | {r['zcrit']:.2f} | {r['n_reject']}/{r['n_tests']} | {r['verdict']} |")


def timing(path, label):
    rows = [json.loads(l) for l in open(path) if l.startswith("{") and "circuit" in l]
    cli = any("stim_cli_native_min_s" in r for r in rows)
    sparse = any("ours_sparse_min_s" in r for r in rows)
    hdr = "| circuit | d | shots | Stim sample_write ptb64 (Mshot/s) | Stim sample() numpy (Mshot/s) |"
    sep = "|---|---|---|---|---|"
    if cli:
        hdr += " Stim native CLI ptb64 (Mshot/s) |"; sep += "---|"
    hdr += " ours dense (Mshot/s) |"; sep += "---|"
    if sparse:
        hdr += " ours sparse+SmallRng (Mshot/s) |"; sep += "---|"
    hdr += " ours/Stim (write) | ours/Stim (numpy) |"; sep += "---|---|"
    if cli:
        hdr += " ours/Stim native |"; sep += "---|"
    hdr += " load |"; sep += "---|"
    print(f"**{label}**\n")
    print(hdr); print(sep)
    for r in rows:
        n = r["shots"]
        best = r.get("ours_best_min_s", r["ours_min_s"])
        line = f"| {'A (ours)' if r['circuit'].startswith('A') else 'B (Stim gen.)'} | {r['d']} | {n} | {n/r['stim_write_min_s']/1e6:.2f} | {n/r['stim_mem_min_s']/1e6:.2f} |"
        if cli:
            line += f" {n/r['stim_cli_native_min_s']/1e6:.2f} |"
        line += f" {n/r['ours_min_s']/1e6:.2f} |"
        if sparse:
            line += f" {n/r['ours_sparse_smallrng_min_s']/1e6:.2f} |"
        line += f" {r['stim_write_min_s']/best:.2f}× | {r['stim_mem_min_s']/best:.2f}× |"
        if cli:
            line += f" {r['stim_cli_native_min_s']/best:.2f}× |"
        line += f" {r.get('load1', float('nan')):.1f} |"
        print(line)
    print()


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "equiv":
        equiv()
    else:
        timing(sys.argv[2], sys.argv[3])
