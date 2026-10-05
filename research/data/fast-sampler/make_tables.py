#!/usr/bin/env python3
"""Markdown tables for research/qec/fast-sampler.md from the timing / equivalence jsonl files.
usage: make_tables.py timing <file.jsonl> | equiv <file.jsonl>..."""
import sys, json


def rows(path):
    for l in open(path):
        l = l.strip()
        if l.startswith("{") and '"note"' not in l:
            yield json.loads(l)


def ms(r, k):
    v = r.get(k + "_min_s")
    return r["shots"] / v / 1e6 if v else float("nan")


def timing(path):
    rs = list(rows(path))
    native = any("stim_cli_net_min_s" in r for r in rs)
    print("| d | p | Stim pip write | " + ("Stim native (AVX2) | Stim DEM sampler pip / native | " if native else "Stim DEM sampler pip | ")
          + "qsim-lab FastSampler | ours / " + ("native Stim" if native else "pip Stim") + " | ours / best Stim | end-to-end ratio | load |")
    print("|---|---|---|" + ("---|---|" if native else "---|") + "---|---|---|---|---|")
    for r in rs:
        ours = ms(r, "ours_blocked_wyrand")
        stim = [ms(r, "stim_write")] + ([ms(r, "stim_cli_net")] if native else [])
        dem = [ms(r, "stim_dem_write")] + ([ms(r, "stim_dem_cli_net")] if native else [])
        best = max(stim + dem)
        ref = stim[-1]
        e2e = r.get("ratio_e2e_vs_stim_cli", float("nan"))
        cells = [f"{r['d']}", f"{r['p']*100:g}%", f"{stim[0]:.2f}"]
        if native:
            cells += [f"{stim[1]:.2f}", f"{dem[0]:.2f} / {dem[1]:.2f}"]
        else:
            cells += [f"{dem[0]:.2f}"]
        cells += [f"{ours:.2f}", f"**{ours/ref:.1f}×**", f"{ours/best:.1f}×", f"{e2e:.1f}×" if native else "–", f"{r['load1']:.1f}"]
        print("| " + " | ".join(cells) + " |")
    print()
    print("Ablation (Mshot/s, same runs; each row adds one change):")
    print()
    cols = [("old_dense_stdrng", "old dense, StdRng"), ("old_sparse_stdrng", "old sparse, StdRng"),
            ("old_sparse_smallrng", "old sparse, Xoshiro"), ("ours_column_xoshiro", "hits, columns, Xoshiro"),
            ("ours_unblocked_xoshiro", "+ hit table"), ("ours_blocked32_xoshiro", "+ blocked"),
            ("ours_blocked_xoshiro", "+ u16 table"), ("ours_blocked_wyrand", "+ wyrand")]
    print("| d | p | " + " | ".join(c[1] for c in cols) + " | total |")
    print("|---|---|" + "---|" * (len(cols) + 1))
    for r in rs:
        v = [ms(r, c[0]) for c in cols]
        print(f"| {r['d']} | {r['p']*100:g}% | " + " | ".join(f"{x:.2f}" for x in v) + f" | {v[-1]/v[0]:.1f}× |")


def equiv(paths):
    print("| direction | d | p | RNG | detectors | DEM support equal | max abs z marg | max abs z pairs (#) | z events mean, var | obs rate ours / Stim | z* | rejections | verdict |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for path in paths:
        for r in rows(path):
            rng = "wyrand" if "wyrand" in r["sampler"] else "Xoshiro256++"
            d = "A: ours→Stim" if r["direction"] == "A" else "B: Stim→ours"
            print(f"| {d} | {r['d']} | {r['p']*100:g}% | {rng} | {r['detectors']} | {'yes' if r['support_equal'] else 'NO'} ({r['support_size']}) | "
                  f"{r['max_abs_z_marg']:.2f} | {r['max_abs_z_pair']:.2f} ({r['pairs']}) | {r['z_mean_events']:+.2f}, {r['z_var_events']:+.2f} | "
                  f"{r['obs_rate_ours']:.4f} / {r['obs_rate_stim']:.4f} | {r['zcrit']:.2f} | {r['n_reject']}/{r['n_tests']} | {r['verdict']} |")


if __name__ == "__main__":
    if sys.argv[1] == "timing":
        timing(sys.argv[2])
    else:
        equiv(sys.argv[2:])
