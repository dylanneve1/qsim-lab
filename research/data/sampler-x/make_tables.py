#!/usr/bin/env python3
"""Markdown tables for research/qec/sampler-x.md from the JSONL files in this directory.
usage: make_tables.py [e2e.jsonl] [throughput.jsonl] [threads.jsonl] [wsweep.jsonl] [old_repro.jsonl]"""
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))


def rows(name):
    path = os.path.join(here, name)
    if not os.path.exists(path):
        return []
    out = []
    for l in open(path):
        l = l.strip()
        if l.startswith("{"):
            r = json.loads(l)
            if not r.get("note"):
                out.append(r)
    return out


def fmt_t(s):
    return f"{s * 1e3:.2f} ms" if s < 1 else f"{s:.2f} s"


def e2e(name="e2e.jsonl"):
    rs = rows(name)
    if not rs:
        return
    print("\n### whole process (min of reps): best native Stim (ptb64 or b8) / ours, (our time vs Stim's)\n")
    shots = sorted({r["shots"] for r in rs})
    for p in sorted({r["p"] for r in rs}):
        print(f"\np = {p}\n")
        print("| d | " + " | ".join(f"{s:.0e}" for s in shots) + " | load |")
        print("|---|" + "---|" * (len(shots) + 1))
        for d in sorted({r["d"] for r in rs}):
            cells, loads = [], []
            for s in shots:
                m = [r for r in rs if r["d"] == d and r["p"] == p and r["shots"] == s]
                if not m:
                    cells.append("–")
                    continue
                r = m[-1]
                loads += [r["load1_before"], r["load1_after"]]
                best = r.get("best_stim_s", r["stim_s"])
                cells.append(f"**{best / r['x_s']:.2f}×** ({fmt_t(r['x_s'])} vs {fmt_t(best)})")
            print(f"| {d} | " + " | ".join(cells) + f" | {min(loads):.0f}–{max(loads):.0f} |")


def throughput(name="throughput.jsonl"):
    rs = rows(name)
    if not rs:
        return
    print("\n### sampling only, Mshot/s (single thread)\n")
    print("| d | p | shots | Stim pip | Stim native ptb64 / b8 (net) | ours old (net) | ours new (net) | ours new (internal) | new / best Stim | new / old | load |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rs, key=lambda r: (r["p"], r["d"])):
        best = max(r["mshots_stim_pip"], r["mshots_stim_net"], r.get("mshots_stim_b8_net", 0))
        print(f"| {r['d']} | {r['p']} | {r['shots']:,} | {r['mshots_stim_pip']:.2f} | "
              f"{r['mshots_stim_net']:.2f} / {r.get('mshots_stim_b8_net', float('nan')):.2f} | "
              f"{r['mshots_fast_net']:.2f} | {r['mshots_x_net']:.2f} | {r['mshots_x_int']:.2f} | "
              f"**{r['mshots_x_int'] / best:.1f}×** | {r['mshots_x_net'] / r['mshots_fast_net']:.2f}× | "
              f"{r['load1_before']:.0f}–{r['load1_after']:.0f} |")


def threads(name="threads.jsonl"):
    rs = rows(name)
    if not rs:
        return
    print("\n### threads\n")
    print("| d | p | threads | CPUs | shots | Gshot/s | detector-shots/s | speed-up | load |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rs, key=lambda r: (r["d"], r["p"], r["threads"])):
        one = [x for x in rs if x["d"] == r["d"] and x["p"] == r["p"] and x["threads"] == 1]
        sp = (r["shots_per_s"] / one[0]["shots_per_s"]) if one else float("nan")
        print(f"| {r['d']} | {r['p']} | {r['threads']} | {r['cpus']} | {r['shots']:.0e} | {r['shots_per_s'] / 1e9:.3f} | "
              f"{r['detector_shots_per_s']:.2e} | {sp:.1f}× | {r['load1_before']:.0f}–{r['load1_after']:.0f} |")


def wsweep(name="wsweep.jsonl"):
    rs = rows(name)
    if not rs:
        return
    print("\n### batch width (Mshot/s)\n")
    print("| d | p | W (shots/batch) | tables | no tables | AVX-512 kernel | load |")
    print("|---|---|---|---|---|---|---|")
    for r in sorted(rs, key=lambda r: (r["d"], r["p"], r["words"])):
        simd = r["mshots_sample_simd"] if r["sample_simd_min_s"] != float("inf") and r["mshots_sample_simd"] > 0 else float("nan")
        print(f"| {r['d']} | {r['p']} | {r['words']} ({64 * r['words']}) | {r['mshots_sample_tables']:.2f} | "
              f"{r['mshots_sample_notables']:.2f} | {simd:.2f} | {r['load1_before']:.0f}–{r['load1_after']:.0f} |")


def old_repro(name="old_repro.jsonl"):
    rs = rows(name)
    if not rs:
        return
    print("\n### fast-sampler.md §3.1 reproduced (old FastSampler, main 6b21728), Mshot/s\n")
    print("| d | p | shots | Stim pip | Stim native (net) | old FastSampler | old / native | end-to-end old / Stim | old compile | load |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rs, key=lambda r: (r["p"], r["d"])):
        print(f"| {r['d']} | {r['p']} | {r['shots']:,} | {r['mshots_stim_write']:.2f} | {r['mshots_stim_cli_net']:.2f} | "
              f"{r['mshots_ours']:.1f} | **{r['ratio_vs_stim_cli_net']:.1f}×** | {r['ratio_e2e_vs_stim_cli']:.2f}× | "
              f"{r['ours_compile_s'] * 1e3:.1f} ms | {r['load1']:.0f} |")


def equivalence(name="equivalence_x.jsonl"):
    rs = rows(name)
    if not rs:
        return
    print("\n### 10^6-shot equivalence with Stim (T0-T3, 1% family-wise per cell)\n")
    print("| direction | d | p | threads | mode | detectors | DEM support equal | max abs z marg | max abs z pairs (#) | z events mean, var | obs rate ours / Stim | z* | rejections | verdict |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    tot_t = tot_r = 0
    for r in rs:
        tot_t += r["n_tests"]
        tot_r += r["n_reject"]
        print(f"| {r['direction']} | {r['d']} | {r['p']} | {r['threads']} | {r['tables']} | {r['detectors']} | "
              f"{'yes' if r['support_equal'] else 'NO'} ({r['support_size']}) | {r['max_abs_z_marg']:.2f} | "
              f"{r['max_abs_z_pair']:.2f} ({r['pairs']}) | {r['z_mean_events']:+.2f}, {r['z_var_events']:+.2f} | "
              f"{r['obs_rate_ours']:.4f} / {r['obs_rate_stim']:.4f} | {r['zcrit']:.2f} | {r['n_reject']}/{r['n_tests']} | {r['verdict']} |")
    print(f"\n{tot_r} rejections in {tot_t} tests over {len(rs)} cells.")


def calibrate(name="calibrate.jsonl"):
    rs = rows(name)
    if not rs:
        return
    print("\n### auto-policy calibration: whole process, ms (min of reps)\n")
    print("| d | shots | frames | compiled, no tables | compiled, tables | fastest | load |")
    print("|---|---|---|---|---|---|---|")
    for r in sorted(rs, key=lambda r: (r["d"], r["shots"])):
        t = {k: r[f"{k}_s"] for k in ("x_frames", "x_notables", "x_tables") if f"{k}_s" in r}
        best = min(t, key=t.get)
        print(f"| {r['d']} | {r['shots']:,} | {t['x_frames'] * 1e3:.2f} | {t['x_notables'] * 1e3:.2f} | "
              f"{t['x_tables'] * 1e3:.2f} | {best[2:]} | {r['load1_before']:.0f}–{r['load1_after']:.0f} |")


if __name__ == "__main__":
    calibrate()
    equivalence()
    old_repro()
    e2e()
    throughput()
    wsweep()
    threads()
