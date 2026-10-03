#!/usr/bin/env python3
"""Logical error rate, schedule A vs schedule B, same decoder, independent samples.

usage: ler_compare.py <color_ler bin> <d> <rounds> <noise> <p1,p2,..> <shots> <threads> <osd_order> <out.jsonl> <name=schedule> ...
Writes one JSON line per (schedule, p) and prints per-p ratios with 95% CIs (log-ratio normal approx).
"""
import sys, json, subprocess, math
L, d, rounds, noise, ps, shots, threads, order, outp = sys.argv[1:10]
scheds = [a.split("=", 1) for a in sys.argv[10:]]
out = open(outp, "a")
res = {}
for p in ps.split(","):
    for k, (name, sch) in enumerate(scheds):
        seed = 1000 * (k + 1) + int(float(p) * 1e5)
        r = json.loads(subprocess.run([L, d, rounds, noise, p, sch, shots, str(seed), threads, order],
                                      capture_output=True, text=True, check=True).stdout)
        r["name"] = name
        out.write(json.dumps(r) + "\n"); out.flush()
        res[(p, name)] = r
    a, b = res[(p, scheds[0][0])], res[(p, scheds[1][0])]
    if a["fails"] and b["fails"]:
        lr = math.log(b["p_L"] / a["p_L"])
        se = math.sqrt(1 / a["fails"] + 1 / b["fails"])
        print(f"p={p}: {scheds[0][0]} p_L/round={a['p_L_round']:.3e} ({a['fails']} fails)  {scheds[1][0]} {b['p_L_round']:.3e} ({b['fails']})  "
              f"ratio {scheds[1][0]}/{scheds[0][0]} = {math.exp(lr):.3f} [{math.exp(lr-1.96*se):.3f}, {math.exp(lr+1.96*se):.3f}]", flush=True)
