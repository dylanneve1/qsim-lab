#!/usr/bin/env python3
"""Offline estimate of end-to-end regret vs the tiering parameter `voi`
(research/simulability/planner-v2.md §3): the tiered decision for each voi (deterministic,
`planner_v2 feat`), its planning cost from the Mac per-tier timings
(feat_mac.jsonl: support, certificate, frame, MPS replay, HSF partition, plus
the O(G) pass), and the chosen engine's measured time (read-out session).
No speculation, no load confound.
    tune_voi.py feat_voi.jsonl feat_mac.jsonl req.jsonl hsfamp.jsonl
"""
import json, sys
import numpy as np
import fit_v2 as fv

VOIS = [0.5, 1, 2, 4, 8, 16, 32, 64, 1e9]
dec = {(r["spec"], str(r["seed"])): r for r in map(json.loads, open(sys.argv[1]))}
mac = {(r["spec"], str(r["seed"])): r for r in map(json.loads, open(sys.argv[2]))}
feat, runs = fv.load(sys.argv[3:], sys.argv[2])
out = []
for rq in ["e", "s1", "s1k", "s100k", "a1", "a1k"]:
    rows = {v: [] for v in VOIS}
    for k, d in dec.items():
        if k not in runs:
            continue
        tr = fv.truth(feat, runs, k, rq)
        if not tr:
            continue
        be, b = min(tr.items(), key=lambda x: x[1])
        if b >= fv.CENS:
            continue
        m = mac[k]
        # the line-split HSF pricing is O(gates) like the support bound
        costs = [m["t_sup"], m["t_cert"], m["t_tier1"] - m["t_sup"], m["t_mps"], m["t_sup"], m["t_hsf"]]
        for v, ch in zip(VOIS, d["choices"][rq]["voi"]):
            if ch is None:
                continue
            e, flags = ch[0], ch[1:]
            plan = m["t_quick"] + sum(c for c, f in zip(costs, flags) if f)
            t = tr.get(e, fv.CENS)
            rows[v].append(((plan + t + fv.EPS) / (b + fv.EPS), (plan + t) / b, (t + fv.EPS) / (b + fv.EPS), plan))
    line = [f"{rq:6s}"]
    for v in VOIS:
        x = np.array(rows[v])
        line.append(f"voi={v:g}: eps {10 ** np.mean(np.log10(x[:, 0])):.3f} (choice {10 ** np.mean(np.log10(x[:, 2])):.3f}, "
                    f"plan med {np.median(x[:, 3]) * 1e3:.3f} ms)")
    out.append("\n   ".join(line))
txt = "\n".join(out)
print(txt)
open("tune_voi.txt", "w").write(txt + "\n")
