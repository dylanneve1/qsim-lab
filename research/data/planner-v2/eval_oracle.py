#!/usr/bin/env python3
"""End-to-end regret against an in-session oracle (research/simulability/planner-v2.md §4):
per instance and request, the planner run (plan + execute, in process) and
the oracle (the engine measured best in the read-out session, forced, no
planning) ran back to back on the two workers, so they share the machine
load. Regret = planner / oracle; epsilon-regret adds 1 ms to both.
    eval_oracle.py OUT.txt e2e2.jsonl [...]
"""
import json, sys
from collections import defaultdict
import numpy as np

EPS, CENS = 1e-3, 20.0
runs = defaultdict(dict)
for p in sys.argv[2:]:
    for l in open(p):
        r = json.loads(l)
        var, rq = r["job"].split(":")
        if var.startswith("force-"):
            var = "oracle"
        t = r["secs"] if r.get("ok") else CENS
        runs[(r["spec"], r["seed"], rq)][var] = (t, r)
out = []
for rq in ["e", "s1k", "s100k", "a1k"]:
    for var in (["v1", "v2nc"] if rq == "e" else ["v2nc"]):
        rows = []
        for (s, seed, q), v in runs.items():
            if q != rq or var not in v or "oracle" not in v:
                continue
            t, r = v[var]
            to = v["oracle"][0]
            rows.append((t / to, (t + EPS) / (to + EPS), to, r.get("plan_secs", 0.0), bool(r.get("aborted")), s))
        x = np.array([r[0] for r in rows]); e = np.array([r[1] for r in rows]); to = np.array([r[2] for r in rows])
        p = np.array([r[3] for r in rows]); ab = sum(r[4] for r in rows)
        w = max(rows, key=lambda r: r[1])
        out.append(f"{rq:6s} {var:5s} n={len(rows)} geo={10 ** np.mean(np.log10(x)):.3f} within2={np.mean(x <= 2):.2f} "
                   f"geo_eps={10 ** np.mean(np.log10(e)):.3f} worst_eps={e.max():.2f} ({w[5]}) aborts={ab} "
                   f"plan med/p90 {np.median(p) * 1e3:.3f}/{np.percentile(p, 90) * 1e3:.2f} ms")
        for lo in (1e-3, 1e-2, 1e-1):
            m = to >= lo
            if m.any():
                out.append(f"          oracle>={lo:g}s n={m.sum()} geo={10 ** np.mean(np.log10(x[m])):.3f} geo_eps={10 ** np.mean(np.log10(e[m])):.3f}")
txt = "\n".join(out)
print(txt)
open(sys.argv[1], "w").write(txt + "\n")
