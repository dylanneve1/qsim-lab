#!/usr/bin/env python3
"""Jobs for the end-to-end session with an in-session oracle (research/
planner-v2.md §4): per instance and request, Planner v2 (cache off), the
engine measured best in the read-out session forced (no planning), and for
expectations Planner v1. Instances in random order, an instance's jobs
back to back (the two workers run them side by side, under the same load).
    gen_e2e_jobs.py feat.jsonl req.jsonl hsfamp.jsonl > jobs.txt
"""
import random, sys
import fit_v2 as fv

feat, runs = fv.load(sys.argv[2:], sys.argv[1])
keys = sorted(k for k in feat if k in runs)
random.Random(11).shuffle(keys)
for k in keys:
    for rq in ["e", "s1k", "s100k", "a1k"]:
        tr = fv.truth(feat, runs, k, rq)
        if not tr:
            continue
        be, b = min(tr.items(), key=lambda x: x[1])
        if b >= fv.CENS:
            continue
        print(f"{k[0]} {k[1]} v2nc:{rq}")
        print(f"{k[0]} {k[1]} force-{be}:{rq}")
        if rq == "e":
            print(f"{k[0]} {k[1]} v1:{rq}")
