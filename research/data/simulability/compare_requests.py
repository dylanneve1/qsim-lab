#!/usr/bin/env python3
"""Same instances, two requests (<Z^n> vs a local Z-product): do state-engine
costs, winners and short-circuits change?   compare_requests.py DIR_A SUFFIX_B
(reads DIR_A/{grid}.csv and DIR_A/{grid}{SUFFIX_B}.csv)"""
import csv, glob, json, math, os, sys, collections
import numpy as np

STATE = ["sv", "sparse", "mps", "hsf", "tableau", "cstate"]
OBS = ["frame", "dense", "auto"]


def load(paths):
    d = collections.defaultdict(dict)
    for p in paths:
        for r in csv.DictReader(open(p)):
            d[(r["spec"], r["seed"])][r["engine"]] = r
    return d


def main():
    root, suf = sys.argv[1], sys.argv[2]
    grids = ["ct24", "ct32", "ctnn0", "brick", "arith", "qaoa"]
    A = load([f"{root}/{g}.csv" for g in grids])
    B = load([f"{root}/{g}{suf}.csv" for g in grids if os.path.exists(f"{root}/{g}{suf}.csv")])
    keys = [k for k in B if k in A]
    print(f"instances with both requests: {len(keys)}")
    ratios = collections.defaultdict(list)
    for k in keys:
        for e in STATE:
            a, b = A[k].get(e), B[k].get(e)
            if a and b and a["status"] == b["status"] == "ok" and float(a["secs"]) >= 1e-3:
                ratios[e].append(float(b["secs"]) / float(a["secs"]))
    for e, r in ratios.items():
        r = np.array(r)
        print(f"  {e:7s} t({suf[1:]})/t(all): median {np.median(r):.3f}  p10 {np.percentile(r,10):.3f}"
              f"  p90 {np.percentile(r,90):.3f}  n={len(r)}")

    def win(rows):
        ok = [(float(rows[e]["secs"]), e) for e in STATE if e in rows and rows[e]["status"] == "ok"]
        return min(ok) if ok else None
    agree = tot = 0
    for k in keys:
        wa, wb = win(A[k]), win(B[k])
        if wa and wb and max(wa[0], wb[0]) >= 1e-3:
            tot += 1
            agree += wa[1] == wb[1]
    print(f"state-engine winner identical under both requests: {agree}/{tot} (best >= 1 ms)")
    for name, D in (("all", A), (suf[1:], B)):
        sc = collections.Counter()
        ok = collections.Counter()
        zero = cert = n = 0
        for k in keys:
            rows = D[k]
            for e in OBS:
                r = rows.get(e)
                if r and r["status"] == "ok":
                    ok[e] += 1
                    if "peak_terms=0" in r["note"] and "switched_at=None" in r["note"]:
                        sc[e] += 1
            r = rows.get("cstate") or next(iter(rows.values()))
            f = json.loads(r["features"])
            n += 1
            cert += bool(f.get("obs_zero"))
            zero += r["ref"] != "" and abs(float(r["ref"])) < 1e-12
        print(f"  request {name:5s}: value exactly 0 on {zero}/{n}, certified {cert}/{n}; "
              f"observable-engine short-circuits " + ", ".join(f"{e} {sc[e]}/{ok[e]}" for e in OBS))


if __name__ == "__main__":
    main()
