#!/usr/bin/env python3
"""Markdown summary of ab.py CSVs: per cell (workload, n, prec, threads) the minimum seconds of
each configuration, the ratio to the first-listed (or --base) configuration, the number of runs
and the 1-min load range during the cell.

usage: ab_summary.py <csv> [<csv> ...] [--base main] [--cfgs a,b,c]
"""
import argparse
import collections
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--base")
    ap.add_argument("--cfgs")
    a = ap.parse_args()
    t = collections.defaultdict(list)
    loads = collections.defaultdict(list)
    order = []
    for f in a.csv:
        for r in csv.DictReader(open(f)):
            cell = (r["workload"], int(r["n"]), r["prec"], int(r["threads"]))
            t[cell + (r["cfg"],)].append(float(r["seconds"]))
            loads[cell] += [float(r["load1_before"]), float(r["load1_after"])]
            if r["cfg"] not in order:
                order.append(r["cfg"])
    cfgs = a.cfgs.split(",") if a.cfgs else order
    base = a.base or cfgs[0]
    print("| workload | n | prec | threads | " + " | ".join(f"{c} s" for c in cfgs)
          + " | " + " | ".join(f"{base}/{c}" for c in cfgs if c != base) + " | runs | load1 |")
    print("|" + "---|" * (5 + 2 * len(cfgs)))
    for cell in sorted(loads):
        mins = {c: min(t[cell + (c,)]) for c in cfgs if t.get(cell + (c,))}
        runs = min((len(t[cell + (c,)]) for c in mins), default=0)
        cols = [f"{mins[c]:.4f}" if c in mins else "" for c in cfgs]
        rat = [f"{mins[base] / mins[c]:.2f}x" if c in mins and base in mins else ""
               for c in cfgs if c != base]
        lo, hi = min(loads[cell]), max(loads[cell])
        print(f"| {cell[0]} | {cell[1]} | {cell[2]} | {cell[3]} | " + " | ".join(cols) + " | "
              + " | ".join(rat) + f" | {runs} | {lo:.1f}-{hi:.1f} |")


if __name__ == "__main__":
    main()
