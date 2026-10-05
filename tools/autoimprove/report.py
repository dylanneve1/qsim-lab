#!/usr/bin/env python3
"""Markdown tables from an autoimprove ledger (JSONL).

usage: report.py LEDGER [--cases SLUG[@BASE7] ...]

Prints: the candidate table (verdict, gate, screen/full geo-means), A/A
calibrations, knob sweeps, and per-case tables for the requested slugs.
"""
import json
import sys


def load(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def g(s):
    if not s or "geo" not in s:
        return ""
    return f"{s['geo']:.3f} / {s['geo_paired_median']:.3f} / {s.get('worst_case', s['min']):.3f}"


def candidates(rows):
    print("| base | candidate | family | verdict | gate cases (worst f64 / f32 err) | screen geo / paired / worst | full geo / paired / worst |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("kind") not in ("patch", None) or r.get("verdict") in (None, "calibration"):
            continue
        if r.get("verdict") == "error":
            continue
        gate = (r.get("gate") or {}).get("ai_gate") or {}
        gs = ""
        if gate.get("cases"):
            gs = f"{gate['cases']} ({gate['worst_f64']:.1e} / {gate['worst_f32']:.1e})"
        elif gate:
            gs = "FAIL"
        print(f"| {r.get('base', '')[:7]} | {r.get('slug')} | {r.get('family', '')} | {r['verdict']} | {gs} | "
              f"{g(r.get('screen'))} | {g(r.get('full'))} |")


def aa(rows):
    print("| run | suite | metric, threads | geo (ratio of mins) | geo (paired median) | min case | max case |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("kind") != "aa":
            continue
        s = r["result"]
        print(f"| {r['id']} | {r.get('suite')} | {s.get('metric', 'wall')}, {s.get('threads')} | {s['geo']:.3f} | "
              f"{s.get('geo_paired_median', float('nan')):.3f} | {s['min']:.3f} | {s['max']:.3f} |")


def knobs(rows):
    print("| base | binary | knobs | geo (ratio of mins) | geo (paired median) | min case | max case |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("kind") != "knobs":
            continue
        s = r["result"]
        print(f"| {r.get('base', '')[:7]} | {r.get('binary', 'base')} | {r['slug'][6:]} | {s['geo']:.3f} | "
              f"{s['geo_paired_median']:.3f} | {s['min']:.3f} | {s['max']:.3f} |")


def confirms(rows):
    print("| base | candidate | threads | geo (ratio of mins) | geo (paired median) | min case | max case |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("kind") != "confirm" or "geo" not in r.get("result", {}):
            continue
        s = r["result"]
        print(f"| {r.get('base', '')[:7]} | {r['slug']} | {s['threads']} (wall) | {s['geo']:.3f} | "
              f"{s['geo_paired_median']:.3f} | {s['min']:.3f} | {s['max']:.3f} |")


def cases(rows, key, part="full"):
    slug, _, base = key.partition("@")
    for r in rows:
        if r.get("slug") != slug or not r.get(part) or (base and not r.get("base", "").startswith(base)):
            continue
        s = r[part]
        unit = "CPU s" if s.get("metric") == "cpu" else "s"
        print(f"\n`{slug}` vs base {r['base'][:7]} ({part}, {s.get('metric', 'wall')}, {s['threads']} thread(s), "
              f"min of {s['reps']}+ per side, loads {[l[0] for l in s['load']]}):\n")
        print(f"| workload | n | prec | base {unit} | cand {unit} | speedup (mins) | paired median | stages base→cand | passes base→cand |")
        print("|---|---|---|---|---|---|---|---|---|")
        for c in s["cases"]:
            ap, bp = c.get("a_plan") or [None, None], c.get("b_plan") or [None, None]
            print(f"| {c['wl']} | {c['n']} | {c['prec']} | {c['a_min']:.4f} | {c['b_min']:.4f} | "
                  f"{c['speedup']:.3f} | {c['paired_median']:.3f} | {ap[0]}→{bp[0]} | {ap[1]}→{bp[1]} |")


def main():
    rows = load(sys.argv[1])
    print("### Candidates\n")
    candidates(rows)
    print("\n### A/A calibration\n")
    aa(rows)
    print("\n### Knob sweeps\n")
    knobs(rows)
    print("\n### Multi-thread confirmation\n")
    confirms(rows)
    if "--cases" in sys.argv:
        for key in sys.argv[sys.argv.index("--cases") + 1:]:
            cases(rows, key)


if __name__ == "__main__":
    main()
