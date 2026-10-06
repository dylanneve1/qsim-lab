#!/usr/bin/env python3
"""Generic peaked-circuit solver: dispatcher.

  1. Parse any QASM 2/3 (gparse).
  2. If exact inverse-segment anchors exist and form consistent involution blocks -> anchor route
     (solve_anchor.py: blocks by conflict sweep, Otsu sections, R ▷ π ▷ P core, max-peak boundary greedy).
  3. Otherwise -> operator route (solve_generic.py: data-driven mirror centre, middle-out compressed
     operator cancellation, exact contraction of whatever is left).
Never reads a target string.  Usage: solve.py FILE.qasm [--json OUT]
"""
import argparse, json, sys
import gparse as G
import struct_probe as SP
import blocks as BL


def route(path):
    n, units, tail = G.parse(path)
    L = SP.layers(n, units)
    anchors = BL.anchor_list(n, units, L)
    blocks = BL.sweep_blocks(n, anchors) if anchors else []
    # a block is usable when its involution covers every wire (unanchored wires would be guesses)
    usable = [b for b, f in blocks if len(f) == n]
    return ('anchor' if usable else 'operator'), len(anchors), len(blocks), len(usable)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("qasm"); ap.add_argument("--json")
    a = ap.parse_args()
    r, na, nb, nu = route(a.qasm)
    print(f"{a.qasm.split('/')[-1]}: {na} anchors, {nb} blocks ({nu} complete) -> {r} route", flush=True)
    log = lambda s: print(s, flush=True)
    if r == 'anchor':
        import solve_anchor
        res = solve_anchor.solve(a.qasm, log=log)
    else:
        import solve_generic
        res = solve_generic.solve(a.qasm, log=log)
    res['route'] = r
    print(f"  PEAK (qubit 0 leftmost): {res['peak']}\n  core peak probability {res['p']:.4f}; {res['seconds']} s; "
          f"peak RSS {res['rss_mb']} MB", flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
