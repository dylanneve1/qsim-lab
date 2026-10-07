#!/usr/bin/env python3
"""T gates in the left (q<=e) / right (q>e) part of the circuit truncated to CZ-depth D (per-qubit CZ count < D)."""
import re, sys
def tcount(qasm, D, e):
    czc = [0] * 70; L = R = 0
    for l in open(qasm):
        m = re.match(r'cz q\[(\d+)\],q\[(\d+)\];', l)
        if m: czc[int(m.group(1))] += 1; czc[int(m.group(2))] += 1; continue
        m = re.match(r'rz\(pi/4\) q\[(\d+)\];', l)
        if m:
            q = int(m.group(1))
            if czc[q] < D:
                if q <= e: L += 1
                else: R += 1
    return L, R
if __name__ == '__main__':
    q = sys.argv[1]
    for D in [24, 32, 40, 48, 56, 70]:
        print(D, [(e,) + tcount(q, D, e) for e in [5, 10, 15, 20, 25, 30, 34, 40, 45, 50, 55, 60, 64]])
