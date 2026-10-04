#!/usr/bin/env python3
"""Corner lemma: d_circ <= d-1 for EVERY single-auxiliary schedule (any order, any depth).
A corner plaquette (weight 4) measured by one auxiliary with sequential CNOTs o1..o4 has the hook
X_{o3 o4} (X on the auxiliary after its 2nd CNOT), which equals X_{o1 o2} times the stabilizer.
Its three possible pairings {o1o2 | o3o4} are checked separately: with only single-qubit data
errors plus that one hook, the minimum logical weight is d-1. So whatever the order, one fault +
(d-2) single faults make a logical; all of these faults are X-half faults of one round (clean data
errors), so the bound holds for the circuit distance of any number of rounds.
usage: corner_bound.py d1,d2,..."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cg_model import Code
from cg_sat import DistServer

srv = DistServer()
for d in map(int, sys.argv[1].split(",")):
    code = Code(d)
    corners = [p for p in code.P if any(len(code.qplaq[q]) == 1 for q in code.pq[p["i"]])]
    assert len(corners) == 3 and all(p["w"] == 4 for p in corners)
    for p in corners:
        qs = code.pq[p["i"]]
        res = []
        for a in qs[1:]:
            pair = (qs[0], a)
            rows = [(q in code.row0, list(code.syn([q]))) for q in range(code.nq)]
            rows.append((code.obs(pair), list(code.syn(pair))))
            w, cnt, nodes, logs = srv.query(code.np, rows, d)
            used = any(len(l) and code.nq in l for l in logs)
            res.append((pair, w, used))
        ok = all(w == d - 1 for _, w, _ in res)
        print(f"d={d} corner plaquette {p['i']} (qubits {qs}): min weight per pairing "
              f"{[(pr, w) for pr, w, _ in res]} -> {'d-1 for every order' if ok else 'NOT ALL d-1'}", flush=True)
