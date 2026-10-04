#!/usr/bin/env python3
"""Minimal set of cut logicals that already makes a cg_sat.py instance UNSAT (deletion-minimal core),
printed in readable form: which hooks/partials each logical needs.
usage: cg_core.py <cnf> (uses <cnf>.logicals.jsonl; same CLI args as the cg_sat run via env CG_ARGS)"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pysat.formula import CNF
from pysat.solvers import Solver
cnf = sys.argv[1]
nbase = int(sys.argv[2])  # number of cut clauses at the end of the CNF
f = CNF(from_file=cnf)
base, cuts = f.clauses[:-nbase], f.clauses[-nbase:]
logs = [json.loads(l) for l in open(cnf + ".logicals.jsonl")]
logs = [l for l in logs if l]  # same order as cuts (non-empty clauses)
top = f.nv
sel = [top + i + 1 for i in range(len(cuts))]
s = Solver(name="glucose4", bootstrap_with=base)
for c, x in zip(cuts, sel):
    s.add_clause(c + [-x])
assert not s.solve(assumptions=sel)
core = sorted(s.get_core())
# deletion-minimise
i = 0
while i < len(core):
    trial = core[:i] + core[i + 1:]
    if not s.solve(assumptions=trial):
        core = sorted(s.get_core()) if len(s.get_core()) < len(trial) else trial
    else:
        i += 1
idx = [x - top - 1 for x in core]
print(json.dumps(dict(cuts=len(cuts), core=len(idx), core_idx=idx)))
for k in idx:
    print(k, "logical of weight", len(logs[k]), ":", logs[k])
