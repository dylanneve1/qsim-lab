#!/usr/bin/env python3
"""Exact Z-memory circuit distance / min-weight count of a schedule given as orders (ox, mz) or a
schedule file, via the symbolic DEM (cg_model.py) and dem_distance.

usage: cg_eval.py <d> <R> <found.json|schedule file|kf> [maxw]
"""
import json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cg_model import Code, mechanisms, to_problem, orders_from_schedule
from cg_sat import DistServer


def load(code, spec):
    if spec == "kf":
        return orders_from_schedule(code, code.kf()) + ([], {})
    txt = open(spec).read().strip()
    if txt.startswith("{"):
        j = json.loads(txt.splitlines()[-1]) if not txt.startswith('{"ox"') else json.loads(txt)
        return j["ox"], j["mz"], j.get("hook_free", []), {int(k): [set(x) for x in v] for k, v in (j.get("split") or {}).items()}
    s = [[int(t) for t in l.split()] for l in txt.splitlines() if l.strip()]
    return orders_from_schedule(code, s) + ([], {})


def evaluate(code, ox, mz, R, maxw=64, srv=None, hook_free=(), split=None):
    srv = srv or DistServer()
    M = mechanisms(code, ox, mz, R, hook_free=hook_free, extra_hooks=split)
    keys, rows, nd = to_problem(code, M, R)
    t = time.time()
    w, cnt, nodes, logs = srv.query(nd, rows, maxw)
    return dict(distance=w, count=cnt, nodes=nodes, mechanisms=len(rows), seconds=time.time() - t,
                example=[keys[j] for j in logs[0]] if logs else None)


if __name__ == "__main__":
    d, R = int(sys.argv[1]), int(sys.argv[2])
    code = Code(d)
    ox, mz, hf, spl = load(code, sys.argv[3])
    maxw = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    r = evaluate(code, ox, mz, R, maxw, hook_free=hf, split=spl)
    print(json.dumps({k: v for k, v in r.items() if k != "example"}))
