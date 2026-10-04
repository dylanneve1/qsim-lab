#!/usr/bin/env python3
"""Exact circuit distance by MaxSAT (pysat RC2), on the Z-sector of our circuit-derived DEM
(with the same twin-certification as circuit_distance.zsector_distance), or on the full DEM.

Encoding: x_j = mechanism j used; each detector: XOR of its x_j = 0 (Tseitin chain), the
observable: XOR = 1; soft clauses (not x_j) weight 1. Optimum = min #mechanisms = distance.
usage: circuit_distance_sat.py file.dem [--full]
"""
import sys, json, time
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
sys.path.insert(0, __import__("os").path.dirname(__file__))
from circuit_distance import read_dem, x_detectors


def sat_distance(nd, ents):
    ents = [e for e in ents if e[0] or e[1] & 1]
    m = len(ents)
    w = WCNF()
    nv = m
    def new():
        nonlocal nv
        nv += 1
        return nv
    rows = [[] for _ in range(nd + 1)]
    for j, (ds, ob, _) in enumerate(ents):
        for i in ds:
            rows[i].append(j + 1)
        if ob & 1:
            rows[nd].append(j + 1)
    for i, r in enumerate(rows):
        target = (i == nd)
        if not r:
            if target:
                w.append([])  # infeasible
            continue
        acc = r[0]
        for v in r[1:]:
            t = new()
            # t = acc xor v
            w.append([-t, acc, v]); w.append([-t, -acc, -v])
            w.append([t, -acc, v]); w.append([t, acc, -v])
            acc = t
        w.append([acc] if target else [-acc])
    for j in range(m):
        w.append([-(j + 1)], weight=1)
    t = time.time()
    with RC2(w) as rc2:
        model = rc2.compute()
        cost = rc2.cost
    chosen = [j for j in range(m) if model[j] > 0]
    return dict(distance=cost, seconds=round(time.time() - t, 2), mechanisms=m,
                chosen=[(ents[j][0], ents[j][1]) for j in chosen])


def zsector(path):
    nd, ents = read_dem(path)
    xs = x_detectors(path)
    zmap = {i: k for k, i in enumerate(i for i in range(nd) if i not in xs)}
    red, pure = {}, set()
    for ds, ob, p in ents:
        zs = tuple(zmap[i] for i in ds if i in zmap)
        key = (zs, ob & 1)
        if not zs and not key[1]:
            continue
        red[key] = p
        if all(i in zmap for i in ds):
            pure.add(key)
    r = sat_distance(len(zmap), [(list(k[0]), k[1], p) for k, p in red.items()])
    r["certified"] = all((tuple(ds), ob & 1) in pure for ds, ob in r["chosen"])
    return r


if __name__ == "__main__":
    if "--full" in sys.argv:
        nd, ents = read_dem(sys.argv[1])
        r = sat_distance(nd, ents)
    else:
        r = zsector(sys.argv[1])
    if "--chosen" not in sys.argv:
        r.pop("chosen")
    print(json.dumps(r))
