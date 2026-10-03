#!/usr/bin/env python3
"""Exact circuit-level distance of a detector error model by integer programming (HiGHS via scipy).

  min sum_j x_j   s.t.  sum_{j ∋ i} x_j = 2 y_i  (every detector i)
                        sum_{j ∋ L} x_j = 2 s + 1 (the Z observable)
  x binary, y, s >= 0 integer.
Same formulation as Kishony-Fowler's ilp_circuit_distance.py (OR-Tools/SCIP); solver here is HiGHS.
Input: the DEM text written by `color_search dem` (or any 'p<TAB>obsmask<TAB>dets' file), or a
.stim file (DEM taken from Stim, for cross-checks).
"""
import sys, time, json
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix


def read_dem(path):
    if path.endswith(".stim"):
        import stim
        c = stim.Circuit.from_file(path)
        dem = c.detector_error_model(decompose_errors=False, flatten_loops=True)
        nd = c.num_detectors
        ents = {}
        for inst in dem.flattened():
            if inst.type != "error":
                continue
            ds, ob = [], 0
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    ds.append(t.val)
                elif t.is_logical_observable_id():
                    ob ^= 1 << t.val
            key = (tuple(sorted(ds)), ob)
            ents[key] = inst.args_copy()[0]
        return nd, [(list(k[0]), k[1], p) for k, p in ents.items()]
    nd, ents = None, []
    for line in open(path):
        if line.startswith("#x"):
            continue
        if line.startswith("#"):
            nd = int(line.split()[-1]); continue
        p, ob, ds = line.rstrip("\n").split("\t")
        ents.append(([int(x) for x in ds.split()] if ds else [], int(ob), float(p)))
    return nd, ents


def distance(nd, ents, time_limit=None, obs_bit=0):
    ents = [e for e in ents if e[0] or (e[1] >> obs_bit) & 1]
    m = len(ents)
    rows, cols = [], []
    for j, (ds, ob, _) in enumerate(ents):
        for i in ds:
            rows.append(i); cols.append(j)
        if (ob >> obs_bit) & 1:
            rows.append(nd); cols.append(j)
    nrow = nd + 1
    # variables: x (m), y (nrow)  ; A x - 2 y = b, b = 0 for detectors, 1 for the observable
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nrow, m)).tocsr()
    from scipy.sparse import hstack, identity
    full = hstack([A, -2 * identity(nrow, format="csr")]).tocsr()
    b = np.zeros(nrow); b[nd] = 1
    c = np.concatenate([np.ones(m), np.zeros(nrow)])
    integrality = np.ones(m + nrow)
    ub = np.concatenate([np.ones(m), np.full(nrow, m // 2 + 1)])
    opts = {"disp": False}
    if time_limit:
        opts["time_limit"] = time_limit
    t = time.time()
    r = milp(c, constraints=LinearConstraint(full, b, b), integrality=integrality,
             bounds=Bounds(np.zeros(m + nrow), ub), options=opts)
    dt = time.time() - t
    if r.x is None:
        return dict(status=r.message, distance=None, seconds=dt)
    chosen = [j for j in range(m) if r.x[j] > 0.5]
    lb = getattr(r, "mip_dual_bound", None)
    return dict(status=r.message, optimal=bool(r.status == 0), distance=len(chosen),
                lower_bound=lb, seconds=round(dt, 2), mechanisms=m,
                chosen=[(ents[j][0], ents[j][1]) for j in chosen])


def x_detectors(path):
    for line in open(path):
        if line.startswith("#x"):
            return set(int(t) for t in line.split()[1:])
    return set()


def zsector_distance(path, time_limit=None):
    """Exact distance via the Z-sector relaxation.

    Dropping the X-type detectors gives a LOWER bound (fewer constraints); mechanisms then
    merge by Z-sector signature, so the ILP is much smaller. If every mechanism of the optimal
    relaxed solution has a twin in the full DEM with the same Z-sector signature and observable
    and an EMPTY X-sector signature, those twins form a full-DEM logical of the same weight, so
    the bound is attained and the result is exact (certified=True)."""
    nd, ents = read_dem(path)
    xs = x_detectors(path)
    zmap = {i: k for k, i in enumerate(i for i in range(nd) if i not in xs)}
    red = {}
    pure = set()
    for ds, ob, p in ents:
        zs = tuple(zmap[i] for i in ds if i in zmap)
        key = (zs, ob & 1)
        if not zs and not key[1]:
            continue
        red[key] = p
        if all(i in zmap for i in ds):
            pure.add(key)
    r = distance(len(zmap), [(list(k[0]), k[1], p) for k, p in red.items()], time_limit)
    if r.get("distance") is None:
        return r
    r["certified"] = all((tuple(ds), ob & 1) in pure for ds, ob in r["chosen"])
    r["zsector_mechanisms"] = len(red)
    return r


if __name__ == "__main__":
    if "--zsector" in sys.argv:
        tl = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
        r = zsector_distance(sys.argv[1], tl)
        if "--quiet" in sys.argv:
            r.pop("chosen", None)
        print(json.dumps(r))
        sys.exit(0)
    nd, ents = read_dem(sys.argv[1])
    tl = float(sys.argv[2]) if len(sys.argv) > 2 else None
    r = distance(nd, ents, tl)
    if "--quiet" in sys.argv:
        r.pop("chosen", None)
    print(json.dumps(r))
