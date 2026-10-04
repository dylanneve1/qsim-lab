#!/usr/bin/env python3
"""Soundness + completeness of the SAT encoding's presence conditions against the CIRCUIT DEM.
For random collision-free schedules (and K-F), for every potential signature of the universe:
    (some presence condition holds under the schedule)  <=>  (signature is in the Rust circuit DEM)
and the circuit DEM has no Z-sector signature outside the universe.
usage: test_presence.py [n_random]"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cg_model import Code, rust_zsector, random_schedule, orders_from_schedule
from cg_sat import universe


def holds(cond, ox, mz):
    pos = lambda lst, a: lst.index(a)
    for kind, owner, a, b in cond:
        lst = ox[owner] if kind == "bx" else mz[owner]
        if not pos(lst, a) < pos(lst, b):
            return False
    return True


n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
bad = 0
for d, R in [(3, 2), (5, 1), (5, 2), (7, 1), (7, 2), (9, 1)]:
    code = Code(d)
    U = universe(code, R, False)
    rng = random.Random(d * 10 + R)
    for name, s in [("kf", code.kf())] + [(f"rand{i}", random_schedule(code, rng)) for i in range(n)]:
        ox, mz = orders_from_schedule(code, s)
        circ = rust_zsector(code, s, R)
        pred = {sig for sig, conds in U.items() if any(holds(c, ox, mz) for c in conds)}
        ok = pred == circ
        bad += not ok
        print(f"d={d} R={R} {name}: universe {len(U)}, predicted present {len(pred)}, circuit {len(circ)}, "
              f"outside universe {len(circ - set(U))} -> {'OK' if ok else 'MISMATCH'}", flush=True)
print("ALL OK" if bad == 0 else f"{bad} MISMATCHES")
sys.exit(1 if bad else 0)
