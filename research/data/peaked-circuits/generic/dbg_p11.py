import sys, numpy as np
sys.path.insert(0,'/tmp/peaked-generic')
import solve_peaked_v2 as v2, solve_anchor as SA
f='/tmp/pk/research/data/peaked-circuits/peaked_circuit_P11_Hqap_98x1999.qasm'
old = v2.Core(f); new = SA.Core(f)
print("L equal:", old.L == new.L)
print("old maps sections:", [s for s,_ in old.maps], "new block centres", [m['centre'] for m in new.maps])
for i,(s,m) in enumerate(old.maps): print(" map", i, "equal:", [m[q] for q in range(98)] == [new.maps[i]['f'][q] for q in range(98)])
print("R0 equal:", set(old.r0) == new.r0, "P0 equal:", set(old.p0) == new.p0)
