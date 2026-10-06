import sys, numpy as np
sys.path.insert(0,'/tmp/peaked-generic')
import solve_peaked as v1, gparse as G
for f in ['/tmp/pk/research/data/peaked-circuits/peaked_circuit_P11_Hqap_98x1999.qasm','/tmp/pk/research/data/peaked-circuits/peaked_circuit_P12_Hqap_98x2457.qasm']:
    n, old = v1.parse(f); n2, new, lone = G.parse_nearest(f)
    same = len(old) == len(new) and all(o[0]==u[0] and o[1]==u[1] and all(np.allclose(o[i], u[i], atol=1e-12) for i in range(2,6)) for o,u in zip(old,new))
    print(f.split('/')[-1], "units identical to strict 5-line parser:", same)
