import sys, collections, numpy as np
sys.path.insert(0,'/tmp/peaked-bound')
from solve_peaked_v2 import Core
D='/tmp/pk/research/data/peaked-circuits/'
c=Core(D+'peaked_circuit_P11_Hqap_98x1999.qasm'); f=dict(c.maps)[2]
lo,hi=c.secs[2]
for w in (8,10,75,f[75],92,f[92],69,f[69]):
    print(w,'f',f[w],[(k,[q for q in c.units[k][:2] if q!=w][0]) for k in range(lo,hi+1) if w in c.units[k][:2]])
