import sys, numpy as np, collections
sys.path.insert(0,'/tmp/peaked-bound')
from solve_peaked_v2 import Core
D='/tmp/pk/research/data/peaked-circuits/'
c=Core(D+'peaked_circuit_P11_Hqap_98x1999.qasm')
def segs(w,lo,hi):
    ks=[k for k in range(lo,hi+1) if w in c.units[k][:2]]
    out=[]
    for k1,k2 in zip(ks,ks[1:]):
        u1=c.units[k1]; u2=c.units[k2]
        post=u1[4] if u1[0]==w else u1[5]; pre=u2[2] if u2[0]==w else u2[3]
        S=pre@post
        out.append((k1,k2,round(abs(S[0,1]),3)))
    return out
for w in (8,10):
    print(w, segs(w,377,913))
