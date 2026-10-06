import sys, time, numpy as np, cProfile, pstats
sys.path.insert(0,'.')
import gparse as G
from mpou import MPOU
CZ=np.diag([1,1,1,-1]).astype(complex)
n,units,tail=G.parse('/tmp/peaked-gen/portal/P5_granite_summit.qasm')
U=[(a,b,CZ@np.kron(Pa,Pb)) for a,b,Pa,Pb in units]
c=946; W=MPOU(n,1e-3,256)
def run():
    t=time.time()
    for i in range(40):
        W.absorb(*U[c+i],'up'); W.absorb(*U[c-1-i],'dn')
        if i%5==4:
            t1=time.time(); g=W.unswap_sweep(); print(i,'unswap gain',g,W.stats(),'sweep %.1fs'%(time.time()-t1),flush=True)
    print('total',time.time()-t)
cProfile.run('run()','prof.out')
pstats.Stats('prof.out').sort_stats('cumtime').print_stats(12)
