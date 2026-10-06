import sys, collections, numpy as np
sys.path.insert(0,'/tmp/peaked-bound')
from solve_peaked_v2 import Core
D='/tmp/pk/research/data/peaked-circuits/'
c=Core(D+'peaked_circuit_P11_Hqap_98x1999.qasm')
lo,hi=c.secs[2]
for w in (13,46,14,59):
    print(w,[(k,c.units[k][:2]) for k in range(lo,hi+1) if w in c.units[k][:2]])
SW=np.eye(4)[[0,2,1,3]].astype(complex)
def kak_like(U):
    # distance-ish: operator Schmidt residual for local and for SWAP*local
    def res(V):
        T=V.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4)
        s=np.linalg.svd(T,compute_uv=False); return 1-s[0]**2/np.sum(s**2)
    return res(U), res(SW@U)
for a,b in [(557,576),(538,654),(654,847),(538,847)]:
    Ua=c.M[a]; Ub=c.M[b]
    wa=c.units[a][:2]; wb=c.units[b][:2]
    if wa!=wb: Ub=SW@Ub@SW
    print(a,b,wa,wb,'local-res, swap-local-res', kak_like(Ub@Ua))
