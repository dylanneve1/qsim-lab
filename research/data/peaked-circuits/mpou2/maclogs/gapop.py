import sys, numpy as np, time, collections
import gparse as G
from mpou2 import MPO2, SWAP
CZ=np.diag([1,1,1,-1]).astype(complex)
n,units,tail=G.parse('P6_titan_pinnacle.qasm')
depth=[0]*n; lay=[]
for a,b,_,_ in units:
    l=max(depth[a],depth[b]); lay.append(l); depth[a]=depth[b]=l+1
lo,hi=int(sys.argv[1]),int(sys.argv[2]); eps=float(sys.argv[3]) if len(sys.argv)>3 else 1e-12
W=MPO2(n,eps=eps,mode='rel',max_bond=4096)
t0=time.time(); mx=0
sel=[k for k in range(len(units)) if lo<=lay[k]<hi]
for i,k in enumerate(sel):
    a,b,Pa,Pb=units[k]
    W.absorb(a,b,CZ@np.kron(Pa,Pb),'up')
    for _ in range(2):
        if W.unswap_sweep(thr=2)<1: break
    mx=max(mx,W.stats()['max_bond'])
for _ in range(10):
    if W.unswap_sweep(thr=2, window=0)<1: break
b=W.bonds()
print('gap',lo,hi,'units',len(sel),'final',W.stats(),'max during',mx,f'{time.time()-t0:.1f}s')
print('bonds',b)
print('pairing mismatches',sum(W.su[s]!=W.sd[s] for s in range(n)))
segs=[]; cur=[W.su[0]]
for s in range(n-1):
    if b[s]==1: segs.append(cur); cur=[W.su[s+1]]
    else: cur.append(W.su[s+1])
segs.append(cur)
print('segments sizes',sorted([len(x) for x in segs],reverse=True)[:12])
