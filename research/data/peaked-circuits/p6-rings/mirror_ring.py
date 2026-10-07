"""Search a time mirror of ring-nn pair occupation: M_r[e,t] (edge e of ring r, CZ layer t) vs M_r'[pi e, 2c - t]
for pi in the dihedral group (and A<->B). Score = normalised overlap; compare to time-shift (non-mirror) correlation baseline."""
from parse import *
G=load()
lay=[0]*62; L=[]
for g in G:
    if g[0]=='cz':
        a,b=g[1]; l=max(lay[a],lay[b]); L.append((l,a,b)); lay[a]=lay[b]=l+1
T=max(lay)
def occ(r):
    R=rings[r]; n=len(R); M=np.zeros((n,T))
    for l,a,b in L:
        if ringof[a]==r and ringof[b]==r:
            i,j=pos[a],pos[b]
            if (i-j)%n==1: M[j,l]+=1
            elif (j-i)%n==1: M[i,l]+=1
    return M  # edge e = (e, e+1)
def dihedral(n):
    for s in range(n):
        yield ('rot',s), (lambda e,s=s: (e+s)%n)
        yield ('ref',s), (lambda e,s=s: (s-e-1)%n)   # edge (e,e+1) -> (s-e-1, s-e)
Ms=[occ(r) for r in range(3)]
def smooth(M,w=1):
    out=M.copy()
    for k in range(1,w+1): out[:,k:]+=M[:,:-k]; out[:,:-k]+=M[:,k:]
    return out
res=[]
for r1,r2 in [(0,0),(1,1),(2,2),(0,1),(1,0)]:
    M1=Ms[r1]; M2=smooth(Ms[r2],1); n=M1.shape[0]
    for name,pi in dihedral(n):
        P=M2[[pi(e) for e in range(n)]]
        for s2 in range(40,2*T-40):   # 2c
            # mirror: M1[e,t] vs P[e, s2-t]
            ts=np.arange(T); tt=s2-ts; ok=(tt>=0)&(tt<T)
            sc=(M1[:,ts[ok]]*P[:,tt[ok]]).sum()/max(1,M1[:,ts[ok]].sum())
            res.append((sc,'mirror',r1,r2,name,s2/2))
        for d in range(-20,21):
            ts=np.arange(T); tt=ts+d; ok=(tt>=0)&(tt<T)
            sc=(M1[:,ts[ok]]*P[:,tt[ok]]).sum()/max(1,M1[:,ts[ok]].sum())
            res.append((sc,'shift',r1,r2,name,d))
res.sort(key=lambda x:-x[0])
for x in res[:25]: print(x)
import collections
for kind in ('mirror','shift'):
    v=np.array([x[0] for x in res if x[1]==kind]); print(kind,'mean',v.mean().round(3),'std',v.std().round(3),'max',v.max().round(3))
