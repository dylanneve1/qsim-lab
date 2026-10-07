import sys, numpy as np, collections
import gparse as G
n,units,tail=G.parse(sys.argv[1]); K=len(units)
depth=[0]*n; lay=[]
for a,b,_,_ in units:
    l=max(depth[a],depth[b]); lay.append(l); depth[a]=depth[b]=l+1
def key(M):
    M=M/np.sqrt(np.linalg.det(M)); 
    if M[0,0].real<0 or (abs(M[0,0].real)<1e-9 and M[0,1].real<0): M=-M
    return tuple(np.round(np.concatenate([M.real.ravel(),M.imag.ravel()]),6))
cnt=collections.Counter()
segs=[]
for k,(a,b,Pa,Pb) in enumerate(units):
    ka,kb=key(Pa),key(Pb); cnt[ka]+=1; cnt[kb]+=1; segs.append((ka,kb))
rep={k for k,c in cnt.items() if c>=3}
print('distinct segs',len(cnt),'repeated(>=3)',len(rep), sorted([c for k,c in cnt.items() if c>=3],reverse=True)[:20])
net=[k for k in range(K) if segs[k][0] in rep and segs[k][1] in rep]
one=[k for k in range(K) if (segs[k][0] in rep) != (segs[k][1] in rep)]
print('units with both segs repeated',len(net),'layers',sorted(collections.Counter(lay[k] for k in net).items()))
print('units with one seg repeated',len(one),'layers',sorted(collections.Counter(lay[k] for k in one).items()))
pairs=collections.Counter(frozenset(units[k][:2]) for k in net)
print('pair multiplicities among net units',collections.Counter(pairs.values()))
# components of net units
par=list(range(n))
def f(x):
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
for k in net: par[f(units[k][0])]=f(units[k][1])
comp=collections.Counter(f(q) for q in range(n)); print('net components',sorted(comp.values(),reverse=True))
print('file idx range of net units', min(net), max(net))
