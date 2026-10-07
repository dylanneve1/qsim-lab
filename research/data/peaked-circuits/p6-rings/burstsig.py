"""Per burst epoch: per ring edge, CZ count. Similarity between bursts (cosine of count vectors) to look for a mirror."""
import numpy as np
from parse import *
from segment import epochs
G=load(); E=epochs(); ne=max(E.values())+1
edges=[]
for r in range(3):
    R=rings[r]; L=len(R)
    for i in range(L): edges.append(frozenset((R[i],R[(i+1)%L])))
eidx={e:i for i,e in enumerate(edges)}
B=sorted(e for e in range(0,ne,2))
V=np.zeros((len(B),len(edges)))
for i,g in enumerate(G):
    if g[0]=='cz' and E[i]%2==0:
        k=frozenset(g[1])
        if k in eidx: V[E[i]//2,eidx[k]]+=1
np.set_printoptions(linewidth=250)
print('CZ per burst',V.sum(1).astype(int))
for r,name in enumerate('ABC'):
    off=sum(len(rings[x]) for x in range(r)); L=len(rings[r])
    print(name); 
    for b in range(len(B)): print('  b%02d'%b,''.join(str(int(x)) for x in V[b,off:off+L]))
Vn=V/np.linalg.norm(V,axis=1,keepdims=True)
S=Vn@Vn.T
print(np.round(S,2))
