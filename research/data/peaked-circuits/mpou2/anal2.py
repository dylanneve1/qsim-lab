import sys, numpy as np, collections
import gparse as G
n,units,tail=G.parse(sys.argv[1]); K=len(units)
depth=[0]*n; lay=[]
for a,b,_,_ in units:
    l=max(depth[a],depth[b]); lay.append(l); depth[a]=depth[b]=l+1
lay=np.array(lay)
# monotonic?
inv=np.sum(np.diff(lay)<0); print('K',K,'layer decreases in file order:',inv)
for c in range(0,K,K//20): print(c, lay[c:c+K//20].min(), lay[c:c+K//20].max())
