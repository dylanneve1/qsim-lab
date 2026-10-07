import json, collections, numpy as np, gparse as G
CZ=np.diag([1,1,1,-1]).astype(complex)
n,units,tail=G.parse('P6_titan_pinnacle.qasm')
rings=json.load(open('P6_rings.json')) if False else None
# rings from frequent pairs (same as ringsim)
cnt=collections.Counter(frozenset(u[:2]) for u in units)
deg=collections.Counter(); heavy=[]
for pr,c in cnt.most_common():
    a,b=tuple(pr)
    if deg[a]<2 and deg[b]<2: heavy.append(pr); deg[a]+=1; deg[b]+=1
par=list(range(n))
def f(x):
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
for pr in heavy: a,b=tuple(pr); par[f(a)]=f(b)
gid={q:f(q) for q in range(n)}
depth=[0]*n; lay=[]
for a,b,_,_ in units:
    l=max(depth[a],depth[b]); lay.append(l); depth[a]=depth[b]=l+1
D=max(lay)+1; ring=[0]*D; cr=[0]*D
for k,(a,b,_,_) in enumerate(units):
    if gid[a]==gid[b]: ring[lay[k]]+=1
    else: cr[lay[k]]+=1
print(' '.join(f'{l}:{ring[l]}/{cr[l]}' for l in range(D)))
