import json, itertools, sys
from geom import code
from collections import Counter
d=11; data,P=code(d)
supp=[[q for q in p['q'] if q is not None] for p in P]
qpl=[[i for i,s in enumerate(supp) if q in s] for q in range(len(data))]
deg=[len(x) for x in qpl]
bnd=[any(deg[q]<3 for q in s) for s in supp]
def syn(S):
    S=set(S); return tuple(sorted(i for i,s in enumerate(supp) if len(S&set(s))%2))
single={syn([q]):q for q in range(len(data))}
hooks={}
for i,s in enumerate(supp):
    if bnd[i]: continue
    for k in (2,3):
        for S in itertools.combinations(s,k):
            hooks.setdefault(syn(S),[]).append((i,S))
lines=open("cg/core11.txt").read().splitlines()[1:]
stat=Counter(); plz=Counter()
for l in lines:
    k,rest=l.split(" logical of weight ")
    w,L=rest.split(" : ",1); L=eval(L)
    nh=0; hp=[]
    for sig,ob in L:
        layers={a for a,b in sig}; assert layers=={1}, layers
        s=tuple(sorted(b for a,b in sig))
        if s in single: continue
        nh+=1; hp.append(sorted({i for i,S in hooks[s]}))
    stat[(int(w),nh)]+=1
    for h in hp: plz[tuple(h)]+=1
print(stat); print(plz.most_common(30))
print("interior plaquettes:",[ (i,(P[i]['x']-2*P[i]['y'])//4,P[i]['y']) for i in range(len(P)) if not bnd[i]])
NBL={(-1,1):'a',(0,1):'b',(1,0):'c',(1,-1):'d',(0,-1):'e',(-1,0):'f'}
uv=lambda q:((data[q][0]-2*data[q][1])//4,data[q][1])
puv=lambda i:((P[i]['x']-2*P[i]['y'])//4,P[i]['y'])
def lab(i,S):
    pu=puv(i); return "".join(sorted(NBL[(uv(q)[0]-pu[0],uv(q)[1]-pu[1])] for q in S))
print()
for l in lines:
    k,rest=l.split(" logical of weight ")
    w,L=rest.split(" : ",1); L=eval(L)
    hs=[];sg=[]
    for sig,ob in L:
        s=tuple(sorted(b for a,b in sig))
        if s in single: sg.append(uv(single[s])); continue
        hs.append(" | ".join(f"P{puv(i)}{'RGB'[P[i]['c']]}:{lab(i,S)}" for i,S in hooks[s]))
    if len(hs)>=2: print(len(hs),"hooks:",hs," singles:",sorted(sg))
