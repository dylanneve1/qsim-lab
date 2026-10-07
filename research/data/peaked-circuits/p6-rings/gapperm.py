"""Per gap epoch: wire permutation from Heisenberg images (Hungarian on support weights). Saves private/../gapperm.json"""
import numpy as np, collections, json
from scipy.optimize import linear_sum_assignment
from gapact import image, G, E
from parse import *
nm=lambda q: 'ABC'[ringof[q]]+str(pos[q])
ne=max(E.values())+1
out={}
for e in range(1,ne,2):
    ops=[G[i] for i in range(len(G)) if E[i]==e]
    S=np.zeros((62,62))
    for q in range(62):
        for p in (1,3):
            im=image(ops,q,p,eps=1e-5,maxterms=40000)
            if im is None: continue
            for k,v in im.items():
                for w,_ in k: S[q,w]+=v*v/len(k)
    r,c=linear_sum_assignment(-S)
    pi={int(q):int(w) for q,w in zip(r,c)}
    sc=[S[q,pi[q]]/2 for q in range(62)]
    moved=[(nm(q),nm(pi[q]),round(sc[q],2)) for q in range(62) if pi[q]!=q]
    print(e,'moved',len(moved),'min score',round(min(sc),2),moved,flush=True)
    out[e]=dict(pi=pi,score=sc)
json.dump(out,open('gapperm.json','w'))
