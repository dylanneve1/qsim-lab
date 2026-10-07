"""Clusters of each epoch's unitary from Heisenberg-image supports (union-find over wires appearing in images with |c|>thr)."""
import sys, numpy as np, collections, json, pickle
from gapact import image, G, E
from parse import *
nm=lambda q: 'ABC'[ringof[q]]+str(pos[q])
thr=float(sys.argv[2]) if len(sys.argv)>2 else 1e-4
eps_list=[int(x) for x in sys.argv[1].split(',')]
for e in eps_list:
    ops=[G[i] for i in range(len(G)) if E[i]==e]
    par=list(range(62))
    def f(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    imgs={}; nb=0
    for q in range(62):
        for p in (1,3):
            im=image(ops,q,p,eps=1e-7,maxterms=300000)
            if im is None: nb+=1; continue
            imgs[(q,p)]=im
            for k,v in im.items():
                if abs(v)>thr:
                    for w,_ in k: par[f(w)]=f(q)
    cl=collections.defaultdict(list)
    for q in range(62): cl[f(q)].append(q)
    sizes=sorted((len(v) for v in cl.values()),reverse=True)
    multi=[sorted(nm(q) for q in v) for v in cl.values() if len(v)>1]
    cross=[m for m in multi if len(set(x[0] for x in m))>1]
    print(e,'blowups',nb,'sizes',sizes[:8],'n1',sizes.count(1),'cross clusters',cross,flush=True)
    pickle.dump(imgs,open(f'private/imgs_e{e}.pkl','wb'))
