"""Compute and cache Heisenberg images (eps 1e-7) of X_q,Z_q through every gap epoch (odd) -> private/imgs_e{e}.pkl"""
import sys, pickle, os
from gapact import image, G, E
ne=max(E.values())+1
for e in range(1,ne,2):
    f=f'private/imgs{os.environ.get("SEG","")}_e{e}.pkl'
    if os.path.exists(f): continue
    ops=[G[i] for i in range(len(G)) if E[i]==e]
    imgs={}; nb=[]
    for q in range(62):
        for p in (1,3):
            im=image(ops,q,p,eps=1e-7,maxterms=300000)
            if im is None: nb.append((q,p)); continue
            imgs[(q,p)]=im
    pickle.dump(imgs,open(f,'wb')); print(e,'done blowups',nb,flush=True)
