"""Per epoch: for each wire q and P in {X,Z}, weight of the Heisenberg image (through that epoch only) on Pauli strings that leave ring(q)."""
import numpy as np, collections, sys
from gapact import image, G, E
from parse import *
nm=lambda q: 'ABC'[ringof[q]]+str(pos[q])
ne=max(E.values())+1
for e in range(ne):
    ops=[G[i] for i in range(len(G)) if E[i]==e]
    rows=[]; tot=0; blow=[]
    for q in range(62):
        for p,pn in ((1,'X'),(3,'Z')):
            im=image(ops,q,p,eps=1e-5,maxterms=40000)
            if im is None: blow.append(nm(q)+pn); continue
            wout=sum(v*v for k,v in im.items() if any(ringof[w]!=ringof[q] for w,_ in k))
            tot+=wout
            if wout>0.02:
                sup=collections.Counter()
                for k,v in im.items():
                    for w,_ in k:
                        if ringof[w]!=ringof[q]: sup[w]+=v*v
                rows.append(f"{nm(q)}{pn}:{wout:.2f}->{','.join(nm(w) for w,s in sup.most_common(3))}")
    print(e,'B' if e%2==0 else 'G',f'cross-weight total {tot:.2f}','blowups',blow,' | '.join(rows),flush=True)
