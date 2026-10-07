"""For every gap epoch: which wires are not factored out (X or Z image not single-site), and the supports of their images."""
import sys, numpy as np, collections, json
from gapact import image, G, E
from parse import *
nm=lambda q: 'ABC'[ringof[q]]+str(pos[q])
ne=max(E.values())+1
out={}
for e in range(ne):
    ops=[G[i] for i in range(len(G)) if E[i]==e]
    if e%2==0 and e!=0 and '--all' not in sys.argv: continue
    rows=[]
    for q in range(62):
        for p,pn in ((1,'X'),(3,'Z')):
            im=image(ops,q,p,eps=1e-4,maxterms=3000)
            if im is None: rows.append((nm(q),pn,'BLOWUP',[])); continue
            w1=sum(v*v for k,v in im.items() if len(k)==1 and k[0][0]==q)
            if w1>0.99: continue
            sup=collections.Counter()
            for k,v in im.items():
                for (w,_) in k: sup[w]+=v*v
            rows.append((nm(q),pn,round(w1,3),sorted([(nm(w),round(s,2)) for w,s in sup.items() if s>0.05 and w!=q],key=lambda x:-x[1])))
    ncz=sum(g[0]=='cz' for g in ops)
    print(f'=== epoch {e} cz {ncz} nonlocal {len(rows)}')
    for r in rows: print('  ',r)
    out[e]=rows
