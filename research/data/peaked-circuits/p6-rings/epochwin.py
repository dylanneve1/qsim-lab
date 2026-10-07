"""Localisation of Heisenberg images through windows of consecutive epochs [e0, e1]. Identity-like windows (U |> U^dag around a
centre) stay local; generic windows scramble (blow-up or low single-site weight).
usage: epochwin.py WIDTH(#epochs) [nwires]"""
import sys, numpy as np, collections
from gapact import image, G, E
W=int(sys.argv[1]); nw=int(sys.argv[2]) if len(sys.argv)>2 else 20
ne=max(E.values())+1
rng=np.random.default_rng(0); wires=sorted(rng.choice(62,nw,replace=False))
for e0 in range(0,ne-W+1):
    e1=e0+W-1
    ops=[G[i] for i in range(len(G)) if e0<=E[i]<=e1]
    sc=[]; nb=0
    for q in wires:
        for p in (1,3):
            im=image(ops,int(q),p,eps=1e-3,maxterms=20000)
            if im is None: sc.append(0.0); nb+=1; continue
            w=collections.defaultdict(float)
            for k,v in im.items():
                if len(k)==1: w[k[0][0]]+=v*v
            sc.append(max(w.values()) if w else 0.0)
    sc=np.array(sc)
    print(f'epochs {e0}-{e1} ({(e0+e1)/2:.1f}) mean {sc.mean():.3f} median {np.median(sc):.3f} >0.9: {(sc>0.9).sum()}/{len(sc)} blowups {nb}',flush=True)
