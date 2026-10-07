"""Localisation scan on the effective circuit: windows of epochs [c-k, c+k] (c = gap epoch) ; sampled wires.
usage: effwin.py EFF.pkl K [nw]"""
import sys, pickle, numpy as np, collections
from effpauli import Prop
ops=pickle.load(open(sys.argv[1],'rb')); ep=pickle.load(open(sys.argv[1]+'.ep','rb')); K=int(sys.argv[2])
nw=int(sys.argv[3]) if len(sys.argv)>3 else 16
P=Prop(ops); ep=np.array(ep); ne=ep.max()+1
wires=sorted(np.random.default_rng(0).choice(62,nw,replace=False))
for c in range(1,ne):
    lo,hi=c-K,c+K
    if lo<0 or hi>=ne: continue
    idx=np.where((ep>=lo)&(ep<=hi))[0]; i0,i1=idx.min(),idx.max()
    sc=[]; nb=0
    for q in wires:
        for p in (1,3):
            im=P.image(i0,i1,int(q),p,eps=1e-3,maxterms=30000)
            if im is None: sc.append(0.0); nb+=1; continue
            w=collections.defaultdict(float)
            for k,v in im.items():
                if len(k)==1: w[k[0][0]]+=v*v
            sc.append(max(w.values()) if w else 0.0)
    sc=np.array(sc)
    print(f'centre {c} epochs {lo}-{hi} mean {sc.mean():.3f} median {np.median(sc):.3f} >0.9 {(sc>0.9).sum()}/{len(sc)} blowups {nb}',flush=True)
