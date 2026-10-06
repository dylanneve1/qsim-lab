import sys, collections, numpy as np
sys.path.insert(0,'/tmp/peaked-bound')
import solve_peaked as v1
from solve_peaked_v2 import Core
D='/tmp/pk/research/data/peaked-circuits/'
for name in ['P11_Hqap_98x1999','P12_Hqap_98x2457']:
    c=Core(D+f'peaked_circuit_{name}.qasm'); maps=dict(c.maps)
    anc=v1.anchors(c.n,c.units)
    for si,f in c.maps:
        lo,hi=c.secs[si]
        cens=[(a[2]+b[1])/2 for a,b in anc if lo<=(a[2]+b[1])/2<=hi]
        print(name,'block',si,lo,hi,'centre min/med/max',min(cens),np.median(cens),max(cens))
        # transpositions
        tr=[(q,f[q]) for q in range(c.n) if q<f[q]]
        print('  transpositions',len(tr))
        # try mirror: for each split point m, compare multiset of pairs first half vs mapped second half
        pr=[tuple(sorted(c.units[k][:2])) for k in range(lo,hi+1)]
        best=None
        for m in range(len(pr)//2-60,len(pr)//2+60):
            A=collections.Counter(pr[:m]); B=collections.Counter(tuple(sorted((f[a],f[b]))) for a,b in pr[m:])
            ov=sum((A&B).values())
            if best is None or ov>best[0]: best=(ov,m)
        print('  best split',best,'of',len(pr))
