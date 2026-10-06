import sys, collections, numpy as np
sys.path.insert(0,'/tmp/peaked-bound')
import solve_peaked as v1
from solve_peaked_v2 import Core
D='/tmp/pk/research/data/peaked-circuits/'
for name in ['P11_Hqap_98x1999','P12_Hqap_98x2457']:
    c=Core(D+f'peaked_circuit_{name}.qasm')
    print(name, 'n',c.n,'units',len(c.units))
    print(' secs',c.secs,[hi-lo+1 for lo,hi in c.secs])
    print(' maps sections',[s for s,_ in c.maps], 'fixed pts',[sum(m[q]==q for q in range(c.n)) for _,m in c.maps])
    anc=v1.anchors(c.n,c.units)
    print(' anchors',len(anc))
    by=collections.Counter()
    for a,b in anc:
        cen=(a[2]+b[1])/2
        for si,(lo,hi) in enumerate(c.secs):
            if lo<=cen<=hi: by[si]+=1
    print(' anchors by centre section',dict(by))
    # layers per section
    for si,(lo,hi) in enumerate(c.secs):
        depth=collections.Counter()
        for k in range(lo,hi+1):
            a,b=c.units[k][:2]; d=max(depth[a],depth[b])+1; depth[a]=depth[b]=d
        pairs=collections.Counter(tuple(sorted(c.units[k][:2])) for k in range(lo,hi+1))
        qs=set(q for k in range(lo,hi+1) for q in c.units[k][:2])
        print(f'  sec{si} [{lo},{hi}] gates {hi-lo+1} depth {max(depth.values())} qubits {len(qs)} distinct pairs {len(pairs)}')
