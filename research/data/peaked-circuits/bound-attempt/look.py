import sys, collections, numpy as np
sys.path.insert(0,'/tmp/peaked-bound')
import solve_peaked as v1
from solve_peaked_v2 import Core
D='/tmp/pk/research/data/peaked-circuits/'
name=sys.argv[1]; si=int(sys.argv[2])
c=Core(D+f'peaked_circuit_{name}.qasm'); f=dict(c.maps).get(si)
lo,hi=c.secs[si]
depth=collections.Counter(); lay=collections.defaultdict(list)
for k in range(lo,hi+1):
    a,b=c.units[k][:2]; d=max(depth[a],depth[b])+1; depth[a]=depth[b]=d; lay[d].append((a,b))
for d in sorted(lay):
    print(d, len(lay[d]), ' '.join(f'{a}-{b}' for a,b in sorted(lay[d])))
print(f and sorted((q,f[q]) for q in range(c.n) if q<f[q]))
