import sys, collections, numpy as np
sys.path.insert(0,'/tmp/peaked-bound')
from solve_peaked_v2 import Core
from zipper import Zipper
D='/tmp/pk/research/data/peaked-circuits/'
def region_cost(c,ks):
    gates=[(c.units[k][:2],c.M[k]) for k in sorted(ks)]
    z=Zipper(gates,len(gates),c.n,kmax=99,tau=1e-12)
    for i in reversed(range(len(gates))): z.absorb('V',i)
    z.peel_all()
    while z.S: z.peel_all(force=True)
    return z.cost, {q:z.Pi[q] for q in range(c.n) if z.Pi[q]!=q}
for name in ['P11_Hqap_98x1999','P12_Hqap_98x2457']:
    c=Core(D+f'peaked_circuit_{name}.qasm')
    last={}; out=[]
    for k,u in enumerate(c.units):
        a,b=u[:2]
        if a in last and b in last and last[a]==last[b]:
            j=last[a]; cost,perm=region_cost(c,[j,k]); 
            sec=[i for i,(lo,hi) in enumerate(c.secs) if lo<=k<=hi][0]
            out.append((cost,j,k,(a,b),sec,perm))
        last[a]=k; last[b]=k
    out.sort()
    print(name,len(out),'same-pair adjacent chains')
    for o in out: print('  cost %.4f  units %d,%d wires %s sec %d perm %s'%o)
