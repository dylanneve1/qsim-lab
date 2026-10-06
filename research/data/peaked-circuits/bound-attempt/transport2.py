"""Cost to carry each boundary transposition CZ (outside the hull) all the way to the nearest CZ
unit on the same pair inside the hull (exact commutation past CZs, paid approximate commutation
past every 1q segment on both wires)."""
import sys, collections, numpy as np
sys.path.insert(0,'/tmp/peaked-bound')
from solve_peaked_v2 import Core
from split_inner import inner_split
from transportlib import prod_cost, seg, CZ
D='/tmp/pk/research/data/peaked-circuits/'
def cross(w,a,S): return prod_cost(CZ@np.kron(S,np.eye(2))@CZ if w==a else CZ@np.kron(np.eye(2),S)@CZ)
res={}
for name in ['P11_Hqap_98x1999','P12_Hqap_98x2457']:
    c=Core(D+f'peaked_circuit_{name}.qasm')
    seq=collections.defaultdict(list)
    for k,u in enumerate(c.units):
        for q in u[:2]: seq[q].append(k)
    tot=0
    for si,f in c.maps:
        inner,before,after,bad=inner_split(c,si)
        tp={frozenset((q,f[q])) for q in range(c.n)}
        costs=[]; nseg=[]
        for k in sorted(before|after):
            p=frozenset(c.units[k][:2])
            if p not in tp: continue
            a,b=c.units[k][:2]
            # target: nearest unit on same pair inside hull in the direction of the hull
            same=[x for x in seq[a] if frozenset(c.units[x][:2])==p and x in inner]
            if not same: costs.append(np.inf); continue
            tgt=min(same) if k in before else max(same)
            cost=0; ns=0
            for w in (a,b):
                s=seq[w]; i=s.index(k); j=s.index(tgt)
                rng=range(i,j) if k in before else range(j,i)
                for t in rng:
                    cost+=cross(w,a,seg(c,w,s[t],s[t+1])); ns+=1
            costs.append(cost); nseg.append(ns)
        cs=np.array(costs); tot+=cs[cs<0.05].sum()
        print(f'{name} sec{si}: {len(cs)} boundary tCZ; to-centre cost: <0.05: n={np.sum(cs<0.05)} sum={cs[cs<0.05].sum():.4f}; >=0.05: n={np.sum(cs>=0.05)} ; median segs crossed {np.median(nseg)}')
        print('   ', np.round(np.sort(cs),4).tolist())
