import sys, collections, numpy as np
sys.path.insert(0,'/tmp/peaked-bound')
import solve_peaked as v1
from solve_peaked_v2 import Core
D='/tmp/pk/research/data/peaked-circuits/'
name=sys.argv[1]
c=Core(D+f'peaked_circuit_{name}.qasm')
anc=v1.anchors(c.n,c.units)
partner=collections.defaultdict(set)
for a,b in anc:
    (w,k1,k2,_),(w2,k1b,k2b,_)=a,b
    partner[k2].add(k1b); partner[k1b].add(k2)
    partner[k1].add(k2b); partner[k2b].add(k1)
print('units with partner',len(partner),'multi',sum(len(v)>1 for v in partner.values()))
def sec(k):
    for i,(lo,hi) in enumerate(c.secs):
        if lo<=k<=hi: return i
bad=0
for si,f in c.maps:
    lo,hi=c.secs[si]
    ks=[k for k in partner if lo<=k<=hi]
    ok=0
    for k in ks:
        for k2 in partner[k]:
            a,b=c.units[k][:2]; a2,b2=c.units[k2][:2]
            if {f[a],f[b]}=={a2,b2}: ok+=1
            else: bad+=1
    print('sec',si,'units',hi-lo+1,'with partner',len(ks),'wire-consistent',ok)
# print example chain: for sec 2 sort partner pairs by centre
si,f=c.maps[0]; lo,hi=c.secs[si]
pairs=sorted({tuple(sorted((k,list(v)[0]))) for k,v in partner.items() if lo<=k<=hi and len(v)==1})
cen=[(a+b)/2 for a,b in pairs]
print('centres',np.percentile(cen,[0,10,25,50,75,90,100]))
print(pairs[:40])
