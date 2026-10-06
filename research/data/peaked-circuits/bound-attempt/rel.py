import sys, collections, numpy as np
sys.path.insert(0,'/tmp/peaked-bound')
import solve_peaked as v1
from solve_peaked_v2 import Core
D='/tmp/pk/research/data/peaked-circuits/'
name=sys.argv[1]; si=int(sys.argv[2])
c=Core(D+f'peaked_circuit_{name}.qasm'); f=dict(c.maps)[si]
lo,hi=c.secs[si]
anc=v1.anchors(c.n,c.units)
links=[]
for a,b in anc:
    (w,k1,k2,_),(w2,k1b,k2b,_)=a,b
    if lo<=k1 and k2b<=hi:
        links.append((w,w2,k1,k2,k1b,k2b))
cnt=collections.Counter()
for w,w2,k1,k2,k1b,k2b in links:
    cnt['w2==w' if w2==w else ('w2==f(w)' if w2==f[w] else 'other')]+=1
print(cnt, 'n links',len(links))
# segment link: (w,k1->k2) inverse (w2,k1b->k2b). Which units: k1,k2 on w; k1b,k2b on w2.
# For links with w2==f(w): check other wires of partner units
c2=collections.Counter()
for w,w2,k1,k2,k1b,k2b in links:
    o=lambda k,x: [q for q in c.units[k][:2] if q!=x][0]
    # mirror: k2<->k1b, k1<->k2b
    r1=(o(k2,w),o(k1b,w2)); r2=(o(k1,w),o(k2b,w2))
    for x,y in (r1,r2):
        c2['same' if x==y else ('f' if f[x]==y else 'other')]+=1
print('partner-unit other wire relation',c2)
ex=[l for l in links if l[1]==l[0]][:10]
print(ex)
