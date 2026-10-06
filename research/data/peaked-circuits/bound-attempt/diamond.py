import sys, collections, numpy as np
sys.path.insert(0,'/tmp/peaked-bound')
import solve_peaked as v1
from solve_peaked_v2 import Core
D='/tmp/pk/research/data/peaked-circuits/'
name=sys.argv[1]; si=int(sys.argv[2])
c=Core(D+f'peaked_circuit_{name}.qasm')
lo,hi=c.secs[si]
anc=v1.anchors(c.n,c.units)
pairs=set()
for a,b in anc:
    (w,k1,k2,_),(w2,k1b,k2b,_)=a,b
    for x,y in ((k2,k1b),(k1,k2b)):
        if lo<=x<=hi and lo<=y<=hi: pairs.add((min(x,y),max(x,y)))
seq=collections.defaultdict(list)
for k in range(lo,hi+1):
    for q in c.units[k][:2]: seq[q].append(k)
def nxt(k):
    out=[]
    for q in c.units[k][:2]:
        s=seq[q]; i=s.index(k)
        if i+1<len(s): out.append(s[i+1])
    return out
def prv(k):
    out=[]
    for q in c.units[k][:2]:
        s=seq[q]; i=s.index(k)
        if i>0: out.append(s[i-1])
    return out
def cone(k,f):
    seen=set(); st=[k]
    while st:
        x=st.pop()
        for y in f(x):
            if y not in seen: seen.add(y); st.append(y)
    return seen
res=[]
for a,b in pairs:
    dm=cone(a,nxt)&cone(b,prv)
    qs=set(q for k in dm for q in c.units[k][:2])|set(c.units[a][:2])|set(c.units[b][:2])
    res.append((len(dm),len(qs),a,b,c.units[a][:2],c.units[b][:2]))
res.sort()
for r in res[:25]: print(r)
print('diamond sizes pct',np.percentile([r[0] for r in res],[0,10,50,90]),'qubits',np.percentile([r[1] for r in res],[0,10,50,90]))
