import sys, collections, numpy as np, time
sys.path.insert(0,'/tmp/peaked-bound')
import solve_peaked as v1
from solve_peaked_v2 import Core
from zipper import Zipper
D='/tmp/pk/research/data/peaked-circuits/'
name=sys.argv[1]; si=int(sys.argv[2]); KMAX=int(sys.argv[3])
c=Core(D+f'peaked_circuit_{name}.qasm')
lo,hi=c.secs[si]
anc=v1.anchors(c.n,c.units)
pairs=set()
for a,b in anc:
    (w,k1,k2,_),(w2,k1b,k2b,_)=a,b
    for x,y in ((k2,k1b),(k1,k2b)):
        if lo<=x<=hi and lo<=y<=hi: pairs.add((min(x,y),max(x,y)))
seq=collections.defaultdict(list)
for k in range(c.secs[1][0],c.secs[-2][1]+1):
    for q in c.units[k][:2]: seq[q].append(k)
def nb(k,d):
    out=[]
    for q in c.units[k][:2]:
        s=seq[q]; i=s.index(k)+d
        if 0<=i<len(s): out.append(s[i])
    return out
def cone(k,d):
    seen=set(); st=[k]
    while st:
        x=st.pop()
        for y in nb(x,d):
            if y not in seen: seen.add(y); st.append(y)
    return seen
def region_cost(ks):
    ks=sorted(ks); n=c.n
    gates=[(c.units[k][:2],c.M[k]) for k in ks]
    z=Zipper(gates,len(gates),n,kmax=99,tau=1e-9)
    for i in reversed(range(len(gates))): z.absorb('V',i)
    k=len(z.S)
    z.peel_all()
    while z.S: z.peel_all(force=True)
    perm={q:z.Pi[q] for q in range(n) if z.Pi[q]!=q}
    return z.cost,k,perm,z.costs
res=[]
for a,b in pairs:
    dm=cone(a,1)&cone(b,-1)
    qs=set(q for k in dm|{a,b} for q in c.units[k][:2])
    res.append((len(qs),len(dm),a,b))
res.sort()
t=time.time()
for nq,nd,a,b in res:
    if nq>KMAX: break
    cost,k,perm,cs=region_cost(cont:=(cone(a,1)&cone(b,-1))|{a,b})
    print(f'pair {a},{b} wires {c.units[a][:2]} diamond {nd} qubits {nq}: cost {cost:.3e} perm {perm} peels {np.round(sorted(cs)[::-1][:4],4).tolist()}',flush=True)
print('time',time.time()-t)
