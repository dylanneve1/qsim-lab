import sys, time, numpy as np, collections
sys.path.insert(0,'/tmp/peaked-bound')
import solve_peaked as v1
from solve_peaked_v2 import Core
from zipper import Zipper
D='/tmp/pk/research/data/peaked-circuits/'
name=sys.argv[1]; si=int(sys.argv[2]); cut=int(sys.argv[3]); kmax=int(sys.argv[4]); tau=float(sys.argv[5])
c=Core(D+f'peaked_circuit_{name}.qasm'); f=dict(c.maps).get(si)
lo,hi=c.secs[si]
gates=[(c.units[k][:2], c.M[k]) for k in range(lo,hi+1)]
t=time.time()
anc=v1.anchors(c.n,c.units)
partner=collections.defaultdict(set)
for a,b in anc:
    (w,k1,k2,_),(w2,k1b,k2b,_)=a,b
    for x,y in ((k2,k1b),(k1,k2b)):
        if lo<=x<=hi and lo<=y<=hi:
            partner[x-lo].add(y-lo); partner[y-lo].add(x-lo)
# cut: W0 = upset of second members
firsts={min(x,y) for x in partner for y in partner[x]}; seconds={max(x,y) for x in partner for y in partner[x]}
print('pairs firsts',len(firsts),'seconds',len(seconds),'overlap',len(firsts&seconds))
if cut<0:
    # downset of firsts
    seq=collections.defaultdict(list)
    for i,(w,m) in enumerate(gates):
        for q in w: seq[q].append(i)
    down=set(); stack=list(firsts)
    while stack:
        i=stack.pop()
        if i in down: continue
        down.add(i)
        for q in gates[i][0]:
            for j in seq[q]:
                if j<i: stack.append(j)
    up=set(); stack=list(seconds)
    while stack:
        i=stack.pop()
        if i in up: continue
        up.add(i)
        for q in gates[i][0]:
            for j in seq[q]:
                if j>i: stack.append(j)
    print('downset',len(down),'upset',len(up),'overlap',len(down&up))
    # reorder gates: downset first (topological order preserved within), then rest
    order=sorted(down)+sorted(set(range(len(gates)))-down)
    remap={o:i for i,o in enumerate(order)}
    gates=[gates[o] for o in order]
    partner={remap[k]:{remap[x] for x in v} for k,v in partner.items()}
    cut=len(down)+lo
z=Zipper(gates, cut-lo, c.n, kmax=kmax, tau=tau, log=print).run(partner=partner)
cs=np.array(z.costs)
print(f'{name} sec{si} cut {cut} kmax {kmax}: total cost {z.cost:.4f}, peels {len(cs)}, >1e-6: {(cs>1e-6).sum()}, forced {z.forced}, maxk {z.maxk}, {time.time()-t:.1f}s')
print(' largest costs', np.sort(cs)[::-1][:15])
if f: print(' Pi == f:', z.Pi==[f[q] for q in range(c.n)], 'mismatches', sum(z.Pi[q]!=f[q] for q in range(c.n)))
dg=[np.linalg.norm(u-np.trace(u)/2*np.eye(2)*0 - u[0,0]*0,2) for u in z.dangling.values()]
print(' dangling 1q gates', len(z.dangling))
