"""Outer-identity test, CONDITIONAL on replacing the inner blocks by their wire maps.
P11: O = sec1 |> f_A(sec3) |> L(sec5)   (should be ~ identity).  P12: O = sec2 |> f_B(sec4)."""
import sys, time, numpy as np, collections
sys.path.insert(0,'/tmp/peaked-bound')
from solve_peaked_v2 import Core
from zipper import Zipper
D='/tmp/pk/research/data/peaked-circuits/'
name=sys.argv[1]; kmax=int(sys.argv[2]); tau=float(sys.argv[3]); look=int(sys.argv[4]) if len(sys.argv)>4 else 1
c=Core(D+f'peaked_circuit_{name}.qasm'); maps=dict(c.maps)
S=len(c.secs)
gates=[]; cut=None
mp=list(range(c.n))
for si in range(1,S-1):
    if si in maps:
        mp=[mp[maps[si][q]] for q in range(c.n)]   # same composition rule as Core.L
        if cut is None: cut=len(gates)
        continue
    lo,hi=c.secs[si]
    for k in range(lo,hi+1):
        a,b=c.units[k][:2]; gates.append(((mp[a],mp[b]),c.M[k]))
print(name,'outer gates',len(gates),'cut',cut, 'L==mp', mp==c.L)
t=time.time()
z=Zipper(gates,cut,c.n,kmax=kmax,tau=tau,log=print).run(lookahead=bool(look))
cs=np.array(z.costs)
print(f'TOTAL cost {z.cost:.4f} peels {len(cs)} nonzero(>1e-9) {(cs>1e-9).sum()} forced {z.forced} maxk {z.maxk} time {time.time()-t:.0f}s')
print('cost quantiles', np.quantile(cs,[0.5,0.9,0.99,1]).round(5).tolist())
print('sorted top', np.sort(cs)[::-1][:20].round(4).tolist())
print('final Pi nontrivial', sum(z.Pi[q]!=q for q in range(c.n)), 'dangling', len(z.dangling))
np.save(f'costs_outer_{name}_{kmax}_{tau}.npy',cs)
