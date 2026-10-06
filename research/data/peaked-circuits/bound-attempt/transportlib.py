"""Rigorous cost of transporting each boundary CZ unit on a transposition pair (w, f(w)) to the
hull boundary, by exact commutation past CZs and approximate commutation past the 1q segments:
cost(S) = min over 1q unitaries P,Q and phase of || CZ (S x I) CZ^dag - e^{ith} P x Q ||  (spectral norm,
computed with the zipper's product-peel routine, exact eigenvalue arc norm).  Summed over all
segments crossed on both wires.  Also returns the CZ unit's own non-CZ content (its 1q gates are
re-assigned to the neighbouring segments exactly)."""
import sys, collections, numpy as np
sys.path.insert(0,'/tmp/peaked-bound')
from solve_peaked_v2 import Core
from split_inner import inner_split
from zipper2 import Zipper
D='/tmp/pk/research/data/peaked-circuits/'
CZ=np.diag([1,1,1,-1]).astype(complex)
def prod_cost(W):
    z=Zipper([((0,1),W)],1,2,kmax=9,tau=0)
    z.absorb('V',0)
    best=9
    for ri in range(2):
        for qi in range(2):
            d,u,A=z.candidate(ri,qi); best=min(best,d)
    return best
def seg(c,w,k1,k2):
    u1=c.units[k1]; u2=c.units[k2]
    post=u1[4] if u1[0]==w else u1[5]; pre=u2[2] if u2[0]==w else u2[3]
    return pre@post

def transport_costs(c, si, f, inner, before, after):
    seq=collections.defaultdict(list)
    for k,u in enumerate(c.units):
        for q in u[:2]: seq[q].append(k)
    tp={frozenset((q,f[q])) for q in range(c.n)}
    out={}
    def cross(w,a,S):
        return prod_cost(CZ@np.kron(S,np.eye(2))@CZ if w==a else CZ@np.kron(np.eye(2),S)@CZ)
    for k in sorted(before|after):
        if frozenset(c.units[k][:2]) not in tp: continue
        a,b=c.units[k][:2]; cost=0
        for w in (a,b):
            s=seq[w]; j=s.index(k)
            if k in before:
                while True:
                    cost+=cross(w,a,seg(c,w,s[j],s[j+1]))
                    if s[j+1] in inner: break
                    j+=1
            else:
                while True:
                    cost+=cross(w,a,seg(c,w,s[j-1],s[j]))
                    if s[j-1] in inner: break
                    j-=1
        out[k]=cost
    return out
