import sys, collections, numpy as np
sys.path.insert(0,'/tmp/peaked-bound')
from solve_peaked_v2 import Core
D='/tmp/pk/research/data/peaked-circuits/'
name=sys.argv[1]; si=int(sys.argv[2])
c=Core(D+f'peaked_circuit_{name}.qasm'); f=dict(c.maps)[si]
lo,hi=c.secs[si]
seq=collections.defaultdict(list)
for k in range(lo,hi+1):
    a,b=c.units[k][:2]; seq[a].append(b); seq[b].append(a)
tot=0;tot2=0
for w in range(c.n):
    s=seq[w]; t=[f[x] for x in seq[f[w]]][::-1]
    # longest suffix of s equal to prefix-reversed ... compare s with t (t = mirror image of f(w)'s sequence)
    # s should equal t if perfect mirror (s = A_w + mid + B_w, t = rev(f(A_fw)+..))
    m=0
    while m<min(len(s),len(t)) and s[-1-m]==t[-1-m]: m+=1
    m2=0
    while m2<min(len(s),len(t)) and s[m2]==t[m2]: m2+=1
    tot+=len(s); tot2+=m+m2
    if w<12: print(w,f[w],len(s),len(t),'tail match',m,'head match',m2,'\n   s',s,'\n   t',t)
print('total',tot,'matched',tot2)
