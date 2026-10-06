import sys, collections, numpy as np
sys.path.insert(0,'/tmp/peaked-bound')
import solve_peaked as v1
from solve_peaked_v2 import Core
D='/tmp/pk/research/data/peaked-circuits/'
def inner_split(c, si):
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
    inner=set()
    for a,b in pairs: inner|=(cone(a,1)&cone(b,-1))|{a,b}
    before=set(); after=set()
    for k in range(lo,hi+1):
        if k in inner: continue
        # before if some inner gate is in its future
        if cone(k,1)&inner: before.add(k)
        elif cone(k,-1)&inner: after.add(k)
        else: before.add(k)   # disconnected: put before
    bad=[k for k in before if cone(k,-1)&inner]
    return inner,before,after,bad
if __name__=='__main__':
    for name in ['P11_Hqap_98x1999','P12_Hqap_98x2457']:
        c=Core(D+f'peaked_circuit_{name}.qasm')
        for si,f in c.maps:
            inner,before,after,bad=inner_split(c,si)
            print(name,'sec',si,'size',c.secs[si][1]-c.secs[si][0]+1,'inner',len(inner),'before',len(before),'after',len(after),'inconsistent',len(bad))
