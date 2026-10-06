import sys, collections, numpy as np
sys.path.insert(0,'/tmp/peaked-bound')
from solve_peaked_v2 import Core
D='/tmp/pk/research/data/peaked-circuits/'
def P(c,si,mp=None):
    lo,hi=c.secs[si]; out=[]
    for k in range(lo,hi+1):
        a,b=c.units[k][:2]
        if mp: a,b=mp[a],mp[b]
        out.append(tuple(sorted((a,b))))
    return out
for name,blocks in [('P11_Hqap_98x1999',None),('P12_Hqap_98x2457',None)]:
    c=Core(D+f'peaked_circuit_{name}.qasm')
    maps=dict(c.maps); S=len(c.secs)
    print(name, 'block sections', list(maps))
    # cumulative map from start to each section: label wire at section si in "reference" frame
    for i in range(S):
        for j in range(i+1,S):
            # compose maps of block sections strictly between i and j
            mp={q:q for q in range(c.n)}
            for s in range(i+1,j):
                if s in maps: mp={q:maps[s][mp[q]] for q in range(c.n)}
            A=collections.Counter(P(c,i)); B=collections.Counter(P(c,j))
            # map pairs in section j back: a wire q in sec j corresponds to wire mp^-1 ... since involutions compose, use inverse
            inv={v:k for k,v in mp.items()}
            Bm=collections.Counter(tuple(sorted((inv[a],inv[b]))) for a,b in P(c,j))
            ov=sum((A&Bm).values()); ov0=sum((A&B).values())
            print(f'  sec{i}({sum(A.values())}) vs sec{j}({sum(B.values())}): overlap mapped {ov}, unmapped {ov0}')
