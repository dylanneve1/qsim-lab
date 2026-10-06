"""Where do the CZ units on transposition pairs (w, f(w)) of each inner block sit?"""
import sys, collections
sys.path.insert(0,'/tmp/peaked-bound')
from solve_peaked_v2 import Core
from split_inner import inner_split
D='/tmp/pk/research/data/peaked-circuits/'
for name in ['P11_Hqap_98x1999','P12_Hqap_98x2457']:
    c=Core(D+f'peaked_circuit_{name}.qasm')
    for si,f in c.maps:
        inner,before,after,bad=inner_split(c,si)
        lo,hi=c.secs[si]
        tp={frozenset((q,f[q])) for q in range(c.n)}
        cnt=collections.Counter()
        per=collections.defaultdict(lambda:[0,0,0])
        for k in range(lo,hi+1):
            p=frozenset(c.units[k][:2])
            if p in tp:
                where=0 if k in inner else (1 if k in before else 2)
                cnt[('inner','before','after')[where]]+=1; per[p][where]+=1
        pat=collections.Counter(tuple(v) for v in per.values())
        print(name,'block sec',si,'transposition-pair CZ units:',dict(cnt),' per-pair pattern (inner,before,after):',dict(pat))
        print('   non-transposition gates in before/after:',sum(1 for k in before|after if frozenset(c.units[k][:2]) not in tp))
