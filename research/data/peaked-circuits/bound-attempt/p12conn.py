import sys, collections
sys.path.insert(0,'/tmp/peaked-bound')
from solve_peaked_v2 import Core
D='/tmp/pk/research/data/peaked-circuits/'
c=Core(D+'peaked_circuit_P12_Hqap_98x2457.qasm'); maps=dict(c.maps); fA=maps[1]; fB=maps[3]
I={q:q for q in range(c.n)}
def P(si,m): lo,hi=c.secs[si]; return collections.Counter(tuple(sorted((m[c.units[k][0]],m[c.units[k][1]]))) for k in range(lo,hi+1))
comp=lambda f,g:{q:f[g[q]] for q in range(c.n)}
for nm,m in [('I',I),('fA',fA),('fB',fB),('fAfB',comp(fA,fB)),('fBfA',comp(fB,fA))]:
    for si,sj in [(2,4),(0,2),(0,4),(2,5),(4,5),(0,5)]:
        print(nm,si,sj,sum((P(si,I)&P(sj,m)).values()))
# split block A in two halves and compare with sec2/sec4
lo,hi=c.secs[1]
for cutf in [0.25,0.5,0.75]:
    m=lo+int((hi-lo)*cutf)
    A1=collections.Counter(tuple(sorted(c.units[k][:2])) for k in range(lo,m)); A2=collections.Counter(tuple(sorted(c.units[k][:2])) for k in range(m,hi+1))
    for nm,mm in [('I',I),('fA',fA),('fB',fB),('fAfB',comp(fA,fB))]:
        print('A split',cutf,nm,'A1~sec2',sum((A1&P(2,mm)).values()),'A1~sec4',sum((A1&P(4,mm)).values()),'A2~sec2',sum((A2&P(2,mm)).values()),'A2~sec4',sum((A2&P(4,mm)).values()))
