# Two-branch classical evaluation of a permutation circuit with one H on |0>.
import sys,time,re
t0=time.perf_counter()
L=[l.strip().rstrip(';') for l in open(sys.argv[1]) if l[:2] in ('h ','x ','cx','cc')]
ops=[]
for l in L:
    g,args=l.split(' ',1); q=[int(x) for x in re.findall(r'\[(\d+)\]',args)]; ops.append((g,q))
t1=time.perf_counter()
br={0:1.0}
for g,q in ops:
    if g=='h':
        assert len(br)==1 and all(not (k>>q[0])&1 for k in br)
        br={k:a for k0,a0 in br.items() for k,a in ((k0,a0/2**.5),(k0|1<<q[0],a0/2**.5))}
    elif g=='x': br={k^(1<<q[0]):a for k,a in br.items()}
    elif g=='cx': br={(k^(1<<q[1])) if (k>>q[0])&1 else k:a for k,a in br.items()}
    elif g=='ccx': br={(k^(1<<q[2])) if (k>>q[0])&1 and (k>>q[1])&1 else k:a for k,a in br.items()}
t2=time.perf_counter()
print('branches',len(br),'parse %.3fs eval %.3fs'%(t1-t0,t2-t1))
N=4611686014132420609; a=7
for k in br:
    x=sum(((k>>(1+j))&1)<<j for j in range(62)); print('ctrl',k&1,'x',x, 'ok' if x==(a if k&1 else 1) else 'BAD', 'anc_clean', (k>>63)==0)
