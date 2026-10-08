import os
ZZP=float(os.environ.get("SU2_ZZP","0.1130625"))
from parse import load
from collections import defaultdict
def logical_map(path):
    ops=load(path); k=0
    while ops[k][0]=='x': k+=1
    lab=list(range(120)); last={}; hop=set(); zz=set(); prev=None
    for name,p,qs in ops[k:]:
        if name=='swap':
            a,b=qs; lab[a],lab[b]=lab[b],lab[a]
        elif name=='cx':
            pr=tuple(sorted((lab[qs[0]],lab[qs[1]]))); last[qs[0]]=pr; last[qs[1]]=pr
        elif name=='rz':
            if abs(abs(p)-0.15)<1e-12: hop.add(last[qs[0]])
            elif abs(p+ZZP)<1e-9 and prev=='cx': zz.add(last[qs[0]])
        prev=name
    deg=defaultdict(set)
    for a,b in hop: deg[a].add(b); deg[b].add(a)
    seen=set(); chains=[]
    for s in range(120):
        if s in seen or len(deg[s])!=1: continue
        c=[s]; seen.add(s)
        while True:
            nx=[n for n in deg[c[-1]] if n not in seen]
            if not nx: break
            c.append(nx[0]); seen.add(nx[0])
        chains.append(c)
    assert len(chains)==2 and all(len(c)==60 for c in chains)
    chains.sort(key=lambda c:c[0])   # chain containing wire 0 = 'i'
    partner={}
    for a,b in zz: partner[a]=b; partner[b]=a
    m={}
    for r in range(60):
        m[chains[0][r]]=(r,0); m[chains[1][r]]=(r,1)
        assert partner[chains[0][r]]==chains[1][r]
    return m
