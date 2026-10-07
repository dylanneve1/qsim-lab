"""Shared cross-ring (and long-range) CZ pairs between gap epochs; aggregate by e1+e2 (mirror centre) and by e2-e1 (shift)."""
from segment import *
from collections import defaultdict
import itertools, sys
E=epochs()
kinds=sys.argv[1] if len(sys.argv)>1 else 'X'
pres=defaultdict(set)
for i,g in enumerate(G):
    if g[0]=='cz' and cls(*g[1]) in kinds:
        pres[E[i]].add(frozenset(g[1]))
eps=sorted(pres)
M={}
for e1,e2 in itertools.combinations(eps,2):
    M[(e1,e2)]=len(pres[e1]&pres[e2])
bysum=defaultdict(int); bydiff=defaultdict(int); npairs=defaultdict(int)
for (e1,e2),v in M.items(): bysum[e1+e2]+=v; bydiff[e2-e1]+=v; npairs[e1+e2]+=1
print('by e1+e2 (centre=sum/2):'); print(' '.join(f'{s/2:g}:{bysum[s]}/{npairs[s]}' for s in sorted(bysum)))
print('by e2-e1:'); print(' '.join(f'{d}:{bydiff[d]}' for d in sorted(bydiff)))
print('top epoch pairs:',sorted(M.items(),key=lambda kv:-kv[1])[:40])
