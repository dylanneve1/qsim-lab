import sys, numpy as np, collections
import gparse as G
n,units,tail=G.parse(sys.argv[1]); K=len(units)
pairs=[(a,b) for a,b,_,_ in units]
print('first 60 pairs:', pairs[:60])
print('around middle:', [(k,pairs[k]) for k in range(930,990)])
# runs with small qubit support
for kmax in (4,6,8):
    runs=[]; cur=set(); start=0
    for k,(a,b) in enumerate(pairs):
        s=cur|{a,b}
        if len(s)>kmax: runs.append((start,k-start,len(cur))); cur={a,b}; start=k
        else: cur=s
    runs.append((start,K-start,len(cur)))
    L=[r[1] for r in runs]
    print('kmax',kmax,'runs',len(runs),'mean len',np.mean(L),'hist',collections.Counter(L).most_common(8))
