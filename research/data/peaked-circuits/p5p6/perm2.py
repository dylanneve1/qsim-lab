import numpy as np,sys,collections
X=np.load(sys.argv[1]); n=int(X[:,:2].max())+1
F=X[:,[8,9,10]]; L=X[:,2].astype(int)
print('G1 |.| max',np.abs(F[:,0]+1j*F[:,1]).max(),'G2 range',F[:,2].min(),F[:,2].max())
D=np.linalg.norm(F[:,None,:]-F[None,:,:],axis=2); np.fill_diagonal(D,9)
nn=D.argmin(1); d=D.min(1)
print('nn quantiles',np.quantile(d,[0,.1,.25,.5,.75,.9]))
cons=collections.defaultdict(collections.Counter)
for i in range(len(X)):
    j=nn[i]
    if d[i]<1e-4 and L[i]<L[j]:
        a,b=X[i,:2].astype(int); c,e=X[j,:2].astype(int)
        for x in (a,b):
            for y in (c,e): cons[x][y]+=1
pi=np.array([cons[x].most_common(1)[0][0] for x in range(n)])
print('pi',pi.tolist(),'bijective',len(set(pi))==n)
np.save(sys.argv[1]+'.pi.npy',pi)
# check: for each block at layer l, find block on (pi a, pi b) and its invariant distance
pairidx=collections.defaultdict(list)
for i in range(len(X)): pairidx[tuple(sorted(X[i,:2].astype(int)))].append(i)
for l in range(L.max()+1):
    res=[]
    for i in np.where(L==l)[0]:
        a,b=X[i,:2].astype(int); key=tuple(sorted((pi[a],pi[b])))
        cand=[j for j in pairidx[key] if j!=i]
        if not cand: res.append('--'); continue
        dd=[(np.linalg.norm(F[i]-F[j]),L[j]) for j in cand]; m=min(dd)
        res.append('%.0e@%d'%m if m[0]<1e-3 else 'x')
    print(l,collections.Counter(r if r in('--','x') else 'ok@%s'%r.split('@')[1] for r in res))
