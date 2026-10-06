import numpy as np,sys,collections
X=np.load(sys.argv[1]); pi=np.load(sys.argv[1]+'.pi.npy'); n=len(pi)
F=X[:,[8,9,10]]; L=X[:,2].astype(int)
special=lambda f: min(np.linalg.norm(f-np.array(s)) for s in [(0,0,1),(0,0,-1),(1,0,3),(-1,0,-3)])
pairidx=collections.defaultdict(list)
for i in range(len(X)): pairidx[tuple(sorted(X[i,:2].astype(int)))].append(i)
matched=np.zeros(len(X),int)-1
dist=np.zeros(len(X))
for i in range(len(X)):
    if L[i]>20: continue
    a,b=X[i,:2].astype(int); key=tuple(sorted((pi[a],pi[b])))
    cand=[j for j in pairidx[key] if L[j]>L[i] and matched[j]<0]
    if not cand: continue
    dd=[np.linalg.norm(F[i]-F[j]) for j in cand]; k=int(np.argmin(dd))
    if dd[k]<1e-3 and not (special(F[i])<0.05 and 17<=L[i]<=24): matched[i]=cand[k]; matched[cand[k]]=i; dist[i]=dist[cand[k]]=dd[k]
print('matched pairs',(matched>=0).sum()//2,'dist quantiles',np.quantile(dist[matched>=0],[0,.5,.9,1]))
np.save(sys.argv[1]+'.match2.npy',matched)
un=np.where(matched<0)[0]
print('unmatched blocks',len(un),'by layer',sorted(collections.Counter(L[un]).items()))
for i in un:
    if 15<=L[i]<=27:
        w=X[i,4:8]
        print(L[i],X[i,:2].astype(int),'ncz',int(X[i,3]),'G1 %.4f%+.4fi G2 %.4f'%tuple(F[i]))
