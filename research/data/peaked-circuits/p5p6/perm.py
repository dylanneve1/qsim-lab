import numpy as np,sys,collections
X=np.load(sys.argv[1]); thr=float(sys.argv[2]) if len(sys.argv)>2 else 1e-4
F=X[:,[7,8,9]]; L=X[:,2].astype(int)
D=np.linalg.norm(F[:,None,:]-F[None,:,:],axis=2); np.fill_diagonal(D,9)
nn=D.argmin(1); d=D.min(1)
pairs=[]
for i in range(len(X)):
    j=nn[i]
    if d[i]<thr and L[i]<L[j] and nn[j]==i:
        pairs.append((i,j))
print('mutual matched pairs',len(pairs))
# edge mapping: {a,b} -> {a',b'}; collect wire-level constraints
cons=collections.defaultdict(collections.Counter)
for i,j in pairs:
    a,b=int(X[i,0]),int(X[i,1]); c,e=int(X[j,0]),int(X[j,1])
    for x in (a,b):
        for y in (c,e): cons[x][y]+=1
n=int(X[:,:2].max())+1
pi={}
for x in range(n):
    if cons[x]:
        top=cons[x].most_common(3); print(x,top, 'layers', sorted(set(L[i] for i,j in pairs if x in X[i,:2]))[:20])
