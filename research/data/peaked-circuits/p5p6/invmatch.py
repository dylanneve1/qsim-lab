import numpy as np,sys
X=np.load(sys.argv[1]); 
print('G1 imag max',np.abs(X[:,8]).max())
F=X[:,[7,8,9]]
D=np.linalg.norm(F[:,None,:]-F[None,:,:],axis=2); np.fill_diagonal(D,9)
nn=D.argmin(1); d=D.min(1)
print('nn dist quantiles',np.quantile(d,[0,.1,.25,.5,.75,.9,1]))
tight=d<1e-6
print('blocks with exact invariant partner',tight.sum(),'of',len(X))
L=X[:,2].astype(int)
import collections
c=collections.Counter()
for i in np.where(tight)[0]:
    j=nn[i]; c[(L[i]+L[j])]+=1
print('layer-sum histogram of exact partners',sorted(c.items()))
for thr in [1e-5,1e-4,1e-3]:
    t=d<thr; c=collections.Counter()
    for i in np.where(t)[0]: c[L[i]+L[nn[i]]]+=1
    print(thr,t.sum(),sorted(c.items()))
# per layer: fraction with partner<1e-4 at layer sum 40/41/42
for l in range(L.max()+1):
    idx=np.where(L==l)[0]
    print(l,len(idx),' '.join('%.0e@%d'%(d[i],L[nn[i]]) for i in idx[:8]))
