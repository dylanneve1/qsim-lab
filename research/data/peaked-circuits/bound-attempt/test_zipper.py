import numpy as np, sys
sys.path.insert(0,'/tmp/peaked-bound')
from zipper import Zipper
rng=np.random.default_rng(1)
def haar(d):
    z=(rng.normal(size=(d,d))+1j*rng.normal(size=(d,d)))/np.sqrt(2); q,r=np.linalg.qr(z); return q*(np.diag(r)/abs(np.diag(r)))
SW=np.eye(4)[[0,2,1,3]].astype(complex)
n=12; V=[]
for _ in range(40):
    a,b=rng.choice(n,2,replace=False); V.append(((int(a),int(b)),haar(4)))
p=rng.permutation(n); sig=list(range(n)); S=[]
for i in range(0,n,2):
    a,b=int(p[i]),int(p[i+1]); sig[a],sig[b]=b,a; S.append(((a,b),SW))
W=[((sig[a],sig[b]),U.conj().T) for (a,b),U in reversed(V)]
for eps in [0,1e-3]:
    W2=[(w,U@ (np.eye(4)+eps*1j*np.diag(rng.normal(size=4)))) for w,U in W]
    z=Zipper(V+S+W2,len(V),n,kmax=8,tau=1e-9 if eps==0 else 1e-2).run()
    print('eps',eps,'cost',z.cost,'Pi==sig',z.Pi==sig,'maxk',z.maxk,'forced',z.forced,'npeel',len(z.costs))
