import numpy as np
from ttn import StarTTN
from scipy.stats import unitary_group
rng=np.random.default_rng(3)
blocks=[[0,3,5],[1,4,7],[2,6,8,9]]; N=10
S=StarTTN(blocks,tol=1e-12,maxa=10**6,dtype=np.complex128)
ops=[]
for t in range(80):
    r=rng.random()
    if r<0.4: ops.append(('u3',(int(rng.integers(N)),),tuple(rng.uniform(-3,3,3))))
    elif r<0.7:
        a,b=rng.choice(N,2,replace=False); ops.append(('cz',(int(a),int(b)),()))
    else:
        k=int(rng.integers(2,4)); qs=tuple(int(x) for x in rng.choice(N,k,replace=False)); ops.append(('U',qs,unitary_group.rvs(2**k,random_state=rng)))
import math
def U3(t,p,l): return np.array([[math.cos(t/2),-np.exp(1j*l)*math.sin(t/2)],[np.exp(1j*p)*math.sin(t/2),np.exp(1j*(p+l))*math.cos(t/2)]])
psi=np.zeros((2,)*N,complex); psi[(0,)*N]=1
for kind,qs,par in ops:
    W=U3(*par) if kind=='u3' else (np.diag([1,1,1,-1]).astype(complex) if kind=='cz' else par)
    k=len(qs); Wt=W.reshape((2,)*2*k)
    psi=np.tensordot(Wt,psi,axes=(list(range(k,2*k)),list(qs))); psi=np.moveaxis(psi,list(range(k)),list(qs))
for op in ops: S.apply(op)
full=S.dense().reshape((2,)*N)*np.exp(S.lognorm)
full=np.moveaxis(full,list(range(N)),blocks[0]+blocks[1]+blocks[2])
print('overlap',abs(np.vdot(full,psi)),'norm',np.linalg.norm(full),'K',S.bonds())
