import numpy as np, sys
sys.path.insert(0,'.')
import czsimp as CS
from blocks2 import u3m
rng=np.random.default_rng(3); n=5
def rand1(q):
    r=rng.random()
    if r<0.35: return ['1',q,np.diag(np.exp(1j*rng.uniform(-3,3,2)))]
    if r<0.6: return ['1',q,np.array([[0,np.exp(1j*rng.uniform(-3,3))],[np.exp(1j*rng.uniform(-3,3)),0]])]
    return ['1',q,u3m(*rng.uniform(-3,3,3))]
ops=[]
for k in range(120):
    if rng.random()<0.5: ops.append(rand1(int(rng.integers(n))))
    else:
        a,b=rng.choice(n,2,replace=False); ops.append(['cz',(int(a),int(b))])
def sim(ops):
    U=np.eye(2**n,dtype=complex).reshape([2]*n+[2**n])
    for o in ops:
        if o[0]=='1':
            U=np.moveaxis(np.tensordot(o[2],U,axes=([1],[o[1]])),0,o[1])
        else:
            a,b=o[1]; idx=[slice(None)]*(n+1); idx[a]=1; idx[b]=1; U=U.copy(); U[tuple(idx)]*=-1
    return U.reshape(2**n,2**n)
U0=sim(ops); ops2,it=CS.simplify(n,[list(o) for o in ops],1e-9)
U1=sim(ops2)
print('cz before',sum(o[0]=='cz' for o in ops),'after',sum(o[0]=='cz' for o in ops2),'fidelity',abs(np.trace(U0.conj().T@U1))/2**n)
