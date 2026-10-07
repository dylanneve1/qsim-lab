import numpy as np, blocksim2 as bs
from scipy.stats import unitary_group
rng=np.random.default_rng(1)
blocks=[[0,3,5],[1,4,7],[2,6,8,9]]; N=10
bs.init(blocks,1e-12,10**6,np.complex128)
ops=[]
for t in range(60):
    r=rng.random()
    if r<0.4: ops.append(('u3',(int(rng.integers(N)),),tuple(rng.uniform(-3,3,3))))
    elif r<0.7:
        a,b=rng.choice(N,2,replace=False); ops.append(('cz',(int(a),int(b)),()))
    else:
        k=int(rng.integers(2,4)); qs=tuple(int(x) for x in rng.choice(N,k,replace=False)); ops.append(('U',qs,unitary_group.rvs(2**k,random_state=rng)))
# dense reference (qubit q = axis q)
psi=np.zeros((2,)*N,complex); psi[(0,)*N]=1
for kind,qs,par in ops:
    if kind=='u3': W=bs.U3(*par).astype(complex)
    elif kind=='cz': W=np.diag([1,1,1,-1]).astype(complex)
    else: W=par
    k=len(qs); Wt=W.reshape((2,)*2*k)
    psi=np.tensordot(Wt,psi,axes=(list(range(k,2*k)),list(qs)))
    psi=np.moveaxis(psi,list(range(k)),list(qs))
for op in ops: bs.apply_op(op)
# contract blocks
A,B,C=bs.T
full=np.einsum('lxa,ayb,bzr->xyz',A,B,C)
full=full.reshape((2,)*N)
# block-local qubit order -> wires
axes=blocks[0]+blocks[1]+blocks[2]
full=np.moveaxis(full,list(range(N)),axes)
print('overlap',abs(np.vdot(full,psi)),'norm',np.linalg.norm(full),'chi',B.shape[0],B.shape[2])
