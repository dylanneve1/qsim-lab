"""If the effective circuit has no cross-ring op: exact per-ring statevector simulation (A, B: 2^20, C: 2^22), complex128.
usage: ringsim.py EFF.pkl TAG -> private/ring_{TAG}_{r}.npy (final ring states) and summary"""
import sys, pickle, math, numpy as np, time
from parse import rings
ops=pickle.load(open(sys.argv[1],'rb')); tag=sys.argv[2]
where={q:(r,k) for r,b in enumerate(rings) for k,q in enumerate(b)}
def U3(t,p,l): return np.array([[math.cos(t/2),-np.exp(1j*l)*math.sin(t/2)],[np.exp(1j*p)*math.sin(t/2),np.exp(1j*(p+l))*math.cos(t/2)]])
cross=[op for op in ops if len({where[q][0] for q in op[1]})>1]
print('cross-ring ops:',len(cross),[op[1] for op in cross][:20],flush=True)
touched={where[q][0] for op in cross for q in op[1]}
print('rings touched by cross ops',touched)
for r in [x for x in range(3) if x not in touched]:
    n=len(rings[r]); psi=np.zeros(2**n,complex); psi[0]=1; t0=time.time()
    for kind,qs,par in ops:
        if where[qs[0]][0]!=r: continue
        ks=[where[q][1] for q in qs]
        if kind=='u3':
            k=ks[0]; v=psi.reshape(2**k,2,-1); psi=np.matmul(U3(*par),v).reshape(-1)
        elif kind=='cz':
            k1,k2=sorted(ks); v=psi.reshape(2**k1,2,2**(k2-k1-1),2,-1); v[:,1,:,1,:]*=-1
        else:
            m=len(ks); X=psi.reshape((2,)*n); Mt=par.reshape((2,)*(2*m))
            xl=list(range(n)); ol=list(range(n))
            for j,k in enumerate(ks): xl[k]=n+m+j; ol[k]=n+j
            psi=np.einsum(Mt,[n+j for j in range(m)]+[n+m+j for j in range(m)],X,xl,ol).reshape(-1)
    p=np.abs(psi)**2; top=np.argsort(p)[::-1][:5]
    print('ring',r,'n',n,'norm',p.sum(),'top probs',p[top],'t=%.0fs'%(time.time()-t0),flush=True)
    np.save(f'private/ring_{tag}_{r}.npy',psi)
