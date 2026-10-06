"""Prototype: recover a hidden wire permutation from an operator W ~ L1 · Pi · M · L2 (M = lossy near-identity)
by the Pauli-transfer score  S[i,j] = (1/9) sum_{P,Q in XYZ} |Tr(W P_i W^dag Q_j)|^2 / d^2  (=1 if Pi(i)=j for
exact Pi·local, 0 otherwise), then the assignment problem. Dense, n<=10, synthetic."""
import numpy as np, itertools
from scipy.optimize import linear_sum_assignment
rng=np.random.default_rng(7)
X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1.,-1.]).astype(complex); I=np.eye(2)
def haar2(): 
    A=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)); q,r=np.linalg.qr(A); return q
def op1(n,q,M):
    out=np.array([[1.]]); 
    for k in range(n): out=np.kron(out, M if k==q else I)
    return out
def op2(n,a,b,M4):
    # dense 2q op on (a,b) via permutation trick
    d=2**n; out=np.zeros((d,d),complex)
    T=M4.reshape(2,2,2,2)
    for x in range(d):
        bits=[(x>>(n-1-k))&1 for k in range(n)]
        for oa in range(2):
            for ob in range(2):
                v=T[oa,ob,bits[a],bits[b]]
                if v==0: continue
                nb=bits.copy(); nb[a]=oa; nb[b]=ob
                y=int(''.join(map(str,nb)),2); out[y,x]+=v
    return out
CZ=np.diag([1,1,1,-1]).astype(complex)
def rand_layer(n,ops):
    for q in range(n): ops.append(op1(n,q,haar2()))
    perm=rng.permutation(n)
    for k in range(0,n-1,2): ops.append(op2(n,int(perm[k]),int(perm[k+1]),CZ))
def score(W,n):
    d=2**n; S=np.zeros((n,n))
    for i in range(n):
        for P in (X,Y,Z):
            WP=W@op1(n,i,P)@W.conj().T
            for j in range(n):
                for Q in (X,Y,Z):
                    S[i,j]+=abs(np.trace(WP@op1(n,j,Q)))**2/d**2
    return S/3
n=8
for noise in (0.0,0.02,0.05,0.1):
    ops=[]
    for _ in range(4): rand_layer(n,ops)
    V=np.eye(2**n)
    for o in ops: V=o@V
    pi=rng.permutation(n)
    # permutation as dense
    d=2**n; Pm=np.zeros((d,d))
    for x in range(d):
        bits=[(x>>(n-1-k))&1 for k in range(n)]; nb=[0]*n
        for k in range(n): nb[pi[k]]=bits[k]
        Pm[int(''.join(map(str,nb)),2),x]=1
    # lossy inverse: V^dag with each 1q gate perturbed by exp(i*noise*H)
    def perturb(o):
        if o.shape==(2**n,2**n) and np.allclose(np.abs(o)@np.ones(2**n), np.abs(o)@np.ones(2**n)) and not np.allclose(o, np.diag(np.diag(o))):
            H=rng.normal(size=(d,d))*0; return o
        return o
    Vd=np.eye(d)
    k1=0
    for o in ops:
        if not np.allclose(o,np.diag(np.diag(o))):   # 1q layer gate (non-diagonal) -> perturb by a random small 1q rotation on a random qubit
            q=int(rng.integers(n)); H=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)); H=(H+H.conj().T)/2; w,v=np.linalg.eigh(H)
            o=op1(n,q,v@np.diag(np.exp(1j*noise*w))@v.conj().T)@o
        Vd=Vd@o.conj().T
    Npert=np.eye(d)
    for q in range(n):
        H=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)); H=(H+H.conj().T)/2
        w,v=np.linalg.eigh(H); Npert=op1(n,q,v@np.diag(np.exp(1j*noise*w))@v.conj().T)@Npert
    L1=np.eye(1); L2=np.eye(1)
    for q in range(n): L1=np.kron(L1,haar2()); L2=np.kron(L2,haar2())
    W=L1@Pm@Vd@V@L2
    fid=abs(np.trace(Vd@V))/d      # = local·Pi·(noise)·local  (V^dag V = I)
    S=score(W,n)
    r,c=linear_sum_assignment(-S)
    rec=np.array(c)
    print(f'noise {noise}: trace fidelity of V^dag V {fid:.3f}; recovered pi correct: {np.array_equal(rec,pi)}   diag score min {S[r,c].min():.3f}  max off-diag {np.max(S-np.where(np.eye(n)[:, :][r][:,None]==0,0,0) * 0 - np.isin(np.arange(n)[None,:],[]) ) if False else np.max(np.where(np.eye(n,dtype=bool)[np.argsort(np.argsort(c))], -1, S)):.3f}')
