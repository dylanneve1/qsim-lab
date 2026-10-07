# Validate the Pauli-path bound on a qubit window [lo,hi) of the full-depth circuit:
# exact Walsh spectrum c_S vs min anticommutation count a found by SA.
import re, sys, numpy as np
F=sys.argv[1]; lo,hi=int(sys.argv[2]),int(sys.argv[3]); D=int(sys.argv[4]) if len(sys.argv)>4 else 999
ops=[]
for line in open(F):
    line=line.strip().rstrip(';')
    if not line or line.startswith(('OPENQASM','include','qreg','creg','barrier','measure')): continue
    m=re.match(r'([a-z]+)(\(([^)]*)\))?\s+(.*)',line)
    qs=[int(x) for x in re.findall(r'q\[(\d+)\]',m.group(4))]
    ops.append((m.group(1),m.group(3),qs))
czl=[0]*70; keep=[]
for nm,arg,qs in ops:
    if nm=='cz':
        d=max(czl[q] for q in qs)+1
        for q in qs: czl[q]=d
        if d>D: continue
        if not all(lo<=q<hi for q in qs): continue
    else:
        if czl[qs[0]]>=D and nm!='cz' and False: pass
    if all(lo<=q<hi for q in qs): keep.append((nm,arg,[q-lo for q in qs]))
ops=keep; n=hi-lo
# exact state vector
psi=np.zeros(2**n,complex); psi[0]=1; psi=psi.reshape([2]*n)
s2=1/np.sqrt(2)
G={'h':np.array([[1,1],[1,-1]])*s2,'s':np.diag([1,1j]),'sdg':np.diag([1,-1j]),
   'sx':0.5*np.array([[1+1j,1-1j],[1-1j,1+1j]]),'sxdg':0.5*np.array([[1-1j,1+1j],[1+1j,1-1j]]),
   'x':np.array([[0,1],[1,0]]),'z':np.diag([1,-1])}
for nm,arg,qs in ops:
    if nm=='cz':
        a,b=qs; idx=[slice(None)]*n; idx[a]=1; idx[b]=1; psi[tuple(idx)]*=-1
    else:
        U=np.diag([1,np.exp(1j*eval(arg.replace('pi','np.pi')))]) if nm=='rz' else G[nm]
        q=qs[0]; psi=np.moveaxis(np.tensordot(U,psi,axes=([1],[q])),0,q)
# qubit i <-> axis i ; p over x with bit i = qubit i
p=np.abs(psi)**2
# Walsh transform
c=p.copy()
for ax in range(n):
    c0=c.take(0,axis=ax); c1=c.take(1,axis=ax)
    c=np.stack([c0+c1,c0-c1],axis=ax)
c=c.reshape(-1); # index bits: axis order -> S as big-endian over axes
nT=sum(1 for o in ops if o[0]=='rz')
cs=np.abs(c[1:]); 
print(f"window {lo}-{hi} n={n} T={nT}  max|c_S|={cs.max():.3e}  sum c^2={np.sum(c[1:]**2):.3f}  #|c|>1e-9: {(cs>1e-9).sum()} of {len(cs)}  rms={np.sqrt(np.mean(cs**2)):.2e}")
# path analysis
NS=n+nT; X=np.zeros((NS,n),np.uint8); Z=np.zeros((NS,n),np.uint8)
for q in range(n): Z[q,q]=1
M0=np.zeros((nT,NS),np.uint8); k=0
for nm,arg,qs in ops:
    if nm=='h': q=qs[0]; X[:,q],Z[:,q]=Z[:,q].copy(),X[:,q].copy()
    elif nm in('sx','sxdg'): q=qs[0]; X[:,q]^=Z[:,q]
    elif nm in('s','sdg'): q=qs[0]; Z[:,q]^=X[:,q]
    elif nm=='cz': a,b=qs; Z[:,a]^=X[:,b]; Z[:,b]^=X[:,a]
    elif nm=='rz': q=qs[0]; M0[k]=X[:,q]; Z[n+k,q]=1; k+=1
Lx=X.T; Lz=Z.T
# rank of LA
A=Lx[:,:n].copy(); r=0
for col in range(n):
    pv=[i for i in range(r,n) if A[i,col]]
    if not pv: continue
    A[[r,pv[0]]]=A[[pv[0],r]]
    for i in range(n):
        if i!=r and A[i,col]: A[i]^=A[r]
    r+=1
print("clifford X-rank",r)
if r<n: sys.exit()
def solve(A,B):
    A=A.copy();B=B.copy();m=A.shape[0]
    for col in range(m):
        pv=next(i for i in range(col,m) if A[i,col]); A[[col,pv]]=A[[pv,col]]; B[[col,pv]]=B[[pv,col]]
        for i in range(m):
            if i!=col and A[i,col]: A[i]^=A[col]; B[i]^=B[col]
    return B
AinvLT=solve(Lx[:,:n],Lx[:,n:])
M=((M0[:,:n].astype(np.int64)@AinvLT+M0[:,n:])%2)
Sm=((Lz[:,:n].astype(np.int64)@AinvLT+Lz[:,n:])%2)
rng=np.random.default_rng(1)
# enumerate small J exhaustively (|J|<=2) and SA
res={}
def evalJ(y):
    v=(M@y)%2
    if ((1-v)*y).sum(): return None
    S=(Sm@y)%2
    if not S.any(): return None
    return int(v.sum()), tuple(S)
from itertools import combinations
for J in list(combinations(range(nT),1))+list(combinations(range(nT),2)):
    y=np.zeros(nT,np.int64); y[list(J)]=1; e=evalJ(y)
    if e: a,S=e; res[S]=min(res.get(S,999),a)
best=sorted(res.items(),key=lambda kv:kv[1])[:8]
amin=best[0][1] if best else None
print("min a (|J|<=2):",amin, " => 2^{-a/2} =",2**(-amin/2) if amin is not None else None)
for S,a in best:
    idx=int(''.join(str(b) for b in S),2)  # axis0 is most significant in reshape(-1)
    print(f"  a={a} |S|={sum(S)} 2^-a/2={2**(-a/2):.3e}  exact|c_S|={abs(c[idx]):.3e}")
# what is the a-value of the actual top exact coefficients?
top=np.argsort(-np.abs(c))[1:6]
print("top exact |c|:",[f"{abs(c[i]):.3e}" for i in top])
