"""Block-MPS Schrodinger simulation of P6 with one block per ring: [A (20q)] - [B (20q)] - [C (22q)] (block order configurable).
Block tensor (l, 2^n, r). Canonical centre always on the middle block. Ring-internal ops are exact; cross-ring CZs expand
the inter-block bonds by 2 and are recompressed by SVD (relative cutoff TOL). Reports bond dims over time.
usage: blocksim.py ORDER(e.g. ABC) TOL MAXCHI [dtype c8|c16] [stop_op]"""
import sys, time, math, json, numpy as np
from parse import load, rings
G=load()
order=sys.argv[1]; TOL=float(sys.argv[2]); MAXCHI=int(sys.argv[3]); dt=np.complex64 if (len(sys.argv)>4 and sys.argv[4]=='c8') else np.complex128
STOP=int(sys.argv[5]) if len(sys.argv)>5 else len(G)
R={'A':rings[0],'B':rings[1],'C':rings[2]}
blk=[R[c] for c in order]; n=[len(b) for b in blk]
where={}
for bi,b in enumerate(blk):
    for k,q in enumerate(b): where[q]=(bi,k)
T=[]
for bi in range(3):
    t=np.zeros((1,2**n[bi],1),dtype=dt); t[0,0,0]=1; T.append(t)
trunc=0.0
def U3(t,p,l): return np.array([[math.cos(t/2),-np.exp(1j*l)*math.sin(t/2)],[np.exp(1j*p)*math.sin(t/2),np.exp(1j*(p+l))*math.cos(t/2)]],dtype=dt)
def view(bi,k):
    l,d,r=T[bi].shape; return T[bi].reshape(l*2**k,2,(d>>(k+1))*r)
def g1(bi,k,M):
    l,d,r=T[bi].shape; v=view(bi,k)
    T[bi]=np.matmul(M,v).reshape(l,d,r)
def cz_in(bi,k1,k2):
    if k1>k2: k1,k2=k2,k1
    l,d,r=T[bi].shape
    v=T[bi].reshape(l*2**k1,2,2**(k2-k1-1),2,(d>>(k2+1))*r)
    v[:,1,:,1,:]*=-1
def _orth_rows(M):
    """M (m x N) = Rm @ Q with Q having orthonormal rows; via small Gram eigh. Returns Rm (m x k), Q (k x N)."""
    Gm=(M@M.conj().T).astype(np.complex128); w,V=np.linalg.eigh(Gm); w=w[::-1]; V=V[:,::-1]
    k=max(1,int(np.sum(w>1e-14*w[0]))); s=np.sqrt(w[:k])
    Q=((V[:,:k].conj().T/s[:,None]).astype(M.dtype))@M
    return (V[:,:k]*s[None,:]).astype(M.dtype), Q
def qr_left(bi):     # make block bi left-orthonormal (columns), push R right
    l,d,r=T[bi].shape
    Rm,Q=_orth_rows(T[bi].reshape(l*d,r).T)     # A^T = Rm Q  -> A = Q^T Rm^T
    m=Q.shape[0]; T[bi]=Q.T.reshape(l,d,m); T[bi+1]=np.tensordot(Rm.T,T[bi+1],axes=(1,0))
def qr_right(bi):    # make block bi right-orthonormal (rows), push R left
    l,d,r=T[bi].shape
    Rm,Q=_orth_rows(T[bi].reshape(l,d*r)); m=Q.shape[0]
    T[bi]=Q.reshape(m,d,r); T[bi-1]=np.tensordot(T[bi-1],Rm,axes=(2,0))
def trunc_bond(which):
    global trunc
    B=T[1]; l,d,r=B.shape
    if which==0:
        Bm=B.reshape(l,d*r); Gm=(Bm@Bm.conj().T).astype(np.complex128)
        w,V=np.linalg.eigh(Gm); w=np.clip(w[::-1],0,None); V=V[:,::-1]
        s=np.sqrt(w); keep=max(1,min(MAXCHI,int(np.sum(s>TOL*s[0]))))
        trunc+=float(w[keep:].sum()/w.sum()); Vk=V[:,:keep].astype(dt)
        T[1]=(Vk.conj().T@Bm).reshape(keep,d,r); T[0]=np.tensordot(T[0],Vk,axes=(2,0))
    else:
        Bm=B.reshape(l*d,r); Gm=(Bm.conj().T@Bm).astype(np.complex128)
        w,V=np.linalg.eigh(Gm); w=np.clip(w[::-1],0,None); V=V[:,::-1]
        s=np.sqrt(w); keep=max(1,min(MAXCHI,int(np.sum(s>TOL*s[0]))))
        trunc+=float(w[keep:].sum()/w.sum()); Vk=V[:,:keep].astype(dt)
        T[1]=(Bm@Vk).reshape(l,d,keep); T[2]=np.tensordot(Vk.conj().T,T[2],axes=(1,0))
def cross_cz(b1,k1,b2,k2):
    if b1>b2: b1,k1,b2,k2=b2,k2,b1,k1
    # left block b1: projectors on k1, new right index (r,k)
    l,d,r=T[b1].shape; v=T[b1].reshape(l*2**k1,2,(d>>(k1+1))*r)
    P=np.zeros(v.shape+(2,),dtype=dt) if False else None
    a0=v.copy(); a0[:,1,:]=0; a1=v.copy(); a1[:,0,:]=0
    X=np.stack([a0.reshape(l,d,r),a1.reshape(l,d,r)],axis=-1).reshape(l,d,2*r)
    # right block b2: Z^k on k2, new left index (l,k)
    l2,d2,r2=T[b2].shape; Y0=T[b2]; Y1=Y0.copy(); vv=Y1.reshape(l2*2**k2,2,(d2>>(k2+1))*r2); vv[:,1,:]*=-1
    Y=np.stack([Y0,Y1],axis=1).reshape(2*l2,d2,r2)
    if b2-b1==2:
        Bm=T[1]; lb,db,rb=Bm.shape
        Bn=np.zeros((lb,2,db,rb,2),dtype=dt); Bn[:,0,:,:,0]=Bm; Bn[:,1,:,:,1]=Bm
        T[1]=Bn.reshape(2*lb,db,2*rb)
    T[b1]=X; T[b2]=Y
    if b1==0: qr_left(0)
    if b2==2: qr_right(2)
    if b1==0: trunc_bond(0)
    if b2==2: trunc_bond(1)
t0=time.time(); hist=[]
for i,g in enumerate(G[:STOP]):
    if g[0]=='u3':
        bi,k=where[g[1][0]]; g1(bi,k,U3(*g[2]))
    else:
        (b1,k1),(b2,k2)=where[g[1][0]],where[g[1][1]]
        if b1==b2: cz_in(b1,k1,k2)
        else: cross_cz(b1,k1,b2,k2)
    if i%100==0 or i==STOP-1:
        c1,c2=T[1].shape[0],T[1].shape[2]; hist.append((i,c1,c2,trunc))
        print(f'op {i} chi {c1} {c2} trunc {trunc:.2e} t={time.time()-t0:.0f}s', flush=True)
np.save(f'private/blocksim_{order}_{TOL}_{MAXCHI}.npy', np.array([T[0],T[1],T[2]],dtype=object), allow_pickle=True)
json.dump(dict(order=order,tol=TOL,maxchi=MAXCHI,hist=hist,blocks=blk),open(f'private/blocksim_{order}_{TOL}_{MAXCHI}.json','w'))
print('done')
