"""Block-MPS simulation (one block per ring, order ABC: A-B-C) of an op list from effcirc.py.
Ops: u3 / cz / ('U', wires, W). Canonical centre on block 1 (B). Cross-block ops via operator-Schmidt MPO, bonds recompressed (TOL).
usage: blocksim2.py OPS.pkl TOL MAXCHI [c8|c16] [tag]"""
import sys, time, math, json, pickle, numpy as np
from parse import rings
TOL=1e-6; MAXCHI=64; dt=np.complex128
blk=[]; n=[]; where={}; T=[]; trunc=0.0
def init(blocks, tol, maxchi, dtype):
    global blk,n,where,T,trunc,TOL,MAXCHI,dt
    TOL,MAXCHI,dt=tol,maxchi,dtype
    blk=blocks; n[:]=[len(b) for b in blk]; where.clear(); T[:]=[]; trunc=0.0
    for bi,b in enumerate(blk):
        for k,q in enumerate(b): where[q]=(bi,k)
    for bi in range(3):
        t=np.zeros((1,2**n[bi],1),dtype=dt); t[0,0,0]=1; T.append(t)
def U3(t,p,l): return np.array([[math.cos(t/2),-np.exp(1j*l)*math.sin(t/2)],[np.exp(1j*p)*math.sin(t/2),np.exp(1j*(p+l))*math.cos(t/2)]],dtype=dt)
def g1(bi,k,M):
    l,d,r=T[bi].shape; v=T[bi].reshape(l*2**k,2,(d>>(k+1))*r); T[bi]=np.matmul(M,v).reshape(l,d,r)
def cz_in(bi,k1,k2):
    if k1>k2: k1,k2=k2,k1
    l,d,r=T[bi].shape; v=T[bi].reshape(l*2**k1,2,2**(k2-k1-1),2,(d>>(k2+1))*r); v[:,1,:,1,:]*=-1
def apply_site(bi,O,ks):
    """O: (Dl, 2^m out, 2^m in, Dr) acting on block-local qubits ks (O's first qubit = ks[0], most significant).
    New block: (l*Dl, d, r*Dr) with combined index (l,Dl) and (r,Dr)."""
    l,d,r=T[bi].shape; nb=n[bi]; m=len(ks); Dl,_,_,Dr=O.shape
    X=T[bi].reshape((l,)+(2,)*nb+(r,))
    src=[1+k for k in ks]; X=np.moveaxis(X,src,list(range(nb+1-m,nb+1)))   # K axes just before r
    X=X.reshape(l,d>>m,2**m,r)
    Y=np.einsum('aoib,lRir->laRorb',O.astype(dt),X,optimize=True)           # (l,Dl,R,2^m,r,Dr)
    Y=Y.reshape((l*Dl,)+(2,)*nb+(r*Dr,))
    Y=np.moveaxis(Y,list(range(nb+1-m,nb+1)),src)
    T[bi]=Y.reshape(l*Dl,d,r*Dr)
def ident_pass(bi,D):
    l,d,r=T[bi].shape
    Bn=np.zeros((l,D,d,r,D),dtype=dt)
    for j in range(D): Bn[:,j,:,:,j]=T[bi]
    T[bi]=Bn.reshape(l*D,d,r*D)
def _orth_rows(M):
    Gm=(M@M.conj().T).astype(np.complex128); w,V=np.linalg.eigh(Gm); w=w[::-1]; V=V[:,::-1]
    k=max(1,int(np.sum(w>1e-14*w[0]))); s=np.sqrt(w[:k])
    Q=((V[:,:k].conj().T/s[:,None]).astype(M.dtype))@M
    return (V[:,:k]*s[None,:]).astype(M.dtype), Q
def qr_left(bi):
    l,d,r=T[bi].shape; Rm,Q=_orth_rows(T[bi].reshape(l*d,r).T)
    m=Q.shape[0]; T[bi]=Q.T.reshape(l,d,m); T[bi+1]=np.tensordot(Rm.T,T[bi+1],axes=(1,0))
def qr_right(bi):
    l,d,r=T[bi].shape; Rm,Q=_orth_rows(T[bi].reshape(l,d*r)); m=Q.shape[0]
    T[bi]=Q.reshape(m,d,r); T[bi-1]=np.tensordot(T[bi-1],Rm,axes=(2,0))
def trunc_bond(which):
    global trunc
    B=T[1]; l,d,r=B.shape
    if which==0:
        Bm=B.reshape(l,d*r); Gm=(Bm@Bm.conj().T).astype(np.complex128)
    else:
        Bm=B.reshape(l*d,r); Gm=(Bm.conj().T@Bm).astype(np.complex128)
    w,V=np.linalg.eigh(Gm); w=np.clip(w[::-1],0,None); V=V[:,::-1]
    s=np.sqrt(w); keep=max(1,min(MAXCHI,int(np.sum(s>TOL*s[0]))))
    trunc+=float(w[keep:].sum()/w.sum()); Vk=V[:,:keep].astype(dt)
    if which==0:
        T[1]=(Vk.conj().T@Bm).reshape(keep,d,r); T[0]=np.tensordot(T[0],Vk,axes=(2,0))
    else:
        T[1]=(Bm@Vk).reshape(l,d,keep); T[2]=np.tensordot(Vk.conj().T,T[2],axes=(1,0))
def mpo_split(W,groups):
    """W on concatenated qubits of groups (list of lists of block-local positions in W order). Returns per-group O (Dl,o,i,Dr)."""
    ms=[len(g) for g in groups]; k=sum(ms)
    X=W.reshape((2,)*k+(2,)*k)
    # order: out_g1,in_g1,out_g2,in_g2,...
    perm=[]; off=0
    for m in ms: perm+= list(range(off,off+m))+list(range(k+off,k+off+m)); off+=m
    X=np.transpose(X,perm); Os=[]; Dl=1; rest=X.reshape(1,-1)
    for gi,m in enumerate(ms[:-1]):
        M=rest.reshape(Dl*4**m,-1); U,S,Vh=np.linalg.svd(M,full_matrices=False)
        keep=int(np.sum(S>1e-12*S[0])); Os.append(U[:,:keep].reshape(Dl,2**m,2**m,keep)); rest=S[:keep,None]*Vh[:keep]; Dl=keep
    m=ms[-1]; Os.append(rest.reshape(Dl,2**m,2**m,1))
    # fix out/in grouping: our reshape of (out bits..., in bits...) per group already contiguous
    return Os
def apply_op(op):
    kind,qs,par=op
    if kind=='u3':
        bi,k=where[qs[0]]; g1(bi,k,U3(*par)); return
    if kind=='cz':
        W=np.diag([1,1,1,-1]).astype(complex)
    else: W=par
    bs=[where[q] for q in qs]
    blocks=sorted(set(b for b,_ in bs))
    if kind=='cz' and len(blocks)==1:
        cz_in(blocks[0],bs[0][1],bs[1][1]); return
    if len(blocks)==1:
        bi=blocks[0]; O=W.reshape(1,W.shape[0],W.shape[1],1); apply_site(bi,O,[k for _,k in bs]); return
    # reorder qubits of W by block
    k=len(qs); order=sorted(range(k),key=lambda j:(bs[j][0],j))
    Wr=W.reshape((2,)*2*k); Wr=np.transpose(Wr,order+[k+j for j in order]).reshape(2**k,2**k)
    groups=[[bs[j][1] for j in order if bs[j][0]==b] for b in blocks]
    Os=mpo_split(Wr,groups)
    for b,O,g in zip(blocks,Os,groups): apply_site(b,O,g)
    if blocks==[0,2]: ident_pass(1,Os[0].shape[3])
    if 0 in blocks: qr_left(0)
    if 2 in blocks: qr_right(2)
    if 0 in blocks or blocks==[1,2]: pass
    if 0 in blocks: trunc_bond(0)
    if 2 in blocks: trunc_bond(1)
if __name__=='__main__':
    ops=pickle.load(open(sys.argv[1],'rb'))
    tag=sys.argv[5] if len(sys.argv)>5 else 'run'
    init([rings[0],rings[1],rings[2]], float(sys.argv[2]), int(sys.argv[3]), np.complex64 if (len(sys.argv)>4 and sys.argv[4]=='c8') else np.complex128)
    t0=time.time(); hist=[]
    for i,op in enumerate(ops):
        apply_op(op)
        if i%200==0 or i==len(ops)-1:
            c1,c2=T[1].shape[0],T[1].shape[2]; hist.append((i,c1,c2,trunc))
            print(f'op {i}/{len(ops)} chi {c1} {c2} trunc {trunc:.2e} t={time.time()-t0:.0f}s', flush=True)
    pickle.dump(dict(T=T,blk=blk,hist=hist,trunc=trunc),open(f'private/bs2_{tag}.pkl','wb'))
    print('done')
