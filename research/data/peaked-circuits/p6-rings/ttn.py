"""Star tree tensor network: one leaf per ring L[r] (2^n_r x a_r, orthonormal columns), core K (a0,a1,a2).
Ring-internal ops are exact on the leaves; cross-ring ops are split into an MPO over rings (operator Schmidt), expanded on the
leaves without materialising the expanded leaf (Gram trick), then the legs are truncated by SVD of the core (TOL / MAXA).
Ops: ('u3',(q,),params) | ('cz',(a,b),()) | ('U',wires,W)."""
import numpy as np, math
class StarTTN:
    def __init__(s, blocks, tol=1e-5, maxa=32, dtype=np.complex64):
        s.blk=blocks; s.n=[len(b) for b in blocks]; s.tol=tol; s.maxa=maxa; s.dt=dtype
        s.where={q:(r,k) for r,b in enumerate(blocks) for k,q in enumerate(b)}
        s.L=[]
        for r in range(3):
            v=np.zeros((2**s.n[r],1),dtype=dtype); v[0,0]=1; s.L.append(v)
        s.K=np.ones((1,1,1),dtype=np.complex128); s.trunc=0.0; s.lognorm=0.0
    # ---- leaf ops ----
    def _apply_leaf(s, L, r, ks, M):
        """apply dense M (2^m x 2^m) on leaf-local qubits ks to the columns of L."""
        n=s.n[r]; a=L.shape[1]; m=len(ks)
        X=L.reshape((2,)*n+(a,))
        X=np.moveaxis(X,ks,list(range(m)))
        sh=X.shape
        Y=(M.astype(s.dt)@X.reshape(2**m,-1)).reshape(sh)
        return np.ascontiguousarray(np.moveaxis(Y,list(range(m)),ks)).reshape(2**n,a)
    def u3(s,q,M):
        r,k=s.where[q]; n=s.n[r]; a=s.L[r].shape[1]
        v=s.L[r].reshape(2**k,2,(2**(n-k-1))*a); s.L[r]=np.matmul(M.astype(s.dt),v).reshape(2**n,a)
    def cz_in(s,r,k1,k2):
        if k1>k2: k1,k2=k2,k1
        n=s.n[r]; a=s.L[r].shape[1]
        v=s.L[r].reshape(2**k1,2,2**(k2-k1-1),2,(2**(n-k2-1))*a); v[:,1,:,1,:]*=-1
    # ---- cross op ----
    def cross(s, W, qs):
        bs=[s.where[q] for q in qs]; rs=sorted(set(r for r,_ in bs)); k=len(qs)
        order=sorted(range(k),key=lambda j:(bs[j][0],j))
        Wr=W.reshape((2,)*2*k); Wr=np.transpose(Wr,order+[k+j for j in order]).reshape(2**k,2**k)
        groups=[[bs[j][1] for j in order if bs[j][0]==r] for r in rs]
        ms=[len(g) for g in groups]
        X=Wr.reshape((2,)*2*k); perm=[]; off=0
        for m in ms: perm+=list(range(off,off+m))+list(range(k+off,k+off+m)); off+=m
        X=np.transpose(X,perm); Os=[]; Dl=1; rest=X.reshape(1,-1)
        for m in ms[:-1]:
            Mx=rest.reshape(Dl*4**m,-1); U,S,Vh=np.linalg.svd(Mx,full_matrices=False)
            kp=int(np.sum(S>1e-12*S[0])); Os.append(U[:,:kp].reshape(Dl,2**m,2**m,kp)); rest=S[:kp,None]*Vh[:kp]; Dl=kp
        Os.append(rest.reshape(Dl,2**ms[-1],2**ms[-1],1))
        # expanded core: leg r -> (a_r, Dl_r, Dr_r) ; deltas between consecutive Dr / Dl
        K=s.K
        if len(rs)==2:
            D=Os[0].shape[3]; r1,r2=rs
            # K[a0,a1,a2] -> leg r1 gets d, leg r2 gets d (delta)
            Kn=np.einsum('abc,de->abcde',K,np.eye(D))   # d on r1, e on r2
            legs={0:'a',1:'b',2:'c'}; Ks=Kn
            # move: leg r1 index pair (a_r1,d), leg r2 (a_r2,e)
            ax=[0,1,2]; src=Ks
            src=np.moveaxis(src,3,r1+1)      # place d right after leg r1
            # now shape has 5 axes; e is last; place after leg r2 (which shifted by 1 if r2>r1)
            pos_r2=r2+1 if r2>r1 else r2
            src=np.moveaxis(src,-1,pos_r2+1)
            sh=list(K.shape); sh[r1]*=D; sh[r2]*=D
            K=src.reshape(sh)
            exp={r1:[(Os[0],'r')], r2:[(Os[1],'l')]}
        else:
            D1=Os[0].shape[3]; D2=Os[1].shape[3]
            Kn=np.einsum('abc,de,fg->adbegcf'.replace('adbegcf','adbegcf'),K,np.eye(D1),np.eye(D2))
            # axes: a,d | b,e,f | c,g  -> need (a,d),(b,e,f),(c,g): einsum output a d b e f? fix explicitly
            Kn=np.einsum('abc,de,fg->adbefcg',K,np.eye(D1),np.eye(D2))
            sh=[K.shape[0]*D1,K.shape[1]*D1*D2,K.shape[2]*D2]; K=Kn.reshape(sh)
            exp={0:[(Os[0],'r')],1:[(Os[1],'m')],2:[(Os[2],'l')]}
        s.K=K
        pend={}
        for gi,r in enumerate(rs):
            pend[r]=s._gram_leaf(r,Os[gi],groups[gi])
        for gi,r in enumerate(rs):
            U=s._truncate_core(r)
            Ops,Cmat,a,T,ks=pend[r]
            CU=(Cmat@U).astype(s.dt)
            L=s.L[r]; Q=None
            for t in range(T):
                Y=s._apply_leaf(L@CU[t*a:(t+1)*a],r,ks,Ops[t])
                if Q is None: Q=Y
                else: Q+=Y
                del Y
            s.L[r]=Q
    def _gram_leaf(s, r, O, ks):
        L=s.L[r]; a=L.shape[1]; Dl,_,_,Dr=O.shape
        terms=[(i,j) for i in range(Dl) for j in range(Dr)]; T=len(terms)
        Ops=[O[i,:,:,j] for i,j in terms]
        Gm=np.zeros((T*a,T*a),dtype=np.complex128)
        for x in range(T):
            for y in range(x,T):
                Z=s._apply_leaf(L,r,ks,Ops[x].conj().T@Ops[y])
                g=(L.conj().T@Z).astype(np.complex128); del Z
                Gm[x*a:(x+1)*a,y*a:(y+1)*a]=g
                if y!=x: Gm[y*a:(y+1)*a,x*a:(x+1)*a]=g.conj().T
        w,V=np.linalg.eigh((Gm+Gm.conj().T)/2); w=w[::-1]; V=V[:,::-1]
        kp=max(1,int(np.sum(w>1e-12*w[0]))); sw=np.sqrt(w[:kp])
        Cmat=V[:,:kp]/sw[None,:]; Rm=(sw[:,None]*V[:,:kp].conj().T)
        K=np.moveaxis(s.K,r,0); sh=K.shape
        Kl=K.reshape(a,T,-1).transpose(1,0,2).reshape(T*a,-1)
        K=(Rm@Kl).reshape((kp,)+sh[1:]); s.K=np.moveaxis(K,0,r)
        return (Ops,Cmat,a,T,ks)
    def _truncate_core(s, r):
        K=np.moveaxis(s.K,r,0); sh=K.shape; M=K.reshape(sh[0],-1)
        U,S,Vh=np.linalg.svd(M,full_matrices=False)
        nrm=np.linalg.norm(S); S=S/nrm; s.lognorm+=np.log(nrm)
        kp=max(1,min(s.maxa,int(np.sum(S>s.tol*S[0]))))
        s.trunc+=float(np.sum(S[kp:]**2))
        K=(S[:kp,None]*Vh[:kp]).reshape((kp,)+sh[1:]); s.K=np.moveaxis(K,0,r)
        return U[:,:kp]
    def apply(s, op):
        kind,qs,par=op
        if kind=='u3':
            t,p,l=par
            M=np.array([[math.cos(t/2),-np.exp(1j*l)*math.sin(t/2)],[np.exp(1j*p)*math.sin(t/2),np.exp(1j*(p+l))*math.cos(t/2)]])
            s.u3(qs[0],M); return
        bs=[s.where[q] for q in qs]; rs=set(r for r,_ in bs)
        if kind=='cz':
            if len(rs)==1: s.cz_in(bs[0][0],bs[0][1],bs[1][1]); return
            W=np.diag([1,1,1,-1]).astype(complex)
        else: W=par
        if len(rs)==1:
            r=bs[0][0]; s.L[r]=s._apply_leaf(s.L[r],r,[k for _,k in bs],W); return
        s.cross(W,qs)
    def bonds(s): return s.K.shape
    def dense(s):
        A,B,C=s.L; return np.einsum('xa,yb,zc,abc->xyz',A,B,C,s.K)
