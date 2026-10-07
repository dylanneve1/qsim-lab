"""Simple exact-ish MPO (site = wire in a chosen order) for applying 1q gates and long-range CZs (bond-2 string), with SVD
recompression (relative cutoff). Site tensor (Dl, out, in, Dr). W represents the product of applied gates (later gates on 'out')."""
import numpy as np, scipy.linalg as sla
class MPO:
    def __init__(s, n, cutoff=1e-10, maxb=4096, dtype=complex):
        s.n=n; s.cut=cutoff; s.maxb=maxb
        s.A=[np.eye(2,dtype=dtype).reshape(1,2,2,1) for _ in range(n)]
        s.trunc=0.0
    def g1(s, i, M, side='out'):
        if side=='out': s.A[i]=np.einsum('ab,lbcr->lacr',M,s.A[i])
        else: s.A[i]=np.einsum('lacr,cb->labr',s.A[i],M)
    def cz(s, i, j, side='out'):
        if i>j: i,j=j,i
        P0=np.diag([1.,0]); P1=np.diag([0,1.]); I=np.eye(2); Z=np.diag([1.,-1])
        for k in range(i,j+1):
            if k==i: O=np.stack([P0,P1],axis=-1)[None]          # (1,2,2,2)
            elif k==j: O=np.stack([I,Z],axis=0)[...,None]       # (2,2,2,1)
            else: O=np.einsum('ab,cd->acdb',np.eye(2),I)        # (2,2,2,2) identity passthrough
            T=s.A[k]
            if side=='out': N=np.einsum('xaby,lbcr->xlacyr',O,T)
            else: N=np.einsum('lacr,xcby->xlabyr',T,O)
            d=N.shape; s.A[k]=N.reshape(d[0]*d[1],d[2],d[3],d[4]*d[5])
        s.compress(i,j)
    def compress(s, i, j):
        # left-to-right QR over [i, j], then right-to-left SVD truncation over [i, j]
        for k in range(i,j):
            T=s.A[k]; Dl,a,b,Dr=T.shape
            Q,R=np.linalg.qr(T.reshape(Dl*a*b,Dr)); s.A[k]=Q.reshape(Dl,a,b,-1)
            s.A[k+1]=np.einsum('xy,yabr->xabr',R,s.A[k+1])
        for k in range(j,i,-1):
            T=s.A[k]; Dl,a,b,Dr=T.shape
            M=T.reshape(Dl,a*b*Dr)
            try: U,S,Vh=sla.svd(M,full_matrices=False,lapack_driver='gesdd')
            except Exception: U,S,Vh=sla.svd(M,full_matrices=False,lapack_driver='gesvd')
            keep=max(1,min(s.maxb,int(np.sum(S>s.cut*S[0]))))
            s.trunc+=float(np.sum(S[keep:]**2)/np.sum(S**2))
            s.A[k]=Vh[:keep].reshape(keep,a,b,Dr)
            s.A[k-1]=np.einsum('labr,rx->labx',s.A[k-1],U[:,:keep]*S[:keep])
    def bonds(s): return [t.shape[3] for t in s.A[:-1]]
