"""Transverse (space-direction) contraction of the folded brick circuit.
Influence matrix IM at bond cut (c-1|c): time-MPS of 2T sites [(k,ket),(k,bra)], phys = operator-Schmidt index mu
of the gate on bond (c-1,c) at step k (rank<=16; 1 if no gate). Column of site c = time-MPO with bond 16 (folded
worldline state (a,b)), initial rho0=|x_c><x_c|, top boundary Tr(O .). Densities <O_r(T)> for all T need separate
top boundaries -> we put the top at T and run one sweep per T (T param)."""
import numpy as np, time
from compile import compile_ops
def split(U):
    U4=U.reshape(4,4,4,4).transpose(0,2,1,3).reshape(16,16)   # (x1',x1),(x2',x2)
    u,s,vh=np.linalg.svd(U4); k=int((s>1e-12*s[0]).sum())
    Lo=(u[:,:k]*np.sqrt(s[:k])).T.reshape(k,4,4); Ro=(np.sqrt(s[:k])[:,None]*vh[:k]).reshape(k,4,4)
    return Lo,Ro
class Circuit:
    def __init__(s,circ,lam,lo,hi,T):
        n0,steps=compile_ops(circ,lam); st=steps[0]; s.lo,s.hi,s.T=lo,hi,T
        s.x0=[int(2*n0[0,r]+n0[1,r]) for r in range(60)]
        s.D={r:np.exp(1j*st['diag'][r]) for r in range(60)}
        s.gate={}; s.par={}
        for li,L in enumerate(st['layers']):
            for a,U in L['G'].items():
                if lo<=a and a+1<hi: s.gate[a]=split(U); s.par[a]=li
    def column(s,c,O=None):
        """time-MPO for site c: list of 2T tensors W[t] (win, mu_left, mu_right, wout)."""
        I1=np.eye(4)[None]
        Lg=s.gate.get(c-1); Rg=s.gate.get(c)
        Rop=Lg[1] if Lg else I1     # site c is right site of bond (c-1,c): uses R-half, index mu (left IM)
        Lop=Rg[0] if Rg else I1     # site c is left site of bond (c,c+1): uses L-half, index mu' (right IM)
        D=np.diag(s.D[c])
        # order within step: layer 0 (even bonds) first
        left_first = (Lg is not None and s.par[c-1]==0) or (Rg is not None and s.par[c]==1)
        if left_first: M=np.einsum('xy,nyz,mzw->mnxw',D,Lop,Rop)   # D . Lop_n . Rop_m  (Rop applied first)
        else:          M=np.einsum('xy,myz,nzw->mnxw',D,Rop,Lop)   # D . Rop_m . Lop_n  (Lop applied first)
        # M[m,n,a',a]
        dm,dn=M.shape[:2]; I4=np.eye(4)
        Wk=np.einsum('mnAa,bB->abmnAB',M,I4).reshape(16,dm,dn,16)          # ket: (a,b)->(a',b)
        Wb=np.einsum('mnBb,aA->abmnAB',M.conj(),I4).reshape(16,dm,dn,16)   # bra: (a,b)->(a,b')
        W=[]
        for k in range(s.T): W+= [Wk.copy(),Wb.copy()]
        x=s.x0[c]; r0=np.zeros(16); r0[5*x]=1     # (a,b)=(x,x) -> index 4x+x
        W[0]=np.einsum('i,imnj->mnj',r0,W[0])[None]
        o=np.eye(4) if O is None else O
        top=o.T.reshape(16)        # Tr(O rho)=sum O[b,a] rho[a,b]
        W[-1]=np.einsum('imnj,j->imn',W[-1],top)[...,None]
        return W
def trivial(T): return [np.ones((1,1,1),complex) for _ in range(2*T)]
def apply(IM,W,chi,eps,side):
    """zip-up IM (time-MPS, legs (l,p,r)) with column W (w_in,mu_left,mu_right,w_out).
    side='L': IM is left IM (contract its p with mu_left), output legs mu_right. side='R': mirror."""
    n=len(IM); C=np.ones((1,1,1),complex); out=[]; Smax=0
    for t in range(n):
        A=IM[t]; Wt=W[t] if side=='L' else W[t].transpose(0,2,1,3)
        T_=np.einsum('nAw,Apb,wpqv->nqbv',C,A,Wt,optimize=True)
        nn,q,b,v=T_.shape; Mx=T_.reshape(nn*q,b*v)
        if t==n-1: out.append(Mx.reshape(nn,q,1)); break
        u,sv,vh=np.linalg.svd(Mx,full_matrices=False)
        k=max(1,min(chi,int((sv>eps*sv[0]).sum())))
        out.append(u[:,:k].reshape(nn,q,k)); C=(sv[:k,None]*vh[:k]).reshape(k,b,v)
    return out
def compress(M,chi,eps):
    """right-to-left SVD truncation sweep of a left-canonical MPS (from zip-up); returns MPS and max temporal entropy"""
    M=list(M); Smax=0; chis=[]; disc=0.0
    for t in range(len(M)-1,0,-1):
        l,p,r=M[t].shape; u,sv,vh=np.linalg.svd(M[t].reshape(l,p*r),full_matrices=False)
        nrm=np.sqrt((sv**2).sum()); w=(sv/nrm)**2
        k=max(1,min(chi,int((sv>eps*sv[0]).sum()))); disc+=w[k:].sum()
        S=-(w[w>1e-300]*np.log(w[w>1e-300])).sum(); Smax=max(Smax,S); chis.append(k)
        M[t]=vh[:k].reshape(k,p,r); M[t-1]=np.einsum('lpr,rk->lpk',M[t-1],u[:,:k]*sv[:k])
    # normalize scale into first tensor stays (IM is not normalized; keep)
    return M,Smax,max(chis) if chis else 1,disc
def contract3(L,W,R):
    E=np.ones((1,1,1),complex)
    for t in range(len(W)):
        E=np.einsum('awc,apb,wpqv,cqd->bvd',E,L[t],W[t],R[t],optimize=True)
    return E[0,0,0]
def sweep_left(cir,chi,eps,upto,verbose=False):
    """left IMs at cuts (c-1|c) for c=lo..upto ; IM_lo trivial"""
    IMs={cir.lo:trivial(cir.T)}; stats={}
    IM=IMs[cir.lo]
    for c in range(cir.lo,upto):
        IM=apply(IM,cir.column(c),2*chi,eps*1e-2,'L'); IM,S,k,d=compress(IM,chi,eps)
        IMs[c+1]=IM; stats[c+1]=(S,k,d)
        if verbose: print(f'   L cut {c}|{c+1}: S_t={S:.3f} chi_t={k} disc={d:.1e}',flush=True)
    return IMs,stats
def sweep_right(cir,chi,eps,downto,verbose=False):
    """right IMs at cuts (c|c+1) stored by key c+1 (same key as left IM at that cut)"""
    IMs={cir.hi:trivial(cir.T)}; stats={}; IM=IMs[cir.hi]
    for c in range(cir.hi-1,downto-1,-1):
        IM=apply(IM,cir.column(c),2*chi,eps*1e-2,'R'); IM,S,k,d=compress(IM,chi,eps)
        IMs[c]=IM; stats[c]=(S,k,d)
        if verbose: print(f'   R cut {c-1}|{c}: S_t={S:.3f} chi_t={k} disc={d:.1e}',flush=True)
    return IMs,stats
def density(cir,Ls,Rs,c):
    x=np.arange(4); O=np.diag((((x>>1)&1)+(x&1)).astype(float))
    num=contract3(Ls[c],cir.column(c,O),Rs[c+1]); den=contract3(Ls[c],cir.column(c),Rs[c+1])
    return (num/den).real, den
