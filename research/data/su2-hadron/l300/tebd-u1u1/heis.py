"""Heisenberg-picture MPO evolution of a rung observable through the brick circuit (dense numpy).
O(t) = U^dag O U, gates applied in reverse order. MPO site tensor A[l, a, b, r] (a=ket,b=bra, 4-dim rung).
Truncation: Frobenius-optimal (canonical form), bond cap D and relative cutoff eps.
Window [lo,hi): gates crossing boundary dropped (hard wall) to compare with exact windows."""
import numpy as np, time
from compile import compile_ops
class MPO:
    def __init__(s,ops_onsite,lo,hi):
        # ops_onsite: dict site->4x4 ; product operator; identity elsewhere
        s.lo,s.hi=lo,hi; s.A=[]
        for r in range(lo,hi):
            M=ops_onsite.get(r,np.eye(4)).astype(complex); s.A.append(M.reshape(1,4,4,1))
        s.c=0; s.err=0.0
    def _qr_right(s,i):   # make A[i] left-canonical, push R to i+1
        A=s.A[i]; l,a,b,r=A.shape; Q,R=np.linalg.qr(A.reshape(l*16,r)); s.A[i]=Q.reshape(l,4,4,-1)
        s.A[i+1]=np.tensordot(R,s.A[i+1],1)
    def _qr_left(s,i):
        A=s.A[i]; l,a,b,r=A.shape; Q,R=np.linalg.qr(A.reshape(l,16*r).T); s.A[i]=Q.T.reshape(-1,4,4,r)
        s.A[i-1]=np.tensordot(s.A[i-1],R.T,1)
    def move(s,to):
        while s.c<to: s._qr_right(s.c); s.c+=1
        while s.c>to: s._qr_left(s.c); s.c-=1
    def gate2(s,i,U,D,eps,center_right=True):
        """O -> U^dag O U on local sites i,i+1 (U 16x16 in basis 4*x_i+x_{i+1})."""
        if s.c not in (i,i+1): s.move(i)
        if s.c==i+1: s.move(i)  # keep simple: center at i
        A,B=s.A[i],s.A[i+1]; l=A.shape[0]; r=B.shape[3]
        T=np.einsum('labm,mcdr->lacbdr',A,B)            # l, a1,a2(ket), b1,b2(bra), r
        T=T.reshape(l,16,16,r)
        T=np.tensordot(np.tensordot(T,U.conj(),axes=([1],[0])),U,axes=([1],[0])).transpose(0,2,3,1)
        T=T.reshape(l,4,4,4,4,r).transpose(0,1,3,2,4,5).reshape(l*16,16*r)   # (l,a1,b1),(a2,b2,r)
        nrm2=np.vdot(T,T).real
        u,sv,vh=np.linalg.svd(T,full_matrices=False)
        w=sv**2; keep=max(1,min(D,int(np.sum(w>eps*eps*w[0]))))
        s.err+=float(w[keep:].sum()/max(nrm2,1e-300))
        u=u[:,:keep]; sv=sv[:keep]; vh=vh[:keep]
        if center_right:
            s.A[i]=u.reshape(l,4,4,keep); s.A[i+1]=(sv[:,None]*vh).reshape(keep,4,4,r); s.c=i+1
        else:
            s.A[i]=(u*sv[None,:]).reshape(l,4,4,keep); s.A[i+1]=vh.reshape(keep,4,4,r); s.c=i
        return keep
    def diag(s,i,ph):   # O -> D^dag O D, D=diag(exp(i ph))
        d=np.exp(1j*ph); s.A[i]=s.A[i]*(d.conj()[None,:,None,None]*d[None,None,:,None])
    def expect(s,x):    # <x|O|x> for product basis state x (list of rung states, len hi-lo)
        v=np.ones(1,complex)
        for i,A in enumerate(s.A): v=v@A[:,x[i],x[i],:]
        return v[0]
    def maxD(s): return max(A.shape[3] for A in s.A)
def rung_states(n0,lo,hi): return [2*n0[0,r]+n0[1,r] for r in range(lo,hi)]
def evolve(r0,lo,hi,steps,nsteps,D,eps,states=(),verbose=False,op=None):
    """Floquet: all steps identical, so applying k steps backward gives O(k). Returns
    vals[k-1][j] = <state_j|O(k)|state_j> + 1 (i.e. <n_i+n_o> on rung r0), errs, Ds."""
    x=np.arange(4); O=np.diag(((x>>1)&1)+(x&1)-1.0) if op is None else op
    m=MPO({r0:O},lo,hi); t0=time.time(); st=steps[0]; vals=[];errs=[];Ds=[]
    for k in range(1,nsteps+1):
        for i in range(hi-lo): m.diag(i,st['diag'][lo+i])
        for L in st['layers'][::-1]:
            bonds=[a for a in sorted(L['G']) if lo<=a and a+1<hi and a+1>=r0-k and a<=r0+k]
            if not bonds: continue
            fwd=abs(m.c-(bonds[0]-lo))<=abs(m.c-(bonds[-1]-lo))
            for a in (bonds if fwd else bonds[::-1]):
                m.move(a-lo); m.gate2(a-lo,L['G'][a],D,eps,center_right=fwd)
        vals.append([m.expect(s_).real+(1 if op is None else 0) for s_ in states]); errs.append(m.err); Ds.append(m.maxD())
        if verbose: print(f'  k={k} D={Ds[-1]} err={m.err:.2e} t={time.time()-t0:.1f} vals={np.round(vals[-1],7)}',flush=True)
    return np.array(vals),errs,Ds
