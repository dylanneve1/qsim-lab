"""Diagnostic for G|MPS>: on exact hard-wall windows compute per step
 - natural occupations per chain (1-RDM eigenvalues), correlation entropy S1=sum h(nu) [bits]
 - i|o entanglement S_io [bits]
 - max cut entropy over the 2L-mode chain in (a) site basis rung-interleaved, (b) natural-orbital basis
   ordered by occupation, interleaved i/o  [bits]"""
import numpy as np, sys, itertools, json
from exact import run
L=int(sys.argv[1]) if __name__=="__main__" else 10; lo=30-L//2; steps=[int(x) for x in sys.argv[2].split(",")] if len(sys.argv)>2 and __name__=="__main__" else list(range(1,21))
def H(p):
    p=p[(p>1e-15)]; return float(-(p*np.log2(p)).sum())
def hb(nu):
    nu=np.clip(nu,1e-16,1-1e-16); return float(-(nu*np.log2(nu)+(1-nu)*np.log2(1-nu)).sum())
def rdm1(rho,B):
    # <c_j^dag c_k> with JW signs; rho = reduced many-body density on chain basis B
    ix={int(s):i for i,s in enumerate(B)}; C=np.zeros((L,L),complex)
    for j in range(L):
        C[j,j]=np.real(np.diag(rho)@((B>>j)&1))
        for k in range(j+1,L):
            m=(((B>>k)&1)==1)&(((B>>j)&1)==0)
            xs=np.nonzero(m)[0]
            if len(xs)==0: continue
            ys=np.array([ix[int(B[x])^(1<<j)^(1<<k)] for x in xs])
            between=np.array([bin(int(B[x])>>(j+1) & ((1<<(k-j-1))-1)).count('1') for x in xs])
            sg=(-1.0)**between
            # c_j^dag c_k |x> = sg |y>  => <c_j^dag c_k> = sum rho[x,y]*sg  (rho=psi psi^dag : rho[x,y]=psi_x psi_y^*)
            C[j,k]=np.sum(rho[xs,ys]*sg); C[k,j]=np.conj(C[j,k])
    return C
def wedge(W,B,N):
    # many-body matrix of single-particle unitary W on fixed-N basis B: M[S,T]=det(W[S,T])
    sets=[np.nonzero((int(b)>>np.arange(L))&1)[0] for b in B]
    S=np.array(sets)            # (dim,N)
    sub=W[S[:,None,:,None],S[None,:,None,:]]   # (dim,dim,N,N)
    return np.linalg.det(sub)
def full(psi,B):
    F=np.zeros((1<<L,1<<L),complex); F[np.ix_(B[0],B[1])]=psi; return F
def maxcut(F,order_i,order_o):
    # F[xi,xo] over 2^L x 2^L; mode orders give positions; build tensor with legs in interleaved order
    T=F.reshape([2]*(2*L))           # legs: xi bits in big-endian? index x=sum bit_k 2^k -> leg (L-1-k)
    legs_i=[L-1-k for k in order_i]; legs_o=[2*L-1-k for k in order_o]
    perm=[x for pair in zip(legs_i,legs_o) for x in pair]
    T=np.transpose(T,perm).reshape(-1)
    Ss=[]
    for c in range(2,2*L,2):
        M=T.reshape(2**c,-1)
        if c<=L: s=np.linalg.svd(M,compute_uv=False)
        else: s=np.linalg.svd(M,compute_uv=False)
        Ss.append(H(s**2))
    return max(Ss),Ss
if __name__=="__main__":
    res=[]
    for circ in ('SCV','meson'):
        def cb(k,psi,B,bits):
            if k not in steps: return
            r=dict(circ=circ,step=k)
            s=np.linalg.svd(psi,compute_uv=False); r['S_io']=H(s**2)
            nos=[];Ws=[]
            for l in (0,1):
                rho=psi@psi.conj().T if l==0 else psi.T@psi.conj()
                C=rdm1(rho,B[l]); ev,W=np.linalg.eigh(C); W=W.conj(); o=np.argsort(-ev); ev=ev[o]; W=W[:,o]
                nos.append(ev); Ws.append(W)
            r['nu_i']=nos[0].tolist(); r['nu_o']=nos[1].tolist(); r['S1']=hb(nos[0])+hb(nos[1])
            r['maxdev']=float(max(np.minimum(n,1-n).max() for n in nos))
            if L<=12:
                F=full(psi,B); r['Scut_site_max'],_=maxcut(F,list(range(L)),list(range(L)))
                # NO basis: psi_NO[S,T] = sum <S|Lam(W^dag)|x> psi[x,y] <T|Lam(W^dag)|y>   (new mode a = sum_j W[j,a] c_j)
                Mi=wedge(Ws[0],B[0],None); Mo=wedge(Ws[1],B[1],None)
                pno=Mi.conj().T@psi@Mo.conj()
                r['norm_no']=float(np.linalg.norm(pno))
                Fn=full(pno,B)
                # modes already ordered by occupation (index 0 = most occupied)
                r['Scut_no_max'],r['Scut_no']=maxcut(Fn,list(range(L)),list(range(L)))
            res.append(r)
            print(f"{circ} L={L} step {k:2d} S_io={r['S_io']:.3f} S1={r['S1']:.3f} maxdev={r['maxdev']:.3f} "
                  +(f"Scut_site={r['Scut_site_max']:.3f} Scut_NO={r['Scut_no_max']:.3f} norm={r['norm_no']:.6f}" if L<=12 else ''),
                  ' nu_i',np.round(nos[0],3),flush=True)
        run(circ,lo,lo+L,1.0,max(steps),cb=cb)
    json.dump(res,open(f'diag_L{L}.json','w'))
