"""Interaction-picture diagnostic: psi_I(k) = U_free(k)^dag psi(k) on exact window; site-basis cut entropies
of psi vs psi_I (rung-interleaved), and i|o entanglement."""
import numpy as np, sys
from exact import run
from diag import maxcut, full, wedge, H
import diag
L=int(sys.argv[1]); lo=30-L//2; steps=[2,4,6,8,10,12,14,16,18,20]
# free single-particle propagator of window per chain: evolve basis via lam=0 runs on 1-particle sectors is awkward;
# instead build W by running exact free dynamics on the single-particle level using the op list.
import pickle
n0,ops=pickle.load(open('ops_SCV.pkl','rb'))
def Wfree(nsteps):
    W=[np.eye(L,dtype=complex),np.eye(L,dtype=complex)]
    out={}
    for o in ops:
        if o[0]=='step':
            out[o[1]]=[w.copy() for w in W]
            if o[1]==nsteps: break
            continue
        if o[0]=='p':
            _,l,s,ph=o
            if lo<=s<hi: W[l][s-lo,:]*=np.exp(1j*ph)
        elif o[0]=='h':
            _,l,s1,s2,V=o
            if lo<=s1<hi and lo<=s2<hi:
                idx=[s1-lo,s2-lo]; W[l][idx,:]=V@W[l][idx,:]
    return out
hi=lo+L; Wt=Wfree(20)
for circ in ('SCV','meson'):
    def cb(k,psi,B,bits):
        if k not in steps: return
        # psi = Lam(W) psi_I  =>  psi_I = Lam(W)^dag psi ; c_j(t)^dag = sum W[j',j]... single-particle amp a -> W a
        Mi=wedge(Wt[k][0],B[0],None); Mo=wedge(Wt[k][1],B[1],None)
        pI=Mi.conj().T@psi@Mo.conj()
        F=full(psi,B); FI=full(pI,B)
        S1,_=maxcut(F,list(range(L)),list(range(L))); S2,_=maxcut(FI,list(range(L)),list(range(L)))
        # sanity: lam=0 psi_I should be the initial product state
        print(f'{circ} L={L} step {k:2d} Scut(psi)={S1:.3f} Scut(psi_I)={S2:.3f} |psi_I|={np.linalg.norm(pI):.6f} S_io={H(np.linalg.svd(psi,compute_uv=False)**2):.3f}',flush=True)
    run(circ,lo,hi,1.0,20,cb=cb)
# sanity with lam=0
def cb0(k,psi,B,bits):
    if k==10:
        Mi=wedge(Wt[k][0],B[0],None); Mo=wedge(Wt[k][1],B[1],None); pI=Mi.conj().T@psi@Mo.conj()
        print('lam=0 check: max |psi_I| entry (should be 1):',np.abs(pI).max())
run('SCV',lo,hi,0.0,10,cb=cb0)
