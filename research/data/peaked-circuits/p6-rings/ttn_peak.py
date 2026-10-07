"""Peak from a saved star TTN (private/ttn_TAG.pkl): per-ring marginal argmax, then conditional refinement; amplitude of the
candidate and of all single-bit flips. Writes the bitstring (q0..q61, char i = bit of qubit i) to private/peak_TAG.txt."""
import sys, pickle, numpy as np
from parse import rings
d=pickle.load(open(f'private/ttn_{sys.argv[1]}.pkl','rb')); L=d['L']; K=d['K'].astype(np.complex128)
nrm=np.einsum('abc,abc->',K,K.conj()).real; K=K/np.sqrt(nrm)
def ring_probs(r, fixed):
    """probabilities over ring r's 2^n configs, other rings either summed (None) or fixed to an index."""
    Kc=K
    for o in range(3):
        if o==r: continue
        if fixed.get(o) is not None:
            v=L[o][fixed[o]].astype(np.complex128)
            Kc=np.tensordot(Kc,v,axes=([o if o<r or Kc.ndim==3 else o-1],[0])) if False else Kc
    # build reduced core vector/matrix explicitly
    idx='abc'; out=idx[r]
    ops=[K,[0,1,2]]; keep=[r]
    for o in range(3):
        if o==r: continue
        if fixed.get(o) is not None:
            ops+= [L[o][fixed[o]].astype(np.complex128),[o]]
        else: keep.append(o)
    Kr=np.einsum(*ops,sorted(keep))    # tensor over legs keep (sorted)
    Kr=np.moveaxis(Kr,sorted(keep).index(r),0).reshape(K.shape[r],-1)
    rho=Kr@Kr.conj().T
    Lr=L[r]
    p=np.real(np.einsum('xa,ab,xb->x',Lr,rho.astype(Lr.dtype),Lr.conj()))
    return p
def amp(ix):
    return np.einsum('abc,a,b,c->',K,L[0][ix[0]].astype(complex),L[1][ix[1]].astype(complex),L[2][ix[2]].astype(complex))
n=[len(r) for r in rings]
ix=[None,None,None]
for r in range(3):
    p=ring_probs(r,{}); ix[r]=int(np.argmax(p)); print('ring',r,'marginal max',p.max(),'sum',p.sum())
for it in range(3):
    for r in range(3):
        p=ring_probs(r,{o:ix[o] for o in range(3) if o!=r}); ix[r]=int(np.argmax(p))
a=amp(ix); P=abs(a)**2
bits=['?']*62
for r in range(3):
    for k,q in enumerate(rings[r]): bits[q]=str((ix[r]>>(n[r]-1-k))&1)
s=''.join(bits)
flips=[]
for q in range(62):
    r=[i for i in range(3) if q in rings[i]][0]; k=rings[r].index(q)
    j=list(ix); j[r]^=1<<(n[r]-1-k); flips.append(abs(amp(j))**2)
print('P(candidate)=%.6f  max single-flip P=%.3e  ratio %.1f'%(P,max(flips),P/max(flips)))
open(f'private/peak_{sys.argv[1]}.txt','w').write(s+'\n')
print('written private/peak_%s.txt'%sys.argv[1])
