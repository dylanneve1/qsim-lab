import os
# exact free-fermion (g=0) entanglement across ladder cuts -> MPS bond dim needed
import numpy as np, gauss, heapq
D=os.environ.get('SU2_CIRCUITS','circuits')+'/'
occ,m,out=gauss.blocks(D+'x_100_SCV.qasm')
# re-run free evolution capturing rho at steps
rho=[np.zeros((60,60),dtype=complex) for _ in range(2)]
for w in range(120):
    s,l=m[w]; rho[l][s,s]=occ[w]
def schmidt_logs(nus):
    # many-body Schmidt probs = prod_k (nu_k or 1-nu_k); return sorted list of top probabilities via heap
    nus=np.clip(np.array(nus),1e-300,1-1e-16)
    base=np.sum(np.log(np.maximum(nus,1-nus))); d=np.abs(np.log(nus)-np.log(1-nus))  # cost to flip
    d=np.sort(d); probs=[];  # enumerate lowest-cost subsets with a heap
    h=[(0.0,-1)];  seen=0; out=[]
    # standard k-smallest subset sums
    h=[(d[0],0)] if len(d) else []; out=[0.0]
    while h and len(out)<20000:
        c,i=heapq.heappop(h); out.append(c)
        if i+1<len(d):
            heapq.heappush(h,(c+d[i+1],i+1)); heapq.heappush(h,(c-d[i]+d[i+1],i+1))
    p=np.exp(base-np.array(out)); return p
def chi_needed(p,eps):
    c=np.cumsum(p); tot=1.0
    return int(np.searchsorted(c,1-eps)+1) if c[-1]>1-eps else '>20000'
step=0
for b in out:
    if 'step' in b:
        step=b['step']
        if step in (4,8,10,12,14,16,18,20):
            r=29; nus=[]
            for l in (0,1):
                ev=np.linalg.eigvalsh(rho[l][:r+1,:r+1]); nus+=list(ev.real)
            nus=np.array(nus); S=-np.sum(nus*np.log2(np.clip(nus,1e-300,1))+(1-nus)*np.log2(np.clip(1-nus,1e-300,1)))
            p=schmidt_logs(nus)
            print(f"step {step:2d} S_center={S:.3f} bits  chi(1e-2)={chi_needed(p,1e-2)} chi(1e-4)={chi_needed(p,1e-4)} chi(1e-6)={chi_needed(p,1e-6)}",flush=True)
        continue
    w=b['w'];U=b['U'];sl=[m[x] for x in w]
    if len(w)==1:
        s,l=sl[0]; ph=U[1,1]/U[0,0]; rho[l][s,:]*=ph; rho[l][:,s]*=np.conj(ph)
    elif sl[0][1]==sl[1][1]:
        (s1,l1),(s2,l2)=sl; V=U[1:3,1:3]/U[0,0]; Vsp=np.array([[V[1,1],V[1,0]],[V[0,1],V[0,0]]]); idx=[s1,s2]; R=rho[l1]
        R[idx,:]=Vsp@R[idx,:]; R[:,idx]=R[:,idx]@Vsp.conj().T
    else:
        (s1,l1),(s2,l2)=sl; ph=np.angle(np.diag(U)); a0=ph[2]-ph[0]; a1=ph[1]-ph[0]
        for (l,s,p) in ((l1,s1,np.exp(1j*a0)),(l2,s2,np.exp(1j*a1))): rho[l][s,:]*=p; rho[l][:,s]*=np.conj(p)
