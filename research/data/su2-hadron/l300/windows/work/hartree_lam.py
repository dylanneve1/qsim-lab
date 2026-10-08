"""Time-dependent Hartree with the inter-chain phase scaled by lam; prints stag_SCV, n_f per step
for lam in (1,-1,2) to show Hartree's lam-dependence (its first-order term is zero, as for the exact
dynamics; its second-order term differs from the exact a2 of pt2.py)."""
import numpy as np, os, json, gauss
def run(circ,lam):
    D=os.environ.get('SU2_CIRCUITS','circuits')+'/'
    occ,m,out=gauss.blocks(D+f'x_100_{circ}.qasm')
    rho=[np.zeros((60,60),dtype=complex) for _ in range(2)]
    for w in range(120):
        s,l=m[w]; rho[l][s,s]=occ[w]
    rec=[]
    for b in out:
        if 'step' in b:
            n=np.array([[rho[l][s,s].real for l in (0,1)] for s in range(60)])
            rec.append(float(sum((-1)**r*(n[r,0]+n[r,1]) for r in range(60)))); continue
        w=b['w']; U=b['U']; sl=[m[x] for x in w]
        if len(w)==1:
            s,l=sl[0]; ph=U[1,1]/U[0,0]; rho[l][s,:]*=ph; rho[l][:,s]*=np.conj(ph)
        elif sl[0][1]==sl[1][1]:
            (s1,l1),(s2,l2)=sl; V=U[1:3,1:3]/U[0,0]; Vsp=np.array([[V[1,1],V[1,0]],[V[0,1],V[0,0]]])
            idx=[s1,s2]; R=rho[l1]; R[idx,:]=Vsp@R[idx,:]; R[:,idx]=R[:,idx]@Vsp.conj().T
        else:
            (s1,l1),(s2,l2)=sl; ph=np.angle(np.diag(U)); a0=ph[2]-ph[0]; a1=ph[1]-ph[0]
            g=ph[3]-ph[2]-ph[1]+ph[0]; g=lam*((g+np.pi)%(2*np.pi)-np.pi)
            n0=rho[l1][s1,s1].real; n1=rho[l2][s2,s2].real
            for (l,s,p) in ((l1,s1,np.exp(1j*(a0+g*n1))),(l2,s2,np.exp(1j*(a1+g*n0)))):
                rho[l][s,:]*=p; rho[l][:,s]*=np.conj(p)
    return rec
if __name__=='__main__':
    res={}
    for lam in (0,1,-1,2):
        res[lam]={c:run(c,lam) for c in ('SCV','meson')}
    json.dump({str(k):v for k,v in res.items()},open('hartree_lam.json','w'))
    for T in range(20):
        f=lambda lam,c: res[lam][c][T]
        nf=lambda lam: f(lam,'meson')-f(lam,'SCV')
        print(f"{T+1:2d} SCV H(1)-H(0) {f(1,'SCV')-f(0,'SCV'):+.6f} H(1)-H(-1) {f(1,'SCV')-f(-1,'SCV'):+.1e} [H(2)-H(0)]/4 {(f(2,'SCV')-f(0,'SCV'))/4:+.6f} | n_f H(1)-H(0) {nf(1)-nf(0):+.6f}")
