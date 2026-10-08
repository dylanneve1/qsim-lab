import gauss, numpy as np
def run_mod(circ):
    occ,m,out=gauss.blocks('/tmp/su2-254/circuits/'+f'x_100_{circ}.qasm')
    rho=[np.zeros((60,60),dtype=complex) for _ in range(2)]
    for w in range(120):
        s,l=m[w]; rho[l][s,s]=occ[w]
    for b in out:
        if 'step' in b:
            n=np.array([[rho[l][s,s].real for l in (0,1)] for s in range(60)])
            stag=sum((-1)**r*(n[r,0]+n[r,1]) for r in range(60))
            if b['step'] == 1: print(f"Step 1: {stag}")
            continue
        w=b['w']; U=b['U']; sl=[m[x] for x in w]
        if len(w)==1: s,l=sl[0]; ph=U[1,1]/U[0,0]; rho[l][s,:]*=ph; rho[l][:,s]*=np.conj(ph)
        else:
            (s1,l1),(s2,l2)=sl
            if l1==l2:
                V=U[1:3,1:3]/U[0,0]; Vsp=np.array([[V[1,1],V[1,0]],[V[0,1],V[0,0]]])
                idx=[s1,s2]; R=rho[l1]
                R[idx,:]=Vsp@R[idx,:]; R[:,idx]=R[:,idx]@Vsp.conj().T
run_mod('meson')
