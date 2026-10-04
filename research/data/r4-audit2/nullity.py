# Independent stabilizer nullity: nu = n - log2 #{Pauli P : |<psi|P|psi>| = 1}.
# For each X-pattern x: v(z) = conj(psi[z]) * psi[z ^ x]; <X^x Z^w> up to phase = sum_z (-1)^{w.z} v(z)
# -> Walsh-Hadamard transform over z (own implementation, numpy).
import numpy as np, sys, csv
def wht(a):
    a=a.copy(); h=1; n=len(a)
    while h<n:
        a=a.reshape(-1,2,h); a=np.stack([a[:,0]+a[:,1],a[:,0]-a[:,1]],1).reshape(-1); h*=2
    return a
def nullity(psi):
    N=len(psi); n=N.bit_length()-1; idx=np.arange(N); cnt=0
    for x in range(N):
        v=np.conj(psi)*psi[idx^x]
        cnt+=int(np.sum(np.abs(wht(v))>1-1e-8))
    return n-np.log2(cnt)
if __name__=='__main__':
    spec,path,n,ref=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
    raw=np.fromfile(path,dtype='<f8'); N=1<<n; G=len(raw)//(2*N)
    S=raw.reshape(G,N,2); S=S[...,0]+1j*S[...,1]
    rows=list(csv.DictReader(open(ref)))
    which=sys.argv[5] if len(sys.argv)>5 else 'ref'
    ks=range(G) if which=='all' else [int(r['gate']) for r in rows if int(r['gate'])>=0]
    refnu={int(r['gate']):float(r['nullity']) for r in rows}
    refd={int(r['gate']):int(r['d']) for r in rows}
    mism=0; numax=0; eq=0; tot=0
    for k in ks:
        nu=nullity(S[k]); numax=max(numax,nu)
        if k in refnu:
            tot+=1
            if abs(nu-refnu[k])>1e-6: mism+=1; print('MISMATCH',k,nu,refnu[k])
            if abs(nu-refd[k])<1e-6: eq+=1
    print(f"{spec}: checked {len(ks)} gates, nu_max={numax:.3f}, ref-checkpoint mismatches {mism}/{tot}, nu==d_k at {eq}/{tot}")
