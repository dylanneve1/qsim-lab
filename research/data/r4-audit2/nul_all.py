import numpy as np,sys
from nullity import wht
n=int(sys.argv[1]); N=1<<n
raw=np.fromfile('st.bin',dtype='<f8'); G=len(raw)//(2*N); S=raw.reshape(G,N,2); S=S[...,0]+1j*S[...,1]
# fast path: nu=0 iff #stabilizers = 2^n. Use support/phase test first: a stabilizer state has flat modulus on an affine support.
mx=0
for k in range(G):
    psi=S[k]; nz=np.nonzero(np.abs(psi)>1e-9)[0]
    if len(nz)<=4:
        # small support: compute nullity restricted (exact): count Paulis via x in differences of support only
        idx=np.arange(N); cnt=0
        xs=set(int(a)^int(b) for a in nz for b in nz)
        for x in xs:
            v=np.conj(psi)*psi[idx^x]; cnt+=int(np.sum(np.abs(wht(v))>1-1e-8))
        nu=n-np.log2(cnt)
    else:
        print('large support at',k,len(nz)); nu=None
    mx=max(mx,nu if nu is not None else 99)
print('gates',G,'max nu',mx)
