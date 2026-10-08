"""Validate pt2.py against exact statevector on small windows: a1,a2 from finite differences in lam."""
import numpy as np, sys, pt2, sv_lam
circ=sys.argv[1]; lo,hi,ns=int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
L=hi-lo; eps=np.array([(-1)**(r+lo) for r in range(L)],float)
D,verts,wT=pt2.collect(circ,lo,hi,ns)
def E(lam): return [float(eps@x.sum(1)) for x in sv_lam.run(circ,lo,hi,lam,ns)]
h=1.0; e0,ep,em=E(0),E(h),E(-h); e2p,e2m=E(2*h),E(-2*h)
for T in range(1,ns+1):
    a0,a1,a2=pt2.pt_coeffs(D,verts,wT,eps,T)
    f1=(ep[T-1]-em[T-1])/(2*h); f2=(ep[T-1]+em[T-1]-2*e0[T-1])/(2*h*h)
    # Richardson for a2: (16 f2(h) - f2(2h))/12... use 4-pt: removes h^2 term
    f2b=(e2p[T-1]+e2m[T-1]-2*e0[T-1])/(8*h*h); f2r=(4*f2-f2b)/3
    print(f"T={T} a0 {a0:.8f} ex {e0[T-1]:.8f} | a1 {a1:+.4e} fd {f1:+.4e} | a2 {a2:+.4e} fd {f2r:+.4e}")
