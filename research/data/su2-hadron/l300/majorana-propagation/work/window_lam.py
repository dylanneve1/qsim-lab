"""Exact statevector on a window of sites [lo,hi) (boundary-crossing gates dropped) for several
lam; per-site quartic residual E(lam) - a0 - a2 lam^2 vs pt2 coefficients on the same window."""
import numpy as np, sys, json, pt2, sv_lam
circ=sys.argv[1]; lo,hi=int(sys.argv[2]),int(sys.argv[3]); ns=int(sys.argv[4]) if len(sys.argv)>4 else 20
lams=[0,1,-1,2,4,8]; L=hi-lo
ex={lam:sv_lam.run(circ,lo,hi,lam,ns) for lam in lams}     # list over steps of (L,2) occupations
D,verts,wT=pt2.collect(circ,lo,hi,ns)
out=[]
for T in range(1,ns+1):
    row={'step':T,'site':{}}
    for r in range(L):
        eps=np.zeros(L); eps[r]=1
        a0,a1,a2=pt2.pt_coeffs(D,verts,wT,eps,T)
        E={lam:float(ex[lam][T-1][r].sum()) for lam in lams}
        # fit E(lam)-a0-a2 lam^2 = a4 lam^4 + a6 lam^6 (+ odd check)
        x=np.array([2.,4.,8.]); y=np.array([E[l]-a0-a2*l*l for l in (2,4,8)])
        c=np.linalg.lstsq(np.vstack([x**4,x**6]).T,y,rcond=None)[0]
        row['site'][lo+r]=dict(a0=a0,a1=a1,a2=a2,E=E,res1=E[1]-a0-a2,odd=E[1]-E[-1],a4fit=c[0],a6fit=c[1])
    out.append(row)
    s=row['site']; res=np.array([s[lo+r]['res1'] for r in range(L)]); a2s=np.array([s[lo+r]['a2'] for r in range(L)])
    a4=np.array([s[lo+r]['a4fit'] for r in range(L)]); odd=max(abs(s[lo+r]['odd']) for r in range(L))
    print(f"T={T:2d} max|a2|/site {abs(a2s).max():.2e} max|res(lam=1)| {abs(res).max():.2e} max|a4fit| {abs(a4).max():.2e} max|E(1)-E(-1)| {odd:.1e}",flush=True)
json.dump(out,open(f'window_{circ}_{lo}_{hi}.json','w'))
