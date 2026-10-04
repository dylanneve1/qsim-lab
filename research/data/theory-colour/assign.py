"""Explicit malign-pair certificates for bottom boundary plaquettes (Boundary Lemma)."""
import sys, itertools
from fam import T, NB
from cover import no_safe, lab
def logicals_for(t):
    m=t.m; I=t.famI; II=t.famII
    tr=lambda j:(3*j+2,0)
    def prod(S,*Ps): return t.plaqprod(S,Ps)
    R=lambda S,k: t.rot(S,k)
    cert={}
    for j in range(0,m-1):            # non-corner trapezoids
        P=tr(j)
        cert[P]={"cf":("A",set(t.A)),"af":(f"I_{j}",I(j)),
                 "ac":(f"II_{j}",II(j)) if j>=1 else (f"rho I_{m-1}",R(I(m-1),1))}
    for j in range(1,m):              # boundary hexagons (3j,1)
        P=(3*j,1); c={}
        c["de"]=("A",set(t.A))
        c["df"]=(f"A.t{j-1}",prod(t.A,tr(j-1)))
        c["ce"]=(f"A.t{j}",prod(t.A,tr(j)))
        c["cf"]=(f"A.t{j-1}.t{j}",prod(t.A,tr(j-1),tr(j)))
        c["cd"]=(f"I_{j}",I(j))
        c["bc"]=(f"II_{j}",II(j))
        c["ef"]=(f"rho^2 II_{m-j}",R(II(m-j),2))
        if j<=m-2:
            c["bd"]=(f"II_{j}.t{j}",prod(II(j),tr(j)))
            c["be"]=(f"rho^2 I_{m-1-j}",R(I(m-1-j),2))
            c["bf"]=(f"rho^2 I_{m-1-j}.t{j-1}",prod(R(I(m-1-j),2),tr(j-1)))
        else:
            K=(3*m-2,2)
            c["bd"]=(f"I_{m-1}.S{K}",prod(I(m-1),K))
            c["be"]=(f"rho^2 I_0.S{K}",prod(R(I(0),2),K))
            c["bf"]=(f"rho II_{m-1}",R(II(m-1),1))
        cert[P]=c
    return cert
def check(d,verbose=False):
    t=T(d); cert=logicals_for(t); ok=True
    for P,c in cert.items():
        sp=t.supp[P]; mal=set()
        for pr,(name,S) in c.items():
            good=t.is_logical(S) and len(S)==d
            I={lab(P,q) for q in sp if q in S}
            if not(good and I==set(pr)): ok=False; print("FAIL",d,P,pr,name,good,sorted(I))
            mal.add(frozenset(q for q in sp if lab(P,q) in pr))
        if not no_safe(sp,mal): ok=False; print("FAIL no_safe",d,P)
    return ok
if __name__=="__main__":
    for d in range(int(sys.argv[1]),int(sys.argv[2])+1,2):
        print(d,"OK" if check(d) else "FAIL")
