# Triangular 6.6.6 colour code in lattice coordinates (u,v): sites u,v>=0, u+v<=L=3m (d=2m+1).
# Plaquette sites: u-v = 2 (mod 3); data: the rest. Support of plaquette = its 6 lattice neighbours in T.
NB=[(-1,1),(0,1),(1,0),(1,-1),(0,-1),(-1,0)]   # offsets a..f (same order as color.rs OFFSETS)
class T:
    def __init__(s,d):
        s.d=d; s.m=(d-1)//2; s.L=3*s.m
        s.sites=[(u,v) for v in range(s.L+1) for u in range(s.L-v+1)]
        s.data={p for p in s.sites if (p[0]-p[1])%3!=2}
        s.plaq=[p for p in s.sites if (p[0]-p[1])%3==2]
        s.supp={P:[(P[0]+a,P[1]+b) for a,b in NB if (P[0]+a,P[1]+b) in s.data] for P in s.plaq}
        s.A={p for p in s.data if p[1]==0}
    def colour(s,P): return "GBR"[P[1]%3]
    def is_logical(s,S):
        S=set(S); assert S<=s.data, S-s.data
        return all(len(S.intersection(s.supp[P]))%2==0 for P in s.plaq) and len(S&s.A)%2==1
    def rho(s,p): u,v=p; return (v,s.L-u-v)          # 120-degree rotation
    def rot(s,S,k=1):
        for _ in range(k%3): S={s.rho(p) for p in S}
        return set(S)
    # Family I: left-side top segment + red string down to the bottom, s=0..m-1
    def famI(s,k):
        c=3*k+1; S={(c,0),(c,1)}
        for t in range(k): S|={(3*(k-t),3*t+3),(3*(k-t)-1,3*t+4)}
        S|={(0,v) for v in range(3*k+3,s.L+1) if v%3!=1}
        return S
    # Family II: bottom right segment + green string from bottom trapezoid (3j+2,0) up-left to the left side, j=0..m-1
    def famII(s,j):
        S={(u,0) for u in range(3*j+3,s.L+1) if u%3!=2}
        for t in range(j): S|={(3*j+1-3*t,3*t+1),(3*j-3*t,3*t+2)}
        S|={(1,3*j),(0,3*j)}
        return S
    def plaqprod(s,S,Ps):
        S=set(S)
        for P in Ps: S^=set(s.supp[P])
        return S
