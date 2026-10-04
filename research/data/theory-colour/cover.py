import sys, itertools
from fam import T, NB
L2="abcdef"
def pool(t, flips=1):
    base={}
    base["A"]=set(t.A)
    for k in range(t.m):
        for r in range(3):
            base[f"rho^{r} I_{k}"]=t.rot(t.famI(k),r); base[f"rho^{r} II_{k}"]=t.rot(t.famII(k),r)
    cur=dict(base)
    for _ in range(flips):
        new={}
        for name,S in cur.items():
            for P in t.plaq:
                sp=t.supp[P]
                if 2*len(S.intersection(sp))==len(sp):
                    new[name+f"*S{P}"]=S^set(sp)
        cur.update(new)
    return cur
def covered_pairs(t,P,pl):
    sp=t.supp[P]; cov={}
    for name,S in pl.items():
        I=[q for q in sp if q in S]
        if len(I)==2:
            key=frozenset(I); cov.setdefault(key,name)
    return cov
def lab(P,q): return L2[NB.index((q[0]-P[0],q[1]-P[1]))]
def no_safe(sp,malign):
    # every order has {o1o2} or {o5o6} (w=6) / every pairing contains a malign pair (w=4)
    w=len(sp)
    for o in itertools.permutations(sp):
        hooks=[frozenset(o[k:]) for k in range(2,w-1)]
        bad=False
        for h in hooks:
            if h in malign or frozenset(sp)-h in malign: bad=True
        if not bad: return False
    return True
if __name__=="__main__":
    for d in map(int,sys.argv[1:]):
        t=T(d); pl=pool(t, flips=int(__import__('os').environ.get('FLIPS','0')))
        bott=[P for P in t.plaq if P[1]<=1]
        allok=True
        for P in sorted(bott):
            cov=covered_pairs(t,P,pl)
            ok=no_safe(t.supp[P],set(cov))
            allok&=ok
            print(d,P,t.colour(P),len(t.supp[P]),"no-safe-order proven" if ok else "NOT", sorted("".join(sorted(lab(P,q) for q in k)) for k in cov))
        print("d",d,"ALL" if allok else "MISSING")
