from fam import T
for d in range(3,42,2):
    t=T(d); ok=True
    for k in range(t.m):
        for name,S in (("I",t.famI(k)),("II",t.famII(k))):
            for r in range(3):
                R=t.rot(S,r)
                if not (t.is_logical(R) and len(R)==d): ok=False; print("FAIL",d,name,k,r,len(R))
    print(d,"ok" if ok else "FAIL")
