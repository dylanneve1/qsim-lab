import json,sys,itertools
for d in map(int,sys.argv[1:]):
    R=json.load(open(f"hookD_d{d}.json"))
    for r in R:
        D={tuple(map(int,k.split(","))):v for k,v in r['D'].items()}
        qs=sorted({q for k in D for q in k}); w=r['w']
        def cls(S):
            S=tuple(sorted(S))
            if len(S)>w-len(S) or (2*len(S)==w and qs[0] not in S):
                S=tuple(sorted(set(qs)-set(S)))
            return D[tuple(x for x in sorted(S))] if tuple(sorted(S)) in D else D[tuple(sorted(S, key=lambda q: qs.index(q)))]
        # D keys use plaquette order; rebuild with that order
        keyof={frozenset(k):v for k,v in D.items()}
        def Dof(S):
            S=frozenset(S); C=frozenset(qs)-S
            return keyof.get(S, keyof.get(C))
        safe=0; best=0
        for o in itertools.permutations(qs):
            m=min(Dof(o[k:]) for k in range(2,w-1))
            best=max(best,m); safe+= m>=d-1
        print(d,r['i'],"RGB"[r['c']],(r['x'],r['y']),"w",w,"bnd" if r['bnd'] else "int","safe orders",safe,"best minD",best)
