import json,sys
from geom import code
OFF="abcdef"
d=int(sys.argv[1]); data,P=code(d); R=json.load(open(f"hookD_d{d}.json"))
for r in R:
    p=P[r['i']]; pos={q:OFF[k] for k,q in enumerate(p['q']) if q is not None}
    bad=sorted("".join(sorted(pos[int(q)] for q in k.split(","))) for k,v in r['D'].items() if v<=d-2)
    print(r['i'],"RGB"[r['c']],(r['x'],r['y']),r['w'],"bnd" if r['bnd'] else "int",bad)
