"""Write P6 with wires relabelled so that wire index = MPO site. usage: relabel.py ORDER out.qasm
ORDER: ring (A,B,C each in ring order), fold (each ring folded 0,L-1,1,L-2,..), twinfold (A folded, B = twin of A's folded order reversed, C folded),
       interleave (A_i, twin(A_i) alternating, folded; then C folded)"""
import sys, re, json
from parse import rings, twin
def fold(R):
    o=[]; i,j=0,len(R)-1
    while i<=j:
        o.append(R[i]);
        if i!=j: o.append(R[j])
        i+=1; j-=1
    return o
def order(name):
    A,B,C=rings
    if name=='ring': return A+B+C
    if name=='fold': return fold(A)+fold(B)+fold(C)
    if name=='twinfold': fa=fold(A); return fa+[twin[q] for q in fa[::-1]]+fold(C)
    if name=='interleave':
        fa=fold(A); o=[]
        for q in fa: o+= [q, twin[q]]
        return o+fold(C)
    if name=='cab': fa=fold(A); return fold(C)+fa+[twin[q] for q in fa[::-1]]
    raise ValueError(name)
if __name__=='__main__':
    o=order(sys.argv[1]); assert sorted(o)==list(range(62))
    new={q:i for i,q in enumerate(o)}
    src=open('/tmp/peaked-gen/portal/P6_titan_pinnacle.qasm').read()
    out='\n'.join(l if l.startswith(('qreg','creg')) else re.sub(r'q\[(\d+)\]', lambda m: f'q[{new[int(m.group(1))]}]', l) for l in src.split('\n'))
    open(sys.argv[2],'w').write(out); json.dump(o,open(sys.argv[2]+'.order.json','w'))
