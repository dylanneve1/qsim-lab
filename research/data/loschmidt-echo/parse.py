import re, math, sys
def parse(fn):
    ops=[]
    for line in open(fn):
        line=line.strip()
        if not line or line.startswith(('OPENQASM','include','gate','}','qubit')) or line.startswith(('s _gate','h _gate')): continue
        if line.startswith('barrier'): ops.append(('barrier',(),None)); continue
        m=re.match(r'(\w+)(?:\(([^)]*)\))?\s+(.*);',line)
        name,arg,qs=m.groups()
        qs=tuple(int(x) for x in re.findall(r'q\[(\d+)\]',qs))
        val=None
        if arg is not None: val=eval(arg,{'pi':math.pi})
        ops.append((name,qs,val))
    return ops
if __name__=='__main__':
    ops=parse(sys.argv[1])
    # segments between barriers
    seg=[];cur=[]
    for o in ops:
        if o[0]=='barrier': seg.append(cur);cur=[]
        else: cur.append(o)
    seg.append(cur)
    for i,s in enumerate(seg):
        from collections import Counter
        c=Counter(o[0] for o in s)
        angs=Counter((o[0],round(o[2],4)) for o in s if o[2] is not None)
        print(i,len(s),dict(c),dict(angs) if len(angs)<8 else len(angs))
    qs=sorted({q for o in ops for q in o[1]})
    print('qubits',len(qs),qs)
