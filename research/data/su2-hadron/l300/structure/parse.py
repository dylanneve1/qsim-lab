import re,sys
def load(path):
    ops=[]
    for line in open(path):
        line=line.strip()
        m=re.match(r'(\w+)(\(([^)]*)\))?\s+(.*);',line)
        if not m or m.group(1) in('OPENQASM','include','qreg'):continue
        qs=[int(x) for x in re.findall(r'q\[(\d+)\]',m.group(4))]
        ops.append((m.group(1),float(m.group(3)) if m.group(3) else None,qs))
    return ops
if __name__=='__main__':
    ops=load(sys.argv[1])
    # initial x
    k=0;init=[]
    while ops[k][0]=='x': init.append(ops[k][2][0]);k+=1
    from collections import Counter
    par=Counter(init); occ=sorted(q for q,c in par.items() if c%2)
    print('init occupied',len(occ),occ)
    # logical labels: q<60 -> i(q), else o(q-60)
    lab={q:(('i',q) if q<60 else ('o',q-60)) for q in range(120)}
    body=ops[k:]
    # track swaps; record 2q interactions (cx pairs) in logical labels
    pairs=Counter(); nswaplayers=0
    xs=Counter()
    for name,p,qs in body:
        if name=='swap':
            a,b=qs; lab[a],lab[b]=lab[b],lab[a]
        elif name=='cx':
            a,b=sorted([lab[qs[0]],lab[qs[1]]]);
            if a[0]==b[0]: pairs[('hop',a[0],abs(a[1]-b[1]))]+=1
            else: pairs[('io',abs(a[1]-b[1]))]+=1
        elif name=='x': xs[lab[qs[0]][0]]+=1
    print(pairs); print('x in body by chain',xs)
    print('final layout', [lab[q] for q in range(120)][:16], '...')
    ok=all(lab[2*r]==('i',r) and lab[2*r+1]==('o',r) for r in range(60))
    print('final: site r on (2r,2r+1) as (i,o)?',ok)
