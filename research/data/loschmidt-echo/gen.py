# Build W_L = A_L^dag Z_S A_L V A_L^dag Z_S A_L  (raw op lists) from a provided circuit's block A.
import math
from parse import parse
from checksym import perq, inv
def blocks(qasm):
    ops=[o for o in parse(qasm) if o[0]!='barrier']
    b=[i for i,o in enumerate(ops) if o[0] in ('s','sdg','sx','sxdg')]
    v0,v1=b[0],b[-1]+1
    pre=ops[:v0]; Vops=ops[v0:v1]; k=len(pre)//2
    A=pre[:k]; B=pre[k:]
    qs=sorted({q for o in ops for q in o[1]})
    pa=perq(A); pb=perq(inv(B)); S=sorted(q for q in qs if pa[q]!=pb[q])
    return A,Vops,S,ops
def split_steps(A):
    # step boundaries: every 3 CZ layers. Count CZ per edge: a step = each edge once.
    edges=sorted({o[1] for o in A if o[0]=='cz'})
    steps=[];cur=[];seen=set()
    for o in A:
        if o[0]=='cz':
            if o[1] in seen: steps.append(cur);cur=[];seen=set()
            seen.add(o[1])
        cur.append(o)
    steps.append(cur); return steps
def zconj(ops,S): return [(n,qs,-v) if (n=='rx' and qs[0] in S) else (n,qs,v) for n,qs,v in ops]
def build_W(A,Vops,S):
    U=A+zconj(inv(A),S)   # U2^dag U1 = Z_S A^dag Z_S A
    return U+Vops+inv(U)
def truncate_A(A6,L):
    from collections import Counter
    deg=Counter(q for o in A6 if o[0]=='cz' for q in o[1]); L6=None
    ncz=Counter(); last={}; out=[]
    tot=Counter(q for o in A6 if o[0]=='cz' for q in o[1])
    # deg per step = number of distinct edges at q
    edeg=Counter(q for e in {o[1] for o in A6 if o[0]=='cz'} for q in e)
    for o in A6:
        ok=True
        for q in o[1]:
            lim=L*edeg[q]
            if ncz[q]<lim: continue
            if ncz[q]==lim and o[0]=='rz' and last.get(q)=='cz': continue
            ok=False
        if ok: out.append(o)
        for q in o[1]:
            if o[0]=='cz': ncz[q]+=1
            last[q]=o[0]
    return out
def blocks_general(qasm):
    ops=[o for o in parse(qasm) if o[0]!='barrier']
    b=[i for i,o in enumerate(ops) if o[0] in ('s','sdg','sx','sxdg')]
    v0,v1=b[0],b[-1]+1
    pre=ops[:v0]; Vops=ops[v0:v1]; k=len(pre)//2
    A=pre[:k]; U2=inv(pre[k:])
    return A,U2,Vops,ops
def build_W_general(A,U2,Vops,L):
    U=truncate_A(A,L)+inv(truncate_A(U2,L))
    return U+Vops+inv(U)
