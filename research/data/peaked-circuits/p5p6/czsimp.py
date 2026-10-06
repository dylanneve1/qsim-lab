"""Clifford-monomial peephole simplifier: 1q gates within tol of diagonal/anti-diagonal are snapped to monomial;
CZ(a,b) pairs separated on both wires only by monomial 1q gates and other CZs cancel (with Z corrections).
Repeats until no change. Writes QASM (u3 via generic 2x2 -> ZYZ) .  usage: czsimp.py IN.qasm TOL OUT.qasm"""
import sys, numpy as np, re
sys.path.insert(0,'/tmp/peaked-p5p6')
from cstruct import load
from blocks2 import u3m
Zm=np.diag([1,-1]).astype(complex)
def monotype(M,tol):
    if abs(M[0,1])<tol and abs(M[1,0])<tol: return 'D'
    if abs(M[0,0])<tol and abs(M[1,1])<tol: return 'A'
    return None
def snap(M,t):
    M=M.copy()
    if t=='D': M[0,1]=M[1,0]=0
    elif t=='A': M[0,0]=M[1,1]=0
    # renormalise phases
    for r in range(2):
        nz=np.abs(M[r])>0; M[r,nz]/=np.abs(M[r,nz])
    return M
def simplify(n,ops,tol):
    # ops: list of ['1',q,M] or ['cz',(a,b)]
    changed=True; it=0
    while changed:
        it+=1; changed=False
        # fuse 1q per wire
        out=[]; pend={}
        for o in ops:
            if o[0]=='1': pend[o[1]]=o[2]@pend.get(o[1],np.eye(2,dtype=complex))
            else:
                for q in o[1]:
                    if q in pend: out.append(['1',q,pend.pop(q)])
                out.append(o)
        for q,M in pend.items(): out.append(['1',q,M])
        ops=out
        # snap monomials
        for o in ops:
            if o[0]=='1':
                t=monotype(o[2],tol)
                if t: o[2]=snap(o[2],t); o.append(t)
        # per-wire index lists
        wl=[[] for _ in range(n)]
        for i,o in enumerate(ops):
            qs=[o[1]] if o[0]=='1' else list(o[1])
            for q in qs: wl[q].append(i)
        posw={}
        for q in range(n):
            for k,i in enumerate(wl[q]): posw[(i,q)]=k
        dead=set(); ins={}   # ins[i] -> list of (q,M) to insert right before op i
        for i,o in enumerate(ops):
            if o[0]!='cz' or i in dead: continue
            a,b=o[1]
            def scan(w,other):
                k=posw[(i,w)]+1; nA=0
                while k<len(wl[w]):
                    j=wl[w][k]; oj=ops[j]
                    if j in dead: k+=1; continue
                    if oj[0]=='1':
                        if len(oj)<4: return None,0
                        if oj[3]=='A': nA+=1
                    else:
                        if set(oj[1])=={w,other}: return j,nA
                    k+=1
                return None,0
            ja,na=scan(a,b); jb,nb=scan(b,a)
            if ja is not None and ja==jb:
                dead.add(i); dead.add(ja); changed=True
                corr=[]
                if na%2: corr.append((b,Zm))
                if nb%2: corr.append((a,Zm))
                ins.setdefault(ja,[]).extend(corr)
        new=[]
        for i,o in enumerate(ops):
            for q,M in ins.get(i,[]): new.append(['1',q,M])
            if i in dead: continue
            new.append(o[:3] if o[0]=='1' else o)
        ops=new
    return ops,it
def zyz(M):
    M=M/np.sqrt(np.linalg.det(M))
    th=2*np.arctan2(abs(M[1,0]),abs(M[0,0]))
    ph=np.angle(M[1,0])-np.angle(M[0,0]) if abs(M[1,0])>1e-15 and abs(M[0,0])>1e-15 else (np.angle(M[1,0])-np.angle(-M[0,1]) if abs(M[0,0])<=1e-15 else 2*np.angle(M[1,1]) )
    if abs(M[0,0])<=1e-15:
        lam=0.0; ph=np.angle(M[1,0])-np.angle(-M[0,1]*np.exp(-1j*lam))
        # u3(pi,ph,lam): [[0,-e^{il}],[e^{ip},0]] ; scale
    elif abs(M[1,0])<=1e-15:
        ph=0.0; lam=np.angle(M[1,1])-np.angle(M[0,0]); th=0.0
    else:
        lam=np.angle(-M[0,1])-np.angle(M[0,0])
    U=u3m(th,ph,lam)
    # check up to phase
    ov=abs(np.trace(U.conj().T@M))/2
    assert ov>1-1e-9, (M,U,ov)
    return th,ph,lam
def run(fn,tol):
    raw=load(fn); n=max(max(o[1]) for o in raw)+1
    ops=[['1',o[1][0],u3m(*o[2])] if o[0]=='u3' else ['cz',tuple(o[1])] for o in raw]
    ops,it=simplify(n,ops,tol)
    return n,ops,it
if __name__=='__main__':
    fn,tol,out=sys.argv[1],float(sys.argv[2]),sys.argv[3]
    n,ops,it=run(fn,tol)
    ncz=sum(o[0]=='cz' for o in ops); print('iterations',it,'remaining cz',ncz,'1q',sum(o[0]=='1' for o in ops))
    with open(out,'w') as f:
        f.write('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[%d];\n'%n)
        for o in ops:
            if o[0]=='1': f.write('u3(%r,%r,%r) q[%d];\n'%(*[float(v) for v in zyz(o[2])],o[1]))
            else: f.write('cz q[%d],q[%d];\n'%o[1])
