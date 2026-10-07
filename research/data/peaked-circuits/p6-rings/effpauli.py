"""Pauli (Heisenberg) propagation through an effective op list (u3 / cz / dense 'U' on <=7 qubits)."""
import numpy as np, collections, math
PM=[np.eye(2),np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]]),np.diag([1.,-1]).astype(complex)]
PB=np.array(PM)  # (4,2,2)
def U3(t,p,l): return np.array([[math.cos(t/2),-np.exp(1j*l)*math.sin(t/2)],[np.exp(1j*p)*math.sin(t/2),np.exp(1j*(p+l))*math.cos(t/2)]])
def op_unitary(op):
    kind,qs,par=op
    if kind=='u3': return U3(*par)
    if kind=='cz': return np.diag([1,1,1,-1]).astype(complex)
    return par
def pauli_decomp(M,k):
    """M (2^k x 2^k) -> real coefficient tensor c[p0..pk-1] with M = sum c P_p0 x ... x P_pk-1."""
    X=M.reshape((2,)*(2*k)); labels=[('o',j) for j in range(k)]+[('i',j) for j in range(k)]
    for j in range(k):
        po=labels.index(('o',j)); pi=labels.index(('i',j))
        X=np.tensordot(PB,X,axes=([2,1],[po,pi]))/2   # sum_{o,i} P[i,o] M[o,i]
        labels=[('p',j)]+[l for l in labels if l not in (('o',j),('i',j))]
    order=[labels.index(('p',j)) for j in range(k)]
    return np.real(np.transpose(X,order))
class Prop:
    def __init__(s,ops): s.ops=ops; s.cache={}
    def act(s,idx,ploc):
        key=(idx,ploc)
        if key in s.cache: return s.cache[key]
        op=s.ops[idx]; W=op_unitary(op); k=len(op[1])
        P=np.ones((1,1),complex)
        for p in ploc: P=np.kron(P,PM[p])
        M=W.conj().T@P@W
        c=pauli_decomp(M,k)
        res=[(tuple(int(x) for x in ix),float(c[ix])) for ix in zip(*np.nonzero(np.abs(c)>1e-12))]
        s.cache[key]=res; return res
    def image(s,i0,i1,q,p,eps=1e-3,maxterms=20000):
        cur={((q,p),):1.0}
        for idx in range(i1,i0-1,-1):
            qs=s.ops[idx][1]; qset=set(qs); new=collections.defaultdict(float)
            for key,c in cur.items():
                d=dict(key)
                if not any(w in d for w in qs): new[key]+=c; continue
                ploc=tuple(d.pop(w,0) for w in qs)
                for pl,v in s.act(idx,ploc):
                    dd=dict(d)
                    for w,pp in zip(qs,pl):
                        if pp: dd[w]=pp
                    new[tuple(sorted(dd.items()))]+=c*v
            cur={k:v for k,v in new.items() if abs(v)>eps}
            if len(cur)>maxterms: return None
        return cur
