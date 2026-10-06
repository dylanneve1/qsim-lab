"""Heisenberg images W^dag P W of single-qubit Paulis through a contiguous range of 2q units (a,b,M) (M on kron(a,b)).
Pauli sums as dict {tuple(sorted((q,p))) : coeff}, p in 1..3 (X,Y,Z). Exact up to eps truncation."""
import numpy as np, sys, json, collections
PM=[np.eye(2),np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1.,-1.])]
P2=[np.kron(PM[i],PM[j]) for i in range(4) for j in range(4)]
def ptm(M):
    # T[i,j]: M^dag P_i M = sum_j T[i,j] P_j  (P_i = P2[i])
    T=np.zeros((16,16))
    for i in range(16):
        A=M.conj().T@P2[i]@M
        for j in range(16): T[i,j]=np.real(np.trace(P2[j]@A))/4
    return T
def image(units, k0, k1, q, p, eps=1e-6, maxterms=200000):
    cur={((q,p),):1.0}
    cache={}
    for k in range(k1,k0-1,-1):   # Heisenberg: last unit first
        a,b,M=int(units[k][0]),int(units[k][1]),np.asarray(units[k][2])
        if k not in cache: cache[k]=ptm(M)
        T=cache[k]; new=collections.defaultdict(float)
        for key,c in cur.items():
            d=dict(key); pa=d.pop(a,0); pb=d.pop(b,0)
            if pa==0 and pb==0: new[key]+=c; continue
            row=T[pa*4+pb]
            for j in np.nonzero(np.abs(row)>1e-12)[0]:
                ja,jb=divmod(j,4); dd=dict(d)
                if ja: dd[a]=ja
                if jb: dd[b]=jb
                new[tuple(sorted(dd.items()))]+=c*row[j]
        cur={k_:v for k_,v in new.items() if abs(v)>eps}
        if len(cur)>maxterms: return None
    return cur
if __name__=='__main__':
    U=np.load(sys.argv[1],allow_pickle=True); k0,k1=int(sys.argv[2]),int(sys.argv[3])
    qs=sorted({int(U[k][0]) for k in range(k0,k1+1)}|{int(U[k][1]) for k in range(k0,k1+1)})
    print('window',k0,k1,'qubits',qs)
    for q in qs:
        out=[]
        for p,nm in ((3,'Z'),(1,'X')):
            im=image(U,k0,k1,q,p)
            if im is None: out.append(nm+':blowup'); continue
            best=max(im.items(),key=lambda kv:abs(kv[1]))
            w1=sum(v*v for k_,v in im.items() if len(k_)==1)
            out.append('%s->%s %.3f (w1 %.3f, %d terms)'%(nm,best[0],best[1],w1,len(im)))
        print(q,' | '.join(out))
