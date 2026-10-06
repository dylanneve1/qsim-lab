"""Commutation-aware exact merging for circuits of 2q units (1q gates absorbed into units).
Repeat: for each unit i, search backwards for an earlier unit j on the same wire pair such that every unit
between them that touches either wire commutes with unit i (exact matrix test). Move i next to j, merge
(U_i @ U_j). If the merged unit is local (operator-Schmidt rank 1), split it and absorb the 1q factors into
the nearest unit on each wire (exact). Iterate to a fixed point.  Validated exactly on random small circuits."""
import sys, numpy as np, itertools, collections
sys.setrecursionlimit(10000)
exec(open(__import__('os').path.join(__import__('os').path.dirname(__file__),'resyn.py')).read().split("if __name__")[0])
I2=np.eye(2)
def to_units(n, ops):
    """absorb all 1q gates into the next 2q unit on the wire (or previous if none after). Returns units [(a,b,M)], with M acting kron(a,b)."""
    units=[]; pend={}
    for k,q,M in ops:
        if k=='1': pend[q]=M@pend.get(q,I2); continue
        a,b=q; M=M@np.kron(pend.pop(a,I2),pend.pop(b,I2)); units.append([a,b,M])
    # trailing: absorb into last unit touching wire
    tail={}
    for w,P in pend.items():
        for u in reversed(units):
            if w in (u[0],u[1]):
                u[2]=(np.kron(P,I2) if u[0]==w else np.kron(I2,P))@u[2]; break
        else: tail[w]=P
    return units, tail
def embed(u, wires):
    """unit u=(a,b,M) as matrix on ordered 'wires' (len 3)."""
    a,b,M=u; k=len(wires); T=M.reshape(2,2,2,2)
    full=np.eye(2**k,dtype=complex).reshape((2,)*(2*k))
    ia,ib=wires.index(a),wires.index(b)
    # apply T on axes (ia,ib) of output side
    res=np.tensordot(T,full,axes=([2,3],[ia,ib]))   # out_a,out_b, remaining...
    rest=[x for x in range(2*k) if x not in (ia,ib)]
    res=np.moveaxis(res,[0,1],[ia,ib])
    return res.reshape(2**k,2**k)
def commute(u,v,tol=1e-6):
    ws=sorted(set([u[0],u[1],v[0],v[1]]))
    if len(set([u[0],u[1]])&set([v[0],v[1]]))==0: return True
    A=embed(u,ws); B=embed(v,ws)
    return np.linalg.norm(A@B-B@A) < tol
def swapM(M): return M.reshape(2,2,2,2).transpose(1,0,3,2).reshape(4,4)
def merge_pass(units, depth=40, tol=1e-5):
    merged=0; removed=0; i=0
    while i < len(units):
        a,b,M=units[i]; found=None
        j=i-1; steps=0
        while j>=0 and steps<depth:
            u=units[j]
            if {u[0],u[1]}=={a,b}: found=j; break
            if {u[0],u[1]}&{a,b}:
                if not commute(units[i],u): break
                steps+=1
            j-=1
        if found is None: i+=1; continue
        u=units[found]
        Mi=M if (u[0],u[1])==(a,b) else swapM(M)
        Mn=Mi@u[2]
        del units[i]; merged+=1
        A,B,err=schmidt_split(Mn)
        if err<tol:
            # local: drop unit, absorb A,B into the next unit after 'found' on each wire (or previous)
            del units[found]; removed+=1
            for w,P in ((u[0],A),(u[1],B)):
                placed=False
                for kk in range(found,len(units)):
                    x=units[kk]
                    if w in (x[0],x[1]):
                        x[2]=x[2]@(np.kron(P,I2) if x[0]==w else np.kron(I2,P)); placed=True; break
                if not placed:
                    for kk in range(found-1,-1,-1):
                        x=units[kk]
                        if w in (x[0],x[1]):
                            x[2]=(np.kron(P,I2) if x[0]==w else np.kron(I2,P))@x[2]; placed=True; break
                if not placed: raise RuntimeError('isolated wire')
            i=max(found-1,0)
        else:
            units[found][2]=Mn
            i=found+1
    return merged, removed
def state_of(n, units, tail):
    psi=np.zeros(2**n,complex); psi[0]=1; T=psi.reshape((2,)*n)
    for a,b,M in units:
        T=np.moveaxis(np.tensordot(M.reshape(2,2,2,2),T,axes=([2,3],[a,b])),[0,1],[a,b])
    for w,P in tail.items(): T=np.moveaxis(np.tensordot(P,T,axes=([1],[w])),0,w)
    return T.reshape(-1)
if __name__=='__main__':
    if sys.argv[1]=='test':
        rng=np.random.default_rng(1)
        for trial in range(20):
            n=6; ops=[]
            for _ in range(60):
                if rng.random()<0.5:
                    q=int(rng.integers(n)); ops.append(('1',q,U3(*rng.uniform(-3,3,3))))
                else:
                    a,b=rng.choice(n,2,replace=False); ops.append(('2',(int(a),int(b)),CZ.copy()))
                    if rng.random()<0.3: ops.append(('2',(int(a),int(b)),CZ.copy()))   # plant cancellations
            u0,t0=to_units(n,ops); s0=state_of(n,[x[:] for x in u0],t0)
            u1=[x[:] for x in u0]; mg,rm=merge_pass(u1); s1=state_of(n,u1,t0)
            assert abs(abs(np.vdot(s0,s1))-1)<1e-9, (trial, abs(np.vdot(s0,s1)))
        print('exactness test passed (20 random circuits)'); sys.exit()
    n,ops=load(sys.argv[1])
    for it in range(5):
        ops,nl=one_pass(n,ops)
        if nl==0: break
    units,tail=to_units(n,ops); print('units',len(units),flush=True)
    for it in range(30):
        mg,rm=merge_pass(units); print('round',it,'merged',mg,'removed local',rm,'units',len(units),flush=True)
        if mg==0: break
    ranks=collections.Counter(int((np.linalg.svd(M.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4),compute_uv=False)>1e-5).sum()) for a,b,M in units)
    print('final units',len(units),'by Schmidt rank',sorted(ranks.items()))
    np.save('/tmp/p6resyn/'+sys.argv[1].split('/')[-1]+'.units.npy', np.array([(a,b,M) for a,b,M in units],dtype=object), allow_pickle=True)
