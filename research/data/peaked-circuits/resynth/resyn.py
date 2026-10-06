"""Exact peephole resynthesis for {1q, CZ} circuits.
Repeat until fixed point:
  - merge maximal runs of 2q gates on the same wire pair (with the 1q gates between them) into one 4x4 U;
  - if U is local (operator-Schmidt rank 1): replace by two 1q gates (exact);
  - (counts invariant classes for the rest).
Prints CZ count per pass. Writes the simplified circuit as JSON (list of ops) for later stages."""
import re, sys, math, json, collections, numpy as np
def U3(t,p,l): return np.array([[math.cos(t/2),-np.exp(1j*l)*math.sin(t/2)],[np.exp(1j*p)*math.sin(t/2),np.exp(1j*(p+l))*math.cos(t/2)]])
def num(x): return float(eval(x,{'pi':math.pi}))
CZ=np.diag([1,1,1,-1]).astype(complex)
def load(f):
    src=open(f).read(); n=int(re.search(r'\[(\d+)\];',src).group(1)); ops=[]
    for l in src.splitlines():
        m=re.match(r'(u3|cz)(?:\(([^)]*)\))?\s+([^;]*);',l.strip())
        if not m: continue
        qs=[int(x) for x in re.findall(r'\[(\d+)\]',m.group(3))]
        ops.append(('1',qs[0],U3(*map(num,m.group(2).split(',')))) if m.group(1)=='u3' else ('2',tuple(qs),CZ.copy()))
    return n,ops
def schmidt_split(U):
    # U (4x4) acting on (a,b) ordered kron(a,b). Return (A,B,err) with U ~ A(x)B.
    R=U.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4)   # (a_out a_in),(b_out b_in)
    u,s,vh=np.linalg.svd(R)
    A=(u[:,0]*np.sqrt(s[0])).reshape(2,2); B=(vh[0]*np.sqrt(s[0])).reshape(2,2)
    return A,B,float(np.sqrt(np.sum(s[1:]**2)))
def one_pass(n,ops,tol=1e-5):
    # normal form: per wire pending 1q; units = 4x4 with pair
    pend={}; out=[]; last={}  # last: wire -> index in out of last 2q op touching it
    def flush(w):
        if w in pend: out.append(('1',w,pend.pop(w)))
    for k,q,M in ops:
        if k=='1': pend[q]=M@pend.get(q,np.eye(2)); continue
        a,b=q
        ia,ib=last.get(a),last.get(b)
        if ia is not None and ia==ib and set(out[ia][1])=={a,b} and all(not(o[0]=='1' and o[1] in (a,b)) for o in out[ia+1:]):
            # merge: previous 2q on same pair with nothing on a,b since (pending 1q are still in pend)
            Pa=pend.pop(a,np.eye(2)); Pb=pend.pop(b,np.eye(2))
            pa,pb=out[ia][1]
            K=np.kron(Pa,Pb) if (pa,pb)==(a,b) else np.kron(Pb,Pa)
            Mq=M if (pa,pb)==(a,b) else M.reshape(2,2,2,2).transpose(1,0,3,2).reshape(4,4)
            out[ia]=('2',(pa,pb),Mq@K@out[ia][2])
            continue
        if a in pend or b in pend:
            Pa=pend.pop(a,np.eye(2)); Pb=pend.pop(b,np.eye(2)); M=M@np.kron(Pa,Pb)
        out.append(('2',(a,b),M)); last[a]=last[b]=len(out)-1
    for w in list(pend): flush(w)
    # replace local 2q ops by 1q ops
    res=[]; nloc=0
    for k,q,M in out:
        if k=='2':
            A,B,err=schmidt_split(M)
            if err<tol:
                res.append(('1',q[0],A)); res.append(('1',q[1],B)); nloc+=1; continue
        res.append((k,q,M))
    return res,nloc
def czcount(ops): return sum(1 for o in ops if o[0]=='2')
if __name__=='__main__':
    n,ops=load(sys.argv[1]); print('start 2q ops',czcount(ops),flush=True)
    for it in range(50):
        ops,nloc=one_pass(n,ops); print('pass',it,'2q units',czcount(ops),'removed local',nloc,flush=True)
        if nloc==0: break
    # class histogram of remaining units via operator-Schmidt rank / entangling power proxy
    sv=[]
    for k,q,M in ops:
        if k=='2':
            R=M.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4); s=np.linalg.svd(R,compute_uv=False); sv.append(int((s>1e-6).sum()))
    print('remaining units by operator-Schmidt rank:',sorted(collections.Counter(sv).items()))
    np.save(sys.argv[1].split('/')[-1]+'.resyn.npy', np.array([(o[0],o[1],o[2]) for o in ops],dtype=object), allow_pickle=True)
