# Exact Z[w] (w=e^{i pi/4}) state vector for a qubit window of the circuit: measure the
# smallest denominator exponent (sde, powers of sqrt2) and CRT bit-size of each amplitude.
import re, sys, numpy as np, math
F=sys.argv[1]; lo,hi=int(sys.argv[2]),int(sys.argv[3]); D=int(sys.argv[4]) if len(sys.argv)>4 else 999
ops=[]
for line in open(F):
    line=line.strip().rstrip(';')
    if not line or line.startswith(('OPENQASM','include','qreg','creg','barrier','measure')): continue
    m=re.match(r'([a-z]+)(\(([^)]*)\))?\s+(.*)',line)
    qs=[int(x) for x in re.findall(r'q\[(\d+)\]',m.group(4))]
    ops.append((m.group(1),m.group(3),qs))
czl=[0]*70; keep=[]
for nm,arg,qs in ops:
    if nm=='cz':
        d=max(czl[q] for q in qs)+1
        for q in qs: czl[q]=d
        if d>D: continue
    if all(lo<=q<hi for q in qs): keep.append((nm,arg,[q-lo for q in qs]))
ops=keep; n=hi-lo; N=1<<n
nT=sum(o[0]=='rz' for o in ops)
# V[k] coefficient arrays (object ints) for w^0..w^3 ; state = (sum_k V[k] w^k)/sqrt2^K
V=[np.zeros(N,dtype=object) for _ in range(4)]
for k in range(4): V[k][:]=0
V[0][0]=1; K=0
def mulw(A,j):  # multiply element arrays A (list of 4) by w^j
    j%=8; R=list(A)
    for _ in range(j): R=[-R[3],R[0],R[1],R[2]]
    return R
def add(A,B): return [A[i]+B[i] for i in range(4)]
idx=np.arange(N)
def split(q):
    b=(idx>>q)&1; return np.nonzero(b==0)[0], np.nonzero(b==1)[0]
cache={}
def halves(q):
    if q not in cache: cache[q]=split(q)
    return cache[q]
def apply1(q,M):  # M: 2x2 of (sign-free) lists of w-powers: entries as list of (coef,int power)
    global V
    i0,i1=halves(q); u=[v[i0] for v in V]; w_=[v[i1] for v in V]
    def comb(row):
        acc=[np.zeros(len(i0),dtype=object) for _ in range(4)]
        for (c,p),X in zip(row,(u,w_)):
            if c==0: continue
            T=mulw(X,p)
            if c<0: T=[-t for t in T]
            acc=add(acc,T)
        return acc
    a=comb(M[0]); b=comb(M[1])
    for k in range(4): V[k][i0]=a[k]; V[k][i1]=b[k]
def divisible_sqrt2(A):
    # A*(w - w^3) all even ?
    P=add(mulw(A,1),[-x for x in mulw(A,3)])
    return P
for nm,arg,qs in ops:
    if nm=='cz':
        a,b=qs; sel=np.nonzero(((idx>>a)&1)&((idx>>b)&1))[0]
        for k in range(4): V[k][sel]=-V[k][sel]
    elif nm=='h': apply1(qs[0],[[(1,0),(1,0)],[(1,0),(-1,0)]]); K+=1
    elif nm=='sx': apply1(qs[0],[[(1,1),(-1,3)],[(-1,3),(1,1)]]); K+=1
    elif nm=='sxdg': apply1(qs[0],[[(-1,3),(1,1)],[(1,1),(-1,3)]]); K+=1
    elif nm in ('s','sdg','rz'):
        if nm=='s': p=2
        elif nm=='sdg': p=6
        else:
            th=eval(arg.replace('pi','math.pi')); p=round(th/(math.pi/4))%8
        q=qs[0]; i0,i1=halves(q); T=mulw([v[i1] for v in V],p)
        for k in range(4): V[k][i1]=T[k]
    else: raise Exception(nm)
    # global reduction by sqrt2 when possible
    while K>0:
        P=divisible_sqrt2(V)
        if all(((x%2)==0).all() for x in P): V=[x//2 for x in P]; K-=1
        else: break
# per-amplitude sde and bit size
sdes=[]; bits=[]
for j in range(N):
    A=[int(V[k][j]) for k in range(4)]; kk=K
    if all(a==0 for a in A): continue
    while kk>0:
        P=[-A[3]*1+0,0,0,0]
        P=add(mulw(A,1),[-x for x in mulw(A,3)])
        if all(x%2==0 for x in P): A=[x//2 for x in P]; kk-=1
        else: break
    sdes.append(kk); bits.append(sum(math.log2(2*abs(a)+1) for a in A))
sdes=np.array(sdes); bits=np.array(bits)
print(f"win {lo}-{hi} D={D} n={n} T={nT} H+sx={sum(o[0] in ('h','sx','sxdg') for o in ops)} globalK={K} sde: mean {sdes.mean():.1f} max {sdes.max()}  CRTbits: mean {bits.mean():.1f} max {bits.max():.1f}  nonzero {len(sdes)}/{N}")
