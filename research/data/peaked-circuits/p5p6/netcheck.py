"""Dense operator of the UNMATCHED blocks in original layers [lo,hi] (original labels, file order), per component;
for each component find the wire permutation tau minimising the operator entanglement of P_tau^-1 M."""
import sys, numpy as np, collections, itertools
sys.path.insert(0,'/tmp/peaked-generic')
import gparse as G, tnoq, relabel as RL
orig, mfn, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
pi=np.load(sys.argv[5]) if len(sys.argv)>5 else None
match=np.load(mfn)
n, units, tail = G.parse(orig)
n_, ops = RL.parse_lines(orig); lay = RL.cz_layers(n_, ops)
owner=[None]*n; blk=[]; bid=[]
for i,o in enumerate(ops):
    if o[0]!='cz': continue
    a,b=o[1]; p=tuple(sorted((a,b)))
    if owner[a] is not None and owner[a]==owner[b] and blk[owner[a]]==p: bid.append(owner[a])
    else: blk.append(p); owner[a]=owner[b]=len(blk)-1; bid.append(len(blk)-1)
czl=[lay[i] for i in range(len(ops)) if ops[i][0]=='cz']
sel=[k for k in range(len(units)) if lo<=czl[k]<=hi and match[bid[k]]<0]
print('selected units',len(sel),'blocks',len(set(bid[k] for k in sel)))
par=list(range(n))
def f(x):
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
for k in sel: a,b=units[k][:2]; par[f(a)]=f(b)
comps=collections.defaultdict(list)
for q in range(n): comps[f(q)].append(q)
print('component sizes',sorted(len(c) for c in comps.values()))
def apply(M,G,qs,m):
    T=M.reshape([2]*m+[2**m]); T=np.moveaxis(T,qs,[0,1]); sh=T.shape
    T=(G@T.reshape(4,-1)).reshape(sh); T=np.moveaxis(T,[0,1],qs); return T.reshape(2**m,2**m)
def ent(M,m):
    out=[]
    for i in range(m):
        T=M.reshape([2]*m+[2]*m); T=np.moveaxis(T,[i,m+i],[0,1]).reshape(4,-1)
        s=np.linalg.svd(T,compute_uv=False); s=s/np.linalg.norm(s); out.append(1-s[0]**2)
    return np.array(out)
def permop(tau,m):  # maps local qubit i -> tau[i]
    P=np.zeros((2**m,2**m))
    for x in range(2**m):
        bits=[(x>>(m-1-i))&1 for i in range(m)]; nb=[0]*m
        for i in range(m): nb[tau[i]]=bits[i]
        y=int(''.join(map(str,nb)),2); P[y,x]=1
    return P
for c in comps.values():
    m=len(c)
    if m<2 or m>8: 
        if m>8: print(c,'too big'); 
        continue
    idx={q:i for i,q in enumerate(c)}
    M=np.eye(2**m,dtype=complex)
    for k in sel:
        a,b=units[k][:2]
        if a in idx: M=apply(M,tnoq.unitG(units[k]),[idx[a],idx[b]],m)
    best=None
    perms=itertools.permutations(range(m)) if m<=6 else []
    if pi is not None:
        tp=tuple(c.index(int(pi[q])) if int(pi[q]) in idx else -1 for q in c)
        if -1 not in tp: perms=list(perms)+[tp, tuple(np.argsort(tp))]
    for tau in perms:
        e=ent(permop(tau,m).T@M,m).max()
        if best is None or e<best[0]: best=(e,tau)
    t=[c[i] for i in best[1]]
    pis=[int(pi[q]) for q in c] if pi is not None else None
    print(c,'best tau ->',t,'max ent %.2e'%best[0],'| identity ent %.2e'%ent(M,m).max(),'| pi:',pis)
