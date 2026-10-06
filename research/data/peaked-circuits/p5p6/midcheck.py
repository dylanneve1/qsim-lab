"""Dense operator of a window [lo,hi] (original block layers) of the relabelled circuit, per connected component.
Reports, per component, the operator-Schmidt spectrum across each single-qubit cut (product of 1q unitaries <=> rank 1)."""
import sys, numpy as np, collections
sys.path.insert(0,'/tmp/peaked-generic'); sys.path.insert(0,'/tmp/peaked-p5p6')
import pairgrow as PG, tnoq
orig, pifn, cut, lo, hi = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
n, units, tail, tags, sigma, nb = PG.build(orig, np.load(pifn), cut)
sel = [k for k,t in enumerate(tags) if t[0]=='S' or lo <= t[2] <= hi]
par=list(range(n))
def f(x):
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
for k in sel: a,b=units[k][:2]; par[f(a)]=f(b)
comps=collections.defaultdict(list)
for q in range(n): comps[f(q)].append(q)
sizes=sorted(len(c) for c in comps.values()); print('window',lo,hi,'units',len(sel),'component sizes',sizes)
if max(sizes)>12: sys.exit()
def apply(M, G, qs, m):  # M: 2^m x 2^m operator, left-multiply by G on local qubits qs
    T=M.reshape([2]*m+[2**m]); T=np.moveaxis(T,qs,[0,1]); sh=T.shape
    T=(G@T.reshape(4,-1)).reshape(sh); T=np.moveaxis(T,[0,1],qs); return T.reshape(2**m,2**m)
for c in comps.values():
    if len(c)<2: continue
    m=len(c); idx={q:i for i,q in enumerate(c)}
    M=np.eye(2**m,dtype=complex)
    for k in sel:
        a,b=units[k][:2]
        if a in idx: M=apply(M,tnoq.unitG(units[k]),[idx[a],idx[b]],m)
    # operator Schmidt across each single qubit
    out=[]
    for i in range(m):
        T=M.reshape([2]*m+[2]*m); T=np.moveaxis(T,[i,m+i],[0,1]).reshape(4,-1)
        s=np.linalg.svd(T,compute_uv=False); s=s/np.linalg.norm(s)
        out.append('%.1e'%(1-s[0]**2))
    print(c,'1-s0^2 per qubit:',' '.join(out))
