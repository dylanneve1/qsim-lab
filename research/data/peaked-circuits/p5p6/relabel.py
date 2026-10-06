"""Exact rewrite C -> C' = SW . C with the wire permutation sigma inserted at an ASAP-layer cut:
ops with block layer <= t unchanged, then explicit SWAPs mapping qubit q -> sigma(q), then later ops with
q -> sigma(q).  Output relation: x_q = x'_{sigma(q)}."""
import re,sys,numpy as np
def parse_lines(fn):
    hdr=[];ops=[]
    for line in open(fn):
        s=line.strip()
        m=re.match(r'u3\(([^)]*)\)\s+\w+\[(\d+)\];',s)
        if m: ops.append(['u3',[int(m.group(2))],m.group(1)]); continue
        m=re.match(r'cz\s+\w+\[(\d+)\],\s*\w+\[(\d+)\];',s)
        if m: ops.append(['cz',[int(m.group(1)),int(m.group(2))],None]); continue
        if s.startswith('qreg'): n=int(re.search(r'\[(\d+)\]',s).group(1))
    return n,ops
def cz_layers(n,ops):
    """block-ASAP layer of each cz (consecutive czs on the same pair = same block)"""
    depth=[0]*n; owner=[None]*n; lay={}; blk=[]  # blk: (pair, layer)
    for i,o in enumerate(ops):
        if o[0]!='cz': continue
        a,b=o[1]; p=tuple(sorted((a,b)))
        if owner[a] is not None and owner[a]==owner[b] and blk[owner[a]][0]==p:
            lay[i]=blk[owner[a]][1]
        else:
            l=max(depth[a],depth[b]); blk.append((p,l)); owner[a]=owner[b]=len(blk)-1
            depth[a]=depth[b]=l+1; lay[i]=l
    return lay
def swaps_for(sigma):
    """list of transpositions (a,b) whose sequential application moves content of q to sigma[q]"""
    n=len(sigma); at=list(range(n))   # at[w] = content currently on wire w
    target={sigma[q]:q for q in range(n)}  # wire w should hold content target[w]
    sw=[]
    for w in range(n):
        c=target[w]
        if at[w]!=c:
            v=at.index(c); sw.append((w,v)); at[w],at[v]=at[v],at[w]
    assert all(at[w]==target[w] for w in range(n))
    return sw
def rewrite(n,ops,t,sigma):
    lay=cz_layers(n,ops)
    # per-op side: cz by layer; u3 by next cz on its wire (after if none)
    side=[None]*len(ops); nxt=[None]*n
    for i in range(len(ops)-1,-1,-1):
        o=ops[i]
        if o[0]=='cz':
            side[i]= 'B' if lay[i]<=t else 'A'
            for q in o[1]: nxt[q]=side[i]
        else:
            side[i]= nxt[o[1][0]] or 'A'
    before=[o for o,s in zip(ops,side) if s=='B']
    after=[[o[0],[sigma[q] for q in o[1]],o[2]] for o,s in zip(ops,side) if s=='A']
    # sanity: per wire, no B after an A
    seen=[False]*n
    for o,s in zip(ops,side):
        for q in o[1]:
            if s=='A': seen[q]=True
            elif seen[q]: raise ValueError('cut not a frontier')
    return before,swaps_for(sigma),after
def emit(n,before,sw,after,fn):
    with open(fn,'w') as f:
        f.write('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[%d];\n'%n)
        for o in before+[None]+after:
            if o is None:
                for a,b in sw: f.write('swap q[%d],q[%d];\n'%(a,b))
                continue
            if o[0]=='u3': f.write('u3(%s) q[%d];\n'%(o[2],o[1][0]))
            else: f.write('cz q[%d],q[%d];\n'%tuple(o[1]))
# ---- test on random small circuit ----
def _sim(n,ops,sw=None):
    psi=np.zeros(2**n,complex); psi[0]=1; psi=psi.reshape([2]*n)
    def ap1(psi,M,q): return np.moveaxis(np.tensordot(M,psi,axes=([1],[q])),0,q)
    for o in ops:
        if o[0]=='u3':
            t,p,l=[eval(x,{'pi':np.pi}) for x in o[2].split(',')]
            M=np.array([[np.cos(t/2),-np.exp(1j*l)*np.sin(t/2)],[np.exp(1j*p)*np.sin(t/2),np.exp(1j*(p+l))*np.cos(t/2)]])
            psi=ap1(psi,M,o[1][0])
        elif o[0]=='cz':
            a,b=o[1]; idx=[slice(None)]*n; idx[a]=1; idx[b]=1; psi=psi.copy(); psi[tuple(idx)]*=-1
        elif o[0]=='swap':
            psi=np.swapaxes(psi,*o[1])
    return psi
def _test():
    rng=np.random.default_rng(1); n=6; ops=[]
    for l in range(8):
        perm=rng.permutation(n)
        for k in range(0,n,2):
            a,b=int(perm[k]),int(perm[k+1])
            for q in (a,b): ops.append(['u3',[q],','.join(str(x) for x in rng.uniform(-3,3,3))])
            ops.append(['cz',[a,b],None]); 
            for q in (a,b): ops.append(['u3',[q],','.join(str(x) for x in rng.uniform(-3,3,3))])
            ops.append(['cz',[a,b],None])
    sigma=[int(x) for x in rng.permutation(n)]
    B,sw,A=rewrite(n,ops,3,sigma)
    ref=_sim(n,ops); new=_sim(n,B+[['swap',[a,b],None] for a,b in sw]+A)
    # x_q = x'_{sigma(q)}  => new axis sigma(q) corresponds to ref axis q
    ref2=np.transpose(ref,np.argsort(sigma))  # axes reorder: new axis w = ref axis sigma^-1(w)
    print('test fidelity',abs(np.vdot(ref2.ravel(),new.ravel())))
if __name__=='__main__':
    if sys.argv[1]=='test': _test(); sys.exit()
    fn,pifn,t,out=sys.argv[1],sys.argv[2],float(sys.argv[3]),sys.argv[4]
    n,ops=parse_lines(fn); pi=np.load(pifn)
    sigma=[int(x) for x in np.argsort(pi)]   # sigma = pi^{-1}
    B,sw,A=rewrite(n,ops,t,sigma)
    print('before ops',len(B),'swaps',len(sw),'after ops',len(A))
    emit(n,B,sw,A,out); np.save(out+'.sigma.npy',np.array(sigma))
