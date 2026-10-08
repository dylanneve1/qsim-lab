"""Exact number-conserving window simulator: psi is (dim_i x dim_o) matrix over fixed-N bitstring bases.
Window sites [lo,hi), gates crossing the window boundary dropped (hard wall). Hop = qubit XY block
(=JW fermion NN hop, no sign). lam scales vertex g."""
import numpy as np, pickle, itertools, scipy.sparse as sp
def basis(L,N):
    st=[sum(1<<k for k in c) for c in itertools.combinations(range(L),N)]
    return np.array(st,dtype=np.int64),{s:i for i,s in enumerate(st)}
def load(circ): return pickle.load(open(f'/tmp/su2-254-opus/ops_{circ}.pkl','rb'))
def run(circ,lo,hi,lam=1.0,nsteps=20,cb=None):
    n0,ops=load(circ); L=hi-lo
    occ=[n0[l,lo:hi] for l in (0,1)]
    B=[];I=[]
    for l in (0,1):
        b,ix=basis(L,int(occ[l].sum())); B.append(b); I.append(ix)
    bits=[((B[l][:,None]>>np.arange(L)[None,:])&1).astype(float) for l in (0,1)]  # (dim,L)
    s0=[I[l][sum(1<<k for k in range(L) if occ[l][k])] for l in (0,1)]
    psi=np.zeros((len(B[0]),len(B[1])),complex); psi[s0[0],s0[1]]=1
    # cache hop pair index lists per (l,site)
    hc={}
    def hoppairs(l,a):
        key=(l,a)
        if key not in hc:
            b=B[l]; m=((b>>a)&1)!=((b>>(a+1))&1)
            src=np.nonzero(m & (((b>>a)&1)==1))[0]   # particle on a, empty a+1
            dst=np.array([I[l][x ^ ((1<<a)|(1<<(a+1)))] for x in b[src]],dtype=np.int64)
            hc[key]=(src,dst)
        return hc[key]
    for o in ops:
        if o[0]=='step':
            if cb: cb(o[1],psi,B,bits)
            if o[1]==nsteps: break
            continue
        if o[0]=='p':
            _,l,s,ph=o
            if not lo<=s<hi: continue
            f=np.exp(1j*ph*bits[l][:,s-lo])
            if l==0: psi*=f[:,None]
            else: psi*=f[None,:]
        elif o[0]=='v':
            _,s,g=o
            if not lo<=s<hi: continue
            psi*=np.exp(1j*lam*g*np.outer(bits[0][:,s-lo],bits[1][:,s-lo]))
        else:
            _,l,s1,s2,V=o
            if not (lo<=s1<hi and lo<=s2<hi): continue
            a=min(s1,s2)-lo
            # V basis (s1,s2): amplitude of particle on s1 / s2
            if s1<s2: Vab=V   # (a,a+1)
            else: Vab=V[::-1,::-1]
            src,dst=hoppairs(l,a)   # src: particle on a ; dst: particle on a+1
            if l==0:
                xa=psi[src].copy(); xb=psi[dst].copy()
                psi[src]=Vab[0,0]*xa+Vab[0,1]*xb; psi[dst]=Vab[1,0]*xa+Vab[1,1]*xb
            else:
                xa=psi[:,src].copy(); xb=psi[:,dst].copy()
                psi[:,src]=Vab[0,0]*xa+Vab[0,1]*xb; psi[:,dst]=Vab[1,0]*xa+Vab[1,1]*xb
    return psi
def dens(psi,bits):
    p=np.abs(psi)**2
    return np.stack([p.sum(1)@bits[0], p.sum(0)@bits[1]],1)   # (L,2)
if __name__=='__main__':
    # validate vs window results from su2-254-win res/*.npy
    import sys
    L=int(sys.argv[1]); lo=30-L//2
    for circ in ('SCV','meson'):
        out=[]
        run(circ,lo,lo+L,1.0,20,cb=lambda k,psi,B,bits: out.append(dens(psi,bits)))
        out=np.array(out)
        import glob
        f=glob.glob(f'/tmp/su2-254-win/work/res/L{L}_{circ}_lam+1.00.npy')
        if f:
            ref=np.load(f[0]); print(circ,'max diff vs su2-254-win window',np.abs(ref-out).max())
        np.save(f'win_L{L}_{circ}.npy',out)
