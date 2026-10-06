"""Variants of the P5 core: seam on/off, perm pi / pi^-1 / identity.  Quick greedy contraction."""
import sys, numpy as np, hashlib, json
sys.path.insert(0,'/tmp/peaked-generic'); sys.path.insert(0,'/tmp/peaked-p5p6')
import quimb.tensor as qtn, cotengra as ctg
import core as CO
def peak(n, psi):
    Z=np.diag([1.,-1.]).astype(complex); I=np.eye(2,dtype=complex)
    bra=psi.conj().reindex({f"s{q}":f"S{q}" for q in range(n)}); base=psi|bra
    opt=ctg.ReusableHyperOptimizer(methods=['greedy'],max_repeats=8,parallel=False,progbar=False)
    def net(q):
        tn=base.copy()
        for p in range(n): tn|=qtn.Tensor(Z if p==q else I,inds=(f"S{p}",f"s{p}"))
        return tn
    nrm=float(np.real(net(-1).contract(all,optimize=opt)))
    zs=np.array([float(np.real(net(q).contract(all,optimize=opt))) for q in range(n)])/nrm
    bits=''.join('0' if z>0 else '1' for z in zs)
    t=psi.copy()
    for q in range(n): t|=qtn.Tensor(np.array([1,0] if bits[q]=='0' else [0,1],dtype=complex),inds=(f"s{q}",))
    p=abs(t.contract(all,optimize=opt))**2/nrm
    return bits,p,zs
if __name__=='__main__':
    orig,pifn=sys.argv[1],sys.argv[2]; rmax,pmin=int(sys.argv[3]),int(sys.argv[4])
    pi=np.load(pifn); n,units,tail,czl=CO.unit_layers(orig)
    for permname,perm in [('pi',pi),('piinv',np.argsort(pi)),('id',np.arange(len(pi)))]:
        for seam in (True,False):
            psi,nR,nP=CO.core_tn(n,units,tail,czl,perm,rmax,pmin,seam)
            bits,p,zs=peak(n,psi)
            print(permname,'seam' if seam else 'noseam','p %.3e'%p,'mean|Z| %.3f'%np.mean(np.abs(zs)),'min|Z| %.3f'%np.min(np.abs(zs)),hashlib.sha256(bits.encode()).hexdigest()[:10],flush=True)
