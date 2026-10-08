"""U(1)xU(1) charge-conserving TEBD (TeNPy) of the exact brick circuit on rung sites (d=4, x=2n_i+n_o).
Per step: Even layer, Odd layer, on-site diagonal D (absorbed into next step's even gates; densities are
diagonal so D does not affect measurements). Gates exact; only error = SVD truncation (chi_max, svd_min).
usage: tebd_sym.py circ chi lam lo hi nsteps [svd_min]"""
import sys, json, time, numpy as np, resource
sys.path.insert(0,'/tmp/su2-254-opus/pylib')
import tenpy.linalg.np_conserved as npc
from tenpy.linalg.charges import ChargeInfo, LegCharge
from tenpy.networks.site import Site
from tenpy.networks.mps import MPS
from compile import compile_ops
circ=sys.argv[1]; chi=int(sys.argv[2]); lam=float(sys.argv[3]); lo=int(sys.argv[4]); hi=int(sys.argv[5]); nsteps=int(sys.argv[6])
svd_min=float(sys.argv[7]) if len(sys.argv)>7 else 1e-10
n0,steps=compile_ops(circ,lam); st=steps[0]; L=hi-lo
chinfo=ChargeInfo([1,1],['Ni','No'])
leg=LegCharge.from_qflat(chinfo,[[0,0],[0,1],[1,0],[1,1]])
x=np.arange(4)
site=Site(leg,['00','01','10','11'],Ni=np.diag(((x>>1)&1).astype(float)),No=np.diag((x&1).astype(float)))
sites=[site]*L; leg=site.leg; print('perm',getattr(site,'perm',None))
psi=MPS.from_product_state(sites,[int(2*n0[0,r]+n0[1,r]) for r in range(lo,hi)],bc='finite')
def mkgate(U):
    return npc.Array.from_ndarray(U.reshape(4,4,4,4),[leg,leg,leg.conj(),leg.conj()],labels=['p0','p1','p0*','p1*'],cutoff=1e-13)
Dg=np.exp(1j*st['diag'])   # (60,4)
gates={}
for li,Lr in enumerate(st['layers']):
    for a,U in Lr['G'].items():
        if not (lo<=a and a+1<hi): continue
        gates[(li,a,False)]=mkgate(U)
        if li==0:  # D (from previous step) then U :  U @ diag(D_a x D_{a+1})
            gates[(li,a,True)]=mkgate(U@np.diag(np.kron(Dg[a],Dg[a+1])))
# sites in the window not covered by an even gate get their D applied as single-site op (hard-wall edges)
cov={s for (li,a,_) in gates if li==0 for s in (a,a+1)}
tp=dict(chi_max=chi,svd_min=svd_min)
rec=[]; t0=time.time(); terr=0.0
for k in range(1,nsteps+1):
    for li,Lr in enumerate(st['layers']):
        for a in sorted(Lr['G']):
            if not (lo<=a and a+1<hi): continue
            i=a-lo; th=psi.get_theta(i,2)
            G=gates[(li,a,li==0 and k>1)]
            th=npc.tensordot(G,th,axes=(['p0*','p1*'],['p0','p1']))
            th.itranspose(['vL','p0','p1','vR']); th=th.combine_legs([['vL','p0'],['p1','vR']],qconj=[+1,-1]); e=psi.set_svd_theta(i,th,trunc_par=tp); terr+=e.eps
    for s_ in range(lo,hi):
        if s_ not in cov:   # this step's D on edge sites not covered by an even gate
            psi.apply_local_op(s_-lo,npc.Array.from_ndarray(np.diag(Dg[s_]),[leg,leg.conj()],labels=['p','p*']))
    ni=psi.expectation_value('Ni'); no=psi.expectation_value('No')
    S=psi.entanglement_entropy(); chis=psi.chi
    rec.append(dict(step=k,ni=ni.tolist(),no=no.tolist(),Smax=float(max(S)),chimax=int(max(chis)),terr=terr,t=time.time()-t0,
                    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))
    print(f"{circ} chi={chi} lam={lam} k={k:2d} Smax={max(S):.3f} chi={max(chis)} terr={terr:.2e} stag={np.sum((-1)**np.arange(lo,hi)*(ni+no)):.7f} t={time.time()-t0:.0f}s rss={rec[-1]['rss']:.2f}GB",flush=True)
    json.dump(dict(circ=circ,chi=chi,lam=lam,lo=lo,hi=hi,svd_min=svd_min,rec=rec),open(f'tebd_{circ}_chi{chi}_lam{lam}_{lo}_{hi}.json','w'))
