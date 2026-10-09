"""U(1)xU(1) charge-conserving TEBD (TeNPy) of the paper's Trotter circuit for the 1D Fermi-Hubbard chain, d=4 sites
(|0>,|up>,|dn>,|updn>; x=n_up+2*n_dn), charges (N_up,N_dn).  Fermionic signs come from a 4-mode Jordan-Wigner embedding
(site-major mode order 0up,0dn,1up,1dn,...), so hopping up past dn on the same site picks up the correct parity sign.
Circuit (fermion frame, equivalent to the paper's fSWAP circuit -- verified in validate.py):
   odd layer : S(dt) U(dt) L(dt)      even layer : L(dt) U(dt) S(dt)   (operators applied left to right in time)
   S = up-hops on even bonds (2j,2j+1) + dn-hops on odd bonds ;  L = dn-hops on even bonds + up-hops on odd bonds
   H1Q ~ total N is a global phase -> dropped.  U = exp(-i dt U n_up n_dn) single-site diagonal.
usage: tebd_fh.py L chi nsteps [svd_min] [dt] [Uint] [tag]"""
import sys, json, time, resource, numpy as np
sys.path.insert(0, '/tmp/su2-254-opus/pylib')
import scipy.linalg as sla
import tenpy.linalg.np_conserved as npc
from tenpy.linalg.charges import ChargeInfo, LegCharge
from tenpy.networks.site import Site
from tenpy.networks.mps import MPS


def mode_ops():
    a = np.array([[0., 1.], [0., 0.]]); Z = np.diag([1., -1.]); I = np.eye(2)   # a=|0><1|, basis (empty,occ)
    def kr(*m):
        o = np.eye(1)
        for x in m: o = np.kron(o, x)
        return o
    c = [kr(a, I, I, I), kr(Z, a, I, I), kr(Z, Z, a, I), kr(Z, Z, Z, a)]   # modes 0up,0dn,1up,1dn
    return c


def hop_h(spin, t=1.0):
    c = mode_ops(); m = 0 if spin == 'u' else 1
    cl, cr = c[m], c[2 + m]
    H = -t * (cl.T @ cr + cr.T @ cl)       # 16x16, index = (site0 x)*4 + (site1 x), x=2nu+nd
    p = np.array([0, 2, 1, 3])             # old x -> tenpy-sorted x' = nu + 2 nd
    P4 = np.zeros((4, 4)); P4[p, np.arange(4)] = 1; P = np.kron(P4, P4)
    return P @ H @ P.T


def make_site():
    chinfo = ChargeInfo([1, 1], ['Nu', 'Nd'])
    leg = LegCharge.from_qflat(chinfo, [[0, 0], [1, 0], [0, 1], [1, 1]])   # x' = n_up + 2 n_dn (sorted order)
    x = np.arange(4)
    nu = (x & 1).astype(float); nd = ((x >> 1) & 1).astype(float)
    site = Site(leg, ['0', 'u', 'd', 'ud'], Nu=np.diag(nu), Nd=np.diag(nd), Nud=np.diag(nu * nd))
    assert site.perm is None or list(site.perm) == [0, 1, 2, 3]
    return site, leg


def run(L, chi, nsteps, svd_min=1e-10, dt=0.2, Uint=-2.0, tag='', outdir='.', quiet=False, init='neel', keep=()):
    site, leg = make_site()
    sites = [site] * L
    # Neel: site 0 = down, site 1 = up, ...  x': up=1, dn=2
    if init == 'neel': st = [2 if i % 2 == 0 else 1 for i in range(L)]
    psi = MPS.from_product_state(sites, st, bc='finite')
    def mk(U4):
        return npc.Array.from_ndarray(U4.reshape(4, 4, 4, 4), [leg, leg, leg.conj(), leg.conj()], labels=['p0', 'p1', 'p0*', 'p1*'], cutoff=1e-14)
    Gu = mk(sla.expm(-1j * dt * hop_h('u'))); Gd = mk(sla.expm(-1j * dt * hop_h('d')))
    Ud = npc.Array.from_ndarray(np.diag(np.exp(-1j * dt * Uint * np.array([0, 0, 0, 1.]))), [leg, leg.conj()], labels=['p', 'p*'])
    def Sgrp(psi, tp):
        eps = 0.
        for b in range(0, L - 1):
            G = Gu if b % 2 == 0 else Gd; eps += apply(psi, b, G, tp)
        return eps
    def Lgrp(psi, tp):
        eps = 0.
        for b in range(0, L - 1):
            G = Gd if b % 2 == 0 else Gu; eps += apply(psi, b, G, tp)
        return eps
    def apply(psi, b, G, tp):
        th = psi.get_theta(b, 2)
        th = npc.tensordot(G, th, axes=(['p0*', 'p1*'], ['p0', 'p1']))
        th.itranspose(['vL', 'p0', 'p1', 'vR'])
        th = th.combine_legs([['vL', 'p0'], ['p1', 'vR']], qconj=[+1, -1])
        return psi.set_svd_theta(b, th, trunc_par=tp).eps
    def Uall(psi):
        for i in range(L): psi.apply_local_op(i, Ud, unitary=True)
    tp = dict(chi_max=chi, svd_min=svd_min)
    kept = {}
    rec = []; t0 = time.time(); terr = 0.; fn = f'{outdir}/tebd_L{L}_chi{chi}{tag}.json'
    for k in range(1, nsteps + 1):
        if k % 2 == 1: terr += Sgrp(psi, tp); Uall(psi); terr += Lgrp(psi, tp)
        else: terr += Lgrp(psi, tp); Uall(psi); terr += Sgrp(psi, tp)
        nu = psi.expectation_value('Nu'); nd = psi.expectation_value('Nd'); dd = psi.expectation_value('Nud')
        S = psi.entanglement_entropy(); chis = psi.chi
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        rec.append(dict(step=k, t=k * dt, nu=np.real(nu).tolist(), nd=np.real(nd).tolist(), dd=np.real(dd).tolist(),
                        Smax=float(max(S)), chimax=int(max(chis)), terr=float(terr), wall=time.time() - t0, rss=rss))
        c = L // 2 - 1 if L % 2 == 0 else L // 2
        if not quiet:
            print(f'L={L} chi={chi} k={k:2d} t={k*dt:.1f} Smax={max(S):.3f} chimax={max(chis)} terr={terr:.2e} '
                  f'n_up[29]={nu[min(29,L-1)]:.6f} n_dn[29]={nd[min(29,L-1)]:.6f} dd[29]={dd[min(29,L-1)]:.6f} wall={time.time()-t0:.0f}s rss={rss:.2f}GB', flush=True)
        if k in keep: kept[k] = psi.copy()
        json.dump(dict(L=L, chi=chi, svd_min=svd_min, dt=dt, U=Uint, rec=rec), open(fn, 'w'))
    return (rec, kept) if keep else rec


if __name__ == '__main__':
    L = int(sys.argv[1]); chi = int(sys.argv[2]); n = int(sys.argv[3])
    svd = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-10
    dt = float(sys.argv[5]) if len(sys.argv) > 5 else 0.2
    U = float(sys.argv[6]) if len(sys.argv) > 6 else -2.0
    tag = sys.argv[7] if len(sys.argv) > 7 else ''
    run(L, chi, n, svd, dt, U, tag)
