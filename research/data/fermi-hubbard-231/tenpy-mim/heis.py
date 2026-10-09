"""Heisenberg-picture MPO (vectorised, doubled d=16 sites, charge q_ket-q_bra) evolution of a local diagonal observable
under the SAME Trotter circuit as tebd_fh.py, plus <Neel|O_H|Neel> by product-state overlap.
O_H^{(k)} = G_1^+ ... G_k^+ O G_k ... G_1 ; conjugate by layer k first, ..., layer 1 last (O <- g^+ O g, per gate).
Layer m (time order): odd: S,U,L ; even: L,U,S.  Heisenberg order is the reverse.
Vectorisation: |O>> = sum_ab O_ab |a>|b>, site index p=4a+b (orig), gate on (ket,bra) = G^+ (x) G^T.
Normalisation: HS-normalised MPS, prefactor 2^(L-1)*||o||, truncation norm loss tracked via exp(sum .5 log(1-eps)).
usage: python heis.py L c obs k chi [svd_min]   (obs in nu, nd, dd)"""
import sys, json, time, resource, numpy as np
sys.path.insert(0, '/tmp/su2-254-opus/pylib')
import scipy.linalg as sla
import tenpy.linalg.np_conserved as npc
from tenpy.linalg.charges import ChargeInfo, LegCharge
from tenpy.networks.site import Site
from tenpy.networks.mps import MPS
import tebd_fh as tf

X = np.arange(4); NU = (X & 1); ND = (X >> 1) & 1       # internal single-site basis x'=nu+2nd


def dsite():
    chinfo = ChargeInfo([1, 1], ['Nu', 'Nd'])
    qflat = [[NU[a] - NU[b], ND[a] - ND[b]] for a in range(4) for b in range(4)]
    leg0 = LegCharge.from_qflat(chinfo, qflat)
    site = Site(leg0, [str(i) for i in range(16)], orig=np.diag(np.arange(16.)))
    d = np.real(np.diag(site.get_op('orig').to_ndarray())).round().astype(int)   # internal j <-> orig d[j]
    return site, site.leg, d


def prodmps(sites, vecs):
    """product MPS from arbitrary single-site vectors (internal basis) that live in the zero-charge sector."""
    j0 = int(np.argmax(np.abs(vecs[0])))
    psi = MPS.from_product_state(sites, [int(np.flatnonzero(np.abs(v) > 0)[0]) for v in vecs], bc='finite')
    for i, v in enumerate(vecs):
        B = psi.get_B(i, form='B')
        nb = npc.Array.from_ndarray(np.asarray(v, dtype=complex).reshape(1, 16, 1), [B.get_leg('vL'), B.get_leg('p'), B.get_leg('vR')], labels=['vL', 'p', 'vR'])
        psi.set_B(i, nb, form='B')
    return psi


def run(L, c, obs, k, chi, dt=0.2, Uint=-2.0, svd_min=1e-10, quiet=False, outdir=None, tag=''):
    site, leg, perm = dsite(); sites = [site] * L
    inv = np.argsort(perm)
    def vec_orig(v): return v[perm]          # orig-ordered vector -> internal-ordered
    o = {'nu': NU.astype(float), 'nd': ND.astype(float), 'dd': (NU * ND).astype(float)}[obs]
    Ivec = np.zeros(16);
    for a in range(4): Ivec[4 * a + a] = 1
    Ovec = np.zeros(16)
    for a in range(4): Ovec[4 * a + a] = o[a]
    fc = np.linalg.norm(Ovec)
    states = [vec_orig(Ivec / 2.) if i != c else vec_orig(Ovec / fc) for i in range(L)]
    psi = prodmps(sites, states)
    lognorm = np.log(2. ** (L - 1) * fc)
    # gates (orig 16x16 time-evolution matrices on two doubled-site pairs)
    Gu = sla.expm(-1j * dt * tf.hop_h('u')); Gd_ = sla.expm(-1j * dt * tf.hop_h('d'))
    def conj_gate(G):
        Gdag = G.conj().T.reshape(4, 4, 4, 4); Gt = G.T.reshape(4, 4, 4, 4)       # [a0 a1 a0' a1']
        F = np.einsum('ijkl,mnop->imjnkolp', Gdag, Gt)                              # [a0 b0 a1 b1 | a0' b0' a1' b1']
        F = F.reshape(16, 16, 16, 16)                                              # p0 p1 p0' p1'
        F = F[np.ix_(perm, perm, perm, perm)]
        return npc.Array.from_ndarray(F, [leg, leg, leg.conj(), leg.conj()], labels=['p0', 'p1', 'p0*', 'p1*'], cutoff=1e-14)
    CG = {'u': conj_gate(Gu), 'd': conj_gate(Gd_)}
    u = np.exp(-1j * dt * Uint * (NU * ND))
    Uop = np.zeros((16, 16), dtype=complex)
    for a in range(4):
        for b in range(4): Uop[4 * a + b, 4 * a + b] = np.conj(u[a]) * u[b]
    Uop = Uop[np.ix_(perm, perm)]
    Uop = npc.Array.from_ndarray(Uop, [leg, leg.conj()], labels=['p', 'p*'])
    tp = dict(chi_max=chi, svd_min=svd_min)
    st = {'eps': 0.}
    def apply(b, G):
        th = psi.get_theta(b, 2)
        th = npc.tensordot(G, th, axes=(['p0*', 'p1*'], ['p0', 'p1']))
        th.itranspose(['vL', 'p0', 'p1', 'vR'])
        th = th.combine_legs([['vL', 'p0'], ['p1', 'vR']], qconj=[+1, -1])
        e = psi.set_svd_theta(b, th, trunc_par=tp).eps
        return e
    def grp(which, j):   # which 'S' or 'L'; j = layers already conjugated
        lo = max(0, c - 2 * j - 2); hi = min(L - 2, c + 2 * j + 1); lg = 0.
        for b in range(lo, hi + 1):
            if which == 'S': G = CG['u'] if b % 2 == 0 else CG['d']
            else: G = CG['d'] if b % 2 == 0 else CG['u']
            e = apply(b, G); lg += 0.5 * np.log(max(1. - e, 1e-300)); st['eps'] += e
        return lg
    def uall(j):
        for i in range(max(0, c - 2 * j - 2), min(L - 1, c + 2 * j + 2) + 1): psi.apply_local_op(i, Uop, unitary=True)
    # product state <Neel|<Neel| in doubled space
    nvec = [None] * L
    for i in range(L):
        x = 2 if i % 2 == 0 else 1
        v = np.zeros(16); v[4 * x + x] = 1; nvec[i] = vec_orig(v)
    bra = prodmps(sites, nvec)
    def expval():
        return np.real(bra.overlap(psi) * np.exp(lognorm))
    rec = []; t0 = time.time()
    for j in range(k):
        m = k - j                     # layer index being conjugated (time-order layer m)
        order = ['L', 'U', 'S'] if m % 2 == 1 else ['S', 'U', 'L']
        for g in order:
            if g == 'U': uall(j)
            else: lognorm += grp(g, j)
        S = psi.entanglement_entropy();
        r = dict(j=j + 1, m=m, Smax=float(max(S)), chimax=int(max(psi.chi)), eps=st['eps'], wall=time.time() - t0, rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
                 val=float(expval()) if j == k - 1 else None)
        rec.append(r)
        if not quiet: print(f"L={L} c={c} {obs} k={k} chi={chi} j={j+1:2d} Smax={r['Smax']:.3f} chimax={r['chimax']} eps={st['eps']:.2e} wall={r['wall']:.0f}s rss={r['rss']:.2f}GB" + (f" VAL={r['val']:.8f}" if r['val'] is not None else ''), flush=True)
    if outdir: json.dump(dict(L=L, c=c, obs=obs, k=k, chi=chi, rec=rec), open(f'{outdir}/heis_L{L}_c{c}_{obs}_k{k}_chi{chi}{tag}.json', 'w'))
    return rec


if __name__ == '__main__':
    L = int(sys.argv[1]); c = int(sys.argv[2]); obs = sys.argv[3]; k = int(sys.argv[4]); chi = int(sys.argv[5])
    run(L, c, obs, k, chi, svd_min=float(sys.argv[6]) if len(sys.argv) > 6 else 1e-10, outdir='/tmp/fh-231-t/heis')
