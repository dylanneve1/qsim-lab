"""<psi| O |psi> where psi = d=4 MPS (tebd2 site, sorted basis x'=nu+2nd) and O = vectorised operator MPS on doubled d=16 sites
(heis2.dsite internal basis).  Left-to-right environment contraction with npc (charge-blocked)."""
import sys, numpy as np, time
sys.path.insert(0, '/tmp/su2-254-opus/pylib')
import tenpy.linalg.np_conserved as npc
import tebd2 as tf, heis2 as hs


def splitter():
    site4, leg4 = tf.make_site()
    dsite, legd, perm = hs.dsite()
    M = np.zeros((16, 4, 4))
    for j in range(16):
        p = perm[j]; M[j, p // 4, p % 4] = 1.0
    # legs [dbl, a.conj(), b]: q_dbl - q_a + q_b = 0
    return npc.Array.from_ndarray(M, [legd.conj(), leg4, leg4.conj()], labels=['pd', 'pa', 'pb'])


def sandwich(psi, Omps, verbose=False):
    L = psi.L; M = splitter(); E = None; t0 = time.time()
    for i in range(L):
        B = psi.get_B(i, 'B'); Bc = B.conj()
        W = Omps.get_B(i, 'B').replace_labels(['vL', 'p', 'vR'], ['wL', 'pd', 'wR'])
        Wm = npc.tensordot(W, M, axes=(['pd'], ['pd']))                 # wL wR pa pb
        if E is None:
            E = npc.Array.from_ndarray(np.ones((1, 1, 1)), [Bc.get_leg('vL*').conj(), W.get_leg('wL').conj(), B.get_leg('vL').conj()],
                                       labels=['e_bra', 'e_op', 'e_ket'])
        T1 = npc.tensordot(E, B, axes=(['e_ket'], ['vL']))               # e_bra e_op p vR
        T2 = npc.tensordot(T1, Wm, axes=(['e_op', 'p'], ['wL', 'pb']))   # e_bra vR wR pa
        T3 = npc.tensordot(T2, Bc, axes=(['e_bra', 'pa'], ['vL*', 'p*']))  # vR wR vR*
        T3.iset_leg_labels(['e_ket', 'e_op', 'e_bra'])
        E = T3
        if verbose: print(i, 'env chi', E.shape, flush=True)
    return complex(E.to_ndarray().ravel()[0])


def value(psi, Omps, lognorm):
    return float(np.real(sandwich(psi, Omps) * np.exp(lognorm)))
