"""Free-fermion (g=0) entanglement diagnostic for the ladder MPS.
For each step and each rung cut, compute S = S_i + S_o from the two 60x60 correlation
matrices and the exact discarded Schmidt weight of an MPS with bond dim chi
(Gaussian Schmidt spectrum = product over single-particle modes)."""
import numpy as np, os, sys, json, itertools
import gauss
D = os.environ.get('SU2_CIRCUITS', '/tmp/su2-254/circuits') + '/'

def evolve_free(circ, g_scale=0.0, record=None):
    occ, m, out = gauss.blocks(D + f'x_100_{circ}.qasm')
    rho = [np.zeros((60, 60), dtype=complex) for _ in range(2)]
    for w in range(120):
        s, l = m[w]; rho[l][s, s] = occ[w]
    step = 0
    for b in out:
        if 'step' in b:
            step = b['step']
            if record: record(step, rho)
            continue
        w = b['w']; U = b['U']; sl = [m[x] for x in w]
        if len(w) == 1:
            s, l = sl[0]; ph = U[1, 1] / U[0, 0]; rho[l][s, :] *= ph; rho[l][:, s] *= np.conj(ph)
        else:
            (s1, l1), (s2, l2) = sl
            if l1 == l2:
                V = U[1:3, 1:3] / U[0, 0]
                Vsp = np.array([[V[1, 1], V[1, 0]], [V[0, 1], V[0, 0]]])
                idx = [s1, s2]; R = rho[l1]
                R[idx, :] = Vsp @ R[idx, :]; R[:, idx] = R[:, idx] @ Vsp.conj().T
            else:
                ph = np.angle(np.diag(U)); a0 = ph[2] - ph[0]; a1 = ph[1] - ph[0]
                p0 = np.exp(1j * a0); p1 = np.exp(1j * a1)
                for (l, s, p) in ((l1, s1, p0), (l2, s2, p1)):
                    rho[l][s, :] *= p; rho[l][:, s] *= np.conj(p)
    return rho

def spectrum_nu(rho, cut):
    nus = []
    for l in (0, 1):
        ev = np.linalg.eigvalsh(rho[l][:cut, :cut]).clip(0, 1); nus += list(ev)
    return np.array(nus)

def entropy(nus):
    v = nus[(nus > 1e-14) & (nus < 1 - 1e-14)]
    return float(-(v * np.log(v) + (1 - v) * np.log(1 - v)).sum())

def disc_weight(nus, chis, nmode=22):
    # Schmidt values: prod over modes of nu or 1-nu ; keep most entangled modes
    p = np.minimum(nus, 1 - nus); order = np.argsort(-p); p = p[order]
    base = np.prod(1 - p[nmode:])  # remaining modes in their dominant config
    pk = p[:nmode]
    lam = np.array([1.0])
    for x in pk:
        lam = np.concatenate([lam * (1 - x), lam * x])
        if lam.size > 2 ** 21:
            lam = np.sort(lam)[::-1][:2 ** 21]
    lam = np.sort(lam)[::-1] * base
    tot = 1.0
    return {c: float(max(tot - lam[:c].sum(), 0.0)) for c in chis}

if __name__ == '__main__':
    chis = [64, 128, 256, 512, 1024, 2048, 4096]
    res = {}
    for circ in ('SCV', 'meson'):
        rows = []
        def rec(step, rho):
            Ss = [entropy(spectrum_nu(rho, c)) for c in range(1, 60)]
            cmax = int(np.argmax(Ss)) + 1
            dw = disc_weight(spectrum_nu(rho, cmax), chis)
            rows.append(dict(step=step, Smax=max(Ss), cut=cmax, dw=dw))
            print(circ, step, f'Smax={max(Ss):.3f} at cut {cmax}', ' '.join(f'chi{c}:{dw[c]:.1e}' for c in chis), flush=True)
        evolve_free(circ, record=rec)
        res[circ] = rows
    json.dump(res, open('ent_free.json', 'w'), indent=1)
