"""Exact statevector of the circuit restricted to sites [lo,hi) (gates crossing the window
boundary dropped), with the inter-chain phase g scaled by lam (free part untouched).
Used to validate pt2.py and tebd_lam.py on small windows."""
import numpy as np, os, gauss

def run(circ, lo, hi, lam=1.0, nsteps=20, path=None):
    if path is None:
        path = os.environ.get('SU2_CIRCUITS', 'circuits') + f'/x_100_{circ}.qasm'
    occ, m, out = gauss.blocks(path)
    L = hi - lo; n = 2 * L
    ax = lambda s, l: 2 * (s - lo) + l
    idx = [0] * n
    for w in range(120):
        s, l = m[w]
        if lo <= s < hi: idx[ax(s, l)] = occ[w]
    psi = np.zeros((2,) * n, dtype=complex); psi[tuple(idx)] = 1
    def sl(assign):
        t = [slice(None)] * n
        for a, v in assign.items(): t[a] = v
        return tuple(t)
    res = []
    for b in out:
        if 'step' in b:
            p = np.abs(psi) ** 2
            res.append(np.array([[p[sl({ax(s + lo, l): 1})].sum() for l in (0, 1)] for s in range(L)]))
            if b['step'] == nsteps: break
            continue
        w = b['w']; U = b['U']; sp = [m[x] for x in w]
        if any(not (lo <= s < hi) for s, _ in sp): continue
        if len(w) == 1:
            a = ax(*sp[0]); psi[sl({a: 0})] *= U[0, 0]; psi[sl({a: 1})] *= U[1, 1]
        else:
            a, c = ax(*sp[0]), ax(*sp[1])
            if sp[0][1] != sp[1][1]:
                ph = np.angle(np.diag(U)); g = ph[3] - ph[2] - ph[1] + ph[0]
                g = (g + np.pi) % (2 * np.pi) - np.pi
                d = np.diag(U) * np.array([1, 1, 1, np.exp(1j * (lam - 1) * g)])
                for i0 in (0, 1):
                    for i1 in (0, 1): psi[sl({a: i0, c: i1})] *= d[2 * i0 + i1]
            else:
                x01 = psi[sl({a: 0, c: 1})].copy(); x10 = psi[sl({a: 1, c: 0})].copy()
                psi[sl({a: 0, c: 1})] = U[1, 1] * x01 + U[1, 2] * x10
                psi[sl({a: 1, c: 0})] = U[2, 1] * x01 + U[2, 2] * x10
                psi[sl({a: 0, c: 0})] *= U[0, 0]; psi[sl({a: 1, c: 1})] *= U[3, 3]
    return res
