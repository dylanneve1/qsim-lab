"""Ladder MPS TEBD (site = rung, d=4) driven by the decoded gauss.blocks gate list, with the
inter-chain phase g scaled by --lam (free part untouched).  lam=0 is the free circuit, whose
exact answer is known from gauss.py, so TEBD(lam)-TEBD(0) cancels most truncation error.
Output: tebd_lam_{circ}_chi{chi}_lam{lam}.json"""
import numpy as np, scipy.linalg as sla, os, json, time, argparse
import gauss
ap = argparse.ArgumentParser()
ap.add_argument('--circ', default='SCV'); ap.add_argument('--chi', type=int, default=64)
ap.add_argument('--lam', type=float, default=1.0); ap.add_argument('--nsteps', type=int, default=8)
ap.add_argument('--cut', type=float, default=1e-14); ap.add_argument('--out', default=None)
a = ap.parse_args()
path = os.environ.get('SU2_CIRCUITS', 'circuits') + f'/x_100_{a.circ}.qasm'
occ, m, blist = gauss.blocks(path)
N = 60
st = [[0, 0] for _ in range(N)]
for w in range(120):
    s, l = m[w]; st[s][l] = occ[w]
A = []
for s in range(N):
    t = np.zeros((1, 4, 1), complex); t[0, 2 * st[s][0] + st[s][1], 0] = 1; A.append(t)
c = 0
def move(to):
    global c
    while c < to:
        Dl, d, Dr = A[c].shape; Q, R = np.linalg.qr(A[c].reshape(Dl * d, Dr)); A[c] = Q.reshape(Dl, d, -1)
        A[c + 1] = np.tensordot(R, A[c + 1], 1); c += 1
    while c > to:
        Dl, d, Dr = A[c].shape; Q, R = np.linalg.qr(A[c].reshape(Dl, d * Dr).T); A[c] = Q.T.reshape(-1, d, Dr)
        A[c - 1] = np.tensordot(A[c - 1], R.T, 1); c -= 1
Zi = np.array([1, 1, -1, -1.]); Zo = np.array([1, -1, 1, -1.])
def measure():
    move(0); out = []
    for s in range(N):
        move(s); p = np.einsum('ajb,ajb->j', A[s], A[s].conj()).real; p = p / p.sum()
        out.append(((1 - p @ Zi) / 2, (1 - p @ Zo) / 2))
    return out
logF = 0.0; maxchi = 1; disc = 0.0; rec = []; t0 = time.time(); maxdw = 0.0
def apply1(s, U4):
    A[s] = np.einsum('jk,akb->ajb', U4, A[s])
def apply2(r, U16):
    global c, logF, maxchi, disc, maxdw
    if c < r: move(r)
    elif c > r + 1: move(r + 1)
    th = np.einsum('ajb,bkc->ajkc', A[r], A[r + 1]); Dl, _, _, Dr = th.shape
    th = np.einsum('xy,ayc->axc', U16, th.reshape(Dl, 16, Dr)).reshape(Dl * 4, 4 * Dr)
    try: u, sv, vh = np.linalg.svd(th, full_matrices=False)
    except np.linalg.LinAlgError: u, sv, vh = sla.svd(th, full_matrices=False, lapack_driver='gesvd')
    w = sv ** 2; tot = w.sum()
    keep = min(a.chi, max(1, int(np.sum(w / tot > a.cut))))
    dw = w[keep:].sum() / tot; disc += dw; logF += np.log1p(-dw); maxdw = max(maxdw, dw)
    sv = sv[:keep] / np.sqrt(w[:keep].sum())
    A[r] = u[:, :keep].reshape(Dl, 4, keep); A[r + 1] = (sv[:, None] * vh[:keep]).reshape(keep, 4, Dr); c = r + 1
    maxchi = max(maxchi, keep)
pend = None   # (r, U16)
def flush():
    global pend
    if pend is not None: apply2(*pend); pend = None
for b in blist:
    if 'step' in b:
        flush(); k = b['step']; z = measure()
        nn = np.array(z)
        stag = float(sum((-1) ** r * (nn[r, 0] + nn[r, 1]) for r in range(N)))
        rec.append(dict(step=k, Q=float(nn.sum()), stag=stag, n=nn.tolist(), fid=float(np.exp(logF)),
                        disc=disc, maxdw=maxdw, maxchi=maxchi, t=time.time() - t0))
        print(f"step {k:2d} stag={stag:.8f} Q={nn.sum():.8f} fid={np.exp(logF):.8f} maxdw={maxdw:.1e} chi={maxchi} t={time.time()-t0:.0f}s", flush=True)
        if k == a.nsteps: break
        continue
    w = b['w']; U = b['U'].copy(); sl = [m[x] for x in w]
    if len(w) == 1:
        s, l = sl[0]; U4 = gauss.embed(U, [l], 2); r0 = s; two = False
    elif sl[0][1] != sl[1][1]:
        (s1, l1), (s2, l2) = sl
        ph = np.angle(np.diag(U)); g = ph[3] - ph[2] - ph[1] + ph[0]; g = (g + np.pi) % (2 * np.pi) - np.pi
        U[3, 3] *= np.exp(1j * (a.lam - 1) * g)
        U4 = gauss.embed(U, [l1, l2], 2); r0 = s1; two = False
    else:
        (s1, l1), (s2, l2) = sl; r0 = min(s1, s2); two = True
        U16 = gauss.embed(U, [2 * (s1 - r0) + l1, 2 * (s2 - r0) + l2], 4)
    if not two:
        if pend is not None and r0 in (pend[0], pend[0] + 1):
            U16b = np.kron(U4, np.eye(4)) if r0 == pend[0] else np.kron(np.eye(4), U4)
            pend = (pend[0], U16b @ pend[1])
        else:
            apply1(r0, U4)
    else:
        if pend is not None and pend[0] != r0: flush()
        if pend is None: pend = (r0, np.eye(16, dtype=complex))
        pend = (r0, U16 @ pend[1])
out = a.out or f'tebd_lam_{a.circ}_chi{a.chi}_lam{a.lam:g}.json'
json.dump(rec, open(out, 'w'))
