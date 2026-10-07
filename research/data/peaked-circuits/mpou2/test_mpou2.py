import numpy as np, sys, time
sys.path.insert(0,'.')
from mpou2 import MPO2, SWAP, mps_marginals
from match2 import match_unswap
rng = np.random.default_rng(1)
def haar4():
    Z = (rng.normal(size=(4,4)) + 1j*rng.normal(size=(4,4))); Q, R = np.linalg.qr(Z); return Q*(np.diag(R)/abs(np.diag(R)))
def dense(n, units):
    U = np.eye(2**n, dtype=complex).reshape([2]*n + [2**n])
    for a, b, G in units:
        T = G.reshape(2,2,2,2); U = np.tensordot(T, U, axes=([2,3],[a,b])); U = np.moveaxis(U, [0,1], [a,b])
    return U.reshape(2**n, 2**n)
def todense(W):
    n=W.n; W.move(0)
    T = W.A[0]
    for s in range(1, n): T = np.tensordot(T, W.A[s], axes=(T.ndim-1, 0))
    T = T.reshape([2]*(2*n)) * np.exp(W.lognorm)
    perm = [2*W.pu[w] for w in range(n)] + [2*W.pd[w]+1 for w in range(n)]
    return np.transpose(T, perm).reshape(2**n, 2**n)
n = 7
units = [(int(a), int(b), haar4()) for a, b in (rng.choice(n, 2, replace=False) for _ in range(30))]
Uex = dense(n, units)
for centre in (0, 12, 30):
    W = MPO2(n, eps=1e-14, mode='sum2', max_bond=10**6)
    L = list(range(centre-1, -1, -1)); R = list(range(centre, len(units)))
    while L or R:
        if R: k = R.pop(0); W.absorb(*units[k], 'up')
        if L: k = L.pop(0); W.absorb(*units[k], 'dn')
        if rng.random() < 0.3: W.unswap_sweep()
        if rng.random() < 0.2: W.swap(int(rng.integers(n-1)), ['up','dn','both'][int(rng.integers(3))])
    D = todense(W)
    print('centre', centre, 'operator error', np.linalg.norm(D - Uex) / np.linalg.norm(Uex), W.stats())
    M = W.state(); p0, nrm = mps_marginals(M)
    psi = Uex[:, 0]; probs = np.abs(psi.reshape([2]*n))**2
    p0ex = [probs.take(0, axis=w).sum() for w in range(n)]
    print('   marginal err', np.max(np.abs(np.array([p0[W.pu[w]] for w in range(n)]) - np.array(p0ex))))
# relabelled mirror with a swap network: must collapse to bond 1
n = 8
units2 = [(int(a), int(b), haar4()) for a, b in (rng.choice(n, 2, replace=False) for _ in range(30))]
perm = rng.permutation(n); sw = []
cur = list(range(n))
for w in range(n):
    want = int(np.where(perm == w)[0][0]); v = cur.index(want)
    while v > w: sw.append((v-1, v)); cur[v-1], cur[v] = cur[v], cur[v-1]; v -= 1
full = units2 + [(a, b, SWAP) for a, b in sw] + [(int(perm[a]), int(perm[b]), G.conj().T) for a, b, G in reversed(units2)]
c = len(units2) + len(sw)//2
for mode in ('greedy','match'):
    W = MPO2(n, eps=1e-12, max_bond=10**6)
    L = list(range(c-1, -1, -1)); R = list(range(c, len(full))); mx = 0
    while L or R:
        if R: k = R.pop(0); W.absorb(*full[k], 'up')
        if L: k = L.pop(0); W.absorb(*full[k], 'dn')
        if mode=='greedy':
            for _ in range(3):
                if W.unswap_sweep(thr=2) == 0: break
        mx = max(mx, W.stats()['max_bond'])
    if mode=='match':
        print(' before match', W.stats()); match_unswap(W, log=print)
        for _ in range(4):
            if W.unswap_sweep(thr=2)==0: break
    print('relabelled mirror', mode, 'final', W.stats(), 'max during', mx, 'err', np.linalg.norm(todense(W)-dense(n,full))/2**(n/2))
