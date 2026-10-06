import numpy as np, sys, time
sys.path.insert(0,'.')
from mpou import MPOU, SWAP
from match import match_unswap
rng = np.random.default_rng(5)
def haar4():
    Z = (rng.normal(size=(4,4)) + 1j*rng.normal(size=(4,4))); Q, R = np.linalg.qr(Z); return Q*(np.diag(R)/abs(np.diag(R)))
def u1(eps):
    H = rng.normal(size=(2,2)) + 1j*rng.normal(size=(2,2)); H = (H + H.conj().T)/2
    w, v = np.linalg.eigh(H); return v @ np.diag(np.exp(1j*eps*w)) @ v.conj().T
n = 8
V = [(int(a), int(b), haar4()) for a, b in (rng.choice(n, 2, replace=False) for _ in range(16))]
perm = rng.permutation(n); sw = []; cur = list(range(n))
for w in range(n):
    want = int(np.where(perm == w)[0][0]); v = cur.index(want)
    while v > w: sw.append((v-1, v)); cur[v-1], cur[v] = cur[v], cur[v-1]; v -= 1
print('perm', perm.tolist(), 'swaps', len(sw))
for eps in (0.0, 0.05, 0.15):
    full = V + [(a, b, SWAP) for a, b in sw] + [(int(perm[a]), int(perm[b]), G.conj().T @ np.kron(u1(eps), u1(eps))) for a, b, G in reversed(V)]
    c = len(V) + len(sw)//2
    W = MPOU(n, cutoff=1e-6, max_bond=4096)
    for k in range(c, len(full)): W.absorb(*full[k], 'up')
    for k in range(c-1, -1, -1): W.absorb(*full[k], 'dn')
    print('eps', eps, 'no-unswap W:', W.stats(), flush=True)
    t = time.time(); pi, sc, nsw = match_unswap(W, log=print)
    print('   perm recovered:', all(pi[w] == perm[w] for w in range(n)), '%.1fs' % (time.time() - t), 'final', W.stats(), flush=True)
