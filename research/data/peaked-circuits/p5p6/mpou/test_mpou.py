import numpy as np, sys
sys.path.insert(0,'.')
from mpou import MPOU, SWAP, mps_marginals, mps_amp
rng = np.random.default_rng(0)
def haar4():
    Z = (rng.normal(size=(4,4)) + 1j*rng.normal(size=(4,4))); Q, R = np.linalg.qr(Z); return Q*(np.diag(R)/abs(np.diag(R)))
def dense(n, units):
    U = np.eye(2**n, dtype=complex).reshape([2]*n + [2**n])
    for a, b, G in units:
        T = G.reshape(2,2,2,2); U = np.tensordot(T, U, axes=([2,3],[a,b])); U = np.moveaxis(U, [0,1], [a,b])
    return U.reshape(2**n, 2**n)
n = 7
units = [(int(a), int(b), haar4()) for a, b in (rng.choice(n, 2, replace=False) for _ in range(30))]
Uex = dense(n, units)
for centre in (0, 12, 30):
    W = MPOU(n, cutoff=1e-12, max_bond=10**6)
    L = list(range(centre-1, -1, -1)); R = list(range(centre, len(units)))
    while L or R:
        if R: k = R.pop(0); W.absorb(*units[k], 'up')
        if L: k = L.pop(0); W.absorb(*units[k], 'dn')
        if rng.random() < 0.3: W.unswap_sweep()
    W.move(0)
    # rebuild dense operator: out logical w at site pu[w], in logical w at site pd[w]
    T = W.A[0]
    for s in range(1, n): T = np.tensordot(T, W.A[s], axes=(T.ndim-1, 0))
    T = T.reshape([2]*(2*n)) * np.exp(W.lognorm)    # axes: u0 d0 u1 d1 ...
    perm = [2*W.pu[w] for w in range(n)] + [2*W.pd[w]+1 for w in range(n)]
    D = np.transpose(T, perm).reshape(2**n, 2**n)
    print('centre', centre, 'operator error', np.linalg.norm(D - Uex) / np.linalg.norm(Uex), W.stats())
    M, su = W.state(); p0, nrm = mps_marginals(M)
    psi = Uex[:, 0]; probs = np.abs(psi.reshape([2]*n))**2
    p0ex = [probs.take(0, axis=w).sum() for w in range(n)]
    p0w = [p0[W.pu[w]] for w in range(n)]
    print('   marginal err', np.max(np.abs(np.array(p0w) - np.array(p0ex))))
# relabelled mirror: U, random swap network, pi U^dag pi^-1 -> should collapse to bond 1 with unswapping
units2 = [(int(a), int(b), haar4()) for a, b in (rng.choice(n, 2, replace=False) for _ in range(25))]
perm = rng.permutation(n); pos = list(range(n)); sw = []
# swap network realising content q -> perm[q]
cur = list(range(n))  # cur[w] = content on wire w
for w in range(n):
    want = int(np.where(perm == w)[0][0])  # content that should end on wire w
    v = cur.index(want)
    while v > w: sw.append((v-1, v)); cur[v-1], cur[v] = cur[v], cur[v-1]; v -= 1
full = units2 + [(a, b, SWAP) for a, b in sw] + [(int(perm[a]), int(perm[b]), G.conj().T) for a, b, G in reversed(units2)]
c = len(units2) + len(sw)//2
W = MPOU(n, cutoff=1e-10, max_bond=10**6)
L = list(range(c-1, -1, -1)); R = list(range(c, len(full)))
mx = 0
while L or R:
    if R: k = R.pop(0); W.absorb(*full[k], 'up')
    if L: k = L.pop(0); W.absorb(*full[k], 'dn')
    for _ in range(3):
        if W.unswap_sweep() == 0: break
    mx = max(mx, W.stats()['max_bond'])
print('relabelled mirror: final', W.stats(), 'max during', mx)
