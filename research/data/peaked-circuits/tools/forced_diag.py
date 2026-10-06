# Diagnostic (mine, parent): at each forced peel in the outer zipper, print the eigenphase
# structure of F^dag E, to test whether the residue is a diagonal controlled phase.
import sys, numpy as np, collections
sys.path.insert(0, '/tmp/peaked-bound')
import zipper2
from zipper2 import Zipper
from solve_peaked_v2 import Core
D = '/tmp/pk/research/data/peaked-circuits/'
name = sys.argv[1]; kmax = int(sys.argv[2])
orig_candidate = Zipper.candidate
def cand(self, ri, qi):
    d, u, A = orig_candidate(self, ri, qi)
    if d > 0.05:
        k = len(self.S)
        F = np.multiply.outer(u, A.reshape((2,) * (2 * k - 2)))
        F = np.moveaxis(F, [0, 1], [ri, k + qi]).reshape(2 ** k, 2 ** k)
        X = F.conj().T @ self.E
        lam = np.linalg.eigvals(X); ph = np.angle(lam)
        # remove best global phase: use phase of the trace
        ph0 = np.angle(np.trace(X)); rel = np.round(((ph - ph0 + np.pi) % (2 * np.pi) - np.pi) / (np.pi / 4), 2)
        offdiag = np.linalg.norm(X - np.diag(np.diag(X))) / np.linalg.norm(X)
        Xg = X * np.exp(-1j * ph0)
        Pm = (Xg * np.sqrt(2) - np.eye(2 ** k)) / 1j       # if X ~ e^{i ph0}(I + iP)/sqrt2
        herm = np.linalg.norm(Pm - Pm.conj().T) / np.linalg.norm(Pm); inv = np.linalg.norm(Pm @ Pm - np.eye(2 ** k)) / np.sqrt(2 ** k)
        # Pauli decomposition weight: project onto single-wire Paulis via partial traces of support
        T = Pm.reshape((2,) * (2 * k)); supp = []
        for w in range(k):
            Mw = np.moveaxis(T, [w, k + w], [0, 1]).reshape(4, -1)
            # wire w is trivial in P iff P = I_w (x) rest
            G = Mw @ Mw.conj().T; ev = np.linalg.eigvalsh(G)
            if 1 - ev[-1] / np.trace(G).real > 1e-3: supp.append(self.S[w])
        print(f'  PAULI-ROT check: herm-err {herm:.3f} P^2=I err {inv:.3f} support wires {supp} (S={self.S})', flush=True)
        print(f'  BIG d={d:.4f} k={k} eig/(pi/4) hist {sorted(collections.Counter(rel.tolist()).items())[:8]} offdiag(compbasis) {offdiag:.3f}', flush=True)
    return d, u, A
Zipper.candidate = cand
c = Core(D + f'peaked_circuit_{name}.qasm'); maps = dict(c.maps)
gates = []; cut = None; mp = list(range(c.n))
for si in range(1, len(c.secs) - 1):
    if si in maps:
        mp = [mp[maps[si][q]] for q in range(c.n)]
        if cut is None: cut = len(gates)
        continue
    lo, hi = c.secs[si]
    for k in range(lo, hi + 1):
        a, b = c.units[k][:2]; gates.append(((mp[a], mp[b]), c.M[k]))
z = Zipper(gates, cut, c.n, kmax=kmax, tau=0.05, log=None).run_paired(maxsteps=int(sys.argv[3]) if len(sys.argv) > 3 else None)
print('TOTAL', z.cost, 'forced', z.forced)
