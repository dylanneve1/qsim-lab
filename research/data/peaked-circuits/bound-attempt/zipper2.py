"""Rigorous mirror-zipper, v2 (see zipper.py docstring).  Changes vs v1:
 * peel bound uses the optimal global phase: ||E - e^{ith} F|| = max_j |e^{i phi_j} - e^{ith}|
   where e^{i phi_j} are the eigenvalues of the unitary F^dag E (F unitary, so this is exact);
   th = centre of the smallest arc containing all phi_j.
 * the product factor F = A (x) u is refined by alternating polar steps (maximises |Tr F^dag E|).
 * incremental screening: after absorbing a gate only (out r, in q) pairs that the gate can
   change are screened; a full screen follows every peel.
"""
import numpy as np, collections
from zipper import Zipper as Z1, polar_unitary, I2

def arc_norm(X):
    """X unitary. Returns (min_th max_j |lam_j - e^{i th}|, th)."""
    lam = np.linalg.eigvals(X)
    ph = np.sort(np.angle(lam))
    gaps = np.diff(np.concatenate([ph, [ph[0] + 2 * np.pi]]))
    j = int(np.argmax(gaps))                      # largest empty gap; arc = complement
    start = ph[(j + 1) % len(ph)]; width = 2 * np.pi - gaps[j]
    th = start + width / 2
    half = width / 2
    return (2 * np.sin(half / 2) if half < np.pi else 2.0), th

class Zipper(Z1):
    def __init__(self, *a, refine=6, **kw):
        super().__init__(*a, **kw)
        self.refine = refine; self.touched = None; self.forced_costs = []; self.nfs = []

    def apply(self, U, wires, left):
        super().apply(U, wires, left)
        self.touched = (set(wires), left)

    def screen(self, full=False):
        k = len(self.S); T = self.E.reshape((2,) * (2 * k)); out = []
        tw = None if (full or self.touched is None) else self.touched
        for ri in range(k):
            for qi in range(k):
                if tw is not None:
                    w, left = tw
                    if left and self.S[ri] not in w and not (self.S[qi] in w): continue
                    if (not left) and self.S[qi] not in w and not (self.S[ri] in w): continue
                M = np.moveaxis(T, [ri, k + qi], [0, 1]).reshape(4, -1)
                G = M @ M.conj().T
                ev = np.linalg.eigvalsh(G)
                out.append((1 - ev[-1] / np.trace(G).real, ri, qi))
        out.sort()
        return out

    def candidate(self, ri, qi):
        k = len(self.S); T = self.E.reshape((2,) * (2 * k))
        Tm = np.moveaxis(T, [ri, k + qi], [0, 1])            # (o_r, i_q, rest_out..., rest_in...)
        M = Tm.reshape(4, -1)
        U_, s, Vh = np.linalg.svd(M, full_matrices=False)
        u = polar_unitary(U_[:, 0].reshape(2, 2))
        D = 2 ** (k - 1)
        Mr = Tm.reshape(2, 2, D, D)
        A = polar_unitary(np.einsum('ab,abij->ij', u.conj(), Mr))
        for _ in range(self.refine):
            u = polar_unitary(np.einsum('abij,ij->ab', Mr, A.conj()))
            A = polar_unitary(np.einsum('ab,abij->ij', u.conj(), Mr))
        F = np.multiply.outer(u, A.reshape((2,) * (2 * k - 2)))
        F = np.moveaxis(F, [0, 1], [ri, k + qi]).reshape(2 ** k, 2 ** k)
        X = F.conj().T @ self.E
        d, th = arc_norm(X)
        # average-case (normalised Frobenius) deviation, phase chosen to maximise overlap; NOT a bound
        self._nf = float(np.sqrt(max(0.0, 2 - 2 * abs(np.trace(X)) / X.shape[0])))
        return d, u, A

    def peel(self, *a):
        super().peel(*a)
        self.touched = None          # force a full screen next time

    def peel_all(self, force=False):
        while self.S:
            sc = self.screen()
            done = False
            for res, ri, qi in sc[:6]:
                if res > 4 * self.tau ** 2 + 1e-12 and not force: break
                d, u, A = self.candidate(ri, qi)
                if d <= self.tau or force:
                    self.peel(ri, qi, d, u, A); done = True
                    self.nfs.append(self._nf)
                    if force: self.forced_costs.append(d)
                    if force and self.log: self.log(f'    forced peel d={d:.3e} k={len(self.S)+1}')
                    break
            if not done: return
            if force: force = len(self.S) > self.kmax

def run_paired(self, maxsteps=None, nlook=8):
    """driver: (1) gates inside the support; (2) V/W frontier gates acting on the same input-frame
    wire pair (mirror partners), preferring pairs touching the support; (3) limited lookahead."""
    steps = 0
    while self.V or self.W:
        fr = sorted(self.frontier(), key=lambda x: self.dist(*x))
        Sset = set(self.S)
        inside = [x for x in fr if all(q in Sset for q in self.in_wires(*x))]
        if inside:
            picks = [inside[0]]
        else:
            vw = collections.defaultdict(list)
            for side, i in fr: vw[frozenset(self.in_wires(side, i))].append((side, i))
            pairs = []
            for key, lst in vw.items():
                vs = [x for x in lst if x[0] == 'V']; ws = [x for x in lst if x[0] == 'W']
                if vs and ws:
                    pairs.append((-len(key & Sset), self.dist(*vs[0]) + self.dist(*ws[0]), vs[0], ws[0]))
            pairs.sort()
            if pairs and (pairs[0][0] < 0 or not self.S):
                picks = [pairs[0][2], pairs[0][3]]
            else:
                cands = sorted(fr, key=lambda x: (sum(q not in Sset for q in self.in_wires(*x)), self.dist(*x)))[:nlook]
                best = None
                for side, i in cands:
                    sn = self.snap()
                    self.absorb(side, i); self.peel_cheap()
                    key = (len(self.S), self.dist(side, i))
                    self.restore(sn)
                    if best is None or key < best[0]: best = (key, (side, i))
                picks = [best[1]]
        for p in picks:
            self.absorb(*p); steps += 1
            self.journal = []
            self.peel_all(); self.journal = []
            if len(self.S) > self.kmax:
                self.forced += 1
                self.peel_all(force=True); self.journal = []
        if self.log and (steps // 20) != ((steps - len(picks)) // 20):
            self.log(f'  step {steps} V {len(self.V)} W {len(self.W)} k {len(self.S)} cost {self.cost:.4f} forced {self.forced}')
        if maxsteps and steps >= maxsteps: break
    self.peel_all()
    while self.S:
        self.peel_all(force=True)
    return self
Zipper.run_paired = run_paired
