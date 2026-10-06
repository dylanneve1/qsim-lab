"""Rigorous mirror-zipper for an (approximately) permutation-valued circuit segment.

Segment G (time-ordered 2-qubit gates). Cut G = W . V (V = gates before the cut, W = after).
Maintain the exact identity  G = W_rem . X . V_rem,  X = Pi . E,  E dense on a small wire set S
(input-frame labels), Pi a wire permutation.  Gates are absorbed into E exactly.  A qubit is
"peeled" when E ~= tau.(A (x) u_q) (tau = transposition, possibly trivial); we replace E by that
product, paying delta = || E - e^{i th} tau(A (x) u) ||_op  (exact spectral norm), push u into an
adjacent gate (exact) and update Pi.  Triangle inequality: ||G - Pi_final.(local 1q)|| <= sum delta.
"""
import numpy as np, collections, math

I2 = np.eye(2, dtype=complex)

def polar_unitary(M):
    try:
        U, s, Vh = np.linalg.svd(M)
    except np.linalg.LinAlgError:
        import scipy.linalg as sl
        U, s, Vh = sl.svd(M, lapack_driver='gesvd')
    return U @ Vh

class Zipper:
    def __init__(self, gates, cut, n, kmax=10, tau=1e-9, log=None):
        # gates: list of (wires(a,b), 4x4) in time order; cut: index; V=gates[:cut], W=gates[cut:]
        self.n = n
        self.g = {i: [tuple(w), np.array(m, dtype=complex)] for i, (w, m) in enumerate(gates)}
        self.V = set(range(cut)); self.W = set(range(cut, len(gates)))
        self.seq = collections.defaultdict(list)
        for i, (w, m) in enumerate(gates):
            for q in w: self.seq[q].append(i)
        self.Pi = list(range(n))      # input-frame wire -> output-frame wire
        self.S = []                   # wires of E (input frame)
        self.E = np.eye(1, dtype=complex)
        self.kmax, self.tau = kmax, tau
        self.cost = 0.0; self.costs = []; self.dangling = collections.defaultdict(lambda: I2.copy())
        self.maxk = 0; self.log = log; self.forced = 0; self.journal = []; self.cut = cut

    # ---------- frontier
    def v_last(self, q):
        for i in reversed(self.seq[q]):
            if i in self.V: return i
        return None
    def w_first(self, q):
        for i in self.seq[q]:
            if i in self.W: return i
        return None
    def frontier(self):
        out = set()
        for q in range(self.n):
            i = self.v_last(q)
            if i is not None and all(self.v_last(x) == i for x in self.g[i][0]): out.add(('V', i))
            i = self.w_first(q)
            if i is not None and all(self.w_first(x) == i for x in self.g[i][0]): out.add(('W', i))
        return out
    def inv(self, p):
        return self.Pi.index(p)
    def in_wires(self, side, i):
        w = self.g[i][0]
        return w if side == 'V' else tuple(self.inv(p) for p in w)

    # ---------- dense ops
    def extend(self, q):
        self.S.append(q); self.E = np.kron(self.E, I2)
        self.maxk = max(self.maxk, len(self.S))
    def apply(self, U, wires, left):
        k = len(self.S); ax = [self.S.index(q) for q in wires]
        T = self.E.reshape((2,) * (2 * k))
        G = U.reshape(2, 2, 2, 2)
        if left:   # E <- U E : contract U's input with E's output axes
            T = np.tensordot(G, T, axes=([2, 3], ax))      # new axes: Uout0,Uout1, remaining...
            rest = [a for a in range(2 * k) if a not in ax]
            perm = [0] * (2 * k)
            # current order: [ax0', ax1', rest...] -> place back
            order = ax + rest
            T = np.moveaxis(T, list(range(2 * k)), order)
        else:      # E <- E U : contract E's input axes with U's output
            axi = [k + a for a in ax]
            T = np.tensordot(T, G, axes=(axi, [0, 1]))      # remaining..., Uin0, Uin1
            rest = [a for a in range(2 * k) if a not in axi]
            order = rest + axi
            T = np.moveaxis(T, list(range(2 * k)), order)
        self.E = T.reshape(2 ** k, 2 ** k)
    def absorb(self, side, i):
        w, U = self.g[i]
        iw = self.in_wires(side, i)
        for q in iw:
            if q not in self.S: self.extend(q)
        self.apply(U, iw, left=(side == 'W'))
        (self.V if side == 'V' else self.W).discard(i)

    # ---------- peeling
    def screen(self):
        """residual (normalised Frobenius) for every (out r, in q) pair."""
        k = len(self.S); T = self.E.reshape((2,) * (2 * k)); out = []
        for ri in range(k):
            for qi in range(k):
                M = np.moveaxis(T, [ri, k + qi], [0, 1]).reshape(4, -1)
                G = M @ M.conj().T
                ev = np.linalg.eigvalsh(G)
                out.append((1 - ev[-1] / np.trace(G).real, ri, qi))
        out.sort()
        return out
    def candidate(self, ri, qi):
        k = len(self.S); T = self.E.reshape((2,) * (2 * k))
        M = np.moveaxis(T, [ri, k + qi], [0, 1]).reshape(4, -1)
        U_, s, Vh = np.linalg.svd(M, full_matrices=False)
        u = polar_unitary(U_[:, 0].reshape(2, 2))                  # (out r, in q)
        A = polar_unitary((Vh[0].conj() * 0 + Vh[0]).reshape(2 ** (k - 1), 2 ** (k - 1)))
        # rebuild F in tensor form with axes (out r, in q, rest_out..., rest_in...)
        F = np.multiply.outer(u, A.reshape((2,) * (2 * k - 2)))
        F = np.moveaxis(F, [0, 1], [ri, k + qi]).reshape(2 ** k, 2 ** k)
        ov = np.vdot(F, self.E); F = F * (ov / abs(ov))
        d = np.linalg.norm(self.E - F, 2)
        return d, u, A
    def peel(self, ri, qi, d, u, A):
        k = len(self.S); r, q = self.S[ri], self.S[qi]
        ov = None
        # A: out S\r , in S\q   (orders of S with those removed). Rename out q -> r (tau), keep S\q order.
        Sout = [x for x in self.S if x != r]; Sin = [x for x in self.S if x != q]
        Sout_ren = [r if x == q else x for x in Sout]
        # reorder out axes to match Sin order
        perm = [Sout_ren.index(x) for x in Sin]
        TA = A.reshape((2,) * (2 * k - 2))
        TA = np.transpose(TA, perm + list(range(k - 1, 2 * k - 2)))
        # global phase: fold into nothing (phases are irrelevant for |amplitude|)
        self.S = Sin; self.E = TA.reshape(2 ** (k - 1), 2 ** (k - 1))
        if r != q:
            Pi = self.Pi[:]; Pi[q], Pi[r] = self.Pi[r], self.Pi[q]; self.Pi = Pi
        self.cost += d; self.costs.append(d)
        # u acts on input-frame wire q (after tau): push to V side, else W side, else dangle
        j = self.v_last(q)
        if j is not None:
            w, U = self.g[j]; uu = np.kron(u, I2) if w[0] == q else np.kron(I2, u)
            self.journal.append((j, U)); self.g[j][1] = uu @ U
        else:
            p = self.Pi[q]; j = self.w_first(p)
            if j is not None:
                w, U = self.g[j]; uu = np.kron(u, I2) if w[0] == p else np.kron(I2, u)
                self.journal.append((j, U)); self.g[j][1] = U @ uu
            else:
                self.journal.append(('d', p, self.dangling[p].copy()))
                self.dangling[p] = u @ self.dangling[p]
    def peel_cheap(self, thr=1e-11):
        """lookahead-only: peel pairs whose Frobenius residual < thr (no op-norm, no cost bookkeeping)"""
        while self.S:
            sc = self.screen()
            if sc[0][0] > thr: return
            res, ri, qi = sc[0]
            k = len(self.S); T = self.E.reshape((2,) * (2 * k))
            M = np.moveaxis(T, [ri, k + qi], [0, 1]).reshape(4, -1)
            U_, s_, Vh = np.linalg.svd(M, full_matrices=False)
            self.peel(ri, qi, 0.0, polar_unitary(U_[:, 0].reshape(2, 2)), polar_unitary(Vh[0].reshape(2 ** (k - 1), 2 ** (k - 1))))
    def peel_all(self, force=False):
        while self.S:
            sc = self.screen()
            done = False
            for res, ri, qi in sc[:6]:
                if res > max(self.tau, 1e-6) * 4 and not force: break
                d, u, A = self.candidate(ri, qi)
                if d <= self.tau or force:
                    self.peel(ri, qi, d, u, A); done = True
                    if force and self.log: self.log(f'    forced peel d={d:.3e} k={len(self.S)+1}')
                    break
            if not done: return
            if force: force = len(self.S) > self.kmax

    # ---------- snapshot
    def snap(self):
        self.journal = []
        return (self.E, list(self.S), list(self.Pi), set(self.V), set(self.W), self.cost, len(self.costs), self.maxk)
    def restore(self, sn):
        self.E, self.S, self.Pi, self.V, self.W, self.cost, nc, self.maxk = sn
        self.S = list(self.S); self.Pi = list(self.Pi); self.V = set(self.V); self.W = set(self.W)
        del self.costs[nc:]
        for item in reversed(self.journal):
            if item[0] == 'd': self.dangling[item[1]] = item[2]
            else: self.g[item[0]][1] = item[1]
        self.journal = []

    # ---------- driver
    def dist(self, side, i):
        return (self.cut - 1 - i) if side == 'V' else (i - self.cut)
    def run(self, partner=None, maxsteps=None, lookahead=True):
        partner = partner or {}
        steps = 0
        while self.V or self.W:
            fr = sorted(self.frontier(), key=lambda x: self.dist(*x))
            inside = [(side, i) for side, i in fr if all(q in self.S for q in self.in_wires(side, i))]
            if inside:
                pick = inside[0]
            else:
                cands = sorted(fr, key=lambda x: (sum(q not in self.S for q in self.in_wires(*x)), self.dist(*x)))
                if lookahead:
                    best = None
                    for side, i in cands[:24]:
                        sn = self.snap()
                        self.absorb(side, i); self.peel_cheap()
                        key = (len(self.S), self.cost, self.dist(side, i))
                        self.restore(sn)
                        if best is None or key < best[0]: best = (key, (side, i))
                    pick = best[1]
                else:
                    pick = cands[0]
            self.absorb(*pick); steps += 1
            self.journal = []
            self.peel_all(); self.journal = []
            if self.log and steps % 10 == 0:
                self.log(f'  step {steps} V {len(self.V)} W {len(self.W)} k {len(self.S)} cost {self.cost:.4f} forced {self.forced}')
            if len(self.S) > self.kmax:
                self.forced += 1
                self.peel_all(force=True); self.journal = []
            if maxsteps and steps >= maxsteps: break
        self.peel_all()
        while self.S:
            self.peel_all(force=True)
        self.journal = []
        return self
