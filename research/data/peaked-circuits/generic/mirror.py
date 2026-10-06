"""Operator-level mirror finder: grow a window W = units[lo:hi] around a candidate centre and track the
Heisenberg images  phi_W(g) = W^dag g W  of all 2n single-qubit generators (X_q, Z_q) as sparse Pauli sums.

If W sits inside an (obfuscated) identity block  T[U] |> U^dag  around its mirror centre, W is close to a
wire permutation and every image stays (close to) a single-qubit Pauli; outside, images spread.
No thresholds on the circuit structure: the only knob is the numerical truncation eps of Pauli coefficients.
"""
import numpy as np
import pauli as PA


class Window:
    def __init__(self, n, units, c, eps=1e-3, layer=None):
        """c: cut position. If layer (list of unit layers) is given, the cut is 'layer < c', else serial k < c."""
        self.n, self.units, self.eps = n, units, eps
        self.renorm = True
        self.seq = [[] for _ in range(n)]
        for k, u in enumerate(units):
            self.seq[u[0]].append(k); self.seq[u[1]].append(k)
        # per-wire pointers: nxt[q] = position in seq[q] of the first gate after the window
        import bisect
        if layer is None:
            self.nxt = [bisect.bisect_left(self.seq[q], c) for q in range(n)]
        else:
            self.nxt = [sum(1 for k in self.seq[q] if layer[k] < c) for q in range(n)]
        self.prv = [self.nxt[q] - 1 for q in range(n)]
        self.inside = set()
        self.img = []
        for q in range(n):
            self.img.append({(1 << q, 0): 1.0 + 0j})   # X_q
            self.img.append({(0, 1 << q): 1.0 + 0j})   # Z_q
        self.T = {}
        self.sup = [self._sup(S) for S in self.img]
        self.cache = {}

    @staticmethod
    def _sup(S):
        m = 0
        for (x, z) in S:
            m |= x | z
        return m

    def table(self, k):
        if k not in self.T:
            a, b, Pa, Pb = self.units[k]
            G = np.diag([1, 1, 1, -1]).astype(complex) @ np.kron(Pa, Pb)
            self.T[k] = PA.conj_table2(G)
        return self.T[k]

    def size(self):
        return sum(len(s) for s in self.img)

    def loss(self):
        return sum(1 - PA.norm2(s) for s in self.img)

    # --- candidate updates (return {gen: new_sum}) ---
    def frontier(self):
        aft, bef = set(), set()
        for q in range(self.n):
            if self.nxt[q] < len(self.seq[q]):
                k = self.seq[q][self.nxt[q]]; a, b = self.units[k][:2]; o = b if a == q else a
                if self.nxt[o] < len(self.seq[o]) and self.seq[o][self.nxt[o]] == k:
                    aft.add(k)
            if self.prv[q] >= 0:
                k = self.seq[q][self.prv[q]]; a, b = self.units[k][:2]; o = b if a == q else a
                if self.prv[o] >= 0 and self.seq[o][self.prv[o]] == k:
                    bef.add(k)
        return sorted(aft), sorted(bef)

    def cand_after(self, k):
        """absorb unit k (applied after the window): phi' = phi_W o phi_b on b's generators."""
        a, b = self.units[k][:2]
        T = self.table(k)
        # images of local Paulis via phi_W
        I = {(0, 0): 1.0 + 0j}
        def loc1(q, s):        # phi_W(i^{xz} X^x Z^z) on wire q
            x, z = s & 1, s >> 1
            out = I
            if x: out = self.img[2 * q]
            if z: out = PA.sum_mul(out, self.img[2 * q + 1], self.eps / 10)
            if x and z: out = {p: 1j * c for p, c in out.items()}
            return out
        La = [loc1(a, s) for s in range(4)]
        Lb = [loc1(b, s) for s in range(4)]
        cache = {}
        new = {}
        for g, s in ((2 * a, 1), (2 * a + 1, 2), (2 * b, 4), (2 * b + 1, 8)):
            acc = {}
            for t, ct in T[s]:
                if t not in cache:
                    cache[t] = PA.sum_mul(La[t & 3], Lb[t >> 2], self.eps / 10)
                for p, c in cache[t].items():
                    acc[p] = acc.get(p, 0) + ct * c
            new[g] = self._norm({p: c for p, c in acc.items() if abs(c) > self.eps})
        return new

    def _norm(self, S):
        if not self.renorm:
            return S
        nrm = sum(abs(c) ** 2 for c in S.values()) ** 0.5
        return {p: c / nrm for p, c in S.items()} if nrm > 0 else S

    def cand_before(self, k):
        """absorb unit k (applied before the window): phi' = phi_a o phi_W on every image."""
        a, b = self.units[k][:2]
        T = self.table(k)
        mask = (1 << a) | (1 << b)
        new = {}
        for g, S in enumerate(self.img):
            if self.sup[g] & mask:
                new[g] = self._norm(PA.conj_sum2(S, T, a, b, self.eps))
        return new

    def delta(self, new):
        return sum(len(s) for s in new.values()) - sum(len(self.img[g]) for g in new)

    @staticmethod
    def wsum(S):
        """mean Pauli weight of the (renormalised) image; truncated norm does not count as progress."""
        nrm = sum(abs(c) ** 2 for c in S.values())
        return sum(abs(c) ** 2 * (x | z).bit_count() for (x, z), c in S.items()) / max(nrm, 1e-300)

    def dweight(self, new):
        return sum(self.wsum(s) for s in new.values()) - sum(self.wsum(self.img[g]) for g in new)

    def weight(self):
        return sum(self.wsum(s) for s in self.img)

    def evaluate(self, side, k):
        """cached (dweight, dterms, new) for a frontier candidate."""
        key = (side, k)
        if key not in self.cache:
            new = self.cand_after(k) if side == 'after' else self.cand_before(k)
            self.cache[key] = (round(self.dweight(new), 9), self.delta(new), new)
        return self.cache[key]

    def commit(self, new, side, k):
        touched = 0
        for g, s in new.items():
            touched |= self.sup[g]
            self.img[g] = s
            self.sup[g] = self._sup(s)
            touched |= self.sup[g]
        changed = set(new)
        drop = []
        for (sd, kk) in self.cache:
            a, b = self.units[kk][:2]
            if kk == k:
                drop.append((sd, kk)); continue
            if sd == 'after':
                if {2 * a, 2 * a + 1, 2 * b, 2 * b + 1} & changed:
                    drop.append((sd, kk))
            else:
                if touched & ((1 << a) | (1 << b)):
                    drop.append((sd, kk))
        for key in drop:
            del self.cache[key]
        a, b = self.units[k][:2]
        for q in (a, b):
            if side == 'after':
                self.nxt[q] += 1
            else:
                self.prv[q] -= 1
        self.inside.add(k)

    def perm(self):
        """If every Z_q image is (dominated by) a single-qubit operator on wire w, return the map q -> w."""
        m = {}
        for q in range(self.n):
            S = self.img[2 * q + 1]
            w = {}
            for (x, z), c in S.items():
                sup = x | z
                if sup and (sup & (sup - 1)) == 0:
                    w[sup.bit_length() - 1] = w.get(sup.bit_length() - 1, 0) + abs(c) ** 2
            if w:
                best = max(w, key=w.get)
                m[q] = (best, w[best])
        return m



def grow(n, units, c, budget, eps=1e-3, max_steps=None, log=None, W=None, layer=None):
    """Greedy growth from the prefix cut c: at every step evaluate every frontier gate on both sides and
    absorb the one whose image term count grows least (ties: smaller |k - c|). Stop when the total term
    count would exceed `budget` or nothing is left."""
    W = W or Window(n, units, c, eps, layer)
    pos = layer if layer is not None else list(range(len(units)))
    steps = 0
    while True:
        aft, bef = W.frontier()
        best = None
        for side, ks in (('after', aft), ('before', bef)):
            for k in ks:
                dw, dt, new = W.evaluate(side, k)
                key = (dw, dt, abs(pos[k] - c), k)
                if best is None or key < best[0]:
                    best = (key, side, k, new)
        if best is None or W.size() + best[0][1] > budget:
            break
        W.commit(best[3], best[1], best[2])
        steps += 1
        if log and steps % log == 0:
            print(f"    step {steps}: inside {len(W.inside)} size {W.size()} weight {W.weight():.1f} loss {W.loss():.3e}", flush=True)
        if max_steps and steps >= max_steps:
            break
    return W
