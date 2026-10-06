"""Middle-out operator cancellation with SVD compression and free wire relabelling.

The window operator W (a contiguous-in-DAG part of the circuit around a mirror centre) is stored as a
tensor network with exactly one tensor per wire slot.  Each tensor carries ONE output leg ('o', q) and ONE
input leg ('i', p) - not necessarily of the same wire - plus bond legs to other tensors.  So a wire
permutation costs nothing: it is just which output leg sits on which tensor.

Absorbing a 2-qubit gate G on wires (a, b):
  after  (W -> G W): contract G into the tensors holding ('o', a), ('o', b);
  before (W -> W G): contract G into the tensors holding ('i', a), ('i', b);
then split the merged tensor back into two by SVD, trying BOTH assignments of the two new gate legs to the
two tensors and keeping the one with the smaller rank (this is what removes swaps for free).
Truncation: singular values below `cutoff` * s_max are dropped (the only approximation).
"""
import itertools, math
import numpy as np

_bond_counter = itertools.count()


def newbond():
    return ('b', next(_bond_counter))


class TNO:
    def __init__(self, n, cutoff=1e-6, max_bond=None):
        self.n, self.cutoff, self.max_bond = n, cutoff, max_bond
        self.T = {}        # tid -> (array, legs)
        self.out_at = {}   # wire -> tid
        self.in_at = {}
        for q in range(n):
            self.T[q] = (np.eye(2, dtype=complex), [('o', q), ('i', q)])
            self.out_at[q] = q; self.in_at[q] = q
        self.trunc_err = 0.0      # accumulated discarded weight (relative, sum of s^2/|s|^2)

    # ------------------------------------------------------------ helpers
    def size(self):
        return sum(a.size for a, _ in self.T.values())

    def max_bond_dim(self):
        m = 1
        for a, legs in self.T.values():
            for d, l in zip(a.shape, legs):
                if l[0] == 'b':
                    m = max(m, d)
        return m

    def neighbours(self, t):
        a, legs = self.T[t]
        bl = {l for l in legs if l[0] == 'b'}
        out = set()
        for u, (b, lg) in self.T.items():
            if u != t and bl & set(lg):
                out.add(u)
        return out

    @staticmethod
    def _contract(A, la, B, lb):
        common = [l for l in la if l in lb]
        ia = [la.index(l) for l in common]; ib = [lb.index(l) for l in common]
        C = np.tensordot(A, B, axes=(ia, ib))
        lc = [l for l in la if l not in common] + [l for l in lb if l not in common]
        return C, lc

    def _split(self, C, lc, left, right):
        """split C (legs lc) into L(left legs + bond) and R(bond + right legs) by truncated SVD."""
        pl = [lc.index(l) for l in left]; pr = [lc.index(l) for l in right]
        Cp = np.transpose(C, pl + pr)
        dl = int(np.prod([C.shape[i] for i in pl])); dr = int(np.prod([C.shape[i] for i in pr]))
        M = Cp.reshape(dl, dr)
        try:
            U, s, Vh = np.linalg.svd(M, full_matrices=False)
        except np.linalg.LinAlgError:
            U, s, Vh = np.linalg.svd(M + 1e-14 * np.random.randn(*M.shape), full_matrices=False)
        tot = float(np.sum(s ** 2))
        keep = max(1, int(np.sum(s > self.cutoff * s[0])))
        if self.max_bond:
            keep = min(keep, self.max_bond)
        err = float(np.sum(s[keep:] ** 2)) / tot if tot > 0 else 0.0
        U, s, Vh = U[:, :keep], s[:keep], Vh[:keep]
        sq = np.sqrt(s)
        bl = newbond()
        Lt = (U * sq).reshape([C.shape[i] for i in pl] + [keep])
        Rt = (sq[:, None] * Vh).reshape([keep] + [C.shape[i] for i in pr])
        return (Lt, left + [bl]), (Rt, [bl] + right), keep, err

    # ------------------------------------------------------------ gate absorption
    def gate(self, G, a, b, side, try_swap=True, commit=True):
        """absorb 4x4 G (first factor = wire a) after ('after') or before ('before') the window.
        Returns the new bond rank between the two tensors."""
        kind = 'o' if side == 'after' else 'i'
        at = self.out_at if side == 'after' else self.in_at
        x, y = at[a], at[b]
        Gt = G.reshape(2, 2, 2, 2)                          # (a_out, b_out, a_in, b_in)
        if side == 'after':
            # new outputs ('o',a),('o',b) ; old outputs contracted
            oa, ob = ('x', a), ('x', b)                        # temporary names for the old legs
            Ga = (Gt, [('o', a), ('o', b), oa, ob])
        else:
            oa, ob = ('x', a), ('x', b)
            Ga = (Gt, [oa, ob, ('i', a), ('i', b)])
        Ax, lx = self.T[x]; Ay, ly = self.T[y]
        lx = [oa if l == (kind, a) else (ob if l == (kind, b) else l) for l in lx]
        ly = [oa if l == (kind, a) else (ob if l == (kind, b) else l) for l in ly]
        if x == y:
            raise RuntimeError("two legs of the same kind on one tensor")
        C, lc = self._contract(Ax, lx, Ay, ly)
        C, lc = self._contract(C, lc, Ga[0], Ga[1])
        newa, newb = (kind, a), (kind, b)
        rest_x = [l for l in lx if l not in (oa, ob) and l not in ly]
        rest_y = [l for l in ly if l not in (oa, ob) and l not in lx]
        best = None
        opts = [(newa, newb)] + ([(newb, newa)] if try_swap else [])
        for gx, gy in opts:
            r = self._split(C, lc, rest_x + [gx], [gy] + rest_y)
            if best is None or r[2] < best[2][2]:
                best = ((gx, gy), None, r)
        (gx, gy), _, (Lx, Ry, keep, err) = best
        if commit:
            self.T[x] = Lx; self.T[y] = Ry
            at[gx[1]] = x; at[gy[1]] = y
            self.trunc_err += err
        return keep

    def compress_pair(self, x, y):
        """re-split the (x, y) pair without a gate (cleans up bonds that became reducible)."""
        Ax, lx = self.T[x]; Ay, ly = self.T[y]
        if not (set(l for l in lx if l[0] == 'b') & set(ly)):
            return
        C, lc = self._contract(Ax, lx, Ay, ly)
        rest_x = [l for l in lx if l not in ly]; rest_y = [l for l in ly if l not in lx]
        Lx, Ry, keep, err = self._split(C, lc, rest_x, rest_y)
        self.T[x] = Lx; self.T[y] = Ry
        self.trunc_err += err

    def drop_trivial_bonds(self):
        """bond legs of dimension 1 are removed (squeezed)."""
        for t, (A, legs) in list(self.T.items()):
            keep = [i for i, (d, l) in enumerate(zip(A.shape, legs)) if not (l[0] == 'b' and d == 1)]
            if len(keep) != len(legs):
                self.T[t] = (A.reshape([A.shape[i] for i in keep]), [legs[i] for i in keep])

    def normalise(self):
        for t, (A, legs) in self.T.items():
            nrm = np.linalg.norm(A)
            if nrm > 0:
                self.T[t] = (A / nrm * math.sqrt(2.0), legs)   # identity site tensor has norm sqrt(2)

    def bond_graph(self):
        owner = {}
        E = {}
        for t, (A, legs) in self.T.items():
            for d, l in zip(A.shape, legs):
                if l[0] == 'b':
                    owner.setdefault(l, []).append((t, d))
        for l, v in owner.items():
            (t1, d), (t2, _) = v
            key = (min(t1, t2), max(t1, t2))
            E[key] = E.get(key, 1) * d
        return E

    def perm(self):
        """wire map if the TNO is a pure permutation: out wire -> in wire, from tensors with no bonds."""
        m = {}
        for t, (A, legs) in self.T.items():
            o = [l[1] for l in legs if l[0] == 'o'][0]; i = [l[1] for l in legs if l[0] == 'i'][0]
            m[o] = i
        return m


# ---------------------------------------------------------------- quimb bridge (canonical compression)
def _ixname(l):
    if l[0] == 'o':
        return f"k{l[1]}"
    if l[0] == 'i':
        return f"b{l[1]}"
    return f"_bd{l[1]}"


def _ixleg(s):
    if s.startswith("k"):
        return ('o', int(s[1:]))
    if s.startswith("b"):
        return ('i', int(s[1:]))
    return ('b', s)


def to_quimb(T):
    import quimb.tensor as qtn
    ts = [qtn.Tensor(A, inds=[_ixname(l) for l in legs], tags={f"S{t}"}) for t, (A, legs) in T.T.items()]
    return qtn.TensorNetwork(ts)


def from_quimb(T, tn):
    newT = {}
    for t in list(T.T):
        tq = tn[f"S{t}"]
        newT[t] = (np.asarray(tq.data), [_ixleg(ix) for ix in tq.inds])
    T.T = newT


def canonical_compress(T, cutoff=1e-3, max_bond=None, method='local-late'):
    from quimb.tensor.tensor_arbgeom_compress import tensor_network_ag_compress
    tn = to_quimb(T)
    tn = tensor_network_ag_compress(tn, max_bond=max_bond, cutoff=cutoff, method=method,
                                    site_tags=[f"S{t}" for t in T.T], canonize=True, equalize_norms=True)
    tn.squeeze_()
    # equalize_norms spreads a global scale factor into tn.exponent; drop it (W is renormalised anyway)
    from_quimb(T, tn)


def unswap_pass(T, sides=('o', 'i'), min_bond=1):
    """KD-style unswapping in arbitrary geometry: for every bonded pair (x, y) try exchanging their output legs,
    their input legs, or both, re-split by SVD and keep the assignment with the smallest bond rank.
    Exchanging legs between tensors is free (it only changes which wire sits on which tensor).
    Returns the number of accepted exchanges."""
    acc = 0
    for (x, y), d in sorted(T.bond_graph().items(), key=lambda kv: -kv[1]):
        if d < min_bond or x not in T.T or y not in T.T:
            continue
        Ax, lx = T.T[x]; Ay, ly = T.T[y]
        if not (set(l for l in lx if l[0] == 'b') & set(ly)):
            continue
        C, lc = T._contract(Ax, lx, Ay, ly)
        rx = [l for l in lx if l not in ly]; ry = [l for l in ly if l not in lx]
        ox = [l for l in rx if l[0] == 'o'][0]; oy = [l for l in ry if l[0] == 'o'][0]
        ix = [l for l in rx if l[0] == 'i'][0]; iy = [l for l in ry if l[0] == 'i'][0]
        opts = []
        for so in ((False, True) if 'o' in sides else (False,)):
            for si in ((False, True) if 'i' in sides else (False,)):
                left = [l for l in rx if l not in (ox, ix)] + [oy if so else ox, iy if si else ix]
                right = [l for l in ry if l not in (oy, iy)] + [ox if so else oy, ix if si else iy]
                r = T._split(C, lc, left, right)
                opts.append((r[2], so or si, so, si, r))
        best = min(opts, key=lambda o: (o[0], o[1]))
        k, moved, so, si, (Lx, Ry, keep, err) = best
        cur = [o for o in opts if not o[1]][0][0]
        if moved and k < cur:
            T.T[x] = Lx; T.T[y] = Ry; T.trunc_err += err
            if so:
                T.out_at[ox[1]], T.out_at[oy[1]] = y, x
            if si:
                T.in_at[ix[1]], T.in_at[iy[1]] = y, x
            acc += 1
    return acc
