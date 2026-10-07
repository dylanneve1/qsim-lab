"""Factored ("reduced") SVD splits for tno.TNO - drop-in replacements for TNO.gate and tno.unswap_pass.

Profiling (logs/prof_tnos_p9_vps.log) shows that the swap-aware P9 growth (tnos.grow_adaptive, unswap=True)
spends ~96% of its time in numpy SVDs of the FULL merged pair matrix  M = C.reshape(dl, dr)  in TNO._split,
although rank(M) <= (pair bond) x (gate rank) is tiny.  Here each side is first QR-reduced over the legs that
stay with it (its other bonds, and for gates also its physical legs that are not touched):
    A_x = Q_x R_x   (Q_x isometric over the legs kept by x)
so  M = (Q_x (x) Q_y) core  with  core = R_x . R_y [. G]  and  svals(M) = svals(core)  exactly.
The SVD is done on the small core, and U / Vh are mapped back with Q_x / Q_y.  Same truncation rule, same
rank and discarded weight (up to round-off); only the cost changes.  dtype follows the tensors (complex128 or
complex64 if the TNO was built with complex64)."""
import math
import numpy as np
import tno as TN


def _qr_reduce(A, legs, keep):
    """A (legs) = Q (keep + [k]) . R ([k] + rest).  Returns None if no reduction is possible."""
    pk = [legs.index(l) for l in keep]
    pr = [i for i in range(len(legs)) if i not in pk]
    dk = math.prod(A.shape[i] for i in pk); dr = math.prod(A.shape[i] for i in pr)
    if not pk or dk <= dr:
        return None
    M = np.transpose(A, pk + pr).reshape(dk, dr)
    Q, R = np.linalg.qr(M)
    k = TN.newbond()
    return ((Q.reshape([A.shape[i] for i in pk] + [Q.shape[1]]), [legs[i] for i in pk] + [k]),
            (R.reshape([R.shape[0]] + [A.shape[i] for i in pr]), [k] + [legs[i] for i in pr]), k)


SVALS_ONLY = True      # level 2: choose among the leg assignments with singular values only (gesdd, no U/V),
                       # then do one full split for the chosen assignment (only if it is committed)


def _rank(self, C, lc, left, right):
    """keep-rank of the split of C into (left | right), same rule as TNO._split (cutoff, max_bond)."""
    pl = [lc.index(l) for l in left]; pr = [lc.index(l) for l in right]
    dl = math.prod(C.shape[i] for i in pl); dr = math.prod(C.shape[i] for i in pr)
    M = np.transpose(C, pl + pr).reshape(dl, dr)
    try:
        s = np.linalg.svd(M, compute_uv=False)
    except np.linalg.LinAlgError:
        s = np.linalg.svd(M + 1e-14 * np.random.randn(*M.shape), compute_uv=False)
    keep = max(1, int(np.sum(s > self.cutoff * s[0])))
    if self.max_bond:
        keep = min(keep, self.max_bond)
    return keep


def _expand(T, red, side):
    """map a split factor back through Q: T=(arr, legs) carries leg k of red."""
    if red is None:
        return T
    (Q, lq), _, k = red
    if side == 'L':
        C, lc = TN.TNO._contract(Q, lq, T[0], T[1])          # keep-legs first, then T's other legs
    else:
        C, lc = TN.TNO._contract(T[0], T[1], Q, lq)
    return (C, lc)


def gate_fast(self, G, a, b, side, try_swap=True, commit=True):
    """= TNO.gate with the merged-pair SVD done on the QR-reduced core."""
    kind = 'o' if side == 'after' else 'i'
    at = self.out_at if side == 'after' else self.in_at
    x, y = at[a], at[b]
    dt = self.T[x][0].dtype
    Gt = np.asarray(G, dtype=dt).reshape(2, 2, 2, 2)
    oa, ob = ('x', a), ('x', b)
    Ga = (Gt, [('o', a), ('o', b), oa, ob]) if side == 'after' else (Gt, [oa, ob, ('i', a), ('i', b)])
    Ax, lx = self.T[x]; Ay, ly = self.T[y]
    lx = [oa if l == (kind, a) else (ob if l == (kind, b) else l) for l in lx]
    ly = [oa if l == (kind, a) else (ob if l == (kind, b) else l) for l in ly]
    if x == y:
        raise RuntimeError("two legs of the same kind on one tensor")
    rest_x = [l for l in lx if l not in (oa, ob) and l not in ly]
    rest_y = [l for l in ly if l not in (oa, ob) and l not in lx]
    rx = _qr_reduce(Ax, lx, rest_x); ry = _qr_reduce(Ay, ly, rest_y)
    Cx = rx[1] if rx else (Ax, lx); Cy = ry[1] if ry else (Ay, ly)
    C, lc = self._contract(Cx[0], Cx[1], Cy[0], Cy[1])
    C, lc = self._contract(C, lc, Ga[0], Ga[1])
    cx = [rx[2]] if rx else rest_x; cy = [ry[2]] if ry else rest_y
    newa, newb = (kind, a), (kind, b)
    best = None
    opts = [(newa, newb)] + ([(newb, newa)] if try_swap else [])
    if SVALS_ONLY and len(opts) > 1:
        ks = [_rank(self, C, lc, cx + [gx], [gy] + cy) for gx, gy in opts]
        gx, gy = opts[int(np.argmin(ks))]          # first minimum, as the strict '<' below
        best = ((gx, gy), self._split(C, lc, cx + [gx], [gy] + cy))
    else:
        for gx, gy in opts:
            r = self._split(C, lc, cx + [gx], [gy] + cy)
            if best is None or r[2] < best[1][2]:
                best = ((gx, gy), r)
    (gx, gy), (Lx, Ry, keep, err) = best
    if commit:
        self.T[x] = _expand(Lx, rx, 'L'); self.T[y] = _expand(Ry, ry, 'R')
        at[gx[1]] = x; at[gy[1]] = y
        self.trunc_err += err
    return keep


def unswap_pass_fast(T, sides=('o', 'i'), min_bond=1):
    """= tno.unswap_pass with the four trial SVDs done on the QR-reduced core."""
    acc = 0
    for (x, y), d in sorted(T.bond_graph().items(), key=lambda kv: -kv[1]):
        if d < min_bond or x not in T.T or y not in T.T:
            continue
        Ax, lx = T.T[x]; Ay, ly = T.T[y]
        if not (set(l for l in lx if l[0] == 'b') & set(ly)):
            continue
        rxl = [l for l in lx if l not in ly]; ryl = [l for l in ly if l not in lx]
        ox = [l for l in rxl if l[0] == 'o'][0]; oy = [l for l in ryl if l[0] == 'o'][0]
        ix = [l for l in rxl if l[0] == 'i'][0]; iy = [l for l in ryl if l[0] == 'i'][0]
        bx = [l for l in rxl if l not in (ox, ix)]; by = [l for l in ryl if l not in (oy, iy)]
        redx = _qr_reduce(Ax, lx, bx); redy = _qr_reduce(Ay, ly, by)
        Cx = redx[1] if redx else (Ax, lx); Cy = redy[1] if redy else (Ay, ly)
        C, lc = T._contract(Cx[0], Cx[1], Cy[0], Cy[1])
        kx = [redx[2]] if redx else bx; ky = [redy[2]] if redy else by
        opts = []
        for so in ((False, True) if 'o' in sides else (False,)):
            for si in ((False, True) if 'i' in sides else (False,)):
                left = kx + [oy if so else ox, iy if si else ix]
                right = ky + [ox if so else oy, ix if si else iy]
                if SVALS_ONLY:
                    opts.append((_rank(T, C, lc, left, right), so or si, so, si, (left, right)))
                else:
                    r = T._split(C, lc, left, right)
                    opts.append((r[2], so or si, so, si, r))
        best = min(opts, key=lambda o: (o[0], o[1]))
        k, moved, so, si, r = best
        cur = [o for o in opts if not o[1]][0][0]
        if moved and k < cur:
            Lx, Ry, keep, err = T._split(C, lc, *r) if SVALS_ONLY else r
            T.T[x] = _expand(Lx, redx, 'L'); T.T[y] = _expand(Ry, redy, 'R'); T.trunc_err += err
            if so:
                T.out_at[ox[1]], T.out_at[oy[1]] = y, x
            if si:
                T.in_at[ix[1]], T.in_at[iy[1]] = y, x
            acc += 1
    return acc


def install(dtype=None):
    """monkeypatch tno: fast gate / unswap_pass; optionally build TNOs in `dtype` (e.g. np.complex64)."""
    TN.TNO.gate = gate_fast
    TN.unswap_pass = unswap_pass_fast
    if dtype is not None:
        set_dtype(dtype)


def set_dtype(dtype):
    """TNO identity tensors in `dtype`; original gate() casts G to the tensor dtype via this wrapper."""
    orig_init = TN.TNO.__init__
    if getattr(orig_init, '_dtype_patched', False):
        orig_init = orig_init._orig

    def __init__(self, n, cutoff=1e-6, max_bond=None):
        orig_init(self, n, cutoff, max_bond)
        for q in range(n):
            A, legs = self.T[q]
            self.T[q] = (A.astype(dtype), legs)
    __init__._dtype_patched = True; __init__._orig = orig_init
    TN.TNO.__init__ = __init__
    g = TN.TNO.gate
    if not getattr(g, '_cast', False):
        def gate(self, G, *a, **k):
            return g(self, np.asarray(G, dtype=next(iter(self.T.values()))[0].dtype), *a, **k)
        gate._cast = True
        TN.TNO.gate = gate
