"""Group-fold pass schedules for the chain sweep (multi-head builder + block executor).

Heads (worldlines) of a group of k neighbouring qubits are compiled together, all in the
same time direction; groups alternate direction (up, down, up, ...), so the pass that
finishes group c at one time-end also starts group c+1 there. Bits are virtual (fresh id per
alloc), each op is tagged (group, head-in-group, processing-time event index)."""
import numpy as np
from chain import *

class MB(B):
    """Builder with fresh virtual bits and op tags."""
    def __init__(s):
        super().__init__(True, {}, [], 0); s.nbits = 0; s.alive = set(); s.tag = None; s.pinned = {}
    def alloc(s):
        c = s.nbits; s.nbits += 1; s.alive.add(c); s.live += 1; s.peak = max(s.peak, s.live); return c
    def free(s, c): s.alive.discard(c); s.live -= 1
    def push(s, op): s.ops.append((op, s.tag))
    def line_gen(s, evs, backward, init, fin, tagbase):
        """Generator version of B.line (same ops); yields before each CZ event, with the
        bond it needs (Z) or None."""
        mode = ('prod', np.array(init, complex))
        seq = evs[::-1] if backward else evs
        t = 0
        for ev in seq:
            if ev[0] != 'G':
                yield ev
                import chain as _c
                L = _c.BOND_LAYER[ev[1]]
                t = L if not backward else _c.DMAX + 1 - L
            s.tag = tagbase + (t,)
            if ev[0] != 'G': s.pinned.pop(tagbase, None)
            if ev[0] == 'P': s.pinned[tagbase] = ev[1]
            if ev[0] == 'G':
                m = ev[1].T if backward else ev[1]
                if mode[0] == 'prod': mode = ('prod', m @ mode[1])
                elif mode[0] == 'tied': mode = ('tied', mode[1], m @ mode[2])
                else: s.u1(mode[1], m)
            elif ev[0] == 'Z':
                a = s.slot_of.pop(ev[1])
                if mode[0] == 'prod':
                    psi = mode[1]
                    s.u1(a, [[psi[0]*S2, psi[0]*S2], [psi[1]*S2, -psi[1]*S2]]); mode = ('live', a)
                elif mode[0] == 'tied':
                    p, u = mode[1], mode[2]; mask = (1 << p) | (1 << a)
                    if np.all(np.abs(u) > 1e-12):
                        b0, b1 = u[0, 0], u[1, 0]; a1 = u[0, 1]/u[0, 0]
                        cc = u[1, 1]*u[0, 0]/(u[1, 0]*u[0, 1])
                        s.u1(a, [[b0*S2, b0*S2], [b1*S2, -b1*S2]])
                        s.phase(1 << p, 1 << p, a1)
                        if abs(cc - 1) > 1e-13: s.phase(mask, mask, cc)
                    else:
                        s.u1(a, [[S2, S2], [S2, -S2]])
                        for bp in range(2):
                            for q in range(2):
                                s.phase(mask, (bp << p) | (q << a), u[q, bp])
                    mode = ('live', a)
                else:
                    p = mode[1]; mask = (1 << p) | (1 << a)
                    s.phase(mask, mask, -1)
                    s.u1(a, [[S2, S2], [0, 0]]); s.free(a)
            else:
                b = ev[1]
                if mode[0] == 'prod':
                    psi = mode[1]; c = s.alloc(); s.slot_of[b] = c
                    s.u1(c, [[psi[0], 0], [psi[1], 0]]); mode = ('tied', c, ID.copy())
                elif mode[0] == 'tied':
                    p, u = mode[1], mode[2]; c = s.alloc(); s.slot_of[b] = c
                    s.u1(c, XM, 1 << p)
                    if not np.array_equal(u, ID): s.u1(c, u)
                    mode = ('tied', c, ID.copy())
                else:
                    p = mode[1]; s.slot_of[b] = p; mode = ('tied', p, ID.copy())
        s.tag = tagbase + (t + 1,)
        if mode[0] == 'prod': s.scale *= fin[0]*mode[1][0] + fin[1]*mode[1][1]
        elif mode[0] == 'tied':
            p, u = mode[1], mode[2]
            for bp in range(2): s.phase(1 << p, bp << p, fin[0]*u[0, bp] + fin[1]*u[1, bp])
        else:
            p = mode[1]; s.u1(p, [[fin[0], fin[1]], [0, 0]]); s.free(p)
        s.pinned.pop(tagbase, None)

def build_groups(lines, x, k, first_dir=False):
    """Compiles qubits in groups of k; group c runs backward iff (c odd) xor first_dir.
    Heads of a group are interleaved round-robin (a head waits for the bond it consumes)."""
    n = len(lines); b = MB(); zero = [1, 0]
    groups = [list(range(c, min(c+k, n))) for c in range(0, n, k)]
    for gi, g in enumerate(groups):
        bw = bool(gi % 2) ^ first_dir
        gens = []
        for hi, q in enumerate(g):
            xi = [0, 1] if (x >> q) & 1 else [1, 0]
            init, fin = (xi, zero) if bw else (zero, xi)
            gens.append([b.line_gen(lines[q], bw, init, fin, (gi, hi)), None, False])
        while not all(h[2] for h in gens):
            for h in gens:
                if h[2]: continue
                while True:
                    if h[1] is not None and h[1][0] == 'Z' and (h[1][1] not in b.slot_of or h[1][1] in b.pinned.values()):
                        break          # needs a bond the previous head has not made yet
                    try:
                        h[1] = next(h[0])
                    except StopIteration:
                        h[2] = True; break

                    # advance one CZ event per round
                    break
    return b, groups

def op_bits(op):
    """(non-diagonal target mask, all touched mask)."""
    if op[0] == 'U':
        _, q, m, ctrl = op
        nd = (1 << q) if (abs(m[0, 1]) > 0 or abs(m[1, 0]) > 0) else 0
        return nd, (1 << q) | ctrl
    return 0, op[1]

def assign_passes(b, groups, k, T0, stride):
    """Pass of each op: group c's ops with processing time t < T0 - stride*head go to pass c,
    the rest to pass c+1 (so pass c+1 = end of group c + start of group c+1)."""
    out = []
    for op, tag in b.ops:
        g, h, t = tag
        out.append(g if t < T0 - stride*h else g + 1)
    return out

def check_legal(ops, passes):
    """The pass-major order is a valid reordering iff every op commutes with every
    program-earlier op placed in a later pass."""
    P = max(passes) + 1
    # scan in program order; for each pass keep OR of nd / touch masks of ops seen so far in later passes
    later_nd = [0]*(P+1); later_t = [0]*(P+1)   # suffix structure: masks of ops seen so far by pass
    nd_by = [0]*P; t_by = [0]*P
    bad = 0
    for (op, _), p in zip(ops, passes):
        nd, t = op_bits(op)
        # earlier ops in passes > p
        snd = 0; st = 0
        for q in range(p+1, P): snd |= nd_by[q]; st |= t_by[q]
        if (nd & st) or (t & snd): bad += 1
        nd_by[p] |= nd; t_by[p] |= t
    return bad

def pass_stats(b, passes):
    """Local (non-diag) bits per pass, bits stored at each boundary."""
    P = max(passes) + 1
    first = {}; last = {}; local = [set() for _ in range(P)]
    for (op, _), p in zip(b.ops, passes):
        nd, t = op_bits(op)
        for v in bits(t):
            first[v] = min(first.get(v, P), p); last[v] = max(last.get(v, -1), p)
        for v in bits(nd): local[p].add(v)
    stored = [sum(1 for v in first if first[v] <= p < last[v]) for p in range(P)]
    # a pass's buffer: its local bits (bits touched only diagonally stay global)
    return [len(l) for l in local], stored, local

def bits(mask):
    out = []; i = 0
    while mask:
        if mask & 1: out.append(i)
        mask >>= 1; i += 1
    return out

def block_execute(b, passes, local_sets, round_fn=None):
    """Out-of-core emulation: the state is a tensor over the bits stored between passes;
    each pass loops over the assignments of its global bits and applies its ops to the
    2^|local| sub-block only. Returns the |0..0> amplitude times the scale."""
    P = max(passes) + 1
    byp = [[] for _ in range(P)]
    for (op, _), p in zip(b.ops, passes): byp[p].append(op)
    last = {}
    for (op, _), p in zip(b.ops, passes):
        for v in bits(op_bits(op)[1]): last[v] = max(last.get(v, -1), p)
    axes = []; st = np.ones((), complex)          # tensor over `axes`
    for p in range(P):
        L = sorted(local_sets[p])
        touched = set()
        for op in byp[p]: touched |= set(bits(op_bits(op)[1]))
        for v in sorted(touched | set(L)):     # bring new bits in as |0>
            if v not in axes:
                axes.append(v); st = np.stack([st, np.zeros_like(st)], axis=-1)
        G = [v for v in axes if v not in L]
        perm = [axes.index(v) for v in G + L]
        st = np.transpose(st, perm); axes = G + L
        shp = st.shape
        st = st.reshape(2**len(G), 2**len(L))
        lpos = {v: i for i, v in enumerate(L)}    # bit i of local index (axis order: last = bit 0)
        nl = len(L)
        lbit = {v: nl - 1 - i for i, v in enumerate(L)}
        gbit = {v: len(G) - 1 - i for i, v in enumerate(G)}
        idx = np.arange(2**nl)
        for gi in range(2**len(G)):
            blk = st[gi].copy()
            gval = lambda v: (gi >> gbit[v]) & 1
            for op in byp[p]:
                if op[0] == 'U':
                    _, q, m, ctrl = op
                    if ctrl and any(gval(v) == 0 for v in bits(ctrl) if v in gbit): continue
                    lc = [v for v in bits(ctrl) if v in lbit]
                    if q in lbit:
                        bq = lbit[q]
                        i0 = idx[((idx >> bq) & 1) == 0]
                        if lc:
                            cm = sum(1 << lbit[v] for v in lc); i0 = i0[(i0 & cm) == cm]
                        i1 = i0 | (1 << bq)
                        a0, a1 = blk[i0].copy(), blk[i1].copy()
                        blk[i0] = m[0, 0]*a0 + m[0, 1]*a1; blk[i1] = m[1, 0]*a0 + m[1, 1]*a1
                    else:
                        assert abs(m[0, 1]) == 0 and abs(m[1, 0]) == 0, 'non-diag op on a global bit'
                        f = m[gval(q), gval(q)]
                        if lc:
                            cm = sum(1 << lbit[v] for v in lc); blk[(idx & cm) == cm] *= f
                        else: blk *= f
                else:
                    _, mask, pat, f = op
                    ok = True; lm = 0; lp = 0
                    for v in bits(mask):
                        want = (pat >> v) & 1
                        if v in gbit:
                            if gval(v) != want: ok = False; break
                        else:
                            lm |= 1 << lbit[v]; lp |= want << lbit[v]
                    if ok: blk[(idx & lm) == lp] *= f
            st[gi] = blk
        st = st.reshape(shp)
        # drop bits that are finished (must be |0>)
        for v in list(axes):
            if last.get(v, P) <= p:
                ax = axes.index(v)
                one = np.take(st, 1, axis=ax)
                assert np.max(np.abs(one), initial=0) <= 1e-12 * max(1e-300, np.max(np.abs(st))), 'freed bit not |0>'
                st = np.take(st, 0, axis=ax); axes.pop(ax)
        if round_fn is not None and p < P - 1: st = round_fn(st)
    assert not axes
    return complex(st) * b.scale
