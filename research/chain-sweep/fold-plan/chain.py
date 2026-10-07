"""Python port of qsim-lab src/engines/chain_sweep.rs (compile + truncate[_window]).
Ops: ('U', q, m(2x2 np), ctrl) or ('P', mask, pat, f)."""
import numpy as np, re
import os
QASM = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nq70_depth70_checks27_doped.qasm')
C1, C0 = 1+0j, 0j
S2 = 1.0
r2 = 2**-0.5
MATS = {
 'h': np.array([[r2, r2], [r2, -r2]], complex),
 's': np.array([[1, 0], [0, 1j]], complex),
 'sx': np.array([[1+1j, 1-1j], [1-1j, 1+1j]], complex)/2,
 'sxdg': np.array([[1-1j, 1+1j], [1+1j, 1-1j]], complex)/2,
 't': np.array([[np.exp(-1j*np.pi/8), 0], [0, np.exp(1j*np.pi/8)]]),
}
ID = np.eye(2, dtype=complex); XM = np.array([[0, 1], [1, 0]], complex)

def parse(path=QASM):
    ops = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith(('OPENQASM', 'include', 'qreg', 'creg', 'measure', 'barrier')):
            continue
        name = line.split()[0]; qs = [int(x) for x in re.findall(r'q\[(\d+)\]', line)]
        if name == 'cz': ops.append(('cz', qs[0], qs[1]))
        elif name.startswith('rz'):
            assert 'pi/4' in name; ops.append(('g', qs[0], MATS['t']))
        else: ops.append(('g', qs[0], MATS[name]))
    return 70, ops

def window(n0, ops, n, lo, hi):
    out = []; depth = [0]*n0
    for op in ops:
        if op[0] == 'cz':
            a, b = op[1], op[2]; L = max(depth[a], depth[b]) + 1; depth[a] = depth[b] = L
            if lo < L <= hi and a < n and b < n: out.append(op)
        else:
            q = op[1]
            if q < n and lo <= depth[q] <= hi: out.append(op)
    return out

def chain(n, ops):
    lines = [[] for _ in range(n)]; bond_edge = []; depth = [0]*n
    global BOND_LAYER
    BOND_LAYER = []
    for op in ops:
        if op[0] == 'cz':
            l, r = sorted(op[1:]); assert r == l+1
            L = max(depth[l], depth[r]) + 1; depth[l] = depth[r] = L
            b = len(bond_edge); bond_edge.append(l); BOND_LAYER.append(L)
            lines[l].append(('P', b)); lines[r].append(('Z', b))
        else:
            lines[op[1]].append(('G', op[2]))
    global DMAX
    DMAX = max(BOND_LAYER) if BOND_LAYER else 0
    return lines, bond_edge

class B:
    def __init__(s, emit, slot_of, used, live):
        s.ops = []; s.emit = emit; s.scale = C1; s.slot_of = dict(slot_of)
        s.used = list(used); s.live = live; s.peak = live
    def alloc(s):
        try: c = s.used.index(False)
        except ValueError: s.used.append(False); c = len(s.used)-1
        s.used[c] = True; s.live += 1; s.peak = max(s.peak, s.live); return c
    def free(s, c): s.used[c] = False; s.live -= 1
    def push(s, op):
        if s.emit: s.ops.append(op)
    def u1(s, q, m, ctrl=0): s.push(('U', q, np.array(m, complex), ctrl))
    def phase(s, mask, pat, f):
        if f != C1: s.push(('P', mask, pat, complex(f)))
    def line(s, evs, backward, init, fin):
        mode = ('prod', np.array(init, complex))
        seq = evs[::-1] if backward else evs
        for ev in seq:
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
            else:  # P
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
        if mode[0] == 'prod': s.scale *= fin[0]*mode[1][0] + fin[1]*mode[1][1]
        elif mode[0] == 'tied':
            p, u = mode[1], mode[2]
            for bp in range(2): s.phase(1 << p, bp << p, fin[0]*u[0, bp] + fin[1]*u[1, bp])
        else:
            p = mode[1]; s.u1(p, [[fin[0], fin[1]], [0, 0]]); s.free(p)

def compile_plan(lines, x):
    n = len(lines); b = B(True, {}, [], 0); zero = [1, 0]
    qops, bw_l, cw = [], [], []
    for i in range(n):
        xi = [0, 1] if (x >> i) & 1 else [1, 0]
        best = (10**9, False)
        for bw in (False, True):
            t = B(False, b.slot_of, b.used, b.live)
            init, fin = (xi, zero) if bw else (zero, xi)
            t.line(lines[i], bw, init, fin)
            if t.peak < best[0]: best = (t.peak, bw)
        bw = best[1]; init, fin = (xi, zero) if bw else (zero, xi)
        qops.append(len(b.ops)); b.line(lines[i], bw, init, fin); cw.append(b.live); bw_l.append(bw)
    qops.append(len(b.ops))
    return dict(width=max(len(b.used), 1), ops=b.ops, scale=b.scale, qubit_ops=qops, backward=bw_l, cut_width=cw)

def apply_op(psi, op, W):
    """psi: flat 2^W array, in place-ish; returns new array."""
    if op[0] == 'U':
        _, q, m, ctrl = op
        v = psi.reshape(2**(W-q-1), 2, 2**q)
        a0, a1 = v[:, 0, :].copy(), v[:, 1, :].copy()
        n0 = m[0, 0]*a0 + m[0, 1]*a1; n1 = m[1, 0]*a0 + m[1, 1]*a1
        if ctrl:
            idx = np.arange(2**W).reshape(2**(W-q-1), 2, 2**q)[:, 0, :]
            sel = (idx & ctrl) == ctrl
            n0 = np.where(sel, n0, a0); n1 = np.where(sel, n1, a1)
        v[:, 0, :] = n0; v[:, 1, :] = n1
    else:
        _, mask, pat, f = op
        idx = np.arange(2**W); psi[(idx & mask) == pat] *= f
    return psi

def run(plan):
    W = plan['width']; psi = np.zeros(2**W, complex); psi[0] = 1
    for op in plan['ops']: apply_op(psi, op, W)
    return psi[0]*plan['scale']

def sv_amp(n, ops, x):
    psi = np.zeros(2**n, complex); psi[0] = 1
    for op in ops:
        if op[0] == 'cz':
            idx = np.arange(2**n); psi[((idx >> op[1]) & 1) & ((idx >> op[2]) & 1) == 1] *= -1
        else:
            q = op[1]; v = psi.reshape(2**(n-q-1), 2, 2**q); m = op[2]
            a0, a1 = v[:, 0, :].copy(), v[:, 1, :].copy()
            v[:, 0, :] = m[0, 0]*a0 + m[0, 1]*a1; v[:, 1, :] = m[1, 0]*a0 + m[1, 1]*a1
    return psi[x]
