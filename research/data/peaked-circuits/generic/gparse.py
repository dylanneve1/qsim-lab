#!/usr/bin/env python3
"""Generic QASM 2/3 front end for peaked circuits.

Output: n, units, tail
  units[k] = (a, b, Pa, Pb) meaning: apply Pa on wire a and Pb on wire b, then CZ(a, b).
             Pa / Pb is the full single-qubit segment accumulated on that wire since the previous
             2-qubit gate on it (all pending 1q gates are folded into the NEXT unit).
  tail[q]  = trailing single-qubit gate on wire q after its last 2-qubit unit (identity if none).
So the circuit unitary is   (⊗_q tail[q]) · unit_{K-1} · ... · unit_0   exactly (up to a global phase).

Every CZ-class 2-qubit gate (cz, cx/cnot, cy, rzz(θ) with θ ≡ ±π/2 mod 2π, any diagonal gate with
interaction phase π, swap → 3 cx) is canonicalised exactly to   (L_a ⊗ L_b) · CZ · (R_a ⊗ R_b);
R is folded into the pending segment of its wire before the CZ, L becomes the start of the next one.
Diagonal local factors of diagonal gates are split symmetrically (half before, half after the CZ)
so that a gate and its inverse canonicalise to mirror-image segments.
"""
import math, re, cmath
import numpy as np

I2 = np.eye(2, dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
S = np.diag([1, 1j])
CZ = np.diag([1, 1, 1, -1]).astype(complex)


def U3(th, ph, la):
    return np.array([[math.cos(th / 2), -cmath.exp(1j * la) * math.sin(th / 2)],
                     [cmath.exp(1j * ph) * math.sin(th / 2), cmath.exp(1j * (ph + la)) * math.cos(th / 2)]])


def RZ(t): return np.diag([cmath.exp(-0.5j * t), cmath.exp(0.5j * t)])
def RX(t): return np.array([[math.cos(t / 2), -1j * math.sin(t / 2)], [-1j * math.sin(t / 2), math.cos(t / 2)]])
def RY(t): return np.array([[math.cos(t / 2), -math.sin(t / 2)], [math.sin(t / 2), math.cos(t / 2)]])
def PH(t): return np.diag([1, cmath.exp(1j * t)])


SX = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2
ONEQ = {
    'u': lambda p: U3(*p), 'u3': lambda p: U3(*p), 'U': lambda p: U3(*p),
    'u2': lambda p: U3(math.pi / 2, p[0], p[1]), 'u1': lambda p: PH(p[0]), 'p': lambda p: PH(p[0]),
    'rz': lambda p: RZ(p[0]), 'rx': lambda p: RX(p[0]), 'ry': lambda p: RY(p[0]),
    'x': lambda p: X, 'y': lambda p: Y, 'z': lambda p: Z, 'h': lambda p: H, 'id': lambda p: I2,
    's': lambda p: S, 'sdg': lambda p: S.conj(), 't': lambda p: PH(math.pi / 4), 'tdg': lambda p: PH(-math.pi / 4),
    'sx': lambda p: SX, 'sxdg': lambda p: SX.conj().T,
}


def num(x):
    return float(eval(x.strip(), {"pi": math.pi, "π": math.pi, "__builtins__": {}}))


def twoq_matrix(name, p):
    """Raw 4x4 matrix (first qubit = most significant) for supported 2-qubit gates."""
    if name == 'cz':
        return CZ.copy()
    if name in ('cx', 'cnot', 'CX'):
        return np.kron(I2, H) @ CZ @ np.kron(I2, H)
    if name == 'cy':
        sdgh = H @ S.conj()  # Y = S X Sdg -> CY = (I⊗S H) CZ (I⊗H Sdg)
        return np.kron(I2, S @ H) @ CZ @ np.kron(I2, sdgh)
    if name == 'rzz':
        t = p[0]
        return np.diag([cmath.exp(-0.5j * t), cmath.exp(0.5j * t), cmath.exp(0.5j * t), cmath.exp(-0.5j * t)])
    if name in ('cp', 'cu1', 'cphase'):
        return np.diag([1, 1, 1, cmath.exp(1j * p[0])])
    if name == 'crz':
        return np.diag([1, 1, cmath.exp(-0.5j * p[0]), cmath.exp(0.5j * p[0])])
    raise ValueError(f"unsupported 2-qubit gate {name}")


def canon_cz(M, tol=1e-9):
    """Exact decomposition M = phase · (La⊗Lb) CZ (Ra⊗Rb) for CZ-class M, returns (La, Lb, Ra, Rb)."""
    if np.allclose(M, np.diag(np.diag(M)), atol=tol):         # diagonal gate
        d = np.diag(M)
        chi = cmath.phase(d[0] * d[3] / (d[1] * d[2]))
        if abs(abs(chi) - math.pi) > 1e-7:
            raise ValueError(f"diagonal gate with interaction phase {chi} is not CZ-class")
        # d = g * diag(1, e^{ib}, e^{ia}, -e^{i(a+b)}): local part P(a)⊗P(b), split half/half
        a = cmath.phase(d[2] / d[0]); b = cmath.phase(d[1] / d[0])
        Ha, Hb = PH(a / 2), PH(b / 2)
        return Ha, Hb, Ha, Hb
    raise ValueError("non-diagonal 2-qubit gate must be given by name (cx/cy/swap)")


GATE_DEFS = {}


def tokenize(path):
    """statements of the file; user `gate name(params) args { body }` definitions are stored in GATE_DEFS."""
    txt = open(path).read()
    txt = re.sub(r"//[^\n]*", "", txt)
    GATE_DEFS.clear()
    def grab(m):
        name, params, args, body = m.group(1), m.group(3), m.group(4), m.group(5)
        GATE_DEFS[name] = ([x.strip() for x in params.split(",")] if params else [],
                           [x.strip() for x in args.split(",")],
                           [" ".join(b.split()) for b in body.split(";") if b.strip()])
        return ""
    txt = re.sub(r"gate\s+(\w+)\s*(\(([^)]*)\))?\s*([^{]+)\{([^}]*)\}", grab, txt)
    for stmt in txt.split(";"):
        s = " ".join(stmt.split())
        if s:
            yield s


def expand(name, p, qs):
    """expand a user-defined gate into primitive (name, params, qubits) ops (recursively)."""
    fp, fa, body = GATE_DEFS[name]
    env = dict(zip(fp, p)); env["pi"] = math.pi
    qmap = dict(zip(fa, qs))
    out = []
    for st in body:
        m = re.match(r"([A-Za-z_][\w]*)\s*(\(([^)]*)\))?\s+(.*)$", st)
        nm, prm, args = m.group(1), m.group(3), m.group(4)
        pp = [float(eval(x.strip(), {"__builtins__": {}}, env)) for x in prm.split(",")] if prm else []
        qq = tuple(qmap[x.strip()] for x in args.split(","))
        if nm in GATE_DEFS:
            out += expand(nm, pp, qq)
        else:
            out.append((nm, pp, qq))
    return out


def parse_ops(path):
    """-> n, ops [(name, params, qubits)] with global qubit indices."""
    regs, n, ops = {}, 0, []
    for s in tokenize(path):
        if s.startswith(("OPENQASM", "include", "barrier", "measure", "creg", "bit")) or "measure" in s:
            continue
        m = re.match(r"qreg\s+(\w+)\s*\[(\d+)\]", s) or re.match(r"qubit\s*\[(\d+)\]\s*(\w+)", s)
        if m:
            if s.startswith("qreg"):
                name, size = m.group(1), int(m.group(2))
            else:
                size, name = int(m.group(1)), m.group(2)
            regs[name] = n; n += size
            continue
        m = re.match(r"([A-Za-z_][\w]*)\s*(\(([^)]*)\))?\s+(.*)$", s)
        if not m:
            raise ValueError(f"cannot parse statement: {s!r}")
        name, params, args = m.group(1), m.group(3), m.group(4)
        p = [num(x) for x in params.split(",")] if params else []
        qs = []
        for a in args.split(","):
            r = re.match(r"(\w+)\s*\[(\d+)\]", a.strip())
            if not r:
                raise ValueError(f"bad qubit argument {a!r} in {s!r}")
            qs.append(regs[r.group(1)] + int(r.group(2)))
        if name in GATE_DEFS:
            ops += expand(name, p, tuple(qs))
        else:
            ops.append((name, p, tuple(qs)))
    return n, ops


def to_units(n, ops):
    """Fold single-qubit gates; canonicalise 2-qubit gates to CZ units."""
    pend = [I2.copy() for _ in range(n)]
    units = []

    def cz_unit(a, b, La, Lb, Ra, Rb):
        units.append((a, b, Ra @ pend[a], Rb @ pend[b]))
        pend[a] = La.copy(); pend[b] = Lb.copy()

    for name, p, qs in ops:
        if len(qs) == 1:
            if name not in ONEQ:
                raise ValueError(f"unsupported 1-qubit gate {name}")
            pend[qs[0]] = ONEQ[name](p) @ pend[qs[0]]
        elif len(qs) == 2:
            a, b = qs
            if name == 'swap':                                 # swap = 3 cx
                for c, t in ((a, b), (b, a), (a, b)):
                    cz_unit(c, t, I2, H, I2, H)
                continue
            if name in ('cx', 'cnot', 'CX'):
                cz_unit(a, b, I2, H, I2, H)
                continue
            if name == 'cy':
                cz_unit(a, b, I2, S @ H, I2, H @ S.conj())
                continue
            M = twoq_matrix(name, p)
            La, Lb, Ra, Rb = canon_cz(M)
            cz_unit(a, b, La, Lb, Ra, Rb)
        else:
            raise ValueError(f"{len(qs)}-qubit gate {name} unsupported")
    return units, pend


def parse(path):
    n, ops = parse_ops(path)
    units, tail = to_units(n, ops)
    return n, units, tail


def unit_matrix(u):
    a, b, Pa, Pb = u
    return CZ @ np.kron(Pa, Pb)


# ---------------------------------------------------------------- dense reference simulation
def apply(state, n, M, qs):
    k = len(qs)
    st = np.moveaxis(state.reshape([2] * n), qs, list(range(k)))
    sh = st.shape
    st = (M @ st.reshape(2 ** k, -1)).reshape(sh)
    return np.moveaxis(st, list(range(k)), qs).reshape(-1)


def ref_state(n, ops, psi0):
    s = psi0.copy()
    for name, p, qs in ops:
        if len(qs) == 1:
            s = apply(s, n, ONEQ[name](p), list(qs))
        elif name == 'swap':
            s = apply(s, n, np.eye(4)[[0, 2, 1, 3]], list(qs))
        else:
            s = apply(s, n, twoq_matrix(name, p), list(qs))
    return s


def unit_state(n, units, tail, psi0):
    s = psi0.copy()
    for u in units:
        s = apply(s, n, unit_matrix(u), [u[0], u[1]])
    for q in range(n):
        s = apply(s, n, tail[q], [q])
    return s


# ---------------------------------------------------------------- nearest-in-file folding (pre/post units)
def to_units_nearest(n, ops):
    """Units (a, b, pre_a, pre_b, post_a, post_b): every 1q gate is attached to the 2q gate on its wire that is
    nearest in FILE order (previous one -> post, next one -> pre; ties -> next). Gates before the first 2q gate
    of a wire go to its pre, after the last one to its post. Canonicalisation local factors are put on the
    side they belong to (R before the CZ, L after). The circuit unitary is unchanged."""
    # first pass: canonicalise 2q ops into CZ + locals, keeping op order
    seq = []   # items: ('1', q, M) or ('2', a, b, La, Lb, Ra, Rb)
    for name, p, qs in ops:
        if len(qs) == 1:
            seq.append(('1', qs[0], ONEQ[name](p)))
        else:
            a, b = qs
            if name == 'swap':
                for c, t in ((a, b), (b, a), (a, b)):
                    seq.append(('2', c, t, I2, H, I2, H))
            elif name in ('cx', 'cnot', 'CX'):
                seq.append(('2', a, b, I2, H, I2, H))
            elif name == 'cy':
                seq.append(('2', a, b, I2, S @ H, I2, H @ S.conj()))
            else:
                La, Lb, Ra, Rb = canon_cz(twoq_matrix(name, p))
                seq.append(('2', a, b, La, Lb, Ra, Rb))
    # positions of 2q gates per wire
    pos2 = [[] for _ in range(n)]
    for i, it in enumerate(seq):
        if it[0] == '2':
            pos2[it[1]].append(i); pos2[it[2]].append(i)
    import bisect
    pre = {}; post = {}
    for i, it in enumerate(seq):
        if it[0] != '1':
            continue
        q, M = it[1], it[2]
        j = bisect.bisect_left(pos2[q], i)
        prv = pos2[q][j - 1] if j > 0 else None
        nxt = pos2[q][j] if j < len(pos2[q]) else None
        if nxt is not None and (prv is None or nxt - i <= i - prv):
            pre.setdefault((nxt, q), []).append(M)
        elif prv is not None:
            post.setdefault((prv, q), []).append(M)
        else:
            pre.setdefault(('lone', q), []).append(M)
    def prod(ms):
        out = I2.copy()
        for M in ms:
            out = M @ out
        return out
    units = []
    idx = {}
    for i, it in enumerate(seq):
        if it[0] != '2':
            continue
        _, a, b, La, Lb, Ra, Rb = it
        units.append((a, b, Ra @ prod(pre.get((i, a), [])), Rb @ prod(pre.get((i, b), [])),
                      prod(post.get((i, a), [])) @ La, prod(post.get((i, b), [])) @ Lb))
    lone = [prod(pre.get(('lone', q), [])) for q in range(n)]   # wires without any 2q gate
    return units, lone


def parse_nearest(path):
    n, ops = parse_ops(path)
    units, lone = to_units_nearest(n, ops)
    return n, units, lone


def unit6_matrix(u):
    a, b, pa, pb, qa, qb = u
    return np.kron(qa, qb) @ CZ @ np.kron(pa, pb)


def unit6_state(n, units, lone, psi0):
    s = psi0.copy()
    for q in range(n):
        s = apply(s, n, lone[q], [q])
    for u in units:
        s = apply(s, n, unit6_matrix(u), [u[0], u[1]])
    return s
