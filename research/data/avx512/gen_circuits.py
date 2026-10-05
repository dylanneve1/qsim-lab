#!/usr/bin/env python3
"""Deterministic circuit generator for the AVX-512 state-vector comparison.

One generator, one gate-list text format, read by every simulator in the study
(qsim-lab via examples/sv_file_bench.rs, qsimcirq, qulacs, qiskit-aer and
lightning.qubit via baselines.py), so all of them run identical circuits.

Format (one gate per line; qubit k = bit k of the state index, little-endian):

    n <num_qubits>
    h q | x q | rx q t | ry q t | rz q t          rz(t) = diag(e^{-it/2}, e^{it/2})
    u3 q theta phi lam      qiskit U: [[c, -e^{i lam} s], [e^{i phi} s, e^{i(phi+lam)} c]], c = cos(theta/2)
    cx c t | cz a b | cp a b t (diag(1,1,1,e^{it})) | swap a b
    rzz a b t               exp(-i t/2 Z(x)Z)
    u4 a b re00 im00 re01 im01 ...   dense 4x4, row-major, local index bit0 = qubit a, bit1 = qubit b

Workloads: qft, brick_cz (Sycamore-like sqrt(X/Y/W) + CZ, 20 cycles), brick_su4
(20 layers of Haar SU(4) on neighbours), qv (n x n quantum volume), qaoa (MaxCut,
random 3-regular graph, p = 6). SU(4) workloads are written twice: `<wl>_<n>.txt`
(exact KAK decomposition into u3 + cx, checked to 1e-12 up to global phase) and
`<wl>_<n>.dense.txt` (u4 gates).

usage: gen_circuits.py <outdir> <wl[,wl...]|all> <n[,n...]>
       gen_circuits.py --selftest
"""
import math
import os
import sys

import numpy as np

WORKLOADS = ["qft", "brick_cz", "brick_su4", "qv", "qaoa"]
SEED = {"qft": 0, "brick_cz": 101, "brick_su4": 202, "qv": 303, "qaoa": 404}
DEPTH = 20  # brick_cz cycles, brick_su4 layers
QAOA_P = 6


# ----------------------------------------------------------------------------- matrices

def u3_mat(th, ph, la):
    c, s = math.cos(th / 2), math.sin(th / 2)
    return np.array([[c, -np.exp(1j * la) * s],
                     [np.exp(1j * ph) * s, np.exp(1j * (ph + la)) * c]], dtype=complex)


def rx_mat(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry_mat(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz_mat(t):
    return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])


H_MAT = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
X_MAT = np.array([[0, 1], [1, 0]], dtype=complex)
Y_MAT = np.array([[0, -1j], [1j, 0]], dtype=complex)


def sqrtm_unitary(m):
    w, v = np.linalg.eig(m)
    return v @ np.diag(np.sqrt(w)) @ np.linalg.inv(v)


# Sycamore single-qubit gates as u3 (equal up to a global phase; checked below)
SYC_1Q = {
    "sx": (math.pi / 2, -math.pi / 2, math.pi / 2),
    "sy": (math.pi / 2, 0.0, 0.0),
    "sw": (math.pi / 2, -math.pi / 4, math.pi / 4),
}
SYC_TARGET = {
    "sx": sqrtm_unitary(X_MAT),
    "sy": sqrtm_unitary(Y_MAT),
    "sw": sqrtm_unitary((X_MAT + Y_MAT) / math.sqrt(2)),
}


def phase_dist(a, b):
    """min over global phases of max |a - e^{i phi} b|."""
    k = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    ph = a[k] / b[k]
    ph /= abs(ph)
    return float(np.abs(a - ph * b).max())


def haar_unitary(d, rng):
    z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / math.sqrt(2)
    q, r = np.linalg.qr(z)
    q = q * (np.diag(r) / np.abs(np.diag(r)))
    return q / np.linalg.det(q) ** (1.0 / d)  # SU(d)


_DECOMP = None


def kak(u):
    """u (4x4, local bit0 = qubit a) -> list of ('u3', which, th, ph, la) / ('cx', c, t)
    with which/c/t in {0 (= a), 1 (= b)}; exact up to a global phase."""
    global _DECOMP
    if _DECOMP is None:
        from qiskit.circuit.library import CXGate
        from qiskit.synthesis import TwoQubitBasisDecomposer
        _DECOMP = TwoQubitBasisDecomposer(CXGate(), euler_basis="U")
    qc = _DECOMP(u)
    out = []
    for ins in qc.data:
        qs = [qc.find_bit(q).index for q in ins.qubits]
        name = ins.operation.name
        if name == "u":
            th, ph, la = (float(p) for p in ins.operation.params)
            out.append(("u3", qs[0], th, ph, la))
        elif name == "cx":
            out.append(("cx", qs[0], qs[1]))
        else:
            raise ValueError(f"unexpected gate {name} in KAK decomposition")
    # check: rebuild in numpy
    v = np.eye(4, dtype=complex)
    for g in out:
        v = two_qubit_of(g) @ v
    err = phase_dist(u, v)
    assert err < 1e-12, f"KAK decomposition error {err}"
    return out


def two_qubit_of(g):
    """4x4 matrix (local bit0 = a, bit1 = b) of a decomposition element."""
    if g[0] == "u3":
        m = u3_mat(*g[2:])
        return np.kron(np.eye(2), m) if g[1] == 0 else np.kron(m, np.eye(2))
    c, t = g[1], g[2]
    m = np.zeros((4, 4), dtype=complex)
    for s in range(4):
        bits = [s & 1, s >> 1]
        if bits[c]:
            bits[t] ^= 1
        m[bits[0] | (bits[1] << 1), s] = 1
    return m


# ----------------------------------------------------------------------------- workloads

def fmt(x):
    return repr(float(x))


def gen_qft(n):
    lines = []
    init = 0x5A5A5A5A & ((1 << n) - 1)
    for q in range(n):
        if (init >> q) & 1:
            lines.append(f"x {q}")
    for j in reversed(range(n)):
        lines.append(f"h {j}")
        for k in reversed(range(j)):
            lines.append(f"cp {k} {j} {fmt(math.pi / (1 << (j - k)))}")
    for j in range(n // 2):
        lines.append(f"swap {j} {n - 1 - j}")
    return lines, None


def gen_brick_cz(n):
    rng = np.random.default_rng(SEED["brick_cz"] + n)
    names = list(SYC_1Q)
    prev = [None] * n
    lines = []
    for cyc in range(DEPTH):
        for q in range(n):
            choices = [g for g in names if g != prev[q]]
            g = choices[rng.integers(len(choices))]
            prev[q] = g
            th, ph, la = SYC_1Q[g]
            lines.append(f"u3 {q} {fmt(th)} {fmt(ph)} {fmt(la)}")
        for q in range(cyc % 2, n - 1, 2):
            lines.append(f"cz {q} {q + 1}")
    return lines, None


def su4_lines(a, b, u):
    dec = [f"u4 {a} {b} " + " ".join(f"{fmt(z.real)} {fmt(z.imag)}" for z in u.ravel())]
    kak_lines = []
    qs = (a, b)
    for g in kak(u):
        if g[0] == "u3":
            kak_lines.append(f"u3 {qs[g[1]]} {fmt(g[2])} {fmt(g[3])} {fmt(g[4])}")
        else:
            kak_lines.append(f"cx {qs[g[1]]} {qs[g[2]]}")
    return kak_lines, dec


def gen_brick_su4(n):
    rng = np.random.default_rng(SEED["brick_su4"] + n)
    lines, dense = [], []
    for layer in range(DEPTH):
        for q in range(layer % 2, n - 1, 2):
            k, d = su4_lines(q, q + 1, haar_unitary(4, rng))
            lines += k
            dense += d
    return lines, dense


def gen_qv(n):
    rng = np.random.default_rng(SEED["qv"] + n)
    lines, dense = [], []
    for _ in range(n):
        perm = rng.permutation(n)
        for i in range(n // 2):
            a, b = int(perm[2 * i]), int(perm[2 * i + 1])
            k, d = su4_lines(a, b, haar_unitary(4, rng))
            lines += k
            dense += d
    return lines, dense


def regular3(n, seed):
    try:
        import networkx as nx
        g = nx.random_regular_graph(3, n, seed=seed)
        return sorted(tuple(sorted(e)) for e in g.edges())
    except ImportError:  # pairing model with rejection
        rng = np.random.default_rng(seed)
        while True:
            pts = rng.permutation(np.repeat(np.arange(n), 3))
            edges = set()
            ok = True
            for i in range(0, len(pts), 2):
                a, b = sorted((int(pts[i]), int(pts[i + 1])))
                if a == b or (a, b) in edges:
                    ok = False
                    break
                edges.add((a, b))
            if ok:
                return sorted(edges)


def gen_qaoa(n):
    rng = np.random.default_rng(SEED["qaoa"] + n)
    edges = regular3(n, SEED["qaoa"] + n)
    gammas = rng.uniform(0, math.pi, QAOA_P)
    betas = rng.uniform(0, math.pi, QAOA_P)
    lines = [f"h {q}" for q in range(n)]
    for p in range(QAOA_P):
        for a, b in edges:
            lines.append(f"rzz {a} {b} {fmt(gammas[p])}")
        for q in range(n):
            lines.append(f"rx {q} {fmt(betas[p])}")
    return lines, None


GEN = {"qft": gen_qft, "brick_cz": gen_brick_cz, "brick_su4": gen_brick_su4,
       "qv": gen_qv, "qaoa": gen_qaoa}


def write(outdir, wl, n):
    lines, dense = GEN[wl](n)
    paths = []
    p = os.path.join(outdir, f"{wl}_{n}.txt")
    with open(p, "w") as f:
        f.write(f"# {wl} n={n} (gen_circuits.py, seed {SEED[wl]}+n)\nn {n}\n")
        f.write("\n".join(lines) + "\n")
    paths.append(p)
    if dense is not None:
        p = os.path.join(outdir, f"{wl}_{n}.dense.txt")
        with open(p, "w") as f:
            f.write(f"# {wl} n={n} dense u4 form (gen_circuits.py, seed {SEED[wl]}+n)\nn {n}\n")
            f.write("\n".join(dense) + "\n")
        paths.append(p)
    return paths


# ----------------------------------------------------------------------------- reader + numpy reference

def read(path):
    """-> (n, [(name, qubits tuple, params tuple)])"""
    n, gates = None, []
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t or t[0].startswith("#"):
                continue
            if t[0] == "n":
                n = int(t[1])
            elif t[0] in ("h", "x"):
                gates.append((t[0], (int(t[1]),), ()))
            elif t[0] in ("rx", "ry", "rz"):
                gates.append((t[0], (int(t[1]),), (float(t[2]),)))
            elif t[0] == "u3":
                gates.append(("u3", (int(t[1]),), tuple(float(x) for x in t[2:5])))
            elif t[0] in ("cx", "cz", "swap"):
                gates.append((t[0], (int(t[1]), int(t[2])), ()))
            elif t[0] in ("cp", "rzz"):
                gates.append((t[0], (int(t[1]), int(t[2])), (float(t[3]),)))
            elif t[0] == "u4":
                v = [float(x) for x in t[3:35]]
                m = np.array(v[0::2]) + 1j * np.array(v[1::2])
                gates.append(("u4", (int(t[1]), int(t[2])), (m.reshape(4, 4),)))
            else:
                raise ValueError(f"{path}: unknown gate {t[0]}")
    return n, gates


def mat1(name, params):
    if name == "h":
        return H_MAT
    if name == "x":
        return X_MAT
    if name == "rx":
        return rx_mat(*params)
    if name == "ry":
        return ry_mat(*params)
    if name == "rz":
        return rz_mat(*params)
    if name == "u3":
        return u3_mat(*params)
    raise KeyError(name)


def mat2(name, params):
    """4x4, local bit0 = first qubit, bit1 = second."""
    if name == "u4":
        return params[0]
    if name == "cx":
        return two_qubit_of(("cx", 0, 1))
    if name == "cz":
        return np.diag([1, 1, 1, -1]).astype(complex)
    if name == "cp":
        return np.diag([1, 1, 1, np.exp(1j * params[0])])
    if name == "swap":
        m = np.zeros((4, 4), dtype=complex)
        for s in range(4):
            m[((s & 1) << 1) | (s >> 1), s] = 1
        return m
    if name == "rzz":
        t = params[0]
        return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t), np.exp(0.5j * t), np.exp(-0.5j * t)])
    raise KeyError(name)


def reference_state(path):
    """Plain numpy state vector (index bit k = qubit k), complex128, n <= 16."""
    n, gates = read(path)
    assert n <= 16
    psi = np.zeros((2,) * n, dtype=complex)  # axis i <-> qubit n-1-i
    psi[(0,) * n] = 1
    ax = lambda q: n - 1 - q
    for name, qs, params in gates:
        if len(qs) == 1:
            m = mat1(name, params)
            psi = np.moveaxis(np.tensordot(m, psi, axes=([1], [ax(qs[0])])), 0, ax(qs[0]))
        else:
            a, b = qs
            m = mat2(name, params).reshape(2, 2, 2, 2)  # [b_out, a_out, b_in, a_in]
            psi = np.tensordot(m, psi, axes=([2, 3], [ax(b), ax(a)]))
            psi = np.moveaxis(psi, [0, 1], [ax(b), ax(a)])
    return psi.reshape(-1)


def selftest():
    for g, (th, ph, la) in SYC_1Q.items():
        e = phase_dist(SYC_TARGET[g], u3_mat(th, ph, la))
        assert e < 1e-12, (g, e)
        print(f"sqrt-{g[1].upper()} as u3: max error up to phase {e:.1e}")
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(50):
        u = haar_unitary(4, rng)
        v = np.eye(4, dtype=complex)
        for g in kak(u):
            v = two_qubit_of(g) @ v
        worst = max(worst, phase_dist(u, v))
    print(f"KAK: 50 Haar SU(4), worst error up to phase {worst:.1e}")
    # dense vs decomposed files give the same state (up to phase)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        for wl in WORKLOADS:
            ps = write(d, wl, 8)
            s = [reference_state(p) for p in ps]
            assert abs(np.linalg.norm(s[0]) - 1) < 1e-12
            if len(s) == 2:
                f = abs(np.vdot(s[0], s[1])) ** 2
                assert abs(f - 1) < 1e-12, (wl, f)
                print(f"{wl}: dense vs KAK file fidelity {f:.15f}")
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
        sys.exit(0)
    outdir, wls, ns = sys.argv[1], sys.argv[2], sys.argv[3]
    wls = WORKLOADS if wls == "all" else wls.split(",")
    os.makedirs(outdir, exist_ok=True)
    for n in (int(x) for x in ns.split(",")):
        for wl in wls:
            for p in write(outdir, wl, n):
                print(p)
