"""An independent dense numpy simulator used as the reference in the tests.

It shares nothing with the Rust engine except the documented conventions
(little-endian qubit order, gate definitions of API.md §5).
"""

from __future__ import annotations

import cmath
import math

import numpy as np

SQ2 = 1 / math.sqrt(2)


def mat1(name, ps=()):
    c, s = (math.cos(ps[0] / 2), math.sin(ps[0] / 2)) if ps else (0.0, 0.0)
    e = cmath.exp
    return {
        "i": np.eye(2),
        "h": np.array([[SQ2, SQ2], [SQ2, -SQ2]]),
        "x": np.array([[0, 1], [1, 0]]),
        "y": np.array([[0, -1j], [1j, 0]]),
        "z": np.diag([1, -1]),
        "s": np.diag([1, 1j]),
        "sdg": np.diag([1, -1j]),
        "t": np.diag([1, e(1j * math.pi / 4)]),
        "tdg": np.diag([1, e(-1j * math.pi / 4)]),
        "sx": 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]),
        "sxdg": 0.5 * np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]),
        "rx": np.array([[c, -1j * s], [-1j * s, c]]) if ps else None,
        "ry": np.array([[c, -s], [s, c]]) if ps else None,
        "rz": np.diag([e(-1j * ps[0] / 2), e(1j * ps[0] / 2)]) if ps else None,
        "p": np.diag([1, e(1j * ps[0])]) if ps else None,
        "u": (
            np.array([[c, -e(1j * ps[2]) * s], [e(1j * ps[1]) * s, e(1j * (ps[1] + ps[2])) * c]])
            if len(ps) == 3
            else None
        ),
    }[name]


def mat_multi(name, ps=()):
    """Matrix indexed by sum_k bit(arg_k) * 2^(len-1-k) (first argument most significant)."""
    if name == "cx":
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    if name == "cz":
        return np.diag([1, 1, 1, -1])
    if name == "swap":
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    if name == "iswap":
        return np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
    if name == "iswapdg":
        return np.array([[1, 0, 0, 0], [0, 0, -1j, 0], [0, -1j, 0, 0], [0, 0, 0, 1]])
    if name == "cp":
        return np.diag([1, 1, 1, cmath.exp(1j * ps[0])])
    if name == "ccx":
        m = np.eye(8)
        m[[6, 7]] = m[[7, 6]]
        return m
    raise KeyError(name)


def apply(state, n, m, qubits):
    """Applies matrix m (first qubit most significant) to `qubits` of a little-endian state."""
    k = len(qubits)
    psi = state.reshape([2] * n)  # axis j <-> qubit n-1-j
    axes = [n - 1 - q for q in qubits]
    t = m.reshape([2] * (2 * k))
    psi = np.tensordot(t, psi, axes=(list(range(k, 2 * k)), axes))
    # tensordot puts the k output axes first, in argument order
    rest = [a for a in range(n) if a not in axes]
    order = axes + rest
    inv = np.argsort(order)
    return psi.transpose(inv).reshape(-1)


def statevector(circuit) -> np.ndarray:
    """Dense state of a unitary qsimlab circuit (terminal measurements ignored)."""
    n = circuit.num_qubits
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1
    for ins in circuit.instructions():
        if ins.name == "measure":
            continue
        assert ins.c_if is None, "reference handles unitary circuits only"
        if len(ins.qubits) == 1:
            m = mat1(ins.name, ins.params)
        else:
            m = mat_multi(ins.name, ins.params)
        psi = apply(psi, n, np.asarray(m, dtype=complex), list(ins.qubits))
    return psi * cmath.exp(1j * circuit.global_phase)


PAULI = {"I": np.eye(2), "X": mat1("x"), "Y": mat1("y"), "Z": mat1("z")}


def pauli_expectation(psi, n, ops):
    """ops: dict qubit -> 'X'/'Y'/'Z'."""
    phi = psi.copy()
    for q, p in ops.items():
        phi = apply(phi, n, PAULI[p].astype(complex), [q])
    return float(np.real(np.vdot(psi, phi)))


def random_circuit(rng, n, depth, gates=None):
    from qsimlab import GATES, Circuit

    names = gates or [g for g, (k, _) in GATES.items() if k <= n]
    c = Circuit(n)
    for _ in range(depth):
        g = names[rng.integers(len(names))]
        k, np_ = GATES[g]
        qs = rng.choice(n, size=k, replace=False).tolist()
        ps = rng.uniform(-math.pi, math.pi, size=np_).tolist()
        c.append(g, qs, ps)
    return c
