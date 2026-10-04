"""Conversions to and from Qiskit, Cirq and Stim.

Every converter imports its framework lazily and raises
:class:`~qsimlab.errors.MissingDependencyError` with an install hint if it is
missing. Qubit ``i`` of the source maps to qsimlab qubit ``i`` (Qiskit: the
circuit's qubit order; Cirq: ``sorted(circuit.all_qubits())`` unless
``qubit_order`` is given; Stim: qubit index). Global phases are kept.

Note on Cirq: Cirq's state vectors are *big-endian* (first qubit most
significant) while qsimlab's are little-endian, so the same circuit gives
bit-reversed state-vector indices; gates and samples per qubit agree.
"""

from __future__ import annotations

import cmath
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .circuit import Circuit
from .errors import MissingDependencyError, UnsupportedOperationError

__all__ = ["from_qiskit", "to_qiskit", "from_cirq", "to_cirq", "from_stim", "to_stim", "from_qasm", "to_qasm"]

#: Gates the engine runs natively; Qiskit circuits are transpiled to this basis.
QISKIT_BASIS = [
    "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg",
    "rx", "ry", "rz", "p", "u", "cx", "cz", "swap", "cp", "ccx",
]


def _require(module: str, extra: str) -> Any:
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as e:  # pragma: no cover - depends on environment
        raise MissingDependencyError(
            f"this converter needs {module!r}: pip install 'qsimlab[{extra}]' (or pip install {extra})"
        ) from e


def from_qasm(source: str) -> Circuit:
    """Same as :meth:`Circuit.from_qasm` (OpenQASM 2.0)."""
    return Circuit.from_qasm(source)


def to_qasm(circuit: Circuit) -> str:
    """Same as :meth:`Circuit.to_qasm` (OpenQASM 2.0)."""
    return circuit.to_qasm()


# --------------------------------------------------------------------------- Qiskit


def from_qiskit(qc: Any, *, transpile: bool = True) -> Circuit:
    """Converts a ``qiskit.QuantumCircuit``.

    The circuit is first transpiled (``optimization_level=0``, no coupling map, so
    no qubit relabelling) to the engine's native basis, then exchanged as
    OpenQASM 2; the global phase is carried over. Parameters must be bound.
    Measurements become records in program order (measurement ``k`` = column ``k``
    of :class:`~qsimlab.sim.SamplesResult`, whatever classical bit Qiskit wrote).
    """
    qiskit = _require("qiskit", "qiskit")
    from qiskit import qasm2

    if qc.parameters:
        raise UnsupportedOperationError(
            f"circuit has unbound parameters {sorted(str(p) for p in qc.parameters)}; "
            "call qc.assign_parameters(...) first"
        )
    tqc = qc
    if transpile:
        tqc = qiskit.transpile(
            qc, basis_gates=QISKIT_BASIS + ["measure", "reset"], optimization_level=0
        )
    out = Circuit.from_qasm(qasm2.dumps(tqc))
    if out.num_qubits < tqc.num_qubits:  # idle qubits after the last register
        wide = Circuit(tqc.num_qubits)
        wide.compose(out)
        out = wide
    out.global_phase = float(tqc.global_phase)
    return out


def to_qiskit(circuit: Circuit) -> Any:
    """Converts to a ``qiskit.QuantumCircuit`` (via OpenQASM 2, global phase kept).

    Conditionals and noise channels have no OpenQASM 2 form and raise
    :class:`~qsimlab.errors.UnsupportedOperationError`.
    """
    _require("qiskit", "qiskit")
    from qiskit import QuantumCircuit

    qc = QuantumCircuit.from_qasm_str(circuit.to_qasm())
    qc.global_phase = circuit.global_phase
    return qc


# --------------------------------------------------------------------------- Cirq


def _u3_from_matrix(m: np.ndarray):
    """Exact ``m = exp(i alpha) * U3(theta, phi, lam)``; returns (theta, phi, lam, alpha)."""
    a, b, c, d = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
    theta = 2.0 * math.atan2(abs(c), abs(a))
    eps = 1e-12
    if abs(a) > eps:
        alpha = cmath.phase(a)
        if abs(c) > eps:
            phi = cmath.phase(c) - alpha
            lam = cmath.phase(-b) - alpha
        else:
            phi = 0.0
            lam = cmath.phase(d) - alpha
    else:
        alpha = cmath.phase(c)
        phi = 0.0
        lam = cmath.phase(-b) - alpha
    return theta, phi, lam, alpha


def _cirq_native(cirq: Any, op: Any, idx: Dict[Any, int], out: Circuit, c_if) -> bool:
    """Appends ``op`` if it maps exactly to an engine gate; returns False otherwise."""
    g = op.gate
    qs = [idx[q] for q in op.qubits]
    if g is None:
        return False

    def is_pow(cls, e):
        return isinstance(g, cls) and getattr(g, "global_shift", 0) == 0 and _close(g.exponent, e)

    simple = [
        (cirq.HPowGate, 1, "h"), (cirq.XPowGate, 1, "x"), (cirq.YPowGate, 1, "y"),
        (cirq.ZPowGate, 1, "z"), (cirq.ZPowGate, 0.5, "s"), (cirq.ZPowGate, -0.5, "sdg"),
        (cirq.ZPowGate, 0.25, "t"), (cirq.ZPowGate, -0.25, "tdg"), (cirq.XPowGate, 0.5, "sx"),
        (cirq.XPowGate, -0.5, "sxdg"), (cirq.CXPowGate, 1, "cx"), (cirq.CZPowGate, 1, "cz"),
        (cirq.SwapPowGate, 1, "swap"), (cirq.ISwapPowGate, 1, "iswap"),
        (cirq.ISwapPowGate, -1, "iswapdg"), (cirq.CCXPowGate, 1, "ccx"),
    ]
    for cls, e, name in simple:
        if is_pow(cls, e):
            out.append(name, qs, c_if=c_if)
            return True
    if isinstance(g, cirq.ZPowGate) and g.global_shift == 0 and not _symbolic(g.exponent):
        out.p(qs[0], math.pi * float(g.exponent), c_if=c_if)
        return True
    if isinstance(g, cirq.CZPowGate) and g.global_shift == 0 and not _symbolic(g.exponent):
        out.cp(qs[0], qs[1], math.pi * float(g.exponent), c_if=c_if)
        return True
    for cls, name in ((cirq.Rx, "rx"), (cirq.Ry, "ry"), (cirq.Rz, "rz")):
        if isinstance(g, cls) and not _symbolic(g._rads):
            out.append(name, qs, (float(g._rads),), c_if=c_if)
            return True
    return False


def _close(x: Any, y: float) -> bool:
    try:
        return abs(float(x) - y) < 1e-12
    except TypeError:
        return False


def _symbolic(x: Any) -> bool:
    try:
        float(x)
        return False
    except TypeError:
        return True


def from_cirq(circuit: Any, *, qubit_order: Optional[Sequence[Any]] = None) -> Circuit:
    """Converts a ``cirq.Circuit`` exactly (global phase included).

    Standard gates map directly; other unitary operations are decomposed by Cirq,
    with single-qubit pieces synthesised exactly as ``u`` gates and the global
    phase corrected against ``cirq.unitary``. Measurements, resets, bit/phase-flip
    and depolarizing channels and simple classical controls (one single-qubit
    measurement key, condition "== 1") are supported. Symbols must be resolved.
    """
    cirq = _require("cirq", "cirq")
    if cirq.is_parameterized(circuit):
        raise UnsupportedOperationError("circuit has unresolved symbols; use cirq.resolve_parameters first")
    qubits = list(qubit_order) if qubit_order is not None else sorted(circuit.all_qubits())
    idx = {q: i for i, q in enumerate(qubits)}
    out = Circuit(len(qubits))
    keys: Dict[str, int] = {}
    for op in circuit.all_operations():
        _from_cirq_op(cirq, op, idx, out, keys, None, depth=0)
    return out


def _from_cirq_op(cirq, op, idx, out: Circuit, keys: Dict[str, int], c_if, depth: int) -> None:
    if depth > 32:
        raise UnsupportedOperationError(f"cannot decompose {op!r}")
    if isinstance(op, cirq.ClassicallyControlledOperation):
        conds = op.classical_controls
        if len(conds) != 1:
            raise UnsupportedOperationError(f"{op!r}: only one classical condition is supported")
        (cond,) = conds
        key = str(getattr(cond, "key", ""))
        if not isinstance(cond, cirq.KeyCondition) or key not in keys:
            raise UnsupportedOperationError(f"{op!r}: unsupported classical condition")
        _from_cirq_op(cirq, op.without_classical_controls(), idx, out, keys, (keys[key], True), depth + 1)
        return
    g = op.gate
    qs = [idx[q] for q in op.qubits]
    if isinstance(g, cirq.MeasurementGate):
        if c_if is not None:
            raise UnsupportedOperationError("conditional measurement is not supported")
        if any(g.full_invert_mask()):
            raise UnsupportedOperationError("measurement invert masks are not supported")
        first = out.num_measurements
        out.measure(*qs)
        if len(qs) == 1:
            keys[str(g.key)] = first
        return
    if isinstance(g, cirq.ResetChannel):
        out.reset(*qs)
        return
    if isinstance(g, cirq.GlobalPhaseGate):
        out.global_phase += cmath.phase(complex(g.coefficient))
        return
    if isinstance(g, cirq.BitFlipChannel):
        out.x_error(qs[0], float(g.p))
        return
    if isinstance(g, cirq.PhaseFlipChannel):
        out.z_error(qs[0], float(g.p))
        return
    if isinstance(g, cirq.DepolarizingChannel):
        if g.num_qubits() == 1:
            out.depolarize1(qs[0], float(g.p))
        elif g.num_qubits() == 2:
            out.depolarize2(qs[0], qs[1], float(g.p))
        else:
            raise UnsupportedOperationError(f"{op!r}: depolarizing on >2 qubits")
        return
    if _cirq_native(cirq, op, idx, out, c_if):
        return
    if not cirq.has_unitary(op):
        raise UnsupportedOperationError(f"cannot convert non-unitary operation {op!r}")
    if len(qs) == 1:
        th, ph, la, alpha = _u3_from_matrix(cirq.unitary(op))
        out.u(qs[0], th, ph, la, c_if=c_if)
        if c_if is None:
            out.global_phase += alpha
        elif abs(alpha) > 1e-12:
            out.p(qs[0], 0.0)  # keep structure; phase on a conditional gate is a controlled phase
            raise UnsupportedOperationError(f"{op!r}: conditional gate with a non-trivial phase")
        return
    target = cirq.unitary(op)
    sub = cirq.decompose_once(op, default=None)
    if sub is None and len(qs) == 2:
        sub = cirq.two_qubit_matrix_to_cz_operations(op.qubits[0], op.qubits[1], target, allow_partial_czs=False)
    if sub is None:
        raise UnsupportedOperationError(f"cannot decompose {op!r}")
    sub = list(sub)
    # correct the global phase of the decomposition exactly
    got = cirq.Circuit(sub).unitary(qubit_order=op.qubits) if sub else np.eye(len(target))
    ov = np.vdot(got.ravel(), target.ravel())
    if abs(abs(ov) - len(target)) > 1e-8 * len(target):
        raise UnsupportedOperationError(f"decomposition of {op!r} is not exact")
    before = out.global_phase
    for s in sub:
        _from_cirq_op(cirq, s, idx, out, keys, c_if, depth + 1)
    if c_if is None:
        out.global_phase = before + (out.global_phase - before) + cmath.phase(ov)
    elif abs(cmath.phase(ov)) > 1e-9:
        raise UnsupportedOperationError(f"{op!r}: conditional gate decomposes with a phase")


def to_cirq(circuit: Circuit) -> Any:
    """Converts to a ``cirq.Circuit`` on ``cirq.LineQubit(0..n-1)``.

    Measurement ``k`` gets key ``"m{k}"``; ``c_if=(k, 1)`` becomes a classical
    control on ``"m{k}"`` (``c_if=(k, 0)`` raises); the global phase becomes a
    global-phase operation.
    """
    cirq = _require("cirq", "cirq")
    q = cirq.LineQubit.range(circuit.num_qubits)
    ops: List[Any] = []
    meas = 0
    for ins in circuit.instructions():
        name, qs, ps = ins.name, [q[i] for i in ins.qubits], ins.params
        if name == "measure":
            ops.append(cirq.measure(qs[0], key=f"m{meas}"))
            meas += 1
            continue
        if name == "reset":
            ops.append(cirq.ResetChannel().on(qs[0]))
            continue
        noise = {
            "x_error": lambda p: cirq.bit_flip(p),
            "z_error": lambda p: cirq.phase_flip(p),
            "y_error": lambda p: cirq.asymmetric_depolarize(p_y=p),
            "depolarize1": lambda p: cirq.depolarize(p),
            "depolarize2": lambda p: cirq.depolarize(p, n_qubits=2),
        }
        if name in noise:
            ops.append(noise[name](ps[0]).on(*qs))
            continue
        op = _cirq_gate(cirq, name, ps).on(*qs)
        if ins.c_if is not None:
            k, v = ins.c_if
            if not v:
                raise UnsupportedOperationError("to_cirq: conditions on a 0 outcome are not supported")
            op = op.with_classical_controls(f"m{k}")
        ops.append(op)
    if circuit.global_phase:
        ops.append(cirq.global_phase_operation(cmath.exp(1j * circuit.global_phase)))
    return cirq.Circuit(ops)


def _cirq_gate(cirq: Any, name: str, ps) -> Any:
    pi = math.pi
    table = {
        "i": lambda: cirq.I, "h": lambda: cirq.H, "x": lambda: cirq.X, "y": lambda: cirq.Y,
        "z": lambda: cirq.Z, "s": lambda: cirq.S, "sdg": lambda: cirq.S**-1, "t": lambda: cirq.T,
        "tdg": lambda: cirq.T**-1, "sx": lambda: cirq.XPowGate(exponent=0.5),
        "sxdg": lambda: cirq.XPowGate(exponent=-0.5), "rx": lambda: cirq.rx(ps[0]),
        "ry": lambda: cirq.ry(ps[0]), "rz": lambda: cirq.rz(ps[0]),
        "p": lambda: cirq.ZPowGate(exponent=ps[0] / pi), "cx": lambda: cirq.CNOT,
        "cz": lambda: cirq.CZ, "swap": lambda: cirq.SWAP, "iswap": lambda: cirq.ISWAP,
        "iswapdg": lambda: cirq.ISWAP**-1, "cp": lambda: cirq.CZPowGate(exponent=ps[0] / pi),
        "ccx": lambda: cirq.TOFFOLI,
    }
    if name == "u":
        th, ph, la = ps
        c, s = math.cos(th / 2), math.sin(th / 2)
        m = np.array(
            [[c, -cmath.exp(1j * la) * s], [cmath.exp(1j * ph) * s, cmath.exp(1j * (ph + la)) * c]]
        )
        return cirq.MatrixGate(m)
    return table[name]()


# --------------------------------------------------------------------------- Stim


def from_stim(circuit: Any) -> Circuit:
    """Converts a ``stim.Circuit`` (or Stim source text): Clifford gates, measurements,
    resets, Pauli noise, ``M(p)`` readout error, detectors and observables;
    ``REPEAT`` blocks are unrolled."""
    return Circuit.from_stim(circuit if isinstance(circuit, str) else str(circuit))


def to_stim(circuit: Circuit) -> Any:
    """Converts to a ``stim.Circuit`` (Clifford + Pauli-noise subset). Use
    :meth:`Circuit.to_stim` for the text without needing ``stim`` installed."""
    stim = _require("stim", "stim")
    return stim.Circuit(circuit.to_stim())
