"""Circuit construction: :class:`Circuit`, the gate table and instructions.

Conventions (``python/API.md`` §5): qubit ``q`` is bit ``q`` of a basis-state
index (little-endian, as in Qiskit); two-qubit gates take the control first;
angles are in radians; measurement ``k`` (program order) writes classical
bit ``k``.

>>> from qsimlab import Circuit
>>> c = Circuit(3).h(0).cx(0, 1).cx(1, 2).measure_all()
>>> c.stats()["gates_2q"], c.num_measurements
(2, 3)
>>> print(Circuit(2).h(0).cx(0, 1).draw())
q0 : ── [H] ──  ■  ────
q1 : ─────────  X  ────
"""

from __future__ import annotations

from typing import Any, Iterable, NamedTuple, Optional, Sequence, Tuple, Union

from . import _native

__all__ = ["Circuit", "Instruction", "GATES", "GATE_ALIASES", "NOISE_CHANNELS"]

#: Gate name -> (number of qubits, number of parameters), every gate of the engine IR.
GATES: dict = dict(_native.GATES)
#: Accepted alternative spellings -> canonical gate name.
GATE_ALIASES: dict = dict(_native.GATE_ALIASES)
#: Noise channel names accepted by :meth:`Circuit.noise`.
NOISE_CHANNELS = ("x_error", "y_error", "z_error", "depolarize1", "depolarize2")

Condition = Union[None, int, Tuple[int, Union[bool, int]]]


class Instruction(NamedTuple):
    """One operation of a circuit, as returned by :meth:`Circuit.instructions`.

    ``name`` is a gate name from :data:`GATES`, ``"measure"``, ``"reset"`` or a
    noise channel; ``params`` holds angles (gates) or the probability (noise);
    ``c_if`` is ``None`` or ``(measurement_index, value)``.
    """

    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...]
    c_if: Optional[Tuple[int, bool]]


def _qubit_list(qubits: Union[int, Iterable[int]]) -> list:
    if isinstance(qubits, int):
        return [qubits]
    return [int(q) for q in qubits]


class Circuit:
    """A quantum circuit on ``num_qubits`` qubits, all starting in ``|0>``.

    Builder methods return ``self`` so they chain. Every gate method accepts
    ``c_if=`` to condition it on a measurement: ``c_if=k`` applies the gate if
    measurement ``k`` read 1, ``c_if=(k, 0)`` if it read 0.

    >>> c = Circuit(2)
    >>> c.h(0).measure(0).x(1, c_if=0).measure(1)
    Circuit(num_qubits=2, ops=4, measurements=2)
    """

    __slots__ = ("_core",)

    def __init__(self, num_qubits: int) -> None:
        if isinstance(num_qubits, bool) or not isinstance(num_qubits, int) or num_qubits < 0:
            raise TypeError("num_qubits must be a non-negative int")
        self._core = _native.CircuitCore(num_qubits)

    @classmethod
    def _wrap(cls, core: Any) -> "Circuit":
        c = cls.__new__(cls)
        c._core = core
        return c

    # ------------------------------------------------------------------ basics
    @property
    def num_qubits(self) -> int:
        """Number of qubits."""
        return self._core.num_qubits

    @property
    def num_measurements(self) -> int:
        """Number of measurements (= classical bits)."""
        return self._core.num_measurements

    @property
    def global_phase(self) -> float:
        """Global phase in radians; every amplitude is multiplied by ``exp(1j * global_phase)``."""
        return self._core.global_phase

    @global_phase.setter
    def global_phase(self, value: float) -> None:
        self._core.global_phase = float(value)

    @property
    def readout_error(self) -> float:
        """Probability that each measurement result is reported flipped (Stim ``M(p)``)."""
        return self._core.readout_error

    @readout_error.setter
    def readout_error(self, p: float) -> None:
        self._core.readout_error = float(p)

    @property
    def detectors(self) -> list:
        """Detectors (lists of absolute measurement indices), for QEC circuits."""
        return self._core.detectors

    @property
    def observables(self) -> list:
        """``observables[k]``: measurement indices whose parity is logical observable ``k``."""
        return self._core.observables

    def __len__(self) -> int:
        return self._core.num_ops

    def __repr__(self) -> str:
        return (
            f"Circuit(num_qubits={self.num_qubits}, ops={len(self)}, "
            f"measurements={self.num_measurements})"
        )

    def __str__(self) -> str:
        return self.draw()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Circuit):
            return NotImplemented
        return self._core == other._core

    __hash__ = None  # type: ignore[assignment]

    def __add__(self, other: "Circuit") -> "Circuit":
        out = self.copy()
        out.compose(other)
        return out

    def __iadd__(self, other: "Circuit") -> "Circuit":
        return self.compose(other)

    def __getstate__(self) -> dict:
        return {
            "num_qubits": self.num_qubits,
            "instructions": [tuple(i) for i in self.instructions()],
            "global_phase": self.global_phase,
            "readout_error": self.readout_error,
            "detectors": self.detectors,
            "observables": self.observables,
            "has_repeats": self._core.has_repeats,
        }

    def __setstate__(self, state: dict) -> None:
        self._core = _native.CircuitCore(state["num_qubits"])
        for name, qubits, params, c_if in state["instructions"]:
            self._append_instruction(name, qubits, params, c_if)
        self._core.global_phase = state["global_phase"]
        self._core.readout_error = state["readout_error"]
        for d in state["detectors"]:
            self._core.add_detector(list(d))
        for k, obs in enumerate(state["observables"]):
            self._core.add_observable(k, list(obs))
        self._core.has_repeats = state["has_repeats"]

    def _append_instruction(self, name, qubits, params, c_if) -> None:
        if name == "measure":
            self._core.append_measure(list(qubits))
        elif name == "reset":
            self._core.append_reset(list(qubits))
        elif name in NOISE_CHANNELS:
            self._core.append_noise(name, list(qubits), params[0])
        else:
            self._core.append_gate(name, list(qubits), list(params), c_if)

    @classmethod
    def from_instructions(
        cls, num_qubits: int, instructions: Iterable[Sequence[Any]]
    ) -> "Circuit":
        """Builds a circuit from ``(name, qubits, params[, c_if])`` tuples (see :meth:`instructions`).

        >>> c = Circuit.from_instructions(2, [("h", (0,), ()), ("cx", (0, 1), ())])
        >>> c == Circuit(2).h(0).cx(0, 1)
        True
        """
        c = cls(num_qubits)
        for ins in instructions:
            name, qubits, params = ins[0], ins[1], ins[2]
            c_if = ins[3] if len(ins) > 3 else None
            c._append_instruction(name, _qubit_list(qubits), list(params), c_if)
        return c

    def instructions(self) -> list:
        """All operations in program order as :class:`Instruction` tuples."""
        return [Instruction(*t) for t in self._core.instructions()]

    def copy(self) -> "Circuit":
        """An independent copy."""
        return Circuit._wrap(self._core.copy())

    def stats(self) -> dict:
        """Summary counts: gates by arity, depth, Clifford/T counts, measurements, noise, ...

        >>> Circuit(2).h(0).t(1).cx(0, 1).stats()["t_gates"]
        1
        """
        return self._core.stats()

    def draw(self) -> str:
        """A text diagram (one line per qubit)."""
        return self._core.draw()

    # ------------------------------------------------------------------ gates
    def append(
        self,
        name: str,
        qubits: Union[int, Sequence[int]],
        params: Sequence[float] = (),
        *,
        c_if: Condition = None,
    ) -> "Circuit":
        """Appends gate ``name`` (any key or alias of :data:`GATES`).

        >>> Circuit(2).append("cp", (0, 1), (0.5,)).instructions()[0].name
        'cp'
        """
        self._core.append_gate(name, _qubit_list(qubits), [float(p) for p in params], c_if)
        return self

    def i(self, q: int, *, c_if: Condition = None) -> "Circuit":
        """Identity."""
        return self.append("i", q, c_if=c_if)

    id = i

    def h(self, q: int, *, c_if: Condition = None) -> "Circuit":
        """Hadamard."""
        return self.append("h", q, c_if=c_if)

    def x(self, q: int, *, c_if: Condition = None) -> "Circuit":
        """Pauli X."""
        return self.append("x", q, c_if=c_if)

    def y(self, q: int, *, c_if: Condition = None) -> "Circuit":
        """Pauli Y."""
        return self.append("y", q, c_if=c_if)

    def z(self, q: int, *, c_if: Condition = None) -> "Circuit":
        """Pauli Z."""
        return self.append("z", q, c_if=c_if)

    def s(self, q: int, *, c_if: Condition = None) -> "Circuit":
        """S = diag(1, i)."""
        return self.append("s", q, c_if=c_if)

    def sdg(self, q: int, *, c_if: Condition = None) -> "Circuit":
        """S† = diag(1, -i)."""
        return self.append("sdg", q, c_if=c_if)

    def t(self, q: int, *, c_if: Condition = None) -> "Circuit":
        """T = diag(1, e^{iπ/4})."""
        return self.append("t", q, c_if=c_if)

    def tdg(self, q: int, *, c_if: Condition = None) -> "Circuit":
        """T† = diag(1, e^{-iπ/4})."""
        return self.append("tdg", q, c_if=c_if)

    def sx(self, q: int, *, c_if: Condition = None) -> "Circuit":
        """√X (Qiskit's ``sx``)."""
        return self.append("sx", q, c_if=c_if)

    def sxdg(self, q: int, *, c_if: Condition = None) -> "Circuit":
        """(√X)†."""
        return self.append("sxdg", q, c_if=c_if)

    def rx(self, q: int, theta: float, *, c_if: Condition = None) -> "Circuit":
        """exp(-iθX/2)."""
        return self.append("rx", q, (theta,), c_if=c_if)

    def ry(self, q: int, theta: float, *, c_if: Condition = None) -> "Circuit":
        """exp(-iθY/2)."""
        return self.append("ry", q, (theta,), c_if=c_if)

    def rz(self, q: int, theta: float, *, c_if: Condition = None) -> "Circuit":
        """exp(-iθZ/2)."""
        return self.append("rz", q, (theta,), c_if=c_if)

    def p(self, q: int, theta: float, *, c_if: Condition = None) -> "Circuit":
        """Phase gate diag(1, e^{iθ}) (OpenQASM ``u1``/``p``)."""
        return self.append("p", q, (theta,), c_if=c_if)

    phase = p

    def u(
        self, q: int, theta: float, phi: float, lam: float, *, c_if: Condition = None
    ) -> "Circuit":
        """OpenQASM ``u3(θ, φ, λ)`` = [[cos θ/2, -e^{iλ} sin θ/2], [e^{iφ} sin θ/2, e^{i(φ+λ)} cos θ/2]]."""
        return self.append("u", q, (theta, phi, lam), c_if=c_if)

    def cx(self, control: int, target: int, *, c_if: Condition = None) -> "Circuit":
        """Controlled X (CNOT)."""
        return self.append("cx", (control, target), c_if=c_if)

    cnot = cx

    def cz(self, a: int, b: int, *, c_if: Condition = None) -> "Circuit":
        """Controlled Z."""
        return self.append("cz", (a, b), c_if=c_if)

    def swap(self, a: int, b: int, *, c_if: Condition = None) -> "Circuit":
        """SWAP."""
        return self.append("swap", (a, b), c_if=c_if)

    def iswap(self, a: int, b: int, *, c_if: Condition = None) -> "Circuit":
        """iSWAP (|01> -> i|10>, |10> -> i|01>)."""
        return self.append("iswap", (a, b), c_if=c_if)

    def iswapdg(self, a: int, b: int, *, c_if: Condition = None) -> "Circuit":
        """iSWAP†."""
        return self.append("iswapdg", (a, b), c_if=c_if)

    def cp(self, a: int, b: int, theta: float, *, c_if: Condition = None) -> "Circuit":
        """Controlled phase diag(1, 1, 1, e^{iθ})."""
        return self.append("cp", (a, b), (theta,), c_if=c_if)

    cphase = cp

    def ccx(self, c1: int, c2: int, target: int, *, c_if: Condition = None) -> "Circuit":
        """Toffoli (doubly controlled X)."""
        return self.append("ccx", (c1, c2, target), c_if=c_if)

    toffoli = ccx

    # --------------------------------------------------- non-unitary operations
    def measure(self, *qubits: int) -> "Circuit":
        """Measures each qubit in the Z basis; measurement ``k`` writes classical bit ``k``."""
        self._core.append_measure(list(qubits))
        return self

    def measure_all(self) -> "Circuit":
        """Measures every qubit, in order."""
        return self.measure(*range(self.num_qubits))

    def reset(self, *qubits: int) -> "Circuit":
        """Resets each qubit to ``|0>``."""
        self._core.append_reset(list(qubits))
        return self

    def noise(self, channel: str, qubits: Union[int, Sequence[int]], p: float) -> "Circuit":
        """Appends a stochastic Pauli channel (one of :data:`NOISE_CHANNELS`) on each qubit (pairs for ``depolarize2``)."""
        self._core.append_noise(channel, _qubit_list(qubits), float(p))
        return self

    def x_error(self, q: Union[int, Sequence[int]], p: float) -> "Circuit":
        """X with probability ``p`` (Stim ``X_ERROR``)."""
        return self.noise("x_error", q, p)

    def y_error(self, q: Union[int, Sequence[int]], p: float) -> "Circuit":
        """Y with probability ``p`` (Stim ``Y_ERROR``)."""
        return self.noise("y_error", q, p)

    def z_error(self, q: Union[int, Sequence[int]], p: float) -> "Circuit":
        """Z with probability ``p`` (Stim ``Z_ERROR``)."""
        return self.noise("z_error", q, p)

    def depolarize1(self, q: Union[int, Sequence[int]], p: float) -> "Circuit":
        """X, Y or Z, each with probability ``p/3`` (Stim ``DEPOLARIZE1``)."""
        return self.noise("depolarize1", q, p)

    def depolarize2(self, a: int, b: int, p: float) -> "Circuit":
        """One of the 15 non-identity two-qubit Paulis, each with probability ``p/15`` (Stim ``DEPOLARIZE2``)."""
        return self.noise("depolarize2", (a, b), p)

    # ------------------------------------------------------------ QEC metadata
    def detector(self, measurements: Iterable[int]) -> int:
        """Declares a detector (parity of absolute measurement indices); returns its index."""
        return self._core.add_detector([int(m) for m in measurements])

    def observable_include(self, index: int, measurements: Iterable[int]) -> "Circuit":
        """Adds measurements to logical observable ``index``."""
        self._core.add_observable(int(index), [int(m) for m in measurements])
        return self

    # ------------------------------------------------------------ composition
    def compose(self, other: "Circuit", qubits: Optional[Sequence[int]] = None) -> "Circuit":
        """Appends ``other`` (its qubit ``i`` on ``qubits[i]``) in place; returns ``self``.

        Measurement indices in conditionals, detectors and observables of
        ``other`` are shifted past this circuit's measurements.
        """
        self._core.extend(other._core, None if qubits is None else list(qubits), 1)
        return self

    def repeat(
        self, body: "Circuit", reps: int, qubits: Optional[Sequence[int]] = None
    ) -> "Circuit":
        """Appends ``body`` ``reps`` times (unrolled).

        The circuit remembers it contains a repeat block, so
        :func:`qsimlab.simulate` runs the repeat-detection pass by default
        (exact fast paths for repeated Clifford/diagonal/small blocks).

        >>> layer = Circuit(2).rx(0, 0.1).cz(0, 1)
        >>> len(Circuit(2).repeat(layer, 100))
        200
        """
        if reps < 0:
            raise ValueError("reps must be non-negative")
        self._core.extend(body._core, None if qubits is None else list(qubits), int(reps))
        return self

    def inverse(self) -> "Circuit":
        """The inverse of a unitary circuit (reversed, each gate inverted, global phase negated)."""
        return Circuit._wrap(self._core.inverse())

    def remove_final_measurements(self) -> "Circuit":
        """A copy without terminal measurements (and without detectors/observables)."""
        return Circuit._wrap(self._core.remove_final_measurements())

    # ------------------------------------------------------------ formats
    def to_qasm(self) -> str:
        """OpenQASM 2.0 source. Conditionals and noise channels have no faithful
        OpenQASM 2 form and raise :class:`~qsimlab.errors.UnsupportedOperationError`;
        the global phase is not representable and is dropped."""
        return self._core.to_qasm()

    @classmethod
    def from_qasm(cls, source: str) -> "Circuit":
        """Parses OpenQASM 2.0: registers (laid out in declaration order), user
        ``gate`` definitions, broadcasting, ``barrier``, ``reset``, ``measure``,
        ``if`` on one-bit registers, and the qelib1.inc gate set plus common
        Qiskit extensions (``rzz``, ``rxx``, ``ryy``, ``rzx``, ``ecr``, ``cu``, ...).

        >>> src = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];'
        >>> Circuit.from_qasm(src) == Circuit(2).h(0).cx(0, 1)
        True
        """
        return cls._wrap(_native.CircuitCore.from_qasm(source))

    def to_stim(self) -> str:
        """Stim circuit text (Clifford gates, measurements, resets, Pauli noise,
        detectors, observables; ``readout_error`` becomes ``M(p)``)."""
        return self._core.to_stim()

    @classmethod
    def from_stim(cls, source: str) -> "Circuit":
        """Parses Stim circuit text (``REPEAT`` blocks are unrolled; ``DETECTOR``
        and ``OBSERVABLE_INCLUDE`` populate :attr:`detectors` / :attr:`observables`).

        >>> c = Circuit.from_stim("H 0\\nCX 0 1\\nM 0 1\\nDETECTOR rec[-2] rec[-1]")
        >>> c.num_measurements, c.detectors
        (2, [[0, 1]])
        """
        return cls._wrap(_native.CircuitCore.from_stim(source))
