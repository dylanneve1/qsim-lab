"""Simulation: :func:`simulate`, :func:`plan`, requests and results.

A *request* says what you want from a circuit; :func:`simulate` picks (or is
told) the engine and returns a result object holding numpy arrays.

>>> import numpy as np
>>> from qsimlab import Circuit, simulate, statevector, amplitudes, samples, expectation
>>> ghz = Circuit(3).h(0).cx(0, 1).cx(1, 2)
>>> np.round(simulate(ghz, amplitudes(["000", "111", 5])).amplitudes, 4)
array([0.7071+0.j, 0.7071+0.j, 0.    +0.j])
>>> simulate(ghz, expectation(["Z0 Z1", "X0 X1 X2", "Z0"])).values.round(6)
array([1., 1., 0.])
>>> r = simulate(ghz, samples(8), seed=7)
>>> r.bits.shape, bool((r.bits == r.bits[:, :1]).all())
((8, 3), True)
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import _native
from .circuit import Circuit

__all__ = [
    "simulate",
    "plan",
    "statevector",
    "amplitudes",
    "samples",
    "expectation",
    "Request",
    "StatevectorRequest",
    "AmplitudesRequest",
    "SamplesRequest",
    "ExpectationRequest",
    "Result",
    "StatevectorResult",
    "AmplitudesResult",
    "SamplesResult",
    "ExpectationResult",
    "Explanation",
    "Budget",
    "NoiseModel",
    "ENGINES",
]

#: ``{engine: (requests it supports, description)}`` for ``simulate(engine=...)``.
ENGINES: Dict[str, Tuple[Tuple[str, ...], str]] = {
    name: (tuple(reqs.split()), desc) for name, reqs, desc in _native.ENGINES
}

BitstringLike = Union[int, str]

# --------------------------------------------------------------------------- requests


@dataclass(frozen=True)
class Request:
    """Base class of the four request types."""

    kind = "request"

    def _payload(self) -> Any:  # pragma: no cover - overridden
        return None


@dataclass(frozen=True)
class StatevectorRequest(Request):
    """The full state vector (``2**n`` amplitudes, little-endian, global phase included)."""

    kind = "statevector"

    def _payload(self) -> Any:
        return None


@dataclass(frozen=True)
class AmplitudesRequest(Request):
    """``<x|ψ>`` for chosen basis states (ints, or bitstrings with qubit 0 rightmost)."""

    bitstrings: Tuple[BitstringLike, ...]
    kind = "amplitudes"

    def _payload(self) -> Any:
        return list(self.bitstrings)


@dataclass(frozen=True)
class SamplesRequest(Request):
    """``shots`` measurement records (one column per measurement, program order)."""

    shots: int
    kind = "samples"

    def _payload(self) -> Any:
        return self.shots


@dataclass(frozen=True)
class ExpectationRequest(Request):
    """``<ψ|P|ψ>`` for each Pauli string ``P`` (``"X0 Z3"`` or dense ``"XIZ"``, qubit 0 rightmost)."""

    paulis: Tuple[str, ...]
    kind = "expectation"

    def _payload(self) -> Any:
        return list(self.paulis)


def statevector() -> StatevectorRequest:
    """Request the full state vector (complex ``ndarray`` of length ``2**n``)."""
    return StatevectorRequest()


def amplitudes(bitstrings: Union[BitstringLike, Iterable[BitstringLike]]) -> AmplitudesRequest:
    """Request ``<x|ψ>`` for basis states ``x``: ints (bit q = qubit q) or ``'0'/'1'`` strings
    whose rightmost character is qubit 0. Works far beyond state-vector sizes when the
    planner finds a cheaper engine (sparse, MPS, HSF path sums)."""
    if isinstance(bitstrings, (int, str, np.integer)):
        bitstrings = [bitstrings]
    return AmplitudesRequest(tuple(int(b) if isinstance(b, np.integer) else b for b in bitstrings))


def samples(shots: int) -> SamplesRequest:
    """Request ``shots`` measurement records. Noise channels, resets, mid-circuit
    measurements and conditionals are simulated exactly, shot by shot or batched."""
    if isinstance(shots, bool) or int(shots) != shots or shots < 0:
        raise ValueError("shots must be a non-negative int")
    return SamplesRequest(int(shots))


def expectation(paulis: Union[str, Iterable[str]]) -> ExpectationRequest:
    """Request expectation values of Pauli strings (see API.md §5 for the syntax)."""
    if isinstance(paulis, str):
        paulis = [paulis]
    return ExpectationRequest(tuple(paulis))


# --------------------------------------------------------------------------- options


@dataclass(frozen=True)
class Budget:
    """Resource limits for a simulation.

    ``memory``: largest register any engine may allocate, as bytes or a string
    like ``"4GiB"`` / ``"512MB"``. ``None`` = the engine's hard cap (32 GiB).

    >>> Budget(memory="1GiB").memory_bytes
    1073741824
    """

    memory: Union[int, str, None] = None

    @property
    def memory_bytes(self) -> Optional[int]:
        if self.memory is None:
            return None
        if isinstance(self.memory, int):
            return self.memory
        return _parse_memory(self.memory)


def _parse_memory(s: str) -> int:
    t = s.strip().replace(" ", "").replace("_", "")
    i = 0
    while i < len(t) and (t[i].isdigit() or t[i] in ".eE+-"):
        if t[i] in "eE" and (i + 1 >= len(t) or not (t[i + 1].isdigit() or t[i + 1] in "+-")):
            break
        i += 1
    num, unit = t[:i], t[i:].lower()
    units = {"": 1, "b": 1, "k": 1e3, "kb": 1e3, "m": 1e6, "mb": 1e6, "g": 1e9, "gb": 1e9,
             "t": 1e12, "tb": 1e12, "kib": 2**10, "mib": 2**20, "gib": 2**30, "tib": 2**40}
    if unit not in units:
        raise ValueError(f"bad memory size {s!r}")
    return int(float(num) * units[unit])


@dataclass(frozen=True)
class NoiseModel:
    """Circuit-level noise applied on top of the circuit's own noise channels (samples only).

    * ``p1``: single-qubit depolarizing after every 1-qubit gate;
    * ``p2``: two-qubit depolarizing after every 2-qubit gate;
    * ``readout``: probability each measurement result is reported flipped;
    * ``reset``: probability each reset leaves ``|1>``.
    """

    p1: float = 0.0
    p2: float = 0.0
    readout: float = 0.0
    reset: float = 0.0

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"NoiseModel.{f.name}={v} is not a probability")

    def _tuple(self) -> Tuple[float, float, float, float]:
        return (self.p1, self.p2, self.readout, self.reset)


# --------------------------------------------------------------------------- results


@dataclass(frozen=True)
class Explanation:
    """What the planner decided and why (``simulate(..., explain=True)`` or :func:`plan`).

    ``ranked`` lists ``(engine, predicted_seconds)`` best first; predictions come
    from cost models fitted on an M1 Pro (one thread), so read them as relative.
    """

    engine: Optional[str]
    ranked: List[Tuple[str, float]]
    plan_seconds: float
    cached: bool
    features: Dict[str, Any]
    notes: List[str]

    def __str__(self) -> str:
        lines = [f"planner choice: {self.engine}"]
        for e, t in self.ranked:
            lines.append(f"  {e:<12} {_fmt_secs(t)}")
        if self.features:
            lines.append("features: " + ", ".join(f"{k}={v}" for k, v in self.features.items()))
        lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)


def _fmt_secs(t: float) -> str:
    if not math.isfinite(t):
        return "n/a"
    for unit, s in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6)):
        if t >= s:
            return f"{t / s:.3g} {unit}"
    return f"{t * 1e9:.3g} ns"


@dataclass(frozen=True)
class Result:
    """Fields shared by every result.

    * ``engine``: the engine that ran (``"pipeline"`` if components used different ones);
    * ``components``: ``[(num_qubits, num_gates, engine)]`` per simulated part;
    * ``seed``: the seed actually used (pass it back for bit-identical samples);
    * ``precision``: ``"f64"`` or ``"f32"``, what was actually used;
    * ``wall_time``: seconds spent in the engine;
    * ``explanation``: an :class:`Explanation` if ``explain=True``.
    """

    engine: str
    components: List[Tuple[int, int, str]]
    seed: int
    precision: str
    wall_time: float
    explanation: Optional[Explanation] = field(default=None, repr=False)


@dataclass(frozen=True)
class StatevectorResult(Result):
    """``state``: complex ``ndarray`` of length ``2**n``; ``state[i]`` is the amplitude of basis index ``i``."""

    state: np.ndarray = field(default=None, repr=False)  # type: ignore[arg-type]

    def probabilities(self) -> np.ndarray:
        """``|state|**2``."""
        return np.abs(self.state) ** 2


@dataclass(frozen=True)
class AmplitudesResult(Result):
    """``amplitudes[k] = <bitstrings[k]|ψ>`` (complex128)."""

    amplitudes: np.ndarray = field(default=None, repr=False)  # type: ignore[arg-type]
    bitstrings: Tuple[BitstringLike, ...] = ()


@dataclass(frozen=True)
class SamplesResult(Result):
    """``bits``: uint8 ``ndarray[shots, m]``, column ``k`` = measurement ``k`` (program order);
    ``measured_qubits[k]`` is the qubit measurement ``k`` read."""

    bits: np.ndarray = field(default=None, repr=False)  # type: ignore[arg-type]
    measured_qubits: Tuple[int, ...] = ()

    @property
    def shots(self) -> int:
        return int(self.bits.shape[0])

    def counts(self, *, as_int: bool = False) -> Dict[Union[str, int], int]:
        """Histogram of records. Keys are bitstrings with measurement 0 **rightmost**
        (Qiskit order), or ints (bit k = measurement k) with ``as_int=True``.

        >>> from qsimlab import Circuit, simulate, samples
        >>> simulate(Circuit(2).x(0).measure_all(), samples(5)).counts()
        {'01': 5}
        """
        m = self.bits.shape[1]
        if m == 0:
            return {("" if not as_int else 0): self.shots} if self.shots else {}
        if m <= 63:
            weights = (np.uint64(1) << np.arange(m, dtype=np.uint64))
            keys = (self.bits.astype(np.uint64) * weights).sum(axis=1)
            vals, cnt = np.unique(keys, return_counts=True)
            ints = [int(v) for v in vals]
        else:
            rows, cnt = np.unique(self.bits, axis=0, return_counts=True)
            ints = [int("".join(str(int(b)) for b in r[::-1]), 2) for r in rows]
        out: Dict[Union[str, int], int] = {}
        for k, c in zip(ints, cnt):
            out[k if as_int else format(k, f"0{m}b")] = int(c)
        return out

    def probabilities(self) -> Dict[str, float]:
        """Empirical probabilities, same keys as :meth:`counts`."""
        return {str(k): v / self.shots for k, v in self.counts().items()}

    def parity(self, columns: Sequence[int]) -> np.ndarray:
        """XOR of the given measurement columns per shot (uint8 ``ndarray[shots]``)."""
        return np.bitwise_xor.reduce(self.bits[:, list(columns)], axis=1)


@dataclass(frozen=True)
class ExpectationResult(Result):
    """``values[k] = <ψ|paulis[k]|ψ>`` (float64)."""

    values: np.ndarray = field(default=None, repr=False)  # type: ignore[arg-type]
    paulis: Tuple[str, ...] = ()


# --------------------------------------------------------------------------- entry points

Engine = str
Precision = str


def _memory_arg(budget: Union[Budget, int, str, None]) -> Union[int, str, None]:
    if budget is None:
        return None
    if isinstance(budget, Budget):
        return budget.memory
    if isinstance(budget, (int, str)):
        return budget
    raise TypeError("budget must be a Budget, an int (bytes) or a string like '4GiB'")


def _check_circuit(circuit: Any) -> Circuit:
    if not isinstance(circuit, Circuit):
        raise TypeError(
            f"expected a qsimlab.Circuit, got {type(circuit).__name__}; "
            "convert with qsimlab.interop.from_qiskit / from_cirq / from_stim or Circuit.from_qasm"
        )
    return circuit


def _explanation(d: Optional[dict]) -> Optional[Explanation]:
    if d is None:
        return None
    return Explanation(
        engine=d["engine"],
        ranked=[tuple(x) for x in d["ranked"]],
        plan_seconds=d["plan_seconds"],
        cached=d["cached"],
        features=dict(d["features"]),
        notes=list(d["notes"]),
    )


def simulate(
    circuit: Circuit,
    request: Request,
    *,
    engine: Engine = "auto",
    precision: Precision = "f64",
    seed: Optional[int] = None,
    budget: Union[Budget, int, str, None] = None,
    threads: Optional[int] = None,
    noise: Optional[NoiseModel] = None,
    repeat: Optional[bool] = None,
    explain: bool = False,
) -> Result:
    """Simulates ``circuit`` for ``request`` and returns the matching result object.

    Parameters
    ----------
    engine
        ``"auto"`` (default): compile passes + Planner v2 choose the cheapest exact
        engine per connected component. Or force one of :data:`ENGINES`; a forced
        engine that cannot do the job raises instead of falling back.
    precision
        ``"f64"`` or ``"f32"`` (honoured by dense state-vector paths; see ``result.precision``).
    seed
        RNG seed for sampling; ``None`` draws one (reported as ``result.seed``).
    budget
        Memory cap: :class:`Budget`, bytes, or a string like ``"8GiB"``.
    threads
        Worker threads for this call (default: :func:`qsimlab.get_num_threads`).
    noise
        A :class:`NoiseModel` applied on top of the circuit's channels (samples only).
    repeat
        Run the repeat-detection pass (default: on iff the circuit was built with
        :meth:`Circuit.repeat` or parsed from Stim with ``REPEAT``).
    explain
        Attach the planner's :class:`Explanation` to the result.

    >>> from qsimlab import Circuit, simulate, expectation
    >>> r = simulate(Circuit(1).ry(0, 0.5), expectation("Z0"), explain=True)
    >>> round(float(r.values[0]), 6), r.explanation is not None
    (0.877583, True)
    """
    circuit = _check_circuit(circuit)
    if not isinstance(request, Request):
        raise TypeError("request must come from statevector(), amplitudes(), samples() or expectation()")
    if seed is not None:
        seed = int(seed) & 0xFFFF_FFFF_FFFF_FFFF
    if noise is not None and not isinstance(noise, NoiseModel):
        raise TypeError("noise must be a qsimlab.NoiseModel")
    d = _native.run(
        circuit._core,
        request.kind,
        request._payload(),
        engine=engine,
        precision=precision,
        seed=seed,
        memory=_memory_arg(budget),
        threads=threads,
        noise=None if noise is None else noise._tuple(),
        repeat=repeat,
        explain=explain,
    )
    common = dict(
        engine=d["engine"],
        components=[tuple(c) for c in d["components"]],
        seed=d["seed"],
        precision=d["precision"],
        wall_time=d["wall_time"],
        explanation=_explanation(d["explanation"]),
    )
    if isinstance(request, StatevectorRequest):
        return StatevectorResult(state=d["data"], **common)
    if isinstance(request, AmplitudesRequest):
        return AmplitudesResult(amplitudes=d["data"], bitstrings=request.bitstrings, **common)
    if isinstance(request, SamplesRequest):
        return SamplesResult(bits=d["data"], measured_qubits=tuple(d["measured_qubits"]), **common)
    assert isinstance(request, ExpectationRequest)
    return ExpectationResult(values=d["data"], paulis=request.paulis, **common)


def plan(
    circuit: Circuit,
    request: Request,
    *,
    budget: Union[Budget, int, str, None] = None,
) -> Explanation:
    """The planner's prediction for ``request`` without running anything.

    >>> from qsimlab import Circuit, plan, samples
    >>> e = plan(Circuit(30).h(0).cx(0, 1).measure_all(), samples(1000))
    >>> e.engine is not None and len(e.ranked) >= 1
    True
    """
    circuit = _check_circuit(circuit)
    d = _native.plan(circuit._core, request.kind, request._payload(), memory=_memory_arg(budget))
    e = _explanation(d)
    assert e is not None
    return e
