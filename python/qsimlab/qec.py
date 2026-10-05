"""Quantum error correction: memory circuits, detector sampling, error models, decoders.

**Provisional** (phase 2, see ``python/API.md`` §1 and §8).

The pipeline is *build → sample → decode → count*::

    >>> import qsimlab.qec as qec
    >>> c = qec.surface_code_memory(3, rounds=3, p=1e-3)
    >>> len(c.detectors), len(c.observables)
    (24, 1)
    >>> dets, obs = qec.sample_detectors(c, 2048, seed=1)
    >>> dets.shape, dets.dtype, obs.shape
    ((2048, 24), dtype('bool'), (2048, 1))
    >>> dem = qec.detector_error_model(c)
    >>> dem.num_detectors, dem.num_observables
    (24, 1)
    >>> qec.circuit_distance(c).distance
    3
    >>> r = qec.logical_error_rate(c, 20_000, seed=2)
    >>> r.errors < 100 and r.shots == 20_000
    True

Conventions (in addition to ``python/API.md`` §5):

* **Detection events** are reported relative to the noiseless reference
  (Stim's convention): a noiseless shot reads all zeros.
* **Bit packing** (``packed=True``) is Stim's ``bit_packed`` layout:
  ``uint8[shots, ceil(n / 8)]``, entry ``k`` is bit ``k % 8`` of byte ``k // 8``.
* **Seeds**: shots are drawn in chunks of 1024 from independent streams
  derived from ``(seed, chunk)``, so a seed gives the same samples for any
  thread count. ``seed=None`` draws a fresh seed from the OS.
* **Noise models** (``noise=``, strength ``p``): ``"cnot"`` (``DEPOLARIZE2(p)``
  after every CNOT, nothing else), ``"uniform"`` (``DEPOLARIZE2(p)`` after CNOTs,
  ``DEPOLARIZE1(p)`` on idle qubits in every moment, readout flip ``p``, reset
  flip ``p``) and ``"si1000"`` (Gidney et al.'s superconducting-inspired model:
  CNOT ``p``, idle ``p/10`` during gate layers and ``2p`` during
  measure/reset layers, readout flip ``5p``, reset flip ``2p``). Every noise
  location is an explicit op of the returned circuit (the readout flip is
  ``circuit.readout_error``), so ``circuit.to_stim()`` is exactly the sampled circuit.
"""

from __future__ import annotations

import math
import os
import re
import secrets
import time
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from ._native import qec as _native_qec
from .circuit import Circuit
from .errors import CircuitError, MissingDependencyError, ParseError, UnsupportedOperationError
from .sim import Result

__all__ = [
    "NOISE_MODELS",
    "DECODERS",
    "CodeLayout",
    "surface_code_memory",
    "repetition_code_memory",
    "color_code_memory",
    "color_code_schedule",
    "DetectorSampler",
    "sample_detectors",
    "DemError",
    "DetectorErrorModel",
    "detector_error_model",
    "DistanceResult",
    "circuit_distance",
    "Decoder",
    "BpOsdDecoder",
    "PyMatchingDecoder",
    "TesseractDecoder",
    "make_decoder",
    "decode",
    "LogicalErrorRate",
    "logical_error_rate",
    "wilson_interval",
]

#: Built-in circuit noise models (see the module docstring).
NOISE_MODELS: Tuple[str, ...] = ("cnot", "uniform", "si1000")

#: Decoders known to :func:`make_decoder`; ``pymatching`` and ``tesseract``
#: need the optional packages of the same name.
DECODERS: Tuple[str, ...] = ("bposd", "pymatching", "tesseract")


def _fresh_seed(seed: Optional[int]) -> int:
    if seed is None:
        return secrets.randbits(64)
    seed = int(seed)
    if not 0 <= seed < 2**64:
        raise ValueError("seed must be in [0, 2**64)")
    return seed


# --------------------------------------------------------------------------- layouts


@dataclass(frozen=True)
class CodeLayout:
    """Where everything sits in a generated memory circuit.

    * ``data_qubits``, ``ancilla_qubits``, ``flag_qubits``: qubit indices;
    * ``qubit_coords``: ``(x, y)`` per qubit;
    * ``detector_coords``: ``float64[D, 3]`` with ``(x, y, round)`` per detector;
    * ``detector_basis``: ``"X"`` or ``"Z"`` per detector (the stabilizer type it checks);
    * ``flag_detectors``: ``bool[D]``, true for flag-qubit detectors (colour code);
    * ``memory_detectors``: indices of the non-flag detectors of the memory basis,
      the sector the logical observable lives in (pass it to
      :func:`circuit_distance` as ``detectors=`` for a much faster search).
    """

    code: str
    distance: int
    rounds: int
    basis: str
    noise: str
    p: float
    data_qubits: Tuple[int, ...]
    ancilla_qubits: Tuple[int, ...]
    flag_qubits: Tuple[int, ...]
    qubit_coords: Tuple[Tuple[float, float], ...] = field(repr=False)
    detector_coords: np.ndarray = field(repr=False)
    detector_basis: Tuple[str, ...] = field(repr=False)
    flag_detectors: np.ndarray = field(repr=False)
    schedule: Optional[Tuple[Tuple[int, ...], ...]] = field(default=None, repr=False)

    @property
    def memory_detectors(self) -> List[int]:
        return [
            i
            for i, b in enumerate(self.detector_basis)
            if b == self.basis and not self.flag_detectors[i]
        ]


def _layout(code: str, d: int, rounds: int, basis: str, noise: str, p: float,
            raw: Dict[str, Any], schedule: Any = None) -> CodeLayout:
    return CodeLayout(
        code=code,
        distance=d,
        rounds=rounds,
        basis=basis.upper(),
        noise=noise,
        p=float(p),
        data_qubits=tuple(raw["data_qubits"]),
        ancilla_qubits=tuple(raw["ancilla_qubits"]),
        flag_qubits=tuple(raw["flag_qubits"]),
        qubit_coords=tuple(tuple(c) for c in raw["qubit_coords"]),
        detector_coords=np.asarray(raw["detector_coords"], dtype=np.float64).reshape(-1, 3),
        detector_basis=tuple(raw["detector_basis"]),
        flag_detectors=np.asarray(raw["flag_detectors"], dtype=bool),
        schedule=None if schedule is None else tuple(tuple(r) for r in schedule),
    )


def _check_noise(noise: str, p: float) -> Tuple[str, float]:
    noise = str(noise).lower()
    if noise not in NOISE_MODELS:
        raise ValueError(f"noise must be one of {NOISE_MODELS}, not {noise!r}")
    return noise, float(p)


# --------------------------------------------------------------------------- generators


def surface_code_memory(
    d: int,
    rounds: Optional[int] = None,
    basis: str = "Z",
    *,
    p: float = 0.0,
    noise: str = "uniform",
    return_layout: bool = False,
) -> Union[Circuit, Tuple[Circuit, CodeLayout]]:
    """Rotated surface code memory experiment.

    ``d*d`` data qubits at ``(2x+1, 2y+1)`` and ``d*d - 1`` auxiliaries
    (Stim's ``surface_code:rotated_memory_*`` layout and hook-safe CNOT order).
    Data start in the memory basis, every round resets the auxiliaries (``R``
    / ``RX``), runs four CNOT layers and measures them (``M`` / ``MX``); the
    data are measured in the memory basis at the end.

    Detectors, in order: per round, first the memory-basis stabilizers
    (round 0: the raw outcome; later: XOR with the previous round), then (from
    round 1) the other basis; finally the memory-basis stabilizers from the
    data. One observable: ``Z_L`` on the ``y = 1`` row (Z memory) or
    ``X_L`` on the ``x = 1`` column (X memory).

    >>> c = surface_code_memory(3, 2, "X", p=1e-3, noise="si1000")
    >>> c.num_qubits, len(c.detectors), c.readout_error
    (17, 16, 0.005)
    """
    d = int(d)
    rounds = d if rounds is None else int(rounds)
    noise, p = _check_noise(noise, p)
    core, raw = _native_qec.surface_code_memory(d, rounds, str(basis), p, noise)
    c = Circuit._wrap(core)
    if return_layout:
        return c, _layout("surface", d, rounds, basis, noise, p, raw)
    return c


def repetition_code_memory(
    d: int,
    rounds: Optional[int] = None,
    *,
    p: float = 0.0,
    noise: str = "uniform",
    return_layout: bool = False,
) -> Union[Circuit, Tuple[Circuit, CodeLayout]]:
    """Bit-flip repetition code memory: data ``0..d-1``, auxiliary ``d+j`` measures
    ``Z_j Z_{j+1}`` every round; detectors as in :func:`surface_code_memory`
    (``(d-1) * (rounds+1)`` of them), observable: the last data qubit.

    >>> c = repetition_code_memory(5, 3, p=0.01)
    >>> c.num_qubits, len(c.detectors)
    (9, 16)
    """
    d = int(d)
    rounds = d if rounds is None else int(rounds)
    noise, p = _check_noise(noise, p)
    core, raw = _native_qec.repetition_code_memory(d, rounds, p, noise)
    c = Circuit._wrap(core)
    if return_layout:
        return c, _layout("repetition", d, rounds, "Z", noise, p, raw)
    return c


# Exact schedules from research/qec/colour-global.md (one row per plaquette, steps
# for positions a..f, 0 = absent). d = 9: circuit distance 8 (K-F: 7) in
# K-F's 6+6-layer design space; d = 11: 7+7 layers, circuit distance 10.
_GLOBAL_SCHEDULES = {
    9: (
        "4 2 6 0 0 5;5 2 4 0 0 3;3 6 2 0 0 5;6 1 2 0 0 3;0 4 3 6 5 0;2 5 6 1 4 3;"
        "1 5 6 2 3 4;1 2 5 6 3 4;4 3 6 5 2 1;4 6 5 3 1 2;6 4 5 3 1 2;5 0 0 4 3 1;"
        "5 6 2 3 1 4;1 5 3 6 2 4;1 6 4 5 3 2;0 5 4 6 2 0;6 2 5 1 3 4;6 5 2 1 4 3;"
        "4 1 2 5 3 6;3 4 5 2 6 1;4 0 0 2 3 1;2 1 4 6 5 3;2 6 3 4 1 5;0 5 4 6 2 0;"
        "2 4 5 6 1 3;4 6 5 2 1 3;5 0 0 3 4 2;6 2 4 3 1 5;0 3 1 2 5 0;4 0 0 3 2 6"
    ),
    11: (
        "4 2 3 0 0 1;5 2 4 0 0 1;4 5 2 0 0 6;5 6 1 0 0 7;4 2 5 0 0 3;0 5 1 7 6 0;"
        "1 3 6 2 4 5;6 2 1 4 5 3;5 1 2 4 3 6;6 1 5 2 4 3;3 4 5 1 6 2;3 5 2 7 1 6;"
        "5 4 6 7 3 1;2 3 1 4 7 6;2 0 0 1 3 5;1 2 7 4 6 3;2 5 3 4 1 6;6 1 5 4 7 2;"
        "6 3 1 4 2 5;0 1 4 2 5 0;5 3 6 2 4 1;6 2 5 3 4 7;5 7 1 2 4 3;4 2 1 5 3 6;"
        "5 6 2 3 4 7;4 5 1 7 2 6;1 0 0 4 2 3;7 6 3 5 2 1;4 5 3 7 1 2;2 5 4 6 7 1;"
        "0 4 3 6 1 0;5 3 6 1 4 2;3 1 4 7 5 6;7 4 6 3 1 5;4 7 2 3 5 6;5 0 0 7 3 6;"
        "4 5 3 2 1 6;7 3 4 2 1 6;0 1 7 5 6 0;6 7 1 2 5 4;4 7 5 3 6 2;4 0 0 5 2 6;"
        "1 5 3 4 7 2;0 4 3 1 5 0;6 0 0 4 2 5"
    ),
}


def _parse_schedule_text(text: str) -> Tuple[List[List[int]], List[bool]]:
    rows, flags = [], []
    for line in text.replace(";", "\n").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        tok = line.split()
        if len(tok) < 6:
            raise ParseError(f"schedule line {line!r}: expected 6 steps")
        try:
            rows.append([int(t) for t in tok[:6]])
        except ValueError as e:
            raise ParseError(f"schedule line {line!r}: {e}") from None
        flags.append(len(tok) > 6 and tok[6].upper() == "F")
    return rows, flags


def color_code_schedule(d: int, schedule: Any = "kf") -> Tuple[List[List[int]], List[bool]]:
    """Resolves a colour-code schedule spec to ``(rows, flag_mask)``.

    ``schedule``: ``"kf"`` (Kishony–Fowler's colour-dependent schedule),
    ``"tri"`` (Lee et al.'s uniform tri-optimal), ``"global"`` (the exact
    global-search schedules of ``research/qec/colour-global.md``: ``d = 9``,
    circuit distance 8, and ``d = 11`` with 7+7 layers, circuit distance 10),
    a path to a ``.sched`` file (one line per plaquette: steps for positions
    a..f, 0 for absent ones, optional trailing ``F`` = flag it), or a
    ``(num_plaquettes, 6)`` array of steps.
    """
    d = int(d)
    if isinstance(schedule, str):
        s = schedule.strip()
        if s.lower() in ("kf", "tri"):
            core, _, rows = _native_qec.color_code_memory(d, 1, "Z", s.lower(), [], 0.0, "cnot")
            return [list(r) for r in rows], [False] * len(rows)
        if s.lower() == "global":
            if d not in _GLOBAL_SCHEDULES:
                raise ValueError(
                    "schedule='global' exists for d = 9 (circuit distance 8 vs K-F's 7) and "
                    "d = 11 (7+7 layers, distance 10); K-F is already optimal at d = 5 and 7"
                )
            return _parse_schedule_text(_GLOBAL_SCHEDULES[d])
        if not os.path.exists(s):
            raise ValueError(f"schedule {schedule!r} is not 'kf', 'tri', 'global' or a file")
        with open(s) as f:
            return _parse_schedule_text(f.read())
    rows = [[int(x) for x in r] for r in np.asarray(schedule).tolist()]
    return rows, [False] * len(rows)


def color_code_memory(
    d: int,
    rounds: Optional[int] = None,
    basis: str = "Z",
    *,
    schedule: Any = "kf",
    flags: Union[bool, str, Sequence[int], Sequence[bool]] = False,
    p: float = 0.0,
    noise: str = "cnot",
    return_layout: bool = False,
) -> Union[Circuit, Tuple[Circuit, CodeLayout]]:
    """Triangular 6.6.6 colour code memory experiment (one auxiliary per plaquette).

    The round structure is Kishony–Fowler's (arXiv:2603.28852): ``CX
    data→aux`` over the schedule's layers, ``M``, ``RX``, ``CX aux→data`` over
    the same layers, ``MX``, ``R``; built by the engine's
    ``qsim_lab::qec::color`` (the circuits of ``research/qec/qec-r4.md`` and
    ``research/qec/colour-global.md``). ``schedule``: see
    :func:`color_code_schedule`. ``flags``: ``False``, ``True`` /
    ``"boundary"`` (a flag qubit on every boundary-touching plaquette, the
    hook-free boundary of ``research/qec/colour-flags.md``), or plaquette indices
    / a boolean mask. Default noise is ``"cnot"`` (K–F's headline model).

    >>> c, lay = color_code_memory(5, 2, p=1e-3, return_layout=True)
    >>> c.num_qubits, len(c.detectors), len(lay.memory_detectors)
    (28, 36, 27)
    """
    d = int(d)
    rounds = d if rounds is None else int(rounds)
    noise, p = _check_noise(noise, p)
    rows, sched_flags = color_code_schedule(d, schedule)
    plaq, boundary = _native_qec.color_code_plaquettes(d)
    npl = len(plaq)
    if flags is True or (isinstance(flags, str) and flags.lower() == "boundary"):
        mask = list(boundary)
    elif flags is False or flags is None:
        mask = [False] * npl
    else:
        fl = list(flags)
        if len(fl) == npl and all(isinstance(x, (bool, np.bool_)) for x in fl):
            mask = [bool(x) for x in fl]
        else:
            mask = [False] * npl
            for i in fl:
                if not 0 <= int(i) < npl:
                    raise ValueError(f"flag plaquette {i} out of range (0..{npl - 1})")
                mask[int(i)] = True
    if len(sched_flags) == npl:
        mask = [a or b for a, b in zip(mask, sched_flags)]
    core, raw, srows = _native_qec.color_code_memory(
        d, rounds, str(basis), rows, mask if any(mask) else [], p, noise
    )
    c = Circuit._wrap(core)
    if return_layout:
        return c, _layout("color", d, rounds, basis, noise, p, raw, srows)
    return c


# --------------------------------------------------------------------------- sampling


class DetectorSampler:
    """A compiled detection-event sampler for a noisy Clifford circuit.

    ``engine``: ``"fast"`` (the Poisson-hit FastSampler of
    ``research/qec/fast-sampler.md``), ``"symphase"`` (the plain SymPhase
    sampler: same distribution, slower) or ``"auto"`` (FastSampler, falling
    back to SymPhase with a :attr:`note` if it rejects the circuit). Compile
    once, sample many times:

    >>> s = DetectorSampler(repetition_code_memory(3, 2, p=0.05))
    >>> s.engine, s.num_detectors, s.num_observables
    ('fast', 6, 1)
    >>> a, _ = s.sample(100, seed=5); b, _ = s.sample(100, seed=5, threads=1)
    >>> bool((a == b).all())
    True
    """

    __slots__ = ("_core",)

    def __init__(self, circuit: Circuit, engine: str = "auto") -> None:
        if not isinstance(circuit, Circuit):
            raise TypeError("circuit must be a qsimlab.Circuit")
        self._core = _native_qec.DetectorSamplerCore(circuit._core, str(engine))

    @property
    def num_detectors(self) -> int:
        return self._core.num_detectors

    @property
    def num_observables(self) -> int:
        return self._core.num_observables

    @property
    def engine(self) -> str:
        """The sampler that runs: ``"fast"`` or ``"symphase"``."""
        return self._core.engine

    @property
    def note(self) -> str:
        """Why the engine differs from the request (empty if it does not)."""
        return self._core.note

    @property
    def compile_time(self) -> float:
        """Seconds spent compiling (SymPhase frame + hit tables)."""
        return self._core.compile_time

    def sample(
        self,
        shots: int,
        *,
        seed: Optional[int] = None,
        packed: bool = False,
        transposed: bool = False,
        threads: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """``(detectors, observables)``: ``bool[shots, D]`` and ``bool[shots, O]``,
        or Stim-style bit-packed ``uint8[shots, ceil(D/8)]`` arrays with ``packed=True``.

        ``transposed=True`` returns detector-major bit-packed arrays
        ``uint8[D, ceil(shots/8)]`` (bit ``s % 8`` of byte ``s // 8`` of row ``r``
        is detector ``r`` in shot ``s``; padding bits are 0). It skips the
        bit transposition, so it is the fastest layout (the Rust bench's
        throughput), and it is what per-detector statistics want. The same
        seed gives the same shots in every layout."""
        shots = int(shots)
        if shots < 0:
            raise ValueError("shots must be >= 0")
        if transposed:
            d, o, _ = self._core.sample(shots, _fresh_seed(seed), True, threads, True)
            return d, o
        d, o, _ = self._core.sample(shots, _fresh_seed(seed), bool(packed), threads)
        if not packed:
            d = d.view(np.bool_)
            o = o.view(np.bool_)
        return d, o

    def __repr__(self) -> str:
        return (
            f"DetectorSampler(engine={self.engine!r}, detectors={self.num_detectors}, "
            f"observables={self.num_observables})"
        )


def sample_detectors(
    circuit: Circuit,
    shots: int,
    *,
    seed: Optional[int] = None,
    engine: str = "auto",
    packed: bool = False,
    transposed: bool = False,
    threads: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Samples detection events and observable flips: ``(dets, obs)``.

    One-shot form of :class:`DetectorSampler` (which compiles once). Arrays
    are ``bool[shots, D]``, ``bool[shots, O]`` (``packed=True``: Stim's
    bit-packed ``uint8``; ``transposed=True``: detector-major bit-packed, the
    fastest layout, see :meth:`DetectorSampler.sample`). Raises :class:`UnsupportedOperationError` for
    non-Clifford gates or classically controlled operations.

    >>> c = repetition_code_memory(3, 1)            # noiseless: all zeros
    >>> dets, obs = sample_detectors(c, 10, seed=0)
    >>> int(dets.sum()), int(obs.sum())
    (0, 0)
    """
    return DetectorSampler(circuit, engine).sample(
        shots, seed=seed, packed=packed, transposed=transposed, threads=threads
    )


# --------------------------------------------------------------------------- DEM


class DemError(NamedTuple):
    """One error mechanism: independent event with ``probability`` that flips
    the ``detectors`` and ``observables`` (sorted index tuples)."""

    probability: float
    detectors: Tuple[int, ...]
    observables: Tuple[int, ...]


_DEM_LINE = re.compile(r"^([A-Za-z_]+)\s*(?:\(([^)]*)\))?\s*(.*)$")


def _xor_merge(a: float, b: float) -> float:
    return a * (1.0 - b) + b * (1.0 - a)


class DetectorErrorModel:
    """A detector error model: independent mechanisms over detectors/observables.

    Built from a circuit by :func:`detector_error_model` (exact: each
    depolarizing channel is converted into independent components the way
    Stim does, mechanisms with equal signatures merged), or parsed from Stim's
    DEM text format with :meth:`from_stim_dem`.

    >>> dem = DetectorErrorModel.from_stim_dem('''
    ...     error(0.1) D0 L0
    ...     error(0.2) D0 D1
    ...     detector D2
    ... ''')
    >>> dem.num_detectors, dem.num_observables, len(dem)
    (3, 1, 2)
    >>> print(dem.to_stim_dem())
    error(0.1) D0 L0
    error(0.2) D0 D1
    detector D2
    """

    __slots__ = ("num_detectors", "num_observables", "errors")

    def __init__(
        self,
        num_detectors: int,
        num_observables: int,
        errors: Iterable[Union[DemError, Tuple[float, Sequence[int], Sequence[int]]]],
    ) -> None:
        self.num_detectors = int(num_detectors)
        self.num_observables = int(num_observables)
        errs: List[DemError] = []
        for e in errors:
            p, d, o = e
            p = float(p)
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"error probability {p} not in [0, 1]")
            d = tuple(sorted(int(x) for x in d))
            o = tuple(sorted(int(x) for x in o))
            if d and (d[0] < 0 or d[-1] >= self.num_detectors):
                raise ValueError(f"error names detector outside 0..{self.num_detectors - 1}")
            if o and (o[0] < 0 or o[-1] >= self.num_observables):
                raise ValueError(f"error names observable outside 0..{self.num_observables - 1}")
            errs.append(DemError(p, d, o))
        self.errors: List[DemError] = errs

    def __len__(self) -> int:
        return len(self.errors)

    def __iter__(self):
        return iter(self.errors)

    def __repr__(self) -> str:
        return (
            f"DetectorErrorModel(detectors={self.num_detectors}, "
            f"observables={self.num_observables}, errors={len(self.errors)})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DetectorErrorModel):
            return NotImplemented
        return (
            self.num_detectors == other.num_detectors
            and self.num_observables == other.num_observables
            and sorted(self.errors) == sorted(other.errors)
        )

    def merged(self) -> Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float]:
        """``{(detectors, observables): probability}`` with equal signatures merged
        (as independent events)."""
        out: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}
        for p, d, o in self.errors:
            k = (d, o)
            out[k] = _xor_merge(out.get(k, 0.0), p)
        return out

    def approx_equal(self, other: "DetectorErrorModel", *, rtol: float = 1e-9,
                     atol: float = 0.0) -> bool:
        """Same detectors/observables and the same merged mechanisms with
        probabilities equal to ``rtol``/``atol``."""
        if (self.num_detectors, self.num_observables) != (
            other.num_detectors,
            other.num_observables,
        ):
            return False
        a, b = self.merged(), other.merged()
        if a.keys() != b.keys():
            return False
        return all(math.isclose(a[k], b[k], rel_tol=rtol, abs_tol=atol) for k in a)

    def matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dense ``(H, L, p)``: ``uint8[D, E]`` check matrix, ``uint8[O, E]``
        observable matrix and ``float64[E]`` probabilities."""
        E = len(self.errors)
        H = np.zeros((self.num_detectors, E), dtype=np.uint8)
        L = np.zeros((self.num_observables, E), dtype=np.uint8)
        p = np.empty(E, dtype=np.float64)
        for j, (pj, d, o) in enumerate(self.errors):
            H[list(d), j] = 1
            L[list(o), j] = 1
            p[j] = pj
        return H, L, p

    # -- Stim text format ------------------------------------------------------

    def to_stim_dem(self) -> str:
        """Stim's DEM text (``error(p) D.. L..`` lines; ``detector`` /
        ``logical_observable`` declarations keep the counts)."""
        lines = []
        used_d, used_o = -1, -1
        for p, d, o in self.errors:
            if d:
                used_d = max(used_d, d[-1])
            if o:
                used_o = max(used_o, o[-1])
            t = " ".join([f"D{x}" for x in d] + [f"L{x}" for x in o])
            lines.append(f"error({p!r}) {t}".rstrip())
        if used_d < self.num_detectors - 1:
            lines.append(f"detector D{self.num_detectors - 1}")
        for k in range(used_o + 1, self.num_observables):
            lines.append(f"logical_observable L{k}")
        return "\n".join(lines)

    @classmethod
    def from_stim_dem(cls, text: str) -> "DetectorErrorModel":
        """Parses Stim's DEM text: ``error``, ``detector``,
        ``logical_observable``, ``shift_detectors``, ``repeat`` blocks and
        ``^`` decomposition separators (components are XOR-ed back together).
        ``detector_separator`` and coordinates are ignored."""
        lines = [ln.split("#", 1)[0].strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln]
        errors: List[Tuple[float, List[int], List[int]]] = []
        state = {"offset": 0, "maxd": -1, "maxo": -1}

        def parse_targets(toks: List[str], lineno: str) -> Tuple[List[int], List[int]]:
            ds: set = set()
            os_: set = set()
            for t in toks:
                if t == "^":
                    continue
                try:
                    if t[0] == "D":
                        ds ^= {int(t[1:]) + state["offset"]}
                    elif t[0] == "L":
                        os_ ^= {int(t[1:])}
                    else:
                        raise ValueError
                except ValueError:
                    raise ParseError(f"DEM line {lineno!r}: bad target {t!r}") from None
            return sorted(ds), sorted(os_)

        def block(start: int, end: int) -> None:
            i = start
            while i < end:
                ln = lines[i]
                mt = _DEM_LINE.match(ln)
                if mt is None:
                    if ln == "}":
                        raise ParseError("DEM: unbalanced '}'")
                    raise ParseError(f"DEM line {ln!r}: cannot parse")
                name = mt.group(1).lower()
                args = mt.group(2) or ""
                rest = mt.group(3)
                toks = rest.split()
                if name == "repeat":
                    count = int(toks[0])
                    if not ln.endswith("{"):
                        raise ParseError(f"DEM line {ln!r}: expected '{{'")
                    depth, j = 1, i + 1
                    while j < end and depth:
                        if lines[j].endswith("{"):
                            depth += 1
                        elif lines[j] == "}":
                            depth -= 1
                        j += 1
                    if depth:
                        raise ParseError("DEM: unterminated repeat block")
                    for _ in range(count):
                        block(i + 1, j - 1)
                    i = j
                    continue
                if name == "error":
                    p = float(args)
                    ds, os_ = parse_targets(toks, ln)
                    errors.append((p, ds, os_))
                    if ds:
                        state["maxd"] = max(state["maxd"], ds[-1])
                    if os_:
                        state["maxo"] = max(state["maxo"], os_[-1])
                elif name in ("detector", "logical_observable"):
                    ds, os_ = parse_targets(toks, ln)
                    if ds:
                        state["maxd"] = max(state["maxd"], ds[-1])
                    if os_:
                        state["maxo"] = max(state["maxo"], os_[-1])
                elif name == "shift_detectors":
                    state["offset"] += int(toks[0]) if toks else 0
                elif name == "detector_separator":
                    pass
                else:
                    raise ParseError(f"DEM line {ln!r}: unknown instruction {name!r}")
                i += 1

        block(0, len(lines))
        return cls(state["maxd"] + 1, state["maxo"] + 1, errors)

    def to_stim(self) -> Any:
        """A ``stim.DetectorErrorModel`` (needs ``stim``)."""
        try:
            import stim
        except ImportError:
            raise MissingDependencyError("to_stim() needs stim (pip install stim)") from None
        return stim.DetectorErrorModel(self.to_stim_dem())

    @classmethod
    def from_stim(cls, dem: Any) -> "DetectorErrorModel":
        """From a ``stim.DetectorErrorModel`` (via its text, loops flattened)."""
        n_d, n_o = dem.num_detectors, dem.num_observables
        out = cls.from_stim_dem(str(dem.flattened()))
        return cls(max(n_d, out.num_detectors), max(n_o, out.num_observables), out.errors)

    def graphlike(self) -> Tuple["DetectorErrorModel", int]:
        """Decomposes every mechanism with more than two detectors into
        mechanisms of the model with one or two detectors whose XOR (detectors
        and observables) reproduces it, as Stim's ``decompose_errors`` does;
        returns ``(graphlike model, number of mechanisms that could not be
        decomposed and were dropped)``. Used by the matching decoders."""
        edges: Dict[Tuple[int, ...], set] = {}
        for p, d, o in self.errors:
            if 1 <= len(d) <= 2:
                edges.setdefault(d, set()).add(o)
        acc: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}
        failed = 0

        def xor_t(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
            return tuple(sorted(set(a) ^ set(b)))

        def split(rem: Tuple[int, ...], target: Tuple[int, ...], depth: int):
            if not rem:
                return [] if not target else None
            if depth > 8:
                return None
            a = rem[0]
            cands = [(a,)] + [(a, b) for b in rem[1:]]
            for part in cands:
                if part not in edges:
                    continue
                rest = tuple(x for x in rem if x not in part)
                for o in edges[part]:
                    sub = split(rest, xor_t(target, o), depth + 1)
                    if sub is not None:
                        return [(part, o)] + sub
            return None

        for p, d, o in self.errors:
            if len(d) <= 2:
                parts = [(d, o)]
            else:
                parts = split(d, o, 0)
                if parts is None:
                    failed += 1
                    continue
            for part in parts:
                acc[part] = _xor_merge(acc.get(part, 0.0), p)
        g = DetectorErrorModel(
            self.num_detectors,
            self.num_observables,
            [(p, d, o) for (d, o), p in sorted(acc.items())],
        )
        return g, failed


def detector_error_model(circuit: Circuit) -> DetectorErrorModel:
    """The circuit-derived detector error model (no decomposition).

    Every Pauli noise channel of the circuit becomes independent mechanisms
    (a flip: one; ``DEPOLARIZE1``/``DEPOLARIZE2``: 3/15 components with
    Stim's exact disjoint→independent conversion), propagated to the
    detectors and observables; mechanisms with equal signatures are merged.
    Equals Stim's ``circuit.detector_error_model()`` of ``circuit.to_stim()``
    up to floating-point rounding. Raises :class:`UnsupportedOperationError`
    if a detector or observable is not deterministic.

    >>> dem = detector_error_model(repetition_code_memory(3, 1, p=0.1, noise="cnot"))
    >>> dem
    DetectorErrorModel(detectors=4, observables=1, errors=9)
    """
    if not isinstance(circuit, Circuit):
        raise TypeError("circuit must be a qsimlab.Circuit")
    nd, no, errs = _native_qec.detector_error_model(circuit._core)
    return DetectorErrorModel(nd, no, errs)


# --------------------------------------------------------------------------- distance


@dataclass(frozen=True)
class DistanceResult:
    """Result of :func:`circuit_distance`.

    * ``distance``: minimum number of error mechanisms that flip the observable
      without firing a detector (``None``: none found up to ``searched_weight``,
      or the search stopped, see ``complete``);
    * ``count``: number of distinct minimum-weight logicals (``count_capped``:
      the count reached the cap);
    * ``example``: one minimum-weight logical as :class:`DemError` mechanisms;
    * ``complete``: the answer is exact (no timeout, node limit or sector caveat);
    * ``certified``: with ``detectors=`` (a sector): the example lifts to the
      full model, so ``distance`` is the full circuit distance (a sector
      search alone gives a lower bound);
    * ``lower_bound``: a proven lower bound on the distance;
    * ``nodes``, ``wall_time``: search effort.
    """

    distance: Optional[int]
    count: int
    count_capped: bool
    example: List[DemError] = field(repr=False)
    complete: bool
    certified: bool
    lower_bound: int
    searched_weight: int
    timed_out: bool
    node_limit_hit: bool
    nodes: int
    wall_time: float


def circuit_distance(
    circuit: Union[Circuit, DetectorErrorModel],
    *,
    max_weight: Optional[int] = None,
    observable: int = 0,
    detectors: Optional[Sequence[int]] = None,
    count_cap: int = 1_000_000,
    node_limit: Optional[int] = None,
    timeout: Optional[float] = None,
) -> DistanceResult:
    """Exact circuit-level distance by branch and bound (``qsim_lab::qec::distance``).

    Searches the detector error model for the smallest set of mechanisms
    that flips ``observable`` and no detector, and counts all such sets.
    ``detectors``: restrict to a sector (e.g. ``layout.memory_detectors``):
    much faster on CSS memories, a lower bound in general, exact when
    ``certified``. ``timeout`` (seconds) switches to iterative deepening over
    the weight; each weight gets the node budget that the measured search
    rate allows in the remaining time, so the limit is approximate (typically
    within a few tens of percent). ``node_limit`` bounds the search nodes.
    On a stop, ``lower_bound`` still reports what was proven.

    >>> r = circuit_distance(surface_code_memory(5, 2, p=1e-3))
    >>> r.distance, r.complete
    (5, True)
    """
    dem = circuit if isinstance(circuit, DetectorErrorModel) else detector_error_model(circuit)
    if not dem.errors:
        raise ValueError("the error model has no mechanisms (is the circuit noiseless?)")
    if not 0 <= observable < max(dem.num_observables, 1):
        raise ValueError(f"observable {observable} out of range")
    mw = 64 if max_weight is None else int(max_weight)
    if mw < 1:
        raise ValueError("max_weight must be >= 1")
    keep = None if detectors is None else [int(x) for x in detectors]
    r = _native_qec.min_weight_logical(
        dem.num_detectors,
        dem.errors,
        int(observable),
        keep,
        mw,
        int(count_cap),
        None if node_limit is None else int(node_limit),
        None if timeout is None else float(timeout),
    )
    w = r["weight"]
    stopped = r["timed_out"] or r["node_limit_hit"]
    certified = bool(r["certified"]) if keep is not None else w is not None
    lb = w if w is not None else r["searched_weight"] + 1
    complete = not stopped and (keep is None or w is None or certified)
    return DistanceResult(
        distance=w,
        count=int(r["count"]),
        count_capped=bool(r["count_capped"]),
        example=[dem.errors[j] for j in r["example"]],
        complete=complete,
        certified=certified,
        lower_bound=int(lb),
        searched_weight=int(r["searched_weight"]),
        timed_out=bool(r["timed_out"]),
        node_limit_hit=bool(r["node_limit_hit"]),
        nodes=int(r["nodes"]),
        wall_time=float(r["wall_time"]),
    )


# --------------------------------------------------------------------------- decoders


def _as_bool_dets(dets: np.ndarray, num_detectors: int, packed: Optional[bool]) -> np.ndarray:
    a = np.asarray(dets)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError("detection events must be a 2-D array (shots, detectors)")
    nb = (num_detectors + 7) // 8
    if packed is None:
        packed = a.dtype == np.uint8 and a.shape[1] == nb and a.shape[1] != num_detectors
    if packed:
        if a.shape[1] != nb:
            raise ValueError(f"packed detection events need {nb} bytes per shot, got {a.shape[1]}")
        return np.unpackbits(a.astype(np.uint8, copy=False), axis=1, count=num_detectors,
                             bitorder="little").astype(bool)
    if a.shape[1] != num_detectors:
        raise ValueError(f"expected {num_detectors} detectors per shot, got {a.shape[1]}")
    return a.astype(bool, copy=False)


def _as_packed_dets(dets: np.ndarray, num_detectors: int, packed: Optional[bool]) -> np.ndarray:
    a = np.asarray(dets)
    nb = (num_detectors + 7) // 8
    if a.ndim == 1:
        a = a[None, :]
    if packed is None:
        packed = a.dtype == np.uint8 and a.shape[1] == nb and a.shape[1] != num_detectors
    if packed:
        if a.shape[1] != nb:
            raise ValueError(f"packed detection events need {nb} bytes per shot, got {a.shape[1]}")
        return np.ascontiguousarray(a, dtype=np.uint8)
    b = _as_bool_dets(a, num_detectors, False)
    return np.packbits(b, axis=1, bitorder="little")


class Decoder:
    """Base class: ``decode(dets) -> bool[shots, num_observables]`` predicted flips."""

    name: str = ""
    num_detectors: int = 0
    num_observables: int = 0

    def decode(self, dets: np.ndarray, *, packed: Optional[bool] = None,
               threads: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(detectors={self.num_detectors}, observables={self.num_observables})"


class BpOsdDecoder(Decoder):
    """Belief propagation + ordered-statistics decoding (``qsim_lab::qec::bposd``:
    normalised min-sum BP, OSD-CS of order ``osd_order``) on the full
    hypergraph DEM; parallel over shots with the GIL released. At most 64
    observables."""

    name = "bposd"

    def __init__(self, dem: DetectorErrorModel, *, max_iter: int = 50,
                 ms_scale: float = 0.625, osd_order: int = 10) -> None:
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables
        self._core = _native_qec.BpOsdCore(
            dem.num_detectors, dem.num_observables, dem.errors,
            int(max_iter), float(ms_scale), int(osd_order),
        )
        self.last_stats: Dict[str, int] = {}

    def decode(self, dets: np.ndarray, *, packed: Optional[bool] = None,
               threads: Optional[int] = None) -> np.ndarray:
        syn = _as_packed_dets(dets, self.num_detectors, packed)
        masks, conv, osd = self._core.decode_packed(syn, threads)
        self.last_stats = {"bp_converged": conv, "osd_calls": osd}
        k = np.arange(self.num_observables, dtype=np.uint64)
        return ((masks[:, None] >> k[None, :]) & np.uint64(1)).astype(bool)


class PyMatchingDecoder(Decoder):
    """Minimum-weight perfect matching via PyMatching (optional dependency).

    Mechanisms with more than two detectors are decomposed into graphlike ones
    (:meth:`DetectorErrorModel.graphlike`); ``dropped`` counts those that
    could not be."""

    name = "pymatching"

    def __init__(self, dem: DetectorErrorModel, **options: Any) -> None:
        try:
            import pymatching
            import scipy.sparse
        except ImportError:
            raise MissingDependencyError(
                "decoder='pymatching' needs pymatching (pip install pymatching)"
            ) from None
        g, self.dropped = dem.graphlike()
        if self.dropped:
            warnings.warn(
                f"{self.dropped} hyperedge mechanisms could not be decomposed into "
                "graphlike ones and are ignored by the matching decoder",
                stacklevel=3,
            )
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables
        errs = [e for e in g.errors if e.detectors and 0.0 < e.probability < 0.5]
        rows, cols, orow, ocol = [], [], [], []
        for j, (_, d, o) in enumerate(errs):
            rows += d
            cols += [j] * len(d)
            orow += o
            ocol += [j] * len(o)
        E = len(errs)
        H = scipy.sparse.csc_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(self.num_detectors, E))
        L = scipy.sparse.csc_matrix(
            (np.ones(len(orow), dtype=np.uint8), (orow, ocol)), shape=(self.num_observables, E))
        p = np.array([e.probability for e in errs], dtype=np.float64)
        self._m = pymatching.Matching.from_check_matrix(
            H, weights=np.log((1 - p) / p), faults_matrix=L, use_virtual_boundary_node=True,
            **options)

    def decode(self, dets: np.ndarray, *, packed: Optional[bool] = None,
               threads: Optional[int] = None) -> np.ndarray:
        b = _as_bool_dets(dets, self.num_detectors, packed).astype(np.uint8)
        return np.asarray(self._m.decode_batch(b), dtype=bool).reshape(len(b), -1)


class TesseractDecoder(Decoder):
    """Tesseract (A* search, near-optimal; optional ``tesseract-decoder`` + ``stim``)."""

    name = "tesseract"

    def __init__(self, dem: DetectorErrorModel, **options: Any) -> None:
        try:
            import stim  # noqa: F401
            from tesseract_decoder import tesseract
        except ImportError:
            raise MissingDependencyError(
                "decoder='tesseract' needs tesseract-decoder and stim "
                "(pip install tesseract-decoder stim)"
            ) from None
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables
        cfg = tesseract.TesseractConfig(dem=dem.to_stim(), **options)
        self._d = cfg.compile_decoder()

    def decode(self, dets: np.ndarray, *, packed: Optional[bool] = None,
               threads: Optional[int] = None) -> np.ndarray:
        b = _as_bool_dets(dets, self.num_detectors, packed)
        out = np.asarray(self._d.decode_batch(b.astype(bool)), dtype=bool)
        return out.reshape(len(b), self.num_observables)


def make_decoder(dem: DetectorErrorModel, decoder: str = "bposd", **options: Any) -> Decoder:
    """A decoder for ``dem``: ``"bposd"`` (built in), ``"pymatching"`` or
    ``"tesseract"`` (optional packages; :class:`MissingDependencyError` if
    absent). ``options`` go to the decoder's constructor."""
    if not isinstance(dem, DetectorErrorModel):
        raise TypeError("dem must be a DetectorErrorModel")
    key = str(decoder).lower()
    if key == "bposd":
        return BpOsdDecoder(dem, **options)
    if key == "pymatching":
        return PyMatchingDecoder(dem, **options)
    if key == "tesseract":
        return TesseractDecoder(dem, **options)
    raise ValueError(f"unknown decoder {decoder!r}; known: {DECODERS}")


def decode(
    dem: DetectorErrorModel,
    dets: np.ndarray,
    decoder: Union[str, Decoder] = "bposd",
    *,
    packed: Optional[bool] = None,
    threads: Optional[int] = None,
    **options: Any,
) -> np.ndarray:
    """Predicted observable flips ``bool[shots, O]`` for detection events
    ``dets`` (``bool[shots, D]`` or bit-packed ``uint8``).

    >>> c = repetition_code_memory(5, 3, p=0.02)
    >>> dets, obs = sample_detectors(c, 1000, seed=3)
    >>> pred = decode(detector_error_model(c), dets)
    >>> float((pred != obs).any(axis=1).mean()) < 0.05
    True
    """
    dec = decoder if isinstance(decoder, Decoder) else make_decoder(dem, decoder, **options)
    return dec.decode(dets, packed=packed, threads=threads)


# --------------------------------------------------------------------------- LER


def wilson_interval(errors: int, shots: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Wilson score interval (default 95%) for ``errors / shots``."""
    if shots <= 0:
        return (0.0, 1.0)
    ph = errors / shots
    den = 1 + z * z / shots
    c = (ph + z * z / (2 * shots)) / den
    h = z * math.sqrt(ph * (1 - ph) / shots + z * z / (4 * shots * shots)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def _per_round(p: float, rounds: int) -> float:
    return 0.5 * (1.0 - max(1.0 - 2.0 * p, 0.0) ** (1.0 / rounds))


@dataclass(frozen=True)
class LogicalErrorRate(Result):
    """Result of :func:`logical_error_rate`. A shot fails if any observable is
    mispredicted.

    * ``errors``, ``shots``, ``rate``; ``ci``: Wilson 95% interval;
    * ``rounds``: if known, ``per_round`` / ``per_round_ci`` convert with
      ``p_L = (1 - (1 - 2 eps)^rounds) / 2``;
    * ``decoder``; ``stats``: decoder statistics (BP convergence, OSD calls).
    """

    errors: int = 0
    shots: int = 0
    rate: float = 0.0
    ci: Tuple[float, float] = (0.0, 1.0)
    decoder: str = ""
    rounds: Optional[int] = None
    per_round: Optional[float] = None
    per_round_ci: Optional[Tuple[float, float]] = None
    stats: Dict[str, Any] = field(default_factory=dict, repr=False)


def logical_error_rate(
    circuit: Circuit,
    shots: int,
    *,
    decoder: Union[str, Decoder] = "bposd",
    seed: Optional[int] = None,
    max_errors: Optional[int] = None,
    rounds: Optional[int] = None,
    engine: str = "auto",
    dem: Optional[DetectorErrorModel] = None,
    threads: Optional[int] = None,
    **decoder_options: Any,
) -> LogicalErrorRate:
    """Monte-Carlo logical error rate of a memory circuit with a decoder.

    Samples ``shots`` (stopping early once ``max_errors`` failures are seen;
    checked every 65,536 shots so results stay seed-reproducible), decodes
    with the circuit's own detector error model (or ``dem``) and counts shots
    where any observable is mispredicted. ``decoder="bposd"`` runs sampling
    and decoding fused in native code; other decoders run in Python chunks.

    >>> c = surface_code_memory(3, 3, p=2e-3)
    >>> r = logical_error_rate(c, 10_000, seed=7, rounds=3)
    >>> 0 <= r.rate < 0.05, r.ci[0] <= r.rate <= r.ci[1]
    (True, True)
    """
    if not isinstance(circuit, Circuit):
        raise TypeError("circuit must be a qsimlab.Circuit")
    shots = int(shots)
    if shots < 0:
        raise ValueError("shots must be >= 0")
    seed = _fresh_seed(seed)
    t0 = time.perf_counter()
    dem = detector_error_model(circuit) if dem is None else dem
    sampler = DetectorSampler(circuit, engine)
    if dem.num_detectors != sampler.num_detectors:
        raise ValueError(
            f"dem has {dem.num_detectors} detectors, the circuit {sampler.num_detectors}"
        )
    dec = decoder if isinstance(decoder, Decoder) else make_decoder(dem, decoder, **decoder_options)
    stats: Dict[str, Any] = {"compile_time": sampler.compile_time}
    if isinstance(dec, BpOsdDecoder):
        r = _native_qec.sample_decode_count(
            sampler._core, dec._core, shots, seed,
            None if max_errors is None else int(max_errors), threads,
        )
        errors, done = int(r["errors"]), int(r["shots"])
        stats.update(bp_converged=r["bp_converged"], osd_calls=r["osd_calls"])
    else:
        errors = done = 0
        chunk = 65_536
        k = 0
        while done < shots and (max_errors is None or errors < max_errors):
            n = min(chunk, shots - done)
            sub = (seed * 0x9E3779B97F4A7C15 + k + 1) % 2**64
            dets, obs = sampler.sample(n, seed=sub, threads=threads)
            pred = dec.decode(dets, packed=False, threads=threads)
            errors += int((pred != obs).any(axis=1).sum())
            done += n
            k += 1
    rate = errors / done if done else 0.0
    ci = wilson_interval(errors, done)
    pr = pr_ci = None
    if rounds:
        pr = _per_round(rate, int(rounds))
        pr_ci = (_per_round(ci[0], int(rounds)), _per_round(ci[1], int(rounds)))
    return LogicalErrorRate(
        engine=sampler.engine,
        components=[(circuit.num_qubits, len(circuit), sampler.engine)],
        seed=seed,
        precision="f64",
        wall_time=time.perf_counter() - t0,
        errors=errors,
        shots=done,
        rate=rate,
        ci=ci,
        decoder=getattr(dec, "name", type(dec).__name__),
        rounds=int(rounds) if rounds else None,
        per_round=pr,
        per_round_ci=pr_ci,
        stats=stats,
    )
