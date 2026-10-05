"""Shor's algorithm at gate level: factoring runs, oracle circuits, resource counts, cost laws.

**Provisional** (phase 2, see ``python/API.md`` §1 and §8).

Every run simulates the semiclassical order-finding circuit exactly (one recycled
control qubit, Griffiths–Niu phase corrections) with a gate-level oracle built
from X/CNOT/Toffoli gates, on the bit-sliced branch engine of
``research/shor/shor.md`` (or the dense/sparse engines for the permutation and
Beauregard oracles)::

    >>> import qsimlab.shor as shor
    >>> r = shor.factor(1_005_973, seed=1)            # 20-bit N, 88 qubits
    >>> r.factors, r.base, r.order
    ((997, 1009), 980062, 41832)
    >>> r.qubits, r.toffoli_gates
    (88, 70720)
    >>> bool((r.runs[0].support_trace <= r.runs[0].predicted_support).all())
    True

The cost of a run is set by the multiplicative order ``r`` of the base, not by
``N`` (the support law, ``research/theory/theory-shor.md`` T1): before round ``i`` the
state holds at most ``B_i = min(2^i, r / gcd(r, 2^(t-i)))`` branches.
:func:`factor` predicts the peak memory from it **before** running and refuses
runs over ``budget`` with :class:`~qsimlab.ResourceLimitError`. The order used
for that prediction is computed classically (Pollard–Brent factoring); the
simulated algorithm never sees it. This is exact simulation, not a factoring
speed-up: for a generic semiprime and a random base, ``r ≈ N/c`` and the work
grows like ``N · n³``.

Oracles (``oracle=``, see :data:`ORACLES`):

* ``"windowed-opt"`` (default): Gidney's windowed table-lookup multiplier with
  the superoptimised blocks of ``research/shor/superopt.md``; ``4n + 4 + w`` qubits;
* ``"windowed-mbu-lookup"`` / ``"windowed-mbu"``: measurement-based
  uncomputation (``research/shor/mbu-shor.md``): fewer Toffolis, X-basis
  measurements with classical fix-ups;
* ``"windowed"``, ``"ripple"``: the round-4 windowed oracle and the Cuccaro
  ripple-carry oracle (``3n + 4`` qubits);
* ``"beauregard"``: Draper/Beauregard QFT arithmetic, ``2n + 3`` qubits, dense
  state (≈ 10-bit N);
* ``"permutation"``: a classical lookup table as the oracle, ``n + 1`` qubits
  (measurement statistics only, not a compilable circuit);
* ``"ge"`` / ``"eh"``: Gidney–Ekerå exponent windowing (``research/shor/ge-shor.md``)
  for Shor's order finding, or the Ekerå–Håstad short-discrete-log schedule
  (``1.5n`` exponent bits, lattice post-processing; balanced semiprimes).
"""

from __future__ import annotations

import math
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ._native import shor as _native_shor
from .circuit import Circuit
from .sim import Budget, Result, _memory_arg, _parse_memory

__all__ = [
    "ORACLES",
    "factor",
    "FactorResult",
    "ShorRun",
    "resource_counts",
    "ResourceCounts",
    "oracle_circuit",
    "OracleCircuit",
    "shor_circuit",
    "exact_distribution",
    "predict_support",
    "SupportPrediction",
    "support_bounds",
    "noisy_success",
    "NoisySuccess",
    "multiplicative_order",
    "carmichael",
]

#: Oracle names accepted by every function of this module.
ORACLES: Tuple[str, ...] = tuple(_native_shor.ORACLES)

_HARD_CAP = 32 << 30


def _default_budget() -> int:
    """``min(32 GiB, half the physical memory)``."""
    try:
        phys = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return _HARD_CAP
    return int(min(_HARD_CAP, phys // 2))


def _budget_bytes(budget: Union[Budget, int, str, None]) -> int:
    m = _memory_arg(budget)
    if m is None:
        return _default_budget()
    b = m if isinstance(m, int) else _parse_memory(m)
    if b <= 0:
        raise ValueError("budget must be positive")
    return int(b)


def _seed(seed: Optional[int]) -> int:
    if seed is None:
        return secrets.randbits(64)
    seed = int(seed)
    if not 0 <= seed < 2**64:
        raise ValueError("seed must be in [0, 2**64)")
    return seed


def _precision(p: str) -> bool:
    if p not in ("f64", "f32"):
        raise ValueError(f"precision must be 'f64' or 'f32', got {p!r}")
    return p == "f32"


def _int(x: Any, name: str) -> int:
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(x).__name__}")
    return int(x)


def support_bounds(order: int, rounds: int) -> np.ndarray:
    """The support law (``research/theory/theory-shor.md`` T1): ``B_i = min(2^i, r / gcd(r, 2^(t-i)))``
    for ``i = 0 … t-1`` — an upper bound on the number of branches before round ``i``,
    attained except with probability ≤ 4/r_odd per round.

    >>> support_bounds(12, 8).tolist()        # r = 12 = 4 · 3, t = 8 rounds
    [1, 2, 3, 3, 3, 3, 3, 6]
    """
    r, t = int(order), int(rounds)
    if r < 1 or t < 1:
        raise ValueError("order and rounds must be positive")
    nu = (r & -r).bit_length() - 1
    return np.array([min(1 << i, r >> min(nu, t - i)) for i in range(t)], dtype=np.uint64)


# --------------------------------------------------------------------------- factoring


@dataclass(frozen=True)
class ShorRun:
    """One order-finding (or Ekerå–Håstad) run.

    * ``base``: the base ``a``; ``measured``: the measured integer (bit ``i`` =
      round ``i``), a tuple ``(k, j)`` for ``"eh"`` (one integer per exponent
      register);
    * ``order`` / ``factors``: what the classical post-processing recovered
      (``None`` if it failed — that is the classical part of Shor failing, e.g.
      an odd order or ``a^(r/2) ≡ −1``);
    * ``qubits``, ``total_gates`` (every gate of the run, control gates
      included), ``toffoli_gates``, ``measurements`` (mid-circuit X-basis
      measurements of the MBU oracles);
    * ``peak_support``: largest number of stored branches;
      ``peak_amplitude_bytes``: amplitude memory of dense/sparse engines, peak
      branch count of the GE engine; ``gate_branch_ops``: gates × branches
      evaluated (the work of the sliced engine);
    * ``true_order``: the multiplicative order computed classically (for the
      memory guard); ``predicted_peak_support`` / ``predicted_bytes``: the
      guard's prediction;
    * ``support_trace``: branches stored before every round (sliced engine,
      ``trace=True``), ``p1_trace``: ``P(control = 1)`` of every round.
    """

    base: int
    measured: Union[int, Tuple[int, ...]]
    order: Optional[int]
    factors: Optional[Tuple[int, int]]
    qubits: int
    total_gates: int
    toffoli_gates: int
    measurements: int
    peak_support: int
    peak_amplitude_bytes: int
    gate_branch_ops: int
    true_order: int
    predicted_peak_support: int
    predicted_bytes: int
    engine: str
    wall_time: float
    support_trace: Optional[np.ndarray] = field(default=None, repr=False)
    p1_trace: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def succeeded(self) -> bool:
        return self.factors is not None

    @property
    def predicted_support(self) -> Optional[np.ndarray]:
        """``B_i`` for the rounds in ``support_trace`` (``None`` without a trace)."""
        if self.support_trace is None:
            return None
        return support_bounds(self.true_order, len(self.support_trace))


@dataclass(frozen=True)
class FactorResult(Result):
    """Result of :func:`factor` (a :class:`qsimlab.sim.Result`).

    ``factors`` is ``(p, q)`` with ``p ≤ q`` or ``None``; ``base``, ``order``,
    ``measured`` and the resource fields describe the last (successful) run;
    ``runs`` lists every attempt (:class:`ShorRun`). ``components`` has one
    ``(qubits, gates, engine)`` entry per run.
    """

    N: int = 0
    oracle: str = ""
    factors: Optional[Tuple[int, int]] = None
    base: Optional[int] = None
    order: Optional[int] = None
    measured: Union[int, Tuple[int, ...], None] = None
    qubits: int = 0
    total_gates: int = 0
    toffoli_gates: int = 0
    peak_support: int = 0
    budget: int = 0
    runs: List[ShorRun] = field(default_factory=list, repr=False)


def _run(d: Dict[str, Any]) -> ShorRun:
    ms = tuple(int(x) for x in d["measured"])
    return ShorRun(
        base=int(d["base"]),
        measured=ms[0] if len(ms) == 1 else ms,
        order=d["order"],
        factors=tuple(d["factors"]) if d["factors"] is not None else None,
        qubits=d["qubits"],
        total_gates=d["total_gates"],
        toffoli_gates=d["toffoli_gates"],
        measurements=d["measurements"],
        peak_support=d["peak_support"],
        peak_amplitude_bytes=d["peak_amplitude_bytes"],
        gate_branch_ops=int(d["gate_branch_ops"]),
        true_order=d["true_order"],
        predicted_peak_support=d["predicted_peak_support"],
        predicted_bytes=int(d["predicted_bytes"]),
        engine=d["engine"],
        wall_time=d["wall_time"],
        support_trace=d["support_trace"],
        p1_trace=d["p1_trace"],
    )


def factor(
    N: int,
    *,
    oracle: str = "windowed-opt",
    window: Optional[int] = None,
    exponent_window: Optional[int] = None,
    base: Optional[int] = None,
    precision: str = "f64",
    seed: Optional[int] = None,
    tries: int = 10,
    budget: Union[Budget, int, str, None] = None,
    engine: str = "auto",
    trace: bool = True,
    threads: Optional[int] = None,
) -> FactorResult:
    """Factor ``N`` by simulating Shor's algorithm at gate level.

    Up to ``tries`` runs; each draws a random base ``a`` (or uses ``base``),
    simulates the whole semiclassical order-finding circuit and post-processes
    the measured integer (continued fractions, small multiples), stopping at
    the first run that yields a factor. With the same ``seed`` the bases and
    measured integers are those of ``qsim run shor --seed <seed>`` (the CLI of
    the Rust crate).

    * ``window``: lookup window ``w`` of the windowed oracles (default 4) or
      the multiplicand window ``w_m`` of ``"ge"``/``"eh"`` (default 3);
      ``exponent_window``: ``w_e`` of ``"ge"``/``"eh"`` (default 2).
    * ``precision``: amplitudes in ``"f64"`` or ``"f32"`` (≈ 35 % less memory;
      distributions within 1e-7 of f64, ``research/shor/shor.md``).
    * ``budget``: refuse (``ResourceLimitError``, with ``needed``/``limit``)
      any run whose predicted peak memory exceeds it. Default:
      ``min(32 GiB, half the physical memory)``.
    * ``engine``: ``"auto"`` (sliced branches for the gate-level X/CNOT/CCX
      oracles, the cheaper of fused dense/sparse for ``"permutation"``, dense
      for ``"beauregard"``), or ``"sliced"``, ``"dense"``, ``"sparse"``.

    ``N`` must be odd, composite, not a prime power, and below ``2^62``.

    >>> r = factor(143, oracle="ripple", seed=3)
    >>> r.factors
    (11, 13)
    >>> pow(r.base, r.order, 143)
    1
    """
    N = _int(N, "N")
    f32 = _precision(precision)
    seed = _seed(seed)
    if base is not None:
        base = _int(base, "base")
    tries = _int(tries, "tries")
    b = _budget_bytes(budget)
    d = _native_shor.factor(
        N,
        oracle,
        window=window,
        exponent_window=exponent_window,
        base=base,
        f32=f32,
        seed=seed,
        tries=tries,
        budget=b,
        engine=engine,
        trace=bool(trace),
        threads=threads,
    )
    runs = [_run(x) for x in d["runs"]]
    last = runs[-1] if runs else None
    engines = {r.engine for r in runs}
    return FactorResult(
        engine=engines.pop() if len(engines) == 1 else ("pipeline" if engines else "none"),
        components=[(r.qubits, r.total_gates, r.engine) for r in runs],
        seed=seed,
        precision=d["precision"],
        wall_time=d["wall_time"],
        N=N,
        oracle=d["oracle"],
        factors=tuple(d["factors"]) if d["factors"] is not None else None,
        base=last.base if last else None,
        order=last.order if last else None,
        measured=last.measured if last else None,
        qubits=last.qubits if last else 0,
        total_gates=last.total_gates if last else 0,
        toffoli_gates=last.toffoli_gates if last else 0,
        peak_support=last.peak_support if last else 0,
        budget=b,
        runs=runs,
    )


# --------------------------------------------------------------------------- counts, circuits


@dataclass(frozen=True)
class ResourceCounts:
    """Whole-run counts of the oracle blocks of one order-finding (or EH) run.

    ``oracle_gates`` counts every operation of the ``rounds`` controlled-``U``
    blocks (X, CNOT, Toffoli, X-basis measurements and their Z/CZ fix-ups);
    ``toffoli``, ``cnot``, ``x``, ``measurements``, ``fixups`` break it down.
    The semiclassical control adds at most 2 H, 1 phase and 1 recycling X per
    round (``control_gates_max``). For the MBU and GE oracles the counts depend
    on the (fixed, seeded) outcome stream of the engine. ``slice_steps``: GE only.
    ``gate_level`` is False for the permutation oracle (no gates).
    """

    N: int
    base: int
    oracle: str
    qubits: int
    rounds: int
    gate_level: bool
    oracle_gates: int
    toffoli: int
    cnot: int
    x: int
    measurements: int
    fixups: int
    slice_steps: Optional[int] = None
    per_round_gates: Optional[np.ndarray] = field(default=None, repr=False)
    per_round_toffoli: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def control_gates_max(self) -> int:
        return 4 * self.rounds


def resource_counts(
    N: int,
    oracle: str = "windowed-opt",
    *,
    window: Optional[int] = None,
    exponent_window: Optional[int] = None,
    base: Optional[int] = None,
    per_round: bool = False,
    threads: Optional[int] = None,
) -> ResourceCounts:
    """Qubits and whole-run gate counts of the circuit, without simulating.

    ``base`` defaults to the smallest base coprime to ``N`` (counts depend on
    it only weakly, through the constants of the lookup tables).

    >>> c = resource_counts(1_005_973, "windowed-opt", base=980_062)
    >>> c.qubits, c.oracle_gates, c.toffoli          # research/data/mbu-shor/counts.txt
    (88, 271380, 70720)
    """
    N = _int(N, "N")
    d = _native_shor.resource_counts(
        N,
        oracle,
        window=window,
        exponent_window=exponent_window,
        base=None if base is None else _int(base, "base"),
        per_round=per_round,
        threads=threads,
    )
    return ResourceCounts(
        N=N,
        base=d["base"],
        oracle=d["oracle"],
        qubits=d["qubits"],
        rounds=d["rounds"],
        gate_level=d["gate_level"],
        oracle_gates=d["oracle_gates"],
        toffoli=d["toffoli"],
        cnot=d["cnot"],
        x=d["x"],
        measurements=d["measurements"],
        fixups=d["fixups"],
        slice_steps=d.get("slice_steps"),
        per_round_gates=d.get("per_round_gates"),
        per_round_toffoli=d.get("per_round_toffoli"),
    )


@dataclass(frozen=True)
class OracleCircuit:
    """A controlled-``U_a`` block: ``|c⟩|x⟩|0…0⟩ → |c⟩|a^c·x mod N⟩|0…0⟩`` for ``x < N``.

    ``control`` is qubit 0, ``work[k]`` holds bit ``k`` of ``x``, every other qubit
    is an ancilla that starts and ends in ``|0⟩``.
    """

    circuit: Circuit = field(repr=False)
    N: int
    a: int
    oracle: str
    control: int
    work: List[int]
    num_qubits: int
    gates: int
    toffolis: int


def oracle_circuit(
    N: int, a: int, oracle: str = "windowed-opt", *, window: Optional[int] = None
) -> OracleCircuit:
    """The controlled-multiplication-by-``a`` block of ``oracle`` as a :class:`~qsimlab.Circuit`.

    Supported: ``"windowed-opt"``, ``"windowed"``, ``"ripple"`` (X/CNOT/CCX only)
    and ``"beauregard"`` (QFT arithmetic). The MBU oracles need multi-bit
    classical feed-forward that a ``Circuit`` cannot express
    (``UnsupportedOperationError``); count them with :func:`resource_counts`.

    >>> o = oracle_circuit(15, 7, "ripple")
    >>> o.num_qubits, o.work, o.toffolis > 0
    (16, [1, 2, 3, 4], True)
    """
    N, a = _int(N, "N"), _int(a, "a")
    d = _native_shor.oracle_circuit(N, a, oracle, window=window)
    c = Circuit._wrap(d["circuit"])
    s = c.stats()
    return OracleCircuit(
        circuit=c,
        N=N,
        a=a,
        oracle=oracle,
        control=d["control"],
        work=list(d["work"]),
        num_qubits=d["num_qubits"],
        gates=s["total_gates"],
        toffolis=s["gates_3q"],
    )


def shor_circuit(N: int, a: int, oracle: str = "ripple", *, window: Optional[int] = None) -> Circuit:
    """The whole semiclassical order-finding circuit as a :class:`~qsimlab.Circuit`.

    Qubit 0 is the recycled control, ``1 … n`` the work register (prepared in
    ``|1⟩``). Round ``i``: recycle (``X`` conditioned on measurement ``i−1``),
    ``H``, controlled ``U^(2^(t−1−i))``, one conditioned phase per earlier bit,
    ``H``, measure. Measurement ``i`` is bit ``i`` of the measured integer, so
    ``simulate(c, samples(k))`` samples the same distribution as :func:`factor`.

    >>> c = shor_circuit(15, 7, "ripple")
    >>> c.num_qubits, c.num_measurements
    (16, 8)
    """
    N, a = _int(N, "N"), _int(a, "a")
    return Circuit._wrap(_native_shor.shor_circuit(N, a, oracle, window=window))


def exact_distribution(
    N: int,
    a: int,
    oracle: str = "permutation",
    *,
    window: Optional[int] = None,
    prune: float = 0.0,
    threads: Optional[int] = None,
) -> np.ndarray:
    """Exact probabilities of the ``2n``-bit measured integer (``float64[2**(2n)]``),
    by walking the whole measurement tree of the semiclassical circuit (``n ≤ 8`` bits;
    10 for ``"permutation"``, 6 for ``"beauregard"``).

    >>> p = exact_distribution(15, 7)
    >>> [int(y) for y in np.flatnonzero(p > 1e-12)], round(float(p[64]), 6)
    ([0, 64, 128, 192], 0.25)
    """
    N, a = _int(N, "N"), _int(a, "a")
    return _native_shor.exact_distribution(
        N, a, oracle, window=window, prune=float(prune), threads=threads
    )


@dataclass(frozen=True)
class SupportPrediction:
    """The support law for ``(N, a)``: ``order`` ``r``, ``nu`` = ν₂(r), ``rounds`` ``t = 2n``,
    ``bounds`` (``B_i``), ``peak`` (``max_i B_i``), ``sum`` (``Σ B_i``; the sliced
    engine's work is ``≈ 2 Ḡ Σ B_i`` gate·branch steps for ``Ḡ`` gates per round),
    and the engine and peak bytes :func:`factor` would predict."""

    order: int
    nu: int
    rounds: int
    peak: int
    sum: int
    engine: str
    predicted_bytes: int
    bounds: np.ndarray = field(repr=False)


def predict_support(
    N: int,
    a: int,
    oracle: str = "windowed-opt",
    *,
    window: Optional[int] = None,
    exponent_window: Optional[int] = None,
    precision: str = "f64",
) -> SupportPrediction:
    """Predict the support trace, peak memory and work of one run, without running it.

    >>> p = predict_support(1_005_973, 980_062)
    >>> p.order, p.nu, p.peak, p.engine
    (41832, 3, 20916, 'sliced')
    """
    d = _native_shor.predict_support(
        _int(N, "N"),
        _int(a, "a"),
        oracle,
        window=window,
        exponent_window=exponent_window,
        f32=_precision(precision),
    )
    return SupportPrediction(
        order=d["order"],
        nu=d["nu"],
        rounds=d["rounds"],
        peak=d["peak"],
        sum=int(d["sum"]),
        engine=d["engine"],
        predicted_bytes=int(d["predicted_bytes"]),
        bounds=d["bounds"],
    )


def multiplicative_order(a: int, N: int) -> int:
    """The multiplicative order of ``a`` modulo ``N`` (classical; Pollard–Brent + Carmichael).

    >>> multiplicative_order(7, 15), multiplicative_order(980_062, 1_005_973)
    (4, 41832)
    """
    r, _, _ = _native_shor.number_theory(_int(N, "N"), _int(a, "a"))
    return r


def carmichael(N: int) -> int:
    """Carmichael's function ``λ(N)`` (the largest order of any unit mod ``N``).

    >>> carmichael(1_537_596_787)
    256252500
    """
    return _native_shor.number_theory(_int(N, "N"), None)[1]


# --------------------------------------------------------------------------- noise


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


@dataclass(frozen=True)
class NoisySuccess(Result):
    """Result of :func:`noisy_success` (a :class:`qsimlab.sim.Result`).

    * ``success``: fraction of trajectories whose post-processing returned a
      factor, with ``ci`` its Wilson 95 % interval;
    * ``order_rate``: the true order was recovered; ``peak_rate``: the measured
      ``y`` is a "good" outcome, ``|y/2^t − s/r| < 1/(2r²)`` (the metric of
      ``research/shor/shor-noise.md``);
    * ``locations``: fault locations ``L`` of the circuit, ``mean_faults``;
    * ``capped``: trajectories abandoned because their support exceeded ``cap``
      (counted as failures);
    * per-trajectory arrays ``faults``, ``factor_ok``, ``order_ok``, ``peak_ok``.
    """

    N: int = 0
    base: int = 0
    p: float = 0.0
    noise: str = ""
    trajectories: int = 0
    locations: int = 0
    order: int = 0
    success: float = 0.0
    ci: Tuple[float, float] = (0.0, 1.0)
    order_rate: float = 0.0
    peak_rate: float = 0.0
    capped: int = 0
    mean_faults: float = 0.0
    faults: Optional[np.ndarray] = field(default=None, repr=False)
    factor_ok: Optional[np.ndarray] = field(default=None, repr=False)
    order_ok: Optional[np.ndarray] = field(default=None, repr=False)
    peak_ok: Optional[np.ndarray] = field(default=None, repr=False)
    measured: Optional[List[Optional[int]]] = field(default=None, repr=False)


def noisy_success(
    N: int,
    a: int,
    p: float,
    *,
    noise: str = "depolarizing",
    trajectories: int = 200,
    oracle: str = "windowed",
    window: Optional[int] = None,
    faults: Optional[int] = None,
    seed: Optional[int] = None,
    reset_ancillas: bool = False,
    cap: int = 1 << 26,
    threads: Optional[int] = None,
) -> NoisySuccess:
    """Success probability of the gate-level circuit under circuit-level Pauli noise.

    Exact Monte-Carlo trajectories (``research/shor/shor-noise.md``): a Pauli fault
    at rate ``p`` after every oracle gate on each of its qubits, and on the
    control (preparation, after each H and the phase correction, readout
    flip). ``noise``: ``"depolarizing"``, ``"bitflip"`` or ``"phaseflip"``.
    ``faults=k`` instead conditions every trajectory on exactly ``k`` faults at
    uniformly random locations (stratified estimates). ``reset_ancillas``: an
    ideal reset of every ancilla after each round. Oracles: ``"windowed"``,
    ``"windowed-opt"``, ``"ripple"`` (≤ 129 qubits). Trajectory ``i`` uses a
    stream derived from ``(seed, i)``: results do not depend on ``threads``.

    >>> r = noisy_success(143, 2, 0.0, trajectories=20, seed=1)
    >>> r.success, r.peak_rate, r.mean_faults
    (1.0, 1.0, 0.0)
    """
    N, a = _int(N, "N"), _int(a, "a")
    trajectories = _int(trajectories, "trajectories")
    if trajectories < 1:
        raise ValueError("trajectories must be ≥ 1")
    seed = _seed(seed)
    t0 = time.perf_counter()
    d = _native_shor.noisy_trajectories(
        N,
        a,
        float(p),
        noise=noise,
        trajectories=trajectories,
        kind=oracle,
        window=window,
        seed=seed,
        faults=None if faults is None else _int(faults, "faults"),
        cap=int(cap),
        reset_ancillas=bool(reset_ancillas),
        threads=threads,
    )
    fok = d["factor_ok"]
    k = int(fok.sum())
    return NoisySuccess(
        engine="sliced-noisy",
        components=[(d["qubits"], d["locations"], "sliced-noisy")],
        seed=seed,
        precision="f64",
        wall_time=time.perf_counter() - t0,
        N=N,
        base=a,
        p=float(p),
        noise=noise,
        trajectories=trajectories,
        locations=d["locations"],
        order=d["order"],
        success=k / trajectories,
        ci=_wilson(k, trajectories),
        order_rate=float(d["order_ok"].mean()),
        peak_rate=float(d["peak_ok"].mean()),
        capped=int(d["capped"].sum()),
        mean_faults=float(d["faults"].mean()),
        faults=d["faults"],
        factor_ok=fok,
        order_ok=d["order_ok"],
        peak_ok=d["peak_ok"],
        measured=[None if y is None else int(y) for y in d["measured"]],
    )
