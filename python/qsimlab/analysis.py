"""Structural analysis of circuits: magic, stabilizer rank, simulability, monitored dynamics.

**Provisional** (phase 2, see ``python/API.md`` §1 and §8).

These functions measure *why* a circuit is hard or easy to simulate, with the
exact invariants the engines use, mostly without simulating it::

    >>> import qsimlab as qs, qsimlab.analysis as an
    >>> c = qs.Circuit(3).h(0).t(0).cx(0, 1).h(2)
    >>> p = an.magic_profile(c)
    >>> p.t_count, p.d, p.f
    (1, 1, 1)
    >>> an.branching_rank(c).rank
    2
    >>> round(an.state_magic(c).m2, 6)          # log2(4/3): one T state's worth
    0.415037
    >>> an.branching_rank(c.copy().t(1)).rank   # T on the partner makes it a stabilizer state again
    1

* :func:`magic_profile`: the magic atlas (``research/simulability/magic-atlas.md``): active
  dimension ``d_k`` of the rotation frame (the exact register size of the
  compressed-state engine), factored dimension ``f_k``, stabilizer entanglement
  of the Clifford skeleton across a cut and the bound ``E + d`` on the true
  entanglement, and an affine bound on the support size.
* :func:`state_magic`: stabilizer nullity and stabilizer 2-Rényi entropy
  ``M2`` of a state (all ``4^n`` Pauli expectations; ``n ≤ 13``).
* :func:`branching_rank`: the number of stabilizer terms of the exact
  low-rank simulator (``research/theory/theory-rank.md``) after every gate, an upper
  bound on the stabilizer rank.
* :func:`simulability`: the planner's features (per-engine log2 work
  estimates) and its explanation (ranked predicted costs per engine).
* :func:`gaussian`: the free-fermion (matchgate) detector: is the circuit a
  fermionic Gaussian circuit under some Jordan–Wigner order, after undoing
  SWAP networks, and how far from it (``docs/ENGINE_GAUSSIAN.md``);
  :func:`gaussian_expectations` runs the Gaussian engine.
* :func:`monitored`: exact simulation of Clifford+T circuits with mid-circuit
  measurements in the rotation frame: ``d(t)``, Born probabilities, and cut
  entropies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._native import analysis as _native_analysis
from .circuit import Circuit
from .sim import Budget, Explanation, Request, Result, _check_circuit, plan, samples, simulate
from .sim import statevector as _sv_request

__all__ = [
    "magic_profile",
    "MagicProfile",
    "state_magic",
    "StateMagic",
    "stabilizer_nullity",
    "stabilizer_renyi_entropy",
    "branching_rank",
    "BranchingRank",
    "simulability",
    "Simulability",
    "monitored",
    "MonitoredResult",
    "gaussian",
    "GaussianReport",
    "gaussian_expectations",
    "GaussianExpectations",
]


# --------------------------------------------------------------------------- magic atlas


@dataclass(frozen=True)
class MagicProfile:
    """Result of :func:`magic_profile` (the magic atlas of a unitary circuit on ``|0^n⟩``).

    * ``t_count``: rotations by odd multiples of π/4; ``rotations``: every
      non-Clifford rotation after lowering to Clifford + Z rotations (Toffoli
      = 7 T) and merging half-π multiples into S;
    * ``d``: final active dimension (dimension of the x-span of the rotation
      axes in the Heisenberg frame); ``2^d`` is the compressed-state register;
      ``d_profile[j]`` is ``d`` after rotation ``j`` (``rotation_gate[j]`` is its
      original gate index);
    * ``f``: largest factored component; ``f_profile[j]`` the component size
      touched by rotation ``j``;
    * ``log2_work`` = ``log2 Σ_j 2^{d_j}`` (amplitude updates of the compressed
      state; −1 for Clifford circuits), ``log2_work_factored`` the same for ``f``;
    * ``e_stab_max``: max stabilizer entanglement (bits) of the Clifford
      skeleton across ``cut``; ``e_bound_max``: max of ``min(E + d, cut, n − cut)``,
      an upper bound on the entanglement of the true state;
    * ``support_log2``: affine upper bound on ``log2 |supp U|0^n⟩|``;
    * ``checkpoints``: ``(gate, rotations, t_count, d, f, e_stab)`` tuples at
      evenly spaced gates.
    """

    num_qubits: int
    gates: int
    lowered_gates: int
    two_qubit_gates: int
    toffolis: int
    rotations: int
    t_count: int
    d: int
    f: int
    log2_work: float
    log2_work_factored: float
    cut: int
    e_stab_max: int
    e_bound_max: int
    support_log2: Optional[int]
    seconds: float
    d_profile: np.ndarray = field(repr=False)
    f_profile: np.ndarray = field(repr=False)
    rotation_gate: np.ndarray = field(repr=False)
    checkpoints: List[Tuple[int, int, int, int, int, int]] = field(repr=False)


def magic_profile(
    circuit: Circuit,
    *,
    checkpoints: int = 64,
    cut: Optional[int] = None,
    entanglement: bool = True,
    support: bool = True,
    threads: Optional[int] = None,
) -> MagicProfile:
    """The magic atlas of ``circuit`` in one ``O(gates · n)`` pass (no simulation).

    Terminal measurements are ignored; mid-circuit measurements, resets and
    noise raise :class:`~qsimlab.UnsupportedOperationError` (use :func:`monitored`).
    ``cut``: the skeleton entanglement is across qubits ``[0, cut)`` vs the rest
    (default ``n // 2``).

    >>> import qsimlab as qs
    >>> qft = qs.Circuit(4)
    >>> for j in range(4):
    ...     _ = qft.h(j)
    ...     for k in range(j + 1, 4):
    ...         _ = qft.cp(k, j, 3.141592653589793 / 2 ** (k - j))
    >>> p = magic_profile(qft)
    >>> p.rotations, p.t_count, p.d, p.f   # each controlled phase lowers to 3 Z rotations
    (18, 9, 3, 1)
    """
    circuit = _check_circuit(circuit)
    d = _native_analysis.magic_profile(
        circuit._core,
        checkpoints=int(checkpoints),
        entanglement=bool(entanglement),
        cut=cut,
        support=bool(support),
        threads=threads,
    )
    return MagicProfile(
        **{k: v for k, v in d.items() if k != "checkpoints"},
        checkpoints=[tuple(x) for x in d["checkpoints"]],
    )


@dataclass(frozen=True)
class StateMagic:
    """``nullity``: stabilizer nullity ``ν = n − log2 |{P : |⟨P⟩| = 1}|`` (Beverland et al.
    2020; 0 iff stabilizer state). ``m2``: stabilizer 2-Rényi entropy
    ``M2 = −log2(Σ_P ⟨P⟩⁴ / 2^n)`` (Leone, Oliviero, Hamma 2022), in bits."""

    nullity: float
    m2: float


def _dense_state(x: Union[Circuit, np.ndarray, Sequence[complex]]) -> np.ndarray:
    if isinstance(x, Circuit):
        if x.num_qubits > 13:
            raise ValueError(f"stabilizer magic needs n ≤ 13 qubits, got {x.num_qubits}")
        return simulate(x, _sv_request()).state.astype(np.complex128)
    v = np.ascontiguousarray(np.asarray(x, dtype=np.complex128).ravel())
    return v


def state_magic(x: Union[Circuit, np.ndarray], *, threads: Optional[int] = None) -> StateMagic:
    """Stabilizer nullity and stabilizer Rényi entropy of a state (``n ≤ 13``).

    ``x`` is a :class:`~qsimlab.Circuit` (its unitary part is simulated from
    ``|0^n⟩``) or a normalised state vector of length ``2^n``. Cost ``O(4^n n)``.

    >>> import numpy as np
    >>> t_state = np.array([1, np.exp(1j * np.pi / 4)]) / np.sqrt(2)
    >>> m = state_magic(t_state)
    >>> m.nullity, round(m.m2, 6)        # M2(|T>) = log2(4/3)
    (1.0, 0.415037)
    """
    v = _dense_state(x)
    nu, m2 = _native_analysis.state_magic(v, threads=threads)
    return StateMagic(nullity=float(nu), m2=float(m2))


def stabilizer_nullity(x: Union[Circuit, np.ndarray]) -> float:
    """:attr:`StateMagic.nullity` of :func:`state_magic`."""
    return state_magic(x).nullity


def stabilizer_renyi_entropy(x: Union[Circuit, np.ndarray]) -> float:
    """:attr:`StateMagic.m2` of :func:`state_magic`."""
    return state_magic(x).m2


# --------------------------------------------------------------------------- branching rank


@dataclass(frozen=True)
class BranchingRank:
    """Result of :func:`branching_rank`.

    ``trace[k]`` is the number of stabilizer terms after gate ``k`` (gates of the
    unitary part, in order), ``rank`` the final value, ``max_rank`` its maximum;
    ``overflow`` is True if ``max_terms`` stopped the run. Events: per term and
    non-Clifford gate, ``branch_events`` (the term split in two),
    ``clifford_events`` (updated in place by a Clifford, Theorem R3),
    ``diagonal_events``; ``merges``/``pair_merges``/``cancellations``: terms
    combined. ``state``: the dense state ``Σ_j c_j|φ_j⟩`` (``state=True``, n ≤ 20).
    """

    rank: int
    max_rank: int
    overflow: bool
    branch_events: int
    clifford_events: int
    diagonal_events: int
    merges: int
    pair_merges: int
    cancellations: int
    trace: np.ndarray = field(repr=False)
    state: Optional[np.ndarray] = field(default=None, repr=False)


def branching_rank(
    circuit: Circuit,
    *,
    max_terms: int = 1 << 16,
    pair_merge: int = 6,
    state: bool = False,
    threads: Optional[int] = None,
) -> BranchingRank:
    """Run the exact branching-rank (low stabilizer rank) simulator on ``circuit``.

    Every non-Clifford gate is a projector gate; a term branches only when the
    gate cannot be absorbed as a Clifford on it, and equal rays are merged, so
    the term count is an upper bound on the stabilizer rank (exact for many
    structured circuits). ``pair_merge``: merge pairs of terms differing in ``≤ s``
    stabilizer generators (0 disables it).

    >>> import qsimlab as qs
    >>> c = qs.Circuit(3).h(0).h(1).h(2).t(0).t(1).t(2)
    >>> branching_rank(c).trace.tolist()
    [1, 1, 1, 2, 4, 8]
    >>> toff = qs.Circuit(3).h(0).h(1).ccx(0, 1, 2)
    >>> branching_rank(toff).rank        # Toffoli on |++0>: two stabilizer terms
    2
    """
    circuit = _check_circuit(circuit)
    d = _native_analysis.branching_rank(
        circuit._core,
        max_terms=int(max_terms),
        pair_merge=int(pair_merge),
        state=bool(state),
        threads=threads,
    )
    return BranchingRank(**d)


# --------------------------------------------------------------------------- simulability

#: feature → engine name (``qsimlab.ENGINES`` spelling) for the log2 work estimates.
_COST_FEATURES: Dict[str, str] = {
    "sv_l": "statevector",
    "sparse_l": "sparse",
    "mps_l": "mps",
    "hsf_l": "hsf",
    "dense_l": "compressed",
    "frame_l": "pauli-frame",
}


@dataclass(frozen=True)
class Simulability:
    """Result of :func:`simulability`.

    * ``features``: the raw feature dict (``research/simulability/simulability.md``):
      ``n, gates, g2, g3, depth2, t_count, rotations, d`` (active dimension),
      ``chi_bits`` (bound on log2 of the MPS bond), ``hsf_k`` (HSF cut bits),
      ``sup`` (affine bound on log2 of the support), and one ``*_l`` log2 work
      estimate per engine;
    * ``log2_costs``: ``{engine: log2 work}`` from those estimates, cheapest first;
    * ``explanation``: the planner's :class:`~qsimlab.sim.Explanation` for
      ``request`` (ranked predicted seconds per engine, from fitted cost models).
    """

    features: Dict[str, Any]
    log2_costs: List[Tuple[str, float]]
    explanation: Explanation

    def __str__(self) -> str:
        lines = ["log2 work estimates (features):"]
        lines += [f"  {e:<12} {c:7.2f}" for e, c in self.log2_costs]
        return "\n".join(lines) + "\n" + str(self.explanation)


def simulability(
    circuit: Circuit,
    request: Optional[Request] = None,
    *,
    hsf: bool = True,
    budget: Union[Budget, int, str, None] = None,
    threads: Optional[int] = None,
) -> Simulability:
    """Simulability features of ``circuit`` plus the planner's explanation for ``request``.

    The features are computed on the unitary part (terminal measurements
    dropped; other non-unitary ops raise ``UnsupportedOperationError``); ``request`` defaults to ``samples(1024)``. ``hsf=False`` skips the
    Kernighan–Lin partition, the slowest feature on large circuits.

    >>> import qsimlab as qs
    >>> c = qs.Circuit(40)
    >>> for q in range(39):
    ...     _ = c.h(q).cx(q, q + 1)
    >>> s = simulability(c)
    >>> s.features["d"], s.features["t_count"], s.explanation.engine
    (0, 0, 'tableau')
    """
    circuit = _check_circuit(circuit)
    f = _native_analysis.features(circuit._core, hsf=bool(hsf), threads=threads)
    costs = sorted(
        ((name, float(f[k])) for k, name in _COST_FEATURES.items() if k in f),
        key=lambda x: x[1],
    )
    ex = plan(circuit, request if request is not None else samples(1024), budget=budget)
    return Simulability(features=dict(f), log2_costs=costs, explanation=ex)


# --------------------------------------------------------------------------- free fermions


@dataclass(frozen=True)
class GaussianReport:
    """Result of :func:`gaussian` (``docs/ENGINE_GAUSSIAN.md`` §2).

    * ``exact``: every fused block is Gaussian in the chosen Jordan–Wigner
      order (to ``tol``): the Gaussian engine applies; ``free``: Gaussian up
      to diagonal interaction phases;
    * ``gaussian_fraction``: Gaussian blocks / all blocks; ``max_residual``:
      largest distance of a non-interaction block from the Gaussian set (1
      for a matchgate on non-adjacent modes or a three-qubit gate);
    * ``interactions``: ``(block, (wire_a, wire_b), (mode_a, mode_b), g)`` for
      every diagonal block ``exp(i g n_a n_b)`` with ``g != 0``;
      ``interaction_total`` = ``Σ|g|``, ``interaction_max``;
    * ``ordering`` (``"identity"``, ``"paths"``, ``"greedy_cover"``),
      ``order[k]`` = wire (initial qubit) on mode ``k``, ``paths`` (chains of
      wires), ``mode_of_qubit[q]`` = mode held by qubit ``q`` at the end;
    * ``swaps_relabelled``: SWAP gates and SWAP-equivalent blocks turned
      into wire renamings; ``number_conserving``: every Gaussian block
      conserves the particle number.
    """

    num_qubits: int
    exact: bool
    free: bool
    blocks: int
    blocks_2q: int
    gaussian_blocks: int
    gaussian_fraction: float
    max_residual: float
    non_gaussian: int
    nonadjacent: int
    interaction_total: float
    interaction_max: float
    swaps_relabelled: int
    ordering: str
    number_conserving: bool
    seconds: float
    interactions: List[Tuple[int, Tuple[int, int], Tuple[int, int], float]] = field(repr=False)
    order: List[int] = field(repr=False)
    paths: List[List[int]] = field(repr=False)
    mode_of_qubit: List[int] = field(repr=False)


def _report(d: Dict[str, Any]) -> GaussianReport:
    return GaussianReport(**{k: d[k] for k in GaussianReport.__dataclass_fields__})


def gaussian(
    circuit: Circuit,
    *,
    tol: float = 1e-10,
    relabel_swaps: bool = True,
    reorder: bool = True,
    threads: Optional[int] = None,
) -> GaussianReport:
    """Free-fermion detector: fuses the gates into blocks and tests each for
    the matchgate property in a Jordan–Wigner order it searches for.

    Takes the unitary part (terminal measurements dropped). ``relabel_swaps``
    treats SWAP gates as wire renamings; ``reorder`` allows an order other
    than the qubit order.

    >>> import qsimlab as qs
    >>> c = qs.Circuit(2).x(0).h(0).h(1).cx(0, 1).rz(1, 0.3).cx(0, 1).h(0).h(1)
    >>> r = gaussian(c)
    >>> r.exact, r.gaussian_fraction, r.blocks
    (True, 1.0, 1)
    >>> gaussian(c.copy().h(0)).exact
    False
    """
    circuit = _check_circuit(circuit)
    d = _native_analysis.gaussian(
        circuit._core, tol=float(tol), relabel_swaps=bool(relabel_swaps),
        reorder=bool(reorder), threads=threads,
    )
    return _report(d)


@dataclass(frozen=True)
class GaussianExpectations:
    """Result of :func:`gaussian_expectations`: ``z[q]`` = ``⟨Z_q⟩``,
    ``zz[i]`` = ``⟨Z_a Z_b⟩`` for ``pairs[i] = (a, b)``, and the detector's
    :class:`GaussianReport`."""

    z: np.ndarray = field(repr=False)
    zz: np.ndarray = field(repr=False)
    report: GaussianReport


def gaussian_expectations(
    circuit: Circuit,
    pairs: Sequence[Tuple[int, int]] = (),
    *,
    drop_interactions: bool = False,
    tol: float = 1e-10,
    threads: Optional[int] = None,
) -> GaussianExpectations:
    """``⟨Z_q⟩`` of every qubit and ``⟨Z_a Z_b⟩`` (Wick's theorem) on the
    Gaussian engine, in ``O(gates · n)`` plus ``O(1)`` per pair.

    The circuit must be exactly Gaussian (``UnsupportedOperationError``
    otherwise). ``drop_interactions=True`` instead sets every diagonal
    interaction phase ``exp(i g n_a n_b)`` to zero, which gives the free-fermion
    part of the circuit. That is an approximation unless ``report.exact``.

    >>> import qsimlab as qs
    >>> c = qs.Circuit(2).x(0).h(0).h(1).cx(0, 1).rz(1, 0.3).cx(0, 1).h(0).h(1)
    >>> e = gaussian_expectations(c, [(0, 1)])
    >>> print(f"{e.z[0]:.6f} {e.z[1]:.6f} {e.zz[0]:.6f}")
    -0.955336 0.955336 -1.000000
    """
    circuit = _check_circuit(circuit)
    d = _native_analysis.gaussian_z(
        circuit._core, pairs=[(int(a), int(b)) for a, b in pairs],
        drop_interactions=bool(drop_interactions), tol=float(tol), threads=threads,
    )
    return GaussianExpectations(z=d["z"], zz=d["zz"], report=_report(d["report"]))


# --------------------------------------------------------------------------- monitored


@dataclass(frozen=True)
class MonitoredResult(Result):
    """Result of :func:`monitored` (a :class:`qsimlab.sim.Result`).

    * ``d``: active dimension after every op (``uint32[len(circuit)]``); the
      state is ``C (|φ⟩ ⊗ |0⟩)`` with ``|φ⟩`` on ``d`` virtual qubits, so ``2^d``
      amplitudes are stored; ``final_d``, ``max_d``;
    * ``outcomes`` / ``qubits`` / ``probabilities``: one entry per measurement
      (program order; resets are not recorded): the outcome, its qubit and the
      Born probability of the observed outcome; ``kinds``: 0 = random in the
      frame (probability 1/2, no amplitude work), 1 = measured on the register
      (``d`` drops by one), 2 = determined;
    * ``entropies``: ``[(op_index, [(lower, upper, s2), ...per cut])]``: bounds
      on every Rényi entropy of each cut (bits) and the exact Rényi-2 entropy
      ``s2`` when the register is tracked and small enough;
    * ``stats``: T gates (activating / in-register), measurement kinds,
      ``element_ops`` (amplitude updates); ``state``: the final state
      (``state=True``), up to a global phase.
    """

    d: Optional[np.ndarray] = field(default=None, repr=False)
    final_d: int = 0
    max_d: int = 0
    outcomes: Optional[np.ndarray] = field(default=None, repr=False)
    qubits: Optional[np.ndarray] = field(default=None, repr=False)
    probabilities: Optional[np.ndarray] = field(default=None, repr=False)
    kinds: Optional[np.ndarray] = field(default=None, repr=False)
    entropies: List[Tuple[int, List[Tuple[float, float, Optional[float]]]]] = field(
        default_factory=list, repr=False
    )
    stats: Dict[str, int] = field(default_factory=dict, repr=False)
    state: Optional[np.ndarray] = field(default=None, repr=False)


def _entropy(e: Sequence[Any]) -> Tuple[float, float, Optional[float]]:
    lo, hi, s2 = e
    if s2 is not None and abs(s2) < 1e-12:
        s2 = 0.0  # rounding of an exactly zero entropy
    return (float(lo), float(hi), s2)


def monitored(
    circuit: Circuit,
    *,
    seed: Optional[int] = None,
    exact: bool = True,
    max_d: int = 24,
    cuts: Optional[Sequence[Sequence[int]]] = None,
    entropy_every: int = 0,
    max_cost_log2: int = 24,
    state: bool = False,
    threads: Optional[int] = None,
) -> MonitoredResult:
    """Simulate a Clifford+T circuit with mid-circuit measurements in the rotation frame.

    Clifford gates update a tableau; a ``T``/``Rz`` gate either activates one
    virtual qubit (``d → d + 1``) or rotates the active register in place; a
    ``Z`` measurement is random in the frame (1/2), determined, or Born-sampled on
    the register (``d → d − 1``). Any gate is accepted (lowered to Clifford +
    Z rotations); measurements, resets, ``c_if`` and Pauli noise channels are
    simulated per shot. One call is one trajectory.

    * ``exact=False`` tracks only the tableau: ``d(t)`` is exact for every
      outcome sequence (it does not depend on the outcomes), register outcomes
      are drawn 50/50 and only entropy bounds are reported; works for any ``n``.
    * ``max_d``: refuse (:class:`~qsimlab.ResourceLimitError`) beyond ``2^max_d``
      amplitudes (≤ 34).
    * ``cuts``: regions (lists of qubits) whose entropy is reported, at the end
      and every ``entropy_every`` ops (default: the first ``n // 2`` qubits).

    >>> import qsimlab as qs
    >>> c = qs.Circuit(2).h(0).t(0).cx(0, 1).measure(1).h(0).t(0)
    >>> r = monitored(c, seed=5)
    >>> r.d.tolist(), r.kinds.tolist()
    ([0, 1, 1, 0, 0, 1], [1])
    """
    circuit = _check_circuit(circuit)
    from .shor import _seed  # one seeding convention for the phase-2 modules

    seed = _seed(seed)
    t0 = time.perf_counter()
    d = _native_analysis.monitored(
        circuit._core,
        seed=seed,
        exact=bool(exact),
        max_d=int(max_d),
        cuts=None if cuts is None else [[int(q) for q in r] for r in cuts],
        entropy_every=int(entropy_every),
        max_cost_log2=int(max_cost_log2),
        state=bool(state),
        threads=threads,
    )
    return MonitoredResult(
        engine="monitored" if exact else "monitored-dimension",
        components=[(circuit.num_qubits, len(circuit), "monitored")],
        seed=seed,
        precision="f64",
        wall_time=time.perf_counter() - t0,
        d=d["d"],
        final_d=d["final_d"],
        max_d=d["stats"]["max_d"],
        outcomes=d["outcomes"],
        qubits=d["qubits"],
        probabilities=d["probabilities"],
        kinds=d["kinds"],
        entropies=[(k, [_entropy(e) for e in v]) for k, v in d["entropies"]],
        stats=dict(d["stats"]),
        state=d["state"],
    )
