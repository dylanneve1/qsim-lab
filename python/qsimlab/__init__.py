"""qsimlab: exact quantum circuit simulation with a cost-model planner.

>>> import qsimlab as qs
>>> bell = qs.Circuit(2).h(0).cx(0, 1)
>>> qs.simulate(bell, qs.statevector()).state.round(3)
array([0.707+0.j, 0.   +0.j, 0.   +0.j, 0.707+0.j])
>>> r = qs.simulate(bell.measure_all(), qs.samples(1000), seed=1)
>>> sorted(r.counts())
['00', '11']

See ``python/API.md`` for the full contract (conventions, errors, threading).
"""

from __future__ import annotations

from . import _native
from ._native import get_num_threads, set_num_threads
from .circuit import GATES, Circuit, Instruction
from .errors import (
    CircuitError,
    EngineAbortedError,
    MissingDependencyError,
    ParseError,
    QsimError,
    QubitIndexError,
    ResourceLimitError,
    UnsupportedOperationError,
)
from .sim import (
    ENGINES,
    AmplitudesResult,
    Budget,
    ExpectationResult,
    Explanation,
    NoiseModel,
    SamplesResult,
    StatevectorResult,
    amplitudes,
    expectation,
    plan,
    samples,
    simulate,
    statevector,
)
from . import analysis, circuit, errors, interop, qec, shor, sim  # noqa: E402

__version__: str = _native.__version__

__all__ = [
    "__version__",
    "Circuit",
    "Instruction",
    "GATES",
    "simulate",
    "plan",
    "statevector",
    "amplitudes",
    "samples",
    "expectation",
    "Budget",
    "NoiseModel",
    "ENGINES",
    "StatevectorResult",
    "AmplitudesResult",
    "SamplesResult",
    "ExpectationResult",
    "Explanation",
    "set_num_threads",
    "get_num_threads",
    "QsimError",
    "CircuitError",
    "QubitIndexError",
    "UnsupportedOperationError",
    "ResourceLimitError",
    "EngineAbortedError",
    "ParseError",
    "MissingDependencyError",
    "circuit",
    "sim",
    "interop",
    "errors",
    "qec",
    "shor",
    "analysis",
]
