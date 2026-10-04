"""Exception hierarchy of qsimlab (see ``python/API.md`` §4).

Every exception derives from :class:`QsimError` and from the closest
built-in exception, so ``except ValueError`` keeps working::

    >>> import qsimlab
    >>> c = qsimlab.Circuit(2)
    >>> try:
    ...     c.h(5)
    ... except IndexError as e:
    ...     print(type(e).__name__)
    QubitIndexError
"""

from __future__ import annotations

__all__ = [
    "QsimError",
    "CircuitError",
    "QubitIndexError",
    "UnsupportedOperationError",
    "ResourceLimitError",
    "EngineAbortedError",
    "ParseError",
    "MissingDependencyError",
]


class QsimError(Exception):
    """Base class of every qsimlab error."""


class CircuitError(QsimError, ValueError):
    """Invalid circuit construction: bad gate name or arity, repeated qubit, bad probability."""


class QubitIndexError(QsimError, IndexError):
    """A qubit, classical bit or basis-state index is out of range."""


class UnsupportedOperationError(QsimError, ValueError):
    """The request, engine or export format cannot handle an operation of the circuit."""


class ResourceLimitError(QsimError, MemoryError):
    """A register or term count would exceed the budget.

    Attributes ``needed`` and ``limit`` hold the requested and allowed amounts
    (bytes for memory, count for Pauli terms) when known.
    """

    needed: int | None = None
    limit: int | None = None


class EngineAbortedError(ResourceLimitError):
    """A forced engine gave up (e.g. the exact MPS would have to truncate)."""


class ParseError(QsimError, ValueError):
    """OpenQASM or Stim source could not be parsed (the message names the line)."""


class MissingDependencyError(QsimError, ImportError):
    """An optional dependency (qiskit, cirq, stim) is needed but not installed."""
