"""Quantum error correction: codes, detector sampling, decoders, logical error rates.

**Provisional** (phase 2, see python/API.md §1 and §8): this module is
owned by the qec module and may change in any release.

Native functions live in qsimlab._native.qec (python/src/qec.rs);
wrap them here with typed, documented Python functions and add them to
__all__.
"""

from __future__ import annotations

from ._native import qec as _native_qec  # noqa: F401

__all__: list = []
