"""Shor's algorithm tooling: modular-exponentiation circuits, period finding, resource counts.

**Provisional** (phase 2, see python/API.md §1 and §8): this module is
owned by the shor module and may change in any release.

Native functions live in qsimlab._native.shor (python/src/shor.rs);
wrap them here with typed, documented Python functions and add them to
__all__.
"""

from __future__ import annotations

from ._native import shor as _native_shor  # noqa: F401

__all__: list = []
