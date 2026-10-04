"""Analysis helpers: observables, fidelities, entropies, distributions, simulability.

**Provisional** (phase 2, see python/API.md §1 and §8): this module is
owned by the analysis module and may change in any release.

Native functions live in qsimlab._native.analysis (python/src/analysis.rs);
wrap them here with typed, documented Python functions and add them to
__all__.
"""

from __future__ import annotations

from ._native import analysis as _native_analysis  # noqa: F401

__all__: list = []
