//! Native half of `qsimlab.analysis` (phase 2). Owned by the analysis module agent.
//!
//! Add `#[pyfunction]`s / `#[pyclass]`es here and register them below;
//! they appear as `qsimlab._native.analysis.<name>` and are wrapped by
//! `python/qsimlab/analysis.py`. Use `crate::circuit::PyCircuit` for circuit
//! arguments, `crate::threads::heavy` for anything slow and
//! `crate::errors::map_sim_err` for engine errors.

use pyo3::prelude::*;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "native part of qsimlab.analysis (phase 2)")?;
    Ok(())
}
