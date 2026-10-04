//! Native half of `qsimlab.qec` (phase 2). Owned by the qec module agent.
//!
//! Add `#[pyfunction]`s / `#[pyclass]`es here and register them below;
//! they appear as `qsimlab._native.qec.<name>` and are wrapped by
//! `python/qsimlab/qec.py`. Use `crate::circuit::PyCircuit` for circuit
//! arguments, `crate::threads::heavy` for anything slow and
//! `crate::errors::map_sim_err` for engine errors.

use pyo3::prelude::*;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "native part of qsimlab.qec (phase 2)")?;
    Ok(())
}
