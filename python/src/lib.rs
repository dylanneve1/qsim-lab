//! PyO3 bindings for qsim-lab: the `qsimlab._native` extension module.
//!
//! Layout (see python/API.md §8):
//! * [`circuit`]: `CircuitCore`, the engine circuit plus metadata;
//! * [`sim`]: `run` / `plan`, the simulation entry points;
//! * [`convert`], [`errors`], [`threads`]: shared helpers for every module;
//! * [`qec`], [`shor`], [`analysis`]: phase-2 domain modules (each owns its
//!   own file and registers the native submodule `qsimlab._native.<name>`).

use pyo3::prelude::*;

pub mod analysis;
pub mod circuit;
pub mod convert;
pub mod errors;
pub mod qec;
pub mod shor;
pub mod sim;
pub mod threads;

/// Adds an empty native submodule `parent.<name>` and returns it.
pub fn add_submodule<'py>(
    parent: &Bound<'py, PyModule>,
    name: &str,
) -> PyResult<Bound<'py, PyModule>> {
    let sub = PyModule::new(parent.py(), name)?;
    parent.add_submodule(&sub)?;
    Ok(sub)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    threads::register(m)?;
    circuit::register(m)?;
    sim::register(m)?;
    qec::register(&add_submodule(m, "qec")?)?;
    shor::register(&add_submodule(m, "shor")?)?;
    analysis::register(&add_submodule(m, "analysis")?)?;
    Ok(())
}
