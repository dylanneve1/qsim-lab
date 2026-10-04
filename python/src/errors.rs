//! Rust error → Python exception mapping.
//!
//! The exception classes live in `qsimlab/errors.py` (they use multiple
//! inheritance, e.g. `ResourceLimitError(QsimError, MemoryError)`, which
//! `create_exception!` cannot express). Rust looks them up lazily by name.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use qsim_lab::SimError;

/// Builds an instance of `qsimlab.errors.<class>` with `msg`, setting the
/// given extra attributes. Falls back to `RuntimeError` if the Python
/// package cannot be imported (only when the extension is used bare).
pub fn qerr(class: &str, msg: impl Into<String>) -> PyErr {
    qerr_with(class, msg, &[])
}

/// [`qerr`] with extra integer attributes (e.g. `needed`, `limit`).
pub fn qerr_with(class: &str, msg: impl Into<String>, attrs: &[(&str, u128)]) -> PyErr {
    let msg = msg.into();
    Python::attach(|py| {
        let build = || -> PyResult<PyErr> {
            let module = py.import("qsimlab.errors")?;
            let ty = module.getattr(class)?;
            let inst = ty.call1((msg.clone(),))?;
            for (k, v) in attrs {
                inst.setattr(*k, *v)?;
            }
            Ok(PyErr::from_value(inst))
        };
        build().unwrap_or_else(|_| PyRuntimeError::new_err(format!("{class}: {msg}")))
    })
}

/// Maps an engine error to the matching Python exception (see python/API.md §4).
pub fn map_sim_err(e: SimError) -> PyErr {
    let msg = e.to_string();
    match e {
        SimError::QubitOutOfRange { .. } | SimError::ClassicalBitOutOfRange { .. } => {
            qerr("QubitIndexError", msg)
        }
        SimError::RepeatedQubit(_) => qerr("CircuitError", msg),
        SimError::Unsupported { .. }
        | SimError::MeasurementNotSupported { .. }
        | SimError::NotSupported { .. } => qerr("UnsupportedOperationError", msg),
        SimError::TooLarge { what, bytes, limit } => {
            if what.contains("aborted") {
                qerr_with(
                    "EngineAbortedError",
                    format!(
                        "the forced engine gave up within the memory budget of {limit} bytes \
                         (e.g. an exact MPS whose bonds outgrow it, or a sparse state that \
                         becomes dense); raise the budget or use engine='auto' [{what}]"
                    ),
                    &[("needed", bytes), ("limit", limit)],
                )
            } else {
                qerr_with(
                    "ResourceLimitError",
                    msg,
                    &[("needed", bytes), ("limit", limit)],
                )
            }
        }
        SimError::TooManyTerms { terms, limit } => qerr_with(
            "ResourceLimitError",
            msg,
            &[("needed", terms as u128), ("limit", limit as u128)],
        ),
        SimError::QasmError(m) => qerr("ParseError", m),
    }
}

/// `CircuitError` (a `ValueError`) for bad arguments.
pub fn circuit_err(msg: impl Into<String>) -> PyErr {
    qerr("CircuitError", msg)
}

/// `UnsupportedOperationError`.
pub fn unsupported(msg: impl Into<String>) -> PyErr {
    qerr("UnsupportedOperationError", msg)
}

/// Plain `ValueError` (argument validation that is not about circuits).
pub fn value_err(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// Maps the engine error and records it in a dict (used by tests).
#[allow(dead_code)]
pub fn describe(py: Python<'_>, e: &SimError) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    d.set_item("message", e.to_string())?;
    Ok(d.unbind())
}
