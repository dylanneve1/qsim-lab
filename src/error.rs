//! One error type for the whole crate.
//!
//! Each subsystem keeps its own precise error ([`SimError`] for running
//! circuits, [`DagError`] for DAG rewrites, [`DemError`] for detector error
//! models, [`StimError`] for `.stim` I/O, [`GroupError`] for finite groups,
//! `MetalError` for the GPU backend).
//! [`Error`] wraps all of them, with `From` conversions, so code that mixes
//! subsystems can use `?` throughout and return [`Result`]:
//!
//! ```
//! use qsim_lab::{io::stim, Circuit, NoiseModel, Result};
//!
//! fn roundtrip(c: &Circuit) -> Result<Circuit> {
//!     let text = stim::to_stim(c, &NoiseModel::default(), &[], &[])?; // StimError -> Error
//!     Ok(stim::parse_stim(&text)?.circuit)
//! }
//! # let mut c = Circuit::new(2);
//! # c.h(0).cnot(0, 1).measure_all();
//! # roundtrip(&c).unwrap();
//! ```

use crate::circuit::SimError;
use crate::dag::DagError;
use crate::io::stim::StimError;
use crate::qec::dem::DemError;
use crate::qec::group_algebra::GroupError;
use std::fmt;

/// Any error produced by `qsim_lab`. See the [module docs](self).
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Running or validating a circuit failed.
    Sim(SimError),
    /// Building or rewriting a DAG failed.
    Dag(DagError),
    /// Building a detector error model failed.
    Dem(DemError),
    /// Reading or writing the `.stim` format failed.
    Stim(StimError),
    /// Building or parsing a finite group failed.
    Group(GroupError),
    /// The Metal (GPU) backend failed.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(crate::engines::metal_sv::MetalError),
    /// Parsing a textual specification (family spec, schedule, ...) failed.
    Parse(String),
}

/// `Result` with [`Error`] as the default error type.
pub type Result<T, E = Error> = std::result::Result<T, E>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Sim(e) => e.fmt(f),
            Error::Dag(e) => e.fmt(f),
            Error::Dem(e) => e.fmt(f),
            Error::Stim(e) => e.fmt(f),
            Error::Group(e) => e.fmt(f),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Error::Metal(e) => e.fmt(f),
            Error::Parse(s) => write!(f, "parse error: {s}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Sim(e) => Some(e),
            Error::Dag(e) => Some(e),
            Error::Dem(e) => Some(e),
            Error::Stim(e) => Some(e),
            Error::Group(e) => Some(e),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Error::Metal(e) => Some(e),
            Error::Parse(_) => None,
        }
    }
}

macro_rules! from_error {
    ($($t:ty => $v:ident),* $(,)?) => {$(
        impl From<$t> for Error {
            fn from(e: $t) -> Self {
                Error::$v(e)
            }
        }
    )*};
}

from_error!(SimError => Sim, DagError => Dag, DemError => Dem, StimError => Stim, GroupError => Group);

#[cfg(all(feature = "metal", target_os = "macos"))]
from_error!(crate::engines::metal_sv::MetalError => Metal);

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Parse(s)
    }
}
