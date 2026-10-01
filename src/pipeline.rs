//! The single entry point: [`simulate`] takes a circuit, a [`Request`] and a
//! [`Budget`] and sends every independent part of the circuit to the cheapest
//! exact engine.
//!
//! ```text
//! simulate(circuit, Request, Budget)
//!   -> compile (src/compile): peephole, SWAP elimination, state propagation,
//!      light cone of the request, connected components, classical suffix
//!   -> per component, rule-based choice (thresholds below):
//!        Clifford                        -> stabilizer tableau
//!        terminal sampling, few outputs  -> Pauli-path marginals
//!        Clifford+T, small active dim d  -> compressed state ([`crate::adaptive`])
//!        everything else                 -> dense state vector, run through
//!                                           the cache-blocked executor
//!   -> combine (product of components, classical suffix)
//! ```
//!
//! Everything here is exact: the engines are the ones already in the crate and
//! the compile passes keep amplitudes (up to the tracked global phase, which
//! [`Request::Amplitudes`] restores) and outcome distributions unchanged.
//!
//! # Rule thresholds (seed for the learned planner)
//!
//! * **Tableau** when the component is Clifford (any size).
//! * **Pauli paths** (terminal sampling only) when `needed <= 12` measured
//!   qubits, `T <= 22` non-Clifford rotations and
//!   `2^needed · 2^T · ⌈n/64⌉ < 2^n / 4` ([`crate::compile::plan`]).
//! * **Adaptive** (compressed state) when the component is unitary with
//!   `n >= ADAPTIVE_MIN_QUBITS`, active dimension `d <= ADAPTIVE_MAX_ACTIVE`
//!   and `d + ADAPTIVE_MARGIN <= n` (so the dense register is at least
//!   `2^ADAPTIVE_MARGIN` times smaller than a full state vector). Used for
//!   terminal samples and Z-product expectations, never for amplitudes: the
//!   compressed state drops global phases.
//! * **State vector** otherwise, f64, blocked executor, refused above
//!   [`Budget::mem_bytes`].
//!
//! The same numbers are recorded in `research/pipeline.md`.

use crate::circuit::{Circuit, SimError};
use crate::compile::plan::{
    compile_sampling, compile_unitary, expectation_z_product, AdaptiveRule, Backend, CompileStats,
    PlanOptions,
};
use crate::shor::Oracle;
use crate::statevector::{state_bytes, MAX_STATE_BYTES};
use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Smallest component (in qubits) for which the compressed state is tried.
pub const ADAPTIVE_MIN_QUBITS: usize = 14;
/// The dense active register must be at least this many qubits smaller than
/// the component.
pub const ADAPTIVE_MARGIN: usize = 3;
/// Largest active register (qubits) the compressed state may allocate.
pub const ADAPTIVE_MAX_ACTIVE: usize = 26;

/// What the caller wants out of the circuit.
#[derive(Clone, Debug, PartialEq)]
pub enum Request {
    /// `shots` independent measurement records (one bool per measurement
    /// in program order), drawn with a seeded RNG.
    Samples { shots: usize, seed: u64 },
    /// `<x|ψ>` for the given basis states `x` (bit `q` = qubit `q`) of a
    /// unitary circuit, including the global phase.
    Amplitudes(Vec<u128>),
    /// `<Z_{q1} Z_{q2} ...>` of the final state of a unitary circuit.
    Expectation(Vec<usize>),
}

/// Resource limits. The state vector itself is additionally capped by
/// [`MAX_STATE_BYTES`]; a budget can only lower that.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    /// Largest dense register (bytes) any engine may allocate.
    pub mem_bytes: u128,
}

impl Default for Budget {
    fn default() -> Self {
        Budget {
            mem_bytes: MAX_STATE_BYTES,
        }
    }
}

/// The answer to a [`Request`].
#[derive(Clone, Debug, PartialEq)]
pub enum Output {
    Samples(Vec<Vec<bool>>),
    Amplitudes(Vec<Complex64>),
    Expectation(f64),
}

/// Result of [`simulate`].
#[derive(Clone, Debug, PartialEq)]
pub struct Simulation {
    pub output: Output,
    /// `(qubits, gates, engine)` per simulated component.
    pub engines: Vec<(usize, usize, Backend)>,
}

/// The compile options the pipeline runs with (all passes and the rule
/// based dispatcher on).
pub fn plan_options() -> PlanOptions {
    PlanOptions {
        adaptive: Some(AdaptiveRule {
            min_qubits: ADAPTIVE_MIN_QUBITS,
            margin: ADAPTIVE_MARGIN,
            max_active: ADAPTIVE_MAX_ACTIVE,
        }),
        ..PlanOptions::default()
    }
}

fn check_budget(stats: &CompileStats, budget: &Budget) -> Result<(), SimError> {
    for &(n, _, backend) in &stats.components {
        let bytes = match backend {
            Backend::StateVector => state_bytes::<f64>(n),
            // The compressed register is checked against its own cap when
            // it is built; its size is at most 2^ADAPTIVE_MAX_ACTIVE.
            _ => 0,
        };
        if bytes > budget.mem_bytes {
            return Err(SimError::TooLarge {
                what: "state vector",
                bytes,
                limit: budget.mem_bytes,
            });
        }
    }
    Ok(())
}

/// Runs `circuit` for `request` on the cheapest exact engines.
///
/// * [`Request::Samples`] accepts any circuit [`Circuit::run`] accepts
///   (mid-circuit measurement, reset, classical control and noise channels
///   are simulated shot by shot per component).
/// * [`Request::Amplitudes`] and [`Request::Expectation`] need a unitary
///   circuit and return [`SimError::NotSupported`] otherwise.
pub fn simulate(
    circuit: &Circuit,
    request: &Request,
    budget: &Budget,
) -> Result<Simulation, SimError> {
    let opts = plan_options();
    match request {
        Request::Samples { shots, seed } => {
            let plan = compile_sampling(circuit, opts)?;
            check_budget(&plan.stats, budget)?;
            let mut rng = StdRng::seed_from_u64(*seed);
            let out = plan.sample::<f64, _>(*shots, &mut rng)?;
            Ok(Simulation {
                output: Output::Samples(out),
                engines: plan.stats.components.clone(),
            })
        }
        Request::Amplitudes(xs) => {
            let plan = compile_unitary(circuit, opts)?;
            check_budget(&plan.stats, budget)?;
            let state = plan.factored::<f64>()?;
            Ok(Simulation {
                output: Output::Amplitudes(xs.iter().map(|&x| state.amplitude(x)).collect()),
                engines: plan.stats.components.clone(),
            })
        }
        Request::Expectation(qs) => {
            let v = expectation_z_product(circuit, qs, opts)?;
            Ok(Simulation {
                output: Output::Expectation(v),
                engines: Vec::new(),
            })
        }
    }
}

/// How `qsim run shor` should run for a modulus.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShorPath {
    pub semiclassical: bool,
    pub sparse: bool,
    /// True if the choice differs from what the flags asked for.
    pub overridden: bool,
}

/// Picks the Shor path: the gate-level 3n-qubit dense circuit if it fits in
/// `mem_bytes`, else the semiclassical one (one recycled control qubit); a
/// dense semiclassical register that does not fit becomes the exact sparse
/// state. `f32` halves the dense amplitude size.
pub fn choose_shor_path(
    n_mod: u64,
    oracle: Oracle,
    semiclassical: bool,
    sparse: bool,
    f32: bool,
    mem_bytes: u128,
) -> ShorPath {
    let m = (64 - n_mod.saturating_sub(1).leading_zeros()) as usize;
    let elem: u128 = if f32 { 8 } else { 16 };
    let fits = |qubits: usize| (1u128 << qubits.min(120)) * elem <= mem_bytes;
    let semi_qubits = match oracle {
        Oracle::Permutation => m + 1,
        Oracle::Beauregard => 2 * m + 3,
    };
    let (mut sc, mut sp) = (semiclassical, sparse);
    if !sc && !fits(3 * m) {
        sc = true;
        sp = true;
    }
    if sc && !sp && !fits(semi_qubits) {
        sp = true;
    }
    ShorPath {
        semiclassical: sc,
        sparse: sp,
        overridden: sc != semiclassical || sp != sparse,
    }
}
