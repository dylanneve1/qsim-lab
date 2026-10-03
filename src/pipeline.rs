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
//!   and `d + ADAPTIVE_MARGIN <= n` (so the dense register is smaller than a full state vector). Used for
//!   terminal samples and Z-product expectations, never for amplitudes: the
//!   compressed state drops global phases.
//! * **State vector** otherwise, f64, blocked executor, refused above
//!   [`Budget::mem_bytes`].
//!
//! The same numbers are recorded in `research/pipeline.md`.

use crate::circuit::{Circuit, Op, SimError};
use crate::compile::plan::{
    compile_sampling, compile_unitary, expectation_z_product, AdaptiveRule, Backend, CompileStats,
    PlanOptions,
};
use crate::compile::repeat::exec::ExecOptions;
use crate::compile::repeat::{DetectOptions, Program};
use crate::compile::validate;
use crate::gate::Gate;
use crate::shor::Oracle;
use crate::statevector::StateVector;
use crate::statevector::{state_bytes, MAX_STATE_BYTES};
use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Smallest component (in qubits) for which the compressed state is tried.
pub const ADAPTIVE_MIN_QUBITS: usize = 12;
/// The dense active register must be at least this many qubits smaller than
/// the component (measured: adaptive ties the state vector at `d = n` and
/// wins from `d = n - 1`, see `research/pipeline.md`).
pub const ADAPTIVE_MARGIN: usize = 1;
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

/// Options of [`simulate_with`]. Everything defaults to off, so
/// `SimOptions::default()` is exactly [`simulate`].
#[derive(Clone, Debug, Default)]
pub struct SimOptions {
    /// Exploit repeated blocks (`compile::repeat`), or `None`.
    pub repeat: Option<RepeatOptions>,
}

/// Options of the repeat pass (see `research/repeat.md`).
#[derive(Clone, Debug)]
pub struct RepeatOptions {
    pub detect: DetectOptions,
    pub exec: ExecOptions,
    /// Ignore circuits where repeats let a simulator skip fewer gates.
    pub min_saved_gates: usize,
    /// Allow the symplectic power for Clifford blocks (not used for
    /// amplitude requests, which need the global phase).
    pub clifford_power: bool,
    /// Skip deterministic steady-state rounds of Clifford circuits with
    /// mid-circuit measurements.
    pub steady_state: bool,
    /// Largest register (qubits) for the dense fast paths.
    pub dense_max_qubits: usize,
}

impl Default for RepeatOptions {
    fn default() -> Self {
        RepeatOptions {
            detect: DetectOptions::default(),
            exec: ExecOptions::default(),
            min_saved_gates: 64,
            clifford_power: true,
            steady_state: true,
            dense_max_qubits: 24,
        }
    }
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
    simulate_with(circuit, request, budget, &SimOptions::default())
}

/// [`simulate`] with options. With `opts.repeat` set, repeated blocks are
/// found and simulated with the exact fast paths of `compile::repeat`;
/// amplitudes, expectation values and outcome distributions are unchanged
/// (amplitudes up to floating-point rounding). Circuits without a useful
/// repeat take the ordinary path.
pub fn simulate_with(
    circuit: &Circuit,
    request: &Request,
    budget: &Budget,
    opts: &SimOptions,
) -> Result<Simulation, SimError> {
    if let Some(ro) = &opts.repeat {
        if let Some(r) = repeat_path(circuit, request, budget, ro) {
            return r;
        }
    }
    simulate_plain(circuit, request, budget)
}

fn is_terminal_unitary(c: &Circuit) -> bool {
    let mut seen_measure = false;
    for op in &c.ops {
        match op {
            Op::Gate(_) if !seen_measure => {}
            Op::Measure(_) => seen_measure = true,
            _ => return false,
        }
    }
    true
}

/// `None`: no useful repeat (or not applicable), use the ordinary path.
fn repeat_path(
    circuit: &Circuit,
    request: &Request,
    budget: &Budget,
    ro: &RepeatOptions,
) -> Option<Result<Simulation, SimError>> {
    use crate::compile::repeat::{cliff, detect, exec, rewrite, Node};
    validate(circuit).ok()?;
    let prog = detect(circuit, &ro.detect);
    let rep = prog.report();
    if rep.saved_gates < ro.min_saved_gates {
        return None;
    }
    let n = circuit.num_qubits;
    let dense_ok = n <= ro.dense_max_qubits && state_bytes::<f64>(n) <= budget.mem_bytes;
    // does any repeat need the dense engine (neither diagonal nor Clifford)?
    fn needs_dense(nodes: &[Node], clifford_ok: bool) -> bool {
        nodes.iter().any(|nd| match nd {
            Node::Repeat { body, .. } => {
                let mut flat = Vec::new();
                fn gs(nodes: &[Node], out: &mut Vec<Gate>) -> bool {
                    nodes.iter().all(|n| match n {
                        Node::Ops(o) => o.iter().all(|op| match op {
                            Op::Gate(g) => {
                                out.push(*g);
                                true
                            }
                            _ => false,
                        }),
                        Node::Repeat { body, .. } => gs(body, out),
                        Node::Param { .. } => false,
                    })
                }
                if !gs(body, &mut flat) {
                    return true;
                }
                let diag = flat.iter().all(|g| {
                    matches!(
                        g,
                        Gate::I(_)
                            | Gate::Z(_)
                            | Gate::S(_)
                            | Gate::Sdg(_)
                            | Gate::T(_)
                            | Gate::Tdg(_)
                            | Gate::Rz(..)
                            | Gate::Phase(..)
                            | Gate::Cz(..)
                            | Gate::CPhase(..)
                    )
                });
                !(diag || (clifford_ok && flat.iter().all(|g| g.is_clifford())))
            }
            Node::Param { .. } => false,
            Node::Ops(_) => false,
        })
    }
    match request {
        Request::Samples { shots, seed } => {
            let mut rng = StdRng::seed_from_u64(*seed);
            if !is_terminal_unitary(circuit) {
                // mid-circuit measurement: Clifford rounds on the tableau
                if !(ro.steady_state && cliff::is_clifford_program(&prog)) {
                    return None;
                }
                let mut out = Vec::with_capacity(*shots);
                for _ in 0..*shots {
                    let (bits, _) =
                        cliff::sample_program(&prog, ro.steady_state, ro.clifford_power, &mut rng)?;
                    out.push(bits);
                }
                return Some(Ok(Simulation {
                    output: Output::Samples(out),
                    engines: vec![(n, circuit.num_gates(), Backend::Tableau)],
                }));
            }
            let measured: Vec<usize> = circuit
                .ops
                .iter()
                .filter_map(|o| match o {
                    Op::Measure(q) => Some(*q),
                    _ => None,
                })
                .collect();
            // Terminal measurements may sit inside repeats too (e.g. the
            // same qubit measured many times): strip them at every depth.
            let gates_only = Program {
                num_qubits: n,
                nodes: strip_measures(&prog.nodes),
            };
            if needs_dense(&gates_only.nodes, ro.clifford_power) && dense_ok {
                let mut run = || -> Result<Vec<Vec<bool>>, SimError> {
                    let mut sv = StateVector::<f64>::try_new(n)?;
                    exec::run_dense(&gates_only, &mut sv, &ro.exec)?;
                    let idx = sv.sample(*shots, &mut rng);
                    Ok(idx
                        .into_iter()
                        .map(|i| measured.iter().map(|&q| (i >> q) & 1 == 1).collect())
                        .collect())
                };
                return Some(run().map(|s| Simulation {
                    output: Output::Samples(s),
                    engines: vec![(n, circuit.num_gates(), Backend::StateVector)],
                }));
            }
            let rw = rewrite(&prog, ro.clifford_power);
            Some(simulate_plain(&rw.circuit, request, budget))
        }
        Request::Amplitudes(xs) => {
            // The global phase matters: no Clifford power.
            if !is_unitary(circuit) || !dense_ok {
                return None;
            }
            let run = || -> Result<Vec<Complex64>, SimError> {
                let mut sv = StateVector::<f64>::try_new(n)?;
                exec::run_dense(
                    &Program {
                        num_qubits: n,
                        nodes: prog.nodes.clone(),
                    },
                    &mut sv,
                    &ro.exec,
                )?;
                Ok(xs.iter().map(|&x| sv.amplitude(x as usize)).collect())
            };
            Some(run().map(|a| Simulation {
                output: Output::Amplitudes(a),
                engines: vec![(n, circuit.num_gates(), Backend::StateVector)],
            }))
        }
        Request::Expectation(_) => {
            if !is_unitary(circuit) {
                return None;
            }
            let rw = rewrite(&prog, ro.clifford_power);
            Some(simulate_plain(&rw.circuit, request, budget))
        }
    }
}

fn is_unitary(c: &Circuit) -> bool {
    c.ops.iter().all(|o| matches!(o, Op::Gate(_)))
}

fn strip_measures(nodes: &[crate::compile::repeat::Node]) -> Vec<crate::compile::repeat::Node> {
    use crate::compile::repeat::Node;
    nodes
        .iter()
        .map(|n| match n {
            Node::Ops(o) => Node::Ops(
                o.iter()
                    .filter(|op| matches!(op, Op::Gate(_)))
                    .copied()
                    .collect(),
            ),
            Node::Repeat { body, reps } => Node::Repeat {
                body: strip_measures(body),
                reps: *reps,
            },
            other => other.clone(),
        })
        .collect()
}

fn simulate_plain(
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
        Oracle::Ripple => 3 * m + 4,
        Oracle::Windowed(w) => 4 * m + 4 + w.min(m),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shor_path_follows_the_memory_cap() {
        let cap = MAX_STATE_BYTES;
        // N = 15: 12 qubits, dense gate-level path fits; flags are respected.
        let p = choose_shor_path(15, Oracle::Permutation, false, false, false, cap);
        assert_eq!(
            p,
            ShorPath {
                semiclassical: false,
                sparse: false,
                overridden: false
            }
        );
        // N ~ 1.6e7: 3n = 72 qubits dense cannot exist -> semiclassical sparse.
        let p = choose_shor_path(16_777_207, Oracle::Permutation, false, false, false, cap);
        assert!(p.semiclassical && p.sparse && p.overridden);
        // semiclassical dense still fits at n = 8 (9 qubits) but not at n = 40.
        let p = choose_shor_path(200, Oracle::Permutation, true, false, false, cap);
        assert!(p.semiclassical && !p.sparse && !p.overridden);
        let p = choose_shor_path(1 << 40 | 1, Oracle::Permutation, true, false, false, cap);
        assert!(p.sparse && p.overridden);
        // explicit sparse is never changed
        let p = choose_shor_path(15, Oracle::Permutation, true, true, false, cap);
        assert!(p.semiclassical && p.sparse && !p.overridden);
    }
}
