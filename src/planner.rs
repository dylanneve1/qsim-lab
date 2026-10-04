//! Planner v0: choose the exact engine for a circuit from fitted per-engine
//! cost models (research/planner.md, research/simulability.md).
//!
//! ```text
//! plan(circuit, request, budget)
//!   1. <Z...Z> provably 0 (x-span certificate, O(gates·n))   -> Zero
//!   2. Clifford circuit                                      -> Tableau
//!   3. argmin_e  2^(a_e + b_e · R_e)  over the applicable state engines
//!      R_e = cheap log2 work estimate (simulability::features):
//!        sv     n + log2 gates
//!        sparse log2 gates + affine support bound
//!        hsf    pruning-aware path count of the KL partition
//!        cstate log2 Σ_j 2^{d_j} (rotation-frame active dimension)
//!        mps    log2 of the replayed operation count (mps_cost::replay
//!               with the best rigorous bond bound)
//! execute(plan)
//!   - MPS and sparse run speculatively: if they exceed `speculate` × the
//!     runner-up's predicted time (or the memory budget) they are aborted
//!     and the runner-up runs instead. The MPS model is the least reliable
//!     one, and this caps what a misprediction can cost.
//!   - `debug_reference`: on small registers, also run the reference state
//!     vector and fail loudly if the plan's value differs.
//! ```
//!
//! The constants `a_e, b_e` are machine specific; [`CostModel::mac_m1`] was
//! fitted on the 314-instance simulability dataset (M1 Pro, single thread,
//! `research/data/planner/`). Everything is exact: every engine either
//! returns the exact value (MPS: to the SVD's numerical rank, ~1e-7) or an
//! error.

use crate::adaptive::{self, AdaptiveOptions, Strategy};
use crate::circuit::{Circuit, SimError};
use crate::mps::Mps;
use crate::mps_cost::{self, BondSource, Estimator};
use crate::pauli_frame::FrameOptions;
use crate::pauli_path::PauliSum;
use crate::simulability::{self, Features};
use crate::sparse::SparseState;
use crate::statevector::StateVectorF64;
use std::time::Instant;

/// An exact engine the planner can choose.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Engine {
    /// The requested Pauli expectation is certified to vanish.
    Zero,
    Tableau,
    StateVector,
    Sparse,
    Mps,
    Hsf,
    /// Clifford frame + dense register on the active qubits
    /// ([`adaptive::CompressedState`]); for expectation values it runs as
    /// [`Strategy::Auto`] (Heisenberg sweep first, ski-rental hand-over).
    Compressed,
}

impl Engine {
    /// The engine name used by [`simulability::run_engine_obs`].
    pub fn name(&self) -> &'static str {
        match self {
            Engine::Zero => "zero",
            Engine::Tableau => "tableau",
            Engine::StateVector => "sv",
            Engine::Sparse => "sparse",
            Engine::Mps => "mps",
            Engine::Hsf => "hsf",
            Engine::Compressed => "cstate",
        }
    }
}

/// The state engines the cost model ranks.
pub const STATE_ENGINES: [Engine; 5] = [
    Engine::StateVector,
    Engine::Sparse,
    Engine::Mps,
    Engine::Hsf,
    Engine::Compressed,
];

/// What the caller wants.
#[derive(Clone, Debug, PartialEq)]
pub enum PlanRequest {
    /// `<Z_{q1} Z_{q2} ...>` of the final state.
    Expectation(Vec<usize>),
    /// Amplitudes with the global phase (the compressed state drops it).
    Amplitudes,
}

/// `log2 seconds = a + b · R`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EngineModel {
    pub a: f64,
    pub b: f64,
}

/// Per-engine cost models plus the units of the MPS work estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostModel {
    pub sv: EngineModel,
    pub sparse: EngineModel,
    pub mps: EngineModel,
    pub hsf: EngineModel,
    pub cstate: EngineModel,
    /// Weight of one SVD work unit relative to QR / product units.
    pub mps_svd_weight: f64,
    /// Fixed cost of one SVD/QR call, in work units.
    pub mps_call_overhead: f64,
}

impl CostModel {
    /// Fitted on the Mac (M1 Pro, one thread) simulability dataset with the
    /// replayed MPS work (research/planner.md §3, `fit_planner.py`).
    pub fn mac_m1() -> Self {
        CostModel {
            sv: EngineModel {
                a: -29.0959,
                b: 0.9325,
            },
            sparse: EngineModel {
                a: -23.1253,
                b: 0.8422,
            },
            mps: EngineModel {
                a: -13.7198,
                b: 0.2697,
            },
            hsf: EngineModel {
                a: -19.6338,
                b: 0.5932,
            },
            cstate: EngineModel {
                a: -28.3955,
                b: 0.9788,
            },
            mps_svd_weight: 8.0,
            mps_call_overhead: 2000.0,
        }
    }

    fn model(&self, e: Engine) -> Option<EngineModel> {
        match e {
            Engine::StateVector => Some(self.sv),
            Engine::Sparse => Some(self.sparse),
            Engine::Mps => Some(self.mps),
            Engine::Hsf => Some(self.hsf),
            Engine::Compressed => Some(self.cstate),
            _ => None,
        }
    }
}

impl Default for CostModel {
    fn default() -> Self {
        CostModel::mac_m1()
    }
}

/// Planner configuration.
#[derive(Clone, Copy, Debug)]
pub struct PlannerConfig {
    /// Largest register (bytes) any engine may allocate.
    pub mem_bytes: u128,
    pub model: CostModel,
    /// Abort MPS/sparse after `speculate` × the runner-up's predicted time
    /// and run the runner-up (`0`: never abort).
    pub speculate: f64,
    /// Never abort before this many seconds (overheads, timer noise).
    pub min_deadline_secs: f64,
    /// Also run the reference state vector (n ≤ `debug_max_qubits`) and
    /// panic if the planned value differs by more than `debug_tol`.
    pub debug_reference: bool,
    pub debug_max_qubits: usize,
    pub debug_tol: f64,
    /// Answer 0 without simulating when the x-span certificate fires.
    pub use_certificate: bool,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        PlannerConfig {
            mem_bytes: 1 << 30,
            model: CostModel::default(),
            speculate: 1.0,
            min_deadline_secs: 2e-3,
            debug_reference: false,
            debug_max_qubits: 20,
            debug_tol: 1e-6,
            use_certificate: true,
        }
    }
}

/// The planner's cheap view of a circuit.
#[derive(Clone, Debug, Default)]
pub struct PlanFeatures {
    pub base: Features,
    /// log2 of the replayed MPS work (best bound), in model units.
    pub mps_r: f64,
    /// Predicted largest MPS bond.
    pub mps_max_bond: usize,
    pub clifford: bool,
}

/// A decision.
#[derive(Clone, Debug)]
pub struct Plan {
    pub engine: Engine,
    /// Applicable engines with their predicted seconds, best first.
    pub ranked: Vec<(Engine, f64)>,
    pub features: PlanFeatures,
    /// Seconds spent planning.
    pub plan_secs: f64,
}

impl Plan {
    /// The best engine other than `e`, if any.
    pub fn runner_up(&self, e: Engine) -> Option<(Engine, f64)> {
        self.ranked.iter().copied().find(|&(x, _)| x != e)
    }
}

/// log2 MPS work of the replayed bond profile.
pub fn mps_work_log2(stats: &crate::mps::MpsStats, m: &CostModel) -> f64 {
    stats
        .total(m.mps_svd_weight, m.mps_call_overhead)
        .max(1.0)
        .log2()
}

/// Computes the planner features (~ms).
pub fn plan_features(
    c: &Circuit,
    obs: &[usize],
    cfg: &PlannerConfig,
) -> Result<PlanFeatures, SimError> {
    let base = simulability::features_for(c, true, obs)?;
    let r = mps_cost::replay(c, BondSource::Bound(Estimator::Best))?;
    Ok(PlanFeatures {
        mps_r: mps_work_log2(&r.stats, &cfg.model),
        mps_max_bond: r.max_bond,
        clifford: c.gates().all(|g| g.is_clifford()),
        base,
    })
}

fn resource(e: Engine, f: &PlanFeatures) -> f64 {
    match e {
        Engine::StateVector => f.base.sv_l,
        Engine::Sparse => f.base.sparse_l,
        Engine::Mps => f.mps_r,
        Engine::Hsf => f.base.hsf_l,
        Engine::Compressed => f.base.dense_l,
        _ => 0.0,
    }
}

fn applicable(e: Engine, f: &PlanFeatures, req: &PlanRequest, mem: u128) -> bool {
    let cap = (mem / 16).max(1).ilog2() as usize;
    let n = f.base.n;
    match e {
        Engine::StateVector | Engine::Hsf => n <= cap,
        Engine::Compressed => f.base.d <= cap.min(30) && matches!(req, PlanRequest::Expectation(_)),
        Engine::Sparse => n <= 64 && (f.base.sup as u32) < 64 && (48u128 << f.base.sup) <= mem,
        Engine::Mps => {
            let chi = f.mps_max_bond as u128;
            (n as u128) * 32 * chi.saturating_mul(chi) <= mem / 4
        }
        Engine::Tableau => f.clifford,
        Engine::Zero => false,
    }
}

/// Chooses the engine (nothing is simulated).
pub fn plan(c: &Circuit, req: &PlanRequest, cfg: &PlannerConfig) -> Result<Plan, SimError> {
    let t0 = Instant::now();
    let n = c.num_qubits;
    let obs: Vec<usize> = match req {
        PlanRequest::Expectation(q) => q.clone(),
        PlanRequest::Amplitudes => (0..n).collect(),
    };
    let f = plan_features(c, &obs, cfg)?;
    let mut ranked: Vec<(Engine, f64)> = STATE_ENGINES
        .iter()
        .filter(|&&e| applicable(e, &f, req, cfg.mem_bytes))
        .map(|&e| {
            let m = cfg.model.model(e).expect("state engine");
            (e, (m.a + m.b * resource(e, &f)).exp2())
        })
        .collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    let engine =
        if matches!(req, PlanRequest::Expectation(_)) && cfg.use_certificate && f.base.obs_zero {
            Engine::Zero
        } else if f.clifford {
            Engine::Tableau
        } else if let Some(&(e, _)) = ranked.first() {
            e
        } else {
            return Err(SimError::TooLarge {
                what: "planner: no exact engine fits the budget",
                bytes: 16u128 << n.min(120),
                limit: cfg.mem_bytes,
            });
        };
    if matches!(engine, Engine::Zero | Engine::Tableau) {
        ranked.insert(0, (engine, 0.0));
    }
    Ok(Plan {
        engine,
        ranked,
        features: f,
        plan_secs: t0.elapsed().as_secs_f64(),
    })
}

/// What [`execute_expectation`] did.
#[derive(Clone, Debug)]
pub struct Execution {
    pub value: f64,
    /// The engine that produced the value.
    pub engine: Engine,
    /// Engines aborted on the way (speculation).
    pub aborted: Vec<Engine>,
    pub secs: f64,
    /// Seconds spent planning.
    pub plan_secs: f64,
    /// The reference value (debug mode only).
    pub reference: Option<f64>,
}

enum Outcome {
    Value(f64),
    Aborted,
}

fn parity(x: u64, mask: u64) -> f64 {
    if (x & mask).count_ones() & 1 == 1 {
        -1.0
    } else {
        1.0
    }
}

fn run_one(
    e: Engine,
    c: &Circuit,
    obs: &[usize],
    cfg: &PlannerConfig,
    deadline: Option<f64>,
) -> Result<Outcome, SimError> {
    let t0 = Instant::now();
    let over = |t0: &Instant| deadline.is_some_and(|d| t0.elapsed().as_secs_f64() > d);
    match e {
        Engine::Zero => Ok(Outcome::Value(0.0)),
        Engine::Mps => {
            let mut m = Mps::new(c.num_qubits, 1 << 20);
            for g in c.gates() {
                m.apply_gate(g)?;
                if m.bytes() as u128 > cfg.mem_bytes / 4 || over(&t0) {
                    return Ok(Outcome::Aborted);
                }
            }
            if 1.0 - m.fidelity_estimate() > 1e-10 {
                return Ok(Outcome::Aborted);
            }
            Ok(Outcome::Value(m.expectation_z_product(obs)))
        }
        Engine::Sparse => {
            if c.num_qubits > 64 {
                return Ok(Outcome::Aborted);
            }
            let max_nnz = (cfg.mem_bytes / 48) as usize;
            let mut s = SparseState::new(c.num_qubits);
            for g in c.gates() {
                s.apply_gate(g)?;
                if s.nnz() > max_nnz || over(&t0) {
                    return Ok(Outcome::Aborted);
                }
            }
            let mask = obs.iter().fold(0u64, |m, &q| m ^ (1u64 << q));
            Ok(Outcome::Value(
                s.iter()
                    .map(|(x, a)| parity(x as u64, mask) * a.norm_sqr())
                    .sum(),
            ))
        }
        Engine::Compressed => {
            let max_d = ((cfg.mem_bytes / 16).max(1).ilog2() as usize).min(30);
            let opt = AdaptiveOptions {
                strategy: Strategy::Auto,
                max_dense_qubits: max_d,
                frame: FrameOptions {
                    max_terms: (cfg.mem_bytes / 64) as usize,
                    ..FrameOptions::default()
                },
                ..AdaptiveOptions::default()
            };
            let o = PauliSum::z_product(c.num_qubits, obs);
            Ok(Outcome::Value(adaptive::expectation(c, &o, &opt)?.value))
        }
        Engine::Tableau | Engine::StateVector | Engine::Hsf => {
            let r = simulability::run_engine_obs(e.name(), c, cfg.mem_bytes, obs)?;
            Ok(Outcome::Value(r.value))
        }
    }
}

/// Runs a [`Plan`] for `<Z_obs>` (see the module docs for speculation and
/// debug mode).
pub fn execute_expectation(
    plan: &Plan,
    c: &Circuit,
    obs: &[usize],
    cfg: &PlannerConfig,
) -> Result<Execution, SimError> {
    let t0 = Instant::now();
    let mut aborted = Vec::new();
    let mut order: Vec<(Engine, f64)> = plan.ranked.clone();
    if order.first().map(|x| x.0) != Some(plan.engine) {
        order.retain(|x| x.0 != plan.engine);
        order.insert(0, (plan.engine, 0.0));
    }
    let mut result = None;
    for i in 0..order.len() {
        let e = order[i].0;
        let speculative = matches!(e, Engine::Mps | Engine::Sparse) && cfg.speculate > 0.0;
        let deadline = if speculative {
            order
                .get(i + 1)
                .map(|&(_, t)| (cfg.speculate * t).max(cfg.min_deadline_secs))
        } else {
            None
        };
        match run_one(e, c, obs, cfg, deadline) {
            Ok(Outcome::Value(v)) => {
                result = Some((v, e));
                break;
            }
            Ok(Outcome::Aborted) => aborted.push(e),
            // a budget error from one engine: try the next one
            Err(SimError::TooLarge { .. }) if i + 1 < order.len() => aborted.push(e),
            Err(err) => return Err(err),
        }
    }
    let (value, engine) = result.ok_or(SimError::TooLarge {
        what: "planner: every engine aborted",
        bytes: 0,
        limit: cfg.mem_bytes,
    })?;
    let secs = t0.elapsed().as_secs_f64();
    let mut reference = None;
    if cfg.debug_reference && c.num_qubits <= cfg.debug_max_qubits {
        let mut sv = StateVectorF64::try_new(c.num_qubits)?;
        for g in c.gates() {
            sv.apply_gate(g)?;
        }
        let mask = obs.iter().fold(0u64, |m, &q| m ^ (1u64 << q));
        let r: f64 = sv
            .amplitudes()
            .iter()
            .enumerate()
            .map(|(x, a)| parity(x as u64, mask) * a.norm_sqr())
            .sum();
        assert!(
            (r - value).abs() <= cfg.debug_tol,
            "planner debug: {:?} gave {value}, reference {r} (plan {:?})",
            engine,
            plan.ranked
        );
        reference = Some(r);
    }
    Ok(Execution {
        value,
        engine,
        aborted,
        secs,
        plan_secs: plan.plan_secs,
        reference,
    })
}

/// Plans and runs `<Z_obs>` of `c|0^n>`.
pub fn expectation(c: &Circuit, obs: &[usize], cfg: &PlannerConfig) -> Result<Execution, SimError> {
    let p = plan(c, &PlanRequest::Expectation(obs.to_vec()), cfg)?;
    execute_expectation(&p, c, obs, cfg)
}
