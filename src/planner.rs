//! Planner v2: choose the exact engine for a circuit and a *request*
//! (expectation value, samples, amplitudes) from fitted per-engine cost
//! models (research/simulability/planner.md, research/simulability/planner-v2.md,
//! research/simulability/simulability.md).
//!
//! ```text
//! plan(circuit, request, config)
//!   tier 0  one O(gates) pass: n, gates, Clifford?, branching gates,
//!           non-Clifford rotations, MPS adjacent-operation count, cache key
//!           -> cache hit: reuse the plan of a structurally equal circuit
//!           -> upper-bound predictions for SV (exact), sparse (support
//!              <= 2^branching), compressed state (d_j <= j)
//!           -> if the best upper bound < voi x (predicted tier-1 cost): run it
//!   tier 1  O(gates·n): affine support bound, rotation-frame d-profile,
//!           vanishing certificate (expectations only)
//!   tier 2  MPS replay with the best rigorous bond bound, only if the MPS
//!           lower bound (every bond 1) beats the best prediction so far and
//!           that prediction > voi x (predicted replay cost)
//!   tier 3  HSF partition (KL), same rule with the HSF lower bound
//!   rank    argmin_e  evolve_e + readout_e(request)
//!           evolve_e  = 2^(a_e + b_e R_e)          (fitted, Mac M1)
//!           readout_e = Σ_i c_{e,i} · T_{e,i}(request) (fitted op counts:
//!             SV sampling 2^n + shots·log shots; MPS perfect sampling
//!             Σ 2 χ_l χ_r per shot; compressed sampler 2^d·d + shots·n;
//!             amplitudes: MPS Σ χ_l χ_r each, HSF its own path model)
//! execute_{expectation,samples,amplitudes}(plan)
//!   MPS and sparse run speculatively (abort after `speculate` × the
//!   runner-up's predicted time, then run the runner-up).
//! ```
//!
//! `PlannerConfig { tiered: false, cache: false, .. }` (see
//! [`PlannerConfig::v1`]) reproduces Planner v0/v1 exactly.
//!
//! Everything is exact: every engine either returns the exact answer
//! (MPS: to the SVD's numerical rank, ~1e-7; samples: exact distribution)
//! or an error. The cache only ever reuses an engine *choice*, never a
//! value.

use crate::circuit::{Circuit, Op, SimError};
use crate::engines::adaptive::{CompressedState, Sampler};
use crate::engines::blocked::BlockConfig;
use crate::engines::hsf::{HsfOptions, HybridSchrodingerFeynman};
use crate::engines::mps::Mps;
use crate::engines::mps_cost::{self, BondSource, Estimator};
use crate::engines::sparse::SparseState;
use crate::engines::stabilizer::Tableau;
use crate::engines::statevector::StateVectorF64;
use crate::engines::tn;
use crate::gate::{is_multiple_of_half_pi, Gate};
use crate::simulability::{self, Features};
use num_complex::Complex64;
use rand::Rng;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// An exact engine the planner can choose.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Engine {
    /// The requested Pauli expectation is certified to vanish.
    Zero,
    /// Stabilizer tableau (Clifford circuits only).
    Tableau,
    /// Dense state vector of all `2^n` amplitudes.
    StateVector,
    /// Sparse state storing only the nonzero amplitudes.
    Sparse,
    /// Matrix product state (exact: no bond truncation).
    Mps,
    /// Hybrid Schrödinger–Feynman: path sum over the gates cut by a bipartition.
    Hsf,
    /// Clifford frame + dense register on the active qubits
    /// ([`crate::engines::adaptive::CompressedState`]).
    Compressed,
    /// Exact tensor-network contraction ([`crate::engines::tn`]): amplitudes
    /// and Z-product expectations (through the light cone) without a state.
    Tn,
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
            Engine::Tn => "tn",
        }
    }

    /// Inverse of [`Engine::name`].
    pub fn from_name(s: &str) -> Option<Engine> {
        Some(match s {
            "zero" => Engine::Zero,
            "tableau" => Engine::Tableau,
            "sv" => Engine::StateVector,
            "sparse" => Engine::Sparse,
            "mps" => Engine::Mps,
            "hsf" => Engine::Hsf,
            "cstate" => Engine::Compressed,
            "tn" => Engine::Tn,
            _ => return None,
        })
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
    /// `count` amplitudes `<x|ψ>` with the global phase (the compressed
    /// state and the tableau drop it, so they are not candidates).
    Amplitudes(usize),
    /// `shots` samples of every qubit in the computational basis.
    Samples(usize),
}

impl PlanRequest {
    fn shots(&self) -> f64 {
        match self {
            PlanRequest::Samples(s) => *s as f64,
            _ => 0.0,
        }
    }
    fn amps(&self) -> f64 {
        match self {
            PlanRequest::Amplitudes(m) => *m as f64,
            _ => 0.0,
        }
    }
}

/// `log2 seconds = a + b · R`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EngineModel {
    /// Intercept of the fit, in log2 seconds.
    pub a: f64,
    /// Slope of log2 seconds per unit of the engine's log2 work estimate `R`.
    pub b: f64,
}

impl EngineModel {
    fn secs(&self, r: f64) -> f64 {
        (self.a + self.b * r).exp2()
    }
}

/// Read-out costs (seconds per operation-count unit) on top of the
/// evolution; research/simulability/planner-v2.md §2 defines the units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReadoutModel {
    /// State vector (and HSF full output) sampling: per amplitude of the
    /// `2^n` pass, and per shot (exponential spacings, merged walk, shuffle).
    pub sv_amp: f64,
    /// Per shot (see `sv_amp`).
    pub sv_shot: f64,
    /// Sparse sampling: per stored amplitude (bound `2^sup`), per shot.
    pub sparse_amp: f64,
    /// Per shot (see `sparse_amp`).
    pub sparse_shot: f64,
    /// MPS canonical form: per `Σ 2 χ_l χ_r min(2χ_r, χ_l)` and per site.
    pub mps_canon: [f64; 2],
    /// MPS perfect sampling, per shot: per `Σ 2 χ_l χ_r` and per site.
    pub mps_shot: [f64; 2],
    /// One MPS amplitude: per `Σ χ_l χ_r` and per site.
    pub mps_amp: [f64; 2],
    /// Compressed-state sampler: build per `2^d (d + 1) + n^2 ⌈n/64⌉`,
    /// per shot per `(d + n) ⌈n/64⌉`.
    pub cs_build: f64,
    /// Per shot (see `cs_build`).
    pub cs_shot: f64,
    /// Tableau sampling: echelon form per `n^2 ⌈n/64⌉`, per shot per
    /// `n (1 + ⌈n/64⌉)`.
    pub tab_build: f64,
    /// Per shot (see `tab_build`).
    pub tab_shot: f64,
    /// One amplitude look-up (state vector; sparse).
    pub lookup: f64,
    /// One amplitude look-up in the sparse state.
    pub sparse_lookup: f64,
    /// HSF amplitudes (no `2^n` output): set-up per `G n` and path sums per
    /// `2^keff G 2^max(n_A, n_B)`.
    pub hsf_amp: [f64; 2],
}

/// Per-engine cost models plus the units of the MPS work estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostModel {
    /// State-vector model (expectation runs), on `Features::sv_l`.
    pub sv: EngineModel,
    /// Sparse-state model (expectation runs), on `Features::sparse_l`.
    pub sparse: EngineModel,
    /// MPS model (expectation runs), on `PlanFeatures::mps_r`.
    pub mps: EngineModel,
    /// HSF model (expectation runs), on `Features::hsf_l`.
    pub hsf: EngineModel,
    /// Compressed-state model (expectation runs), on `Features::dense_l`.
    pub cstate: EngineModel,
    /// Weight of one SVD work unit relative to QR / product units.
    pub mps_svd_weight: f64,
    /// Fixed cost of one SVD/QR call, in work units.
    pub mps_call_overhead: f64,
    /// Read-out costs added for samples and amplitudes.
    pub readout: ReadoutModel,
    /// Evolution-only models (`[sv, sparse, mps, hsf, cstate]`; HSF
    /// includes the full `2^n` output) for samples and amplitudes; the
    /// models above are fitted on whole expectation runs, as in v1.
    pub state: [EngineModel; 5],
    /// Tensor-network contraction model.
    pub tn: TnModel,
}

/// Tensor-network time model (research/simulability/tn.md §6): one
/// contraction takes `2^(a + b log2 C) + c0 + c1 · tensors` seconds for the
/// sliced contraction cost `C` (complex multiply-adds) of the planned tree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TnModel {
    /// Intercept, log2 seconds.
    pub a: f64,
    /// Slope per log2 of the contraction cost.
    pub b: f64,
    /// Fixed seconds per contraction (network build, simplification, plan).
    pub c0: f64,
    /// Seconds per tensor of the simplified network.
    pub c1: f64,
}

impl TnModel {
    /// Predicted seconds of one contraction of cost `10^log10_cost`.
    pub fn secs(&self, log10_cost: f64, tensors: usize) -> f64 {
        let l2 = log10_cost / std::f64::consts::LOG10_2;
        (self.a + self.b * l2).exp2() + self.c0 + self.c1 * tensors as f64
    }
}

impl Default for TnModel {
    /// Fitted on the 16-vCPU Xeon (one thread) and converted to M1 units with
    /// the measured state-vector speed ratio (research/simulability/tn.md §6).
    fn default() -> Self {
        TnModel {
            a: -29.9,
            b: 1.0,
            c0: 2e-4,
            c1: 2e-6,
        }
    }
}

impl ReadoutModel {
    /// Fitted on the Mac (M1 Pro, one thread; research/simulability/planner-v2.md §2,
    /// `fit_v2.py`, all 350 instances).
    pub fn mac_m1() -> Self {
        ReadoutModel {
            sv_amp: 1.4455e-9,
            sv_shot: 3.5906e-8,
            sparse_amp: 4.6809e-9,
            sparse_shot: 3.1852e-8,
            mps_canon: [3.1776e-11, 1.1146e-6],
            mps_shot: [1.1203e-10, 5.1838e-8],
            mps_amp: [1.6409e-10, 3.2896e-8],
            cs_build: 5.1520e-9,
            cs_shot: 7.2439e-9,
            tab_build: 1.4713e-7,
            tab_shot: 6.5697e-9,
            lookup: 1.1694e-8,
            sparse_lookup: 1.0637e-8,
            hsf_amp: [1.5968e-6, 1.0771e-11],
        }
    }
}

impl CostModel {
    /// Planner v2 (research/simulability/planner-v2.md §2): every model refitted on the
    /// v2 Mac session (M1 Pro, one thread, 350 instances): expectation
    /// models on whole runs, `state` models on the evolution alone.
    pub fn mac_m1() -> Self {
        let m = |a: f64, b: f64| EngineModel { a, b };
        CostModel {
            sv: m(-29.1444, 0.91994),
            sparse: m(-21.7869, 0.76945),
            mps: m(-18.9644, 0.48306),
            hsf: m(-20.1377, 0.60326),
            cstate: m(-28.5166, 0.98593),
            mps_svd_weight: 8.0,
            mps_call_overhead: 1000.0,
            readout: ReadoutModel::mac_m1(),
            state: [
                m(-29.2010, 0.92070),
                m(-21.7932, 0.76926),
                m(-18.9003, 0.47820),
                m(-20.1377, 0.60326),
                m(-28.5187, 0.98508),
            ],
            tn: TnModel::default(),
        }
    }

    /// Planner v1's constants (research/simulability/planner.md §3: the round-4 sweep,
    /// MPS refitted on the replayed work), for [`PlannerConfig::v1`].
    pub fn mac_m1_v1() -> Self {
        let m = |a: f64, b: f64| EngineModel { a, b };
        CostModel {
            sv: m(-29.0959, 0.9325),
            sparse: m(-23.1253, 0.8422),
            mps: m(-18.7136, 0.4755),
            hsf: m(-19.6338, 0.5932),
            cstate: m(-28.3955, 0.9788),
            ..CostModel::mac_m1()
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

/// Predicted seconds of each planning tier (fitted on the Mac,
/// research/simulability/planner-v2.md §3; RMSE 0.12-0.37 decades).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeatureCost {
    /// Tier 1a, affine support bound: fixed + per gate.
    pub support: [f64; 2],
    /// Tier 1b, rotation-frame profile (and the certificate): fixed + per
    /// gate + per rotation·qubit.
    pub frame: [f64; 3],
    /// MPS replay: fixed + per adjacent step·qubit.
    pub mps: [f64; 2],
    /// HSF Kernighan–Lin partition: fixed + per gate·qubit².
    pub hsf: [f64; 2],
    /// Tensor-network network build and quick tree search: fixed + per gate.
    pub tn: [f64; 2],
}

impl Default for FeatureCost {
    fn default() -> Self {
        FeatureCost {
            support: [8.96e-6, 7.39e-8],
            frame: [1.254e-5, 2.153e-7, 1.461e-8],
            mps: [7.26e-5, 4.33e-8],
            hsf: [3.67e-3, 7.34e-9],
            tn: [2e-3, 2e-5],
        }
    }
}

/// Planner configuration.
#[derive(Clone, Copy, Debug)]
pub struct PlannerConfig {
    /// Largest register (bytes) any engine may allocate.
    pub mem_bytes: u128,
    /// Cost models used to rank engines.
    pub model: CostModel,
    /// Abort MPS/sparse after `speculate` × the runner-up's predicted time
    /// and run the runner-up (`0`: never abort).
    pub speculate: f64,
    /// Never abort before this many seconds (overheads, timer noise).
    pub min_deadline_secs: f64,
    /// Also run the reference state vector (n ≤ `debug_max_qubits`) and
    /// panic if the planned value differs by more than `debug_tol`
    /// (expectations and amplitudes; samples are checked by the tests).
    pub debug_reference: bool,
    /// Largest circuit (qubits) the debug reference runs on; larger circuits skip the check.
    pub debug_max_qubits: usize,
    /// Largest allowed deviation between planned and reference values.
    pub debug_tol: f64,
    /// Answer 0 without simulating when the x-span certificate fires.
    pub use_certificate: bool,
    /// Probe-or-solve (expectations only): when MPS is not the first
    /// choice, run the MPS with bond cap `probe_cap` for at most
    /// `probe_frac` × the best predicted time. If it finishes without
    /// truncating, it *is* the exact answer; if it truncates, its bond
    /// trace replaces the bound in the MPS prediction and the engines are
    /// re-ranked. `None` (default) plans from the bounds alone.
    pub probe_cap: Option<u32>,
    /// Fraction of the best predicted time the probe may run.
    pub probe_frac: f64,
    /// v1 staging (used when `tiered` is false): the MPS replay is computed
    /// only if the cheapest O(gates · n) prediction is at least
    /// `mps_feature_min_secs`, the HSF partition only above
    /// `hsf_feature_min_secs`, and a state vector predicted below
    /// `sv_shortcut_secs` runs without any feature.
    pub mps_feature_min_secs: f64,
    /// See `mps_feature_min_secs`.
    pub hsf_feature_min_secs: f64,
    /// See `mps_feature_min_secs`.
    pub sv_shortcut_secs: f64,
    /// v2 tiered planning (module docs). A tier is computed only when the
    /// best prediction so far exceeds `voi` × its predicted cost and (MPS,
    /// HSF) the engine's lower bound beats that prediction.
    pub tiered: bool,
    /// Value-of-information factor for expectation and sample requests (see `tiered`).
    pub voi: f64,
    /// `voi` for amplitude requests (their MPS/HSF features matter more:
    /// the state vector and sparse look-ups rarely win).
    pub voi_amplitudes: f64,
    /// Per-tier planning-cost models used by the `voi` test.
    pub feature_cost: FeatureCost,
    /// Reuse plans of structurally equal circuits (same gates and qubits,
    /// same Clifford class of every angle, same request size bucket).
    pub cache: bool,
    /// Consider the tensor-network engine (v2 tiers only) for amplitudes
    /// and expectation values.
    pub tn: bool,
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
            probe_cap: None,
            probe_frac: 0.2,
            mps_feature_min_secs: 1e-3,
            hsf_feature_min_secs: 5e-3,
            sv_shortcut_secs: 3e-4,
            tiered: true,
            voi: 8.0,
            voi_amplitudes: 4.0,
            feature_cost: FeatureCost::default(),
            cache: true,
            tn: true,
        }
    }
}

impl PlannerConfig {
    /// Planner v1 (research/simulability/planner.md): v1 staging, no cache.
    pub fn v1() -> Self {
        PlannerConfig {
            tiered: false,
            cache: false,
            tn: false,
            model: CostModel::mac_m1_v1(),
            ..Default::default()
        }
    }

    fn fingerprint(&self) -> u64 {
        let mut h = Fnv::new();
        h.u128(self.mem_bytes);
        for x in [
            self.speculate,
            self.min_deadline_secs,
            self.probe_frac,
            self.mps_feature_min_secs,
            self.hsf_feature_min_secs,
            self.sv_shortcut_secs,
            self.voi,
            self.voi_amplitudes,
            self.model.sv.a,
            self.model.mps.a,
            self.model.readout.sv_amp,
            self.model.readout.mps_shot[0],
        ] {
            h.u64(x.to_bits());
        }
        h.u64(
            u64::from(self.use_certificate)
                | (u64::from(self.tiered) << 1)
                | (u64::from(self.tn) << 2),
        );
        h.u64(self.probe_cap.map_or(u64::MAX, u64::from));
        h.0
    }
}

/// The O(gates) tier-0 view of a circuit.
#[derive(Clone, Debug, Default)]
pub struct QuickFeatures {
    /// Number of qubits.
    pub n: usize,
    /// Unitary gates (`Op::Gate`); other ops are skipped.
    pub gates: usize,
    /// Two-qubit gates.
    pub g2: usize,
    /// True when every gate is Clifford.
    pub clifford: bool,
    /// Upper bound on the non-Clifford rotations of the rotation frame.
    pub rotations: usize,
    /// Branching (non-monomial) one-qubit gates: `log2 nnz ≤ branching`.
    pub branching: usize,
    /// Adjacent two-qubit applications of the MPS engine (SWAP routing and
    /// decompositions included): a lower bound on its SVD calls.
    pub mps_steps: usize,
    /// Structural hash (gates, qubits, Clifford class of every angle).
    pub key: u64,
}

struct Fnv(u64);
impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn u64(&mut self, x: u64) {
        self.0 = (self.0 ^ x).wrapping_mul(0x0100_0000_01b3);
        self.0 ^= self.0 >> 29;
    }
    fn u128(&mut self, x: u128) {
        self.u64(x as u64);
        self.u64((x >> 64) as u64);
    }
}

fn angle_class(t: f64) -> u64 {
    u64::from(is_multiple_of_half_pi(t)) | (u64::from(is_multiple_of_half_pi(t / 2.0)) << 1)
}

/// Tier 0: one pass over the gates, no allocation.
pub fn quick_features(c: &Circuit) -> QuickFeatures {
    let n = c.num_qubits;
    let mut q = QuickFeatures {
        n,
        clifford: true,
        ..Default::default()
    };
    let mut h = Fnv::new();
    h.u64(n as u64);
    let nonhalf = |t: f64| usize::from(!is_multiple_of_half_pi(t));
    for op in &c.ops {
        let g = match op {
            Op::Gate(g) => g,
            _ => {
                h.u64(u64::MAX);
                continue;
            }
        };
        q.gates += 1;
        q.clifford &= g.is_clifford();
        use Gate::*;
        let (code, class, a, b, t3): (u64, u64, usize, usize, usize) = match *g {
            I(x) => (0, 0, x, 0, 0),
            H(x) => (1, 0, x, 0, 0),
            X(x) => (2, 0, x, 0, 0),
            Y(x) => (3, 0, x, 0, 0),
            Z(x) => (4, 0, x, 0, 0),
            S(x) => (5, 0, x, 0, 0),
            Sdg(x) => (6, 0, x, 0, 0),
            T(x) => (7, 0, x, 0, 0),
            Tdg(x) => (8, 0, x, 0, 0),
            Sx(x) => (9, 0, x, 0, 0),
            Sxdg(x) => (10, 0, x, 0, 0),
            Rx(x, t) => (11, angle_class(t), x, 0, 0),
            Ry(x, t) => (12, angle_class(t), x, 0, 0),
            Rz(x, t) => (13, angle_class(t), x, 0, 0),
            Phase(x, t) => (14, angle_class(t), x, 0, 0),
            U(x, t, p, l) => (
                15,
                angle_class(t) | angle_class(p) << 2 | angle_class(l) << 4,
                x,
                0,
                0,
            ),
            Cnot(x, y) => (16, 0, x, y, 0),
            Cz(x, y) => (17, 0, x, y, 0),
            Swap(x, y) => (18, 0, x, y, 0),
            ISwap(x, y) => (19, 0, x, y, 0),
            ISwapdg(x, y) => (20, 0, x, y, 0),
            CPhase(x, y, t) => (21, angle_class(t / 2.0), x, y, 0),
            Ccx(x, y, z) => (22, 0, x, y, z),
        };
        h.u64(code | class << 8 | (a as u64) << 16 | (b as u64) << 36);
        if t3 != 0 || code == 22 {
            h.u64(t3 as u64);
        }
        // rotations (upper bound: before merging) and branching gates
        q.rotations += match *g {
            T(_) | Tdg(_) => 1,
            Rz(_, t) | Phase(_, t) | Rx(_, t) | Ry(_, t) => nonhalf(t),
            U(_, t, p, l) => nonhalf(t) + nonhalf(p) + nonhalf(l),
            CPhase(_, _, t) => 3 * nonhalf(t / 2.0),
            Ccx(..) => 7,
            _ => 0,
        };
        if g.arity() == 1 && g.diagonal_1q().is_none() && !matches!(g, X(_) | Y(_) | I(_)) {
            q.branching += 1;
        }
        // MPS adjacent applications (as mps_cost::replay routes them)
        let span = |x: usize, y: usize| (2 * x.abs_diff(y)).saturating_sub(1);
        match *g {
            Cnot(x, y) | Cz(x, y) | Swap(x, y) => {
                q.g2 += 1;
                q.mps_steps += span(x, y);
            }
            ISwap(x, y) | ISwapdg(x, y) | CPhase(x, y, _) => {
                q.g2 += 1;
                // decomposed into two-qubit parts by the MPS engine
                q.mps_steps += span(x, y);
            }
            Ccx(x, y, z) => {
                // Clifford+T decomposition: CNOTs (x,y)x1? use the exact list
                for p in crate::gate::toffoli_clifford_t(x, y, z) {
                    if let Cnot(u, v) = p {
                        q.mps_steps += span(u, v);
                    }
                }
            }
            _ => {}
        }
    }
    q.key = h.0;
    q
}

/// The planner's cheap view of a circuit.
#[derive(Clone, Debug, Default)]
pub struct PlanFeatures {
    /// Simulability features (see [`Features`]); fields of tiers that were not computed keep their defaults.
    pub base: Features,
    /// log2 of the replayed MPS work (best bound), in model units.
    pub mps_r: f64,
    /// Predicted largest MPS bond.
    pub mps_max_bond: usize,
    /// True when every gate is Clifford (the tableau is then a candidate).
    pub clifford: bool,
    /// Tier-0 features (v2).
    pub quick: QuickFeatures,
    /// Predicted bonds of the final MPS (empty until the replay ran).
    pub mps_bonds: Vec<usize>,
    /// Deepest tier computed: 0 quick, 1 O(G n), 2 + MPS replay, 3 + HSF.
    pub tier: u8,
    /// Which features were computed (v2): support bound, certificate,
    /// frame profile, MPS replay, HSF on the line split, HSF Kernighan–Lin
    /// partition.
    pub computed: [bool; 6],
    /// The HSF partition the plan priced (the line split of tier 3a, or the
    /// Kernighan–Lin partition of tier 3b); the HSF engine runs on exactly
    /// this partition. `None`: the engine computes its own KL partition.
    pub hsf_split: Option<Vec<bool>>,
    /// The tensor-network tier (tier 4): the tree the engine runs.
    pub tn: Option<TnFeature>,
}

/// The tensor-network tier's result.
#[derive(Clone, Debug)]
pub struct TnFeature {
    /// log10 of the sliced contraction cost of one amplitude, or of the
    /// doubled light-cone network of an expectation (complex multiply-adds).
    pub log10_cost: f64,
    /// log2 of the largest intermediate of one slice (entries).
    pub log2_size: f64,
    /// Tensors of the simplified network.
    pub tensors: usize,
    /// The contraction tree and slicing found while planning; the engine
    /// contracts along it (and searches again only if it does not fit).
    pub path: Arc<tn::Path>,
}

/// A decision.
#[derive(Clone, Debug)]
pub struct Plan {
    /// The chosen engine.
    pub engine: Engine,
    /// Applicable engines with their predicted seconds, best first.
    pub ranked: Vec<(Engine, f64)>,
    /// The features the decision was based on.
    pub features: PlanFeatures,
    /// Seconds spent planning.
    pub plan_secs: f64,
    /// Probe-or-solve answered the request during planning (MPS, exact).
    pub solved: Option<f64>,
    /// What the probe did: `(finished, truncated)`.
    pub probe: Option<(bool, bool)>,
    /// Seconds per tier `[quick, tier 1, MPS replay, HSF and TN]` (v2).
    pub stage_secs: [f64; 4],
    /// The plan came from the cache.
    pub cached: bool,
}

impl Plan {
    /// The best engine other than `e`, if any.
    pub fn runner_up(&self, e: Engine) -> Option<(Engine, f64)> {
        self.ranked.iter().copied().find(|&(x, _)| x != e)
    }
}

/// log2 MPS work of the replayed bond profile.
pub fn mps_work_log2(stats: &crate::engines::mps::MpsStats, m: &CostModel) -> f64 {
    stats
        .total(m.mps_svd_weight, m.mps_call_overhead)
        .max(1.0)
        .log2()
}

/// Computes all planner features (~ms; see [`plan`] for the staged
/// version it actually uses).
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
        mps_bonds: r.final_bonds,
        quick: quick_features(c),
        tier: 3,
        computed: [true; 6],
        hsf_split: None,
        tn: None,
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

fn words(n: usize) -> f64 {
    n.div_ceil(64).max(1) as f64
}

/// Read-out operation counts of the MPS final state: `(canonicalisation,
/// one shot, one amplitude)`.
pub fn mps_readout_units(bonds: &[usize]) -> (f64, f64, f64) {
    let n = bonds.len() + 1;
    let (mut canon, mut shot, mut amp) = (0.0, 0.0, 0.0);
    for q in 0..n {
        let dl = if q == 0 { 1 } else { bonds[q - 1] } as f64;
        let dr = if q + 1 == n { 1 } else { bonds[q] } as f64;
        canon += 2.0 * dl * dr * (2.0 * dr).min(dl);
        shot += 2.0 * dl * dr;
        amp += dl * dr;
    }
    (canon, shot, amp)
}

/// Sorted uniforms are drawn in O(shots) (exponential spacings), so a
/// shot costs a constant on top of the `2^n` pass.
fn shot_units(s: f64) -> f64 {
    s
}

/// HSF amplitude units: `(G, 2^keff G 2^max(n_A, n_B))`.
pub fn hsf_amp_units(f: &Features) -> (f64, f64) {
    let g = f.gates.max(1) as f64;
    let big = f.hsf_na.max(f.hsf_nb) as f64;
    (g, (f.hsf_keff as f64 + big).exp2() * g)
}

/// Predicted seconds of `e` for `req` (evolution + read-out).
pub fn predict_secs(e: Engine, f: &PlanFeatures, req: &PlanRequest, m: &CostModel) -> f64 {
    if e == Engine::Tn {
        let Some(t) = &f.tn else {
            return f64::INFINITY;
        };
        let one = m.tn.secs(t.log10_cost, t.tensors);
        return match req {
            PlanRequest::Amplitudes(k) => one * (*k).max(1) as f64,
            PlanRequest::Expectation(_) => one,
            PlanRequest::Samples(_) => f64::INFINITY,
        };
    }
    let ro = &m.readout;
    let n = f.base.n;
    let s = req.shots();
    let am = req.amps();
    if e == Engine::Hsf && matches!(req, PlanRequest::Amplitudes(_)) {
        let (g, paths) = hsf_amp_units(&f.base);
        return ro.hsf_amp[0] * g * n as f64 + ro.hsf_amp[1] * paths;
    }
    if e == Engine::Tableau {
        let nn = n as f64;
        return match req {
            PlanRequest::Samples(_) => {
                ro.tab_build * nn * nn * words(n) + ro.tab_shot * s * nn * (1.0 + words(n))
            }
            _ => 0.0,
        };
    }
    let Some(model) = m.model(e) else {
        return 0.0;
    };
    let model = match (req, e) {
        (PlanRequest::Expectation(_), _) => model,
        (_, Engine::StateVector) => m.state[0],
        (_, Engine::Sparse) => m.state[1],
        (_, Engine::Mps) => m.state[2],
        (_, Engine::Hsf) => m.state[3],
        _ => m.state[4],
    };
    let evolve = model.secs(resource(e, f));
    let read = match (e, req) {
        (_, PlanRequest::Expectation(_)) => 0.0,
        (Engine::StateVector, PlanRequest::Samples(_)) => {
            ro.sv_amp * (n as f64).exp2() + ro.sv_shot * shot_units(s)
        }
        (Engine::Hsf, PlanRequest::Samples(_)) => {
            ro.sv_amp * (n as f64).exp2() + ro.sv_shot * shot_units(s)
        }
        (Engine::Sparse, PlanRequest::Samples(_)) => {
            ro.sparse_amp * (f.base.sup.min(n) as f64).exp2() + ro.sparse_shot * shot_units(s)
        }
        (Engine::Mps, PlanRequest::Samples(_)) => {
            let (c, sh, _) = mps_readout_units(&f.mps_bonds);
            let nn = n as f64;
            ro.mps_canon[0] * c
                + ro.mps_canon[1] * nn
                + s * (ro.mps_shot[0] * sh + ro.mps_shot[1] * nn)
        }
        (Engine::Mps, PlanRequest::Amplitudes(_)) => {
            let (_, _, a) = mps_readout_units(&f.mps_bonds);
            am * (ro.mps_amp[0] * a + ro.mps_amp[1] * n as f64)
        }
        (Engine::Compressed, PlanRequest::Samples(_)) => {
            let d = f.base.d as f64;
            let nn = n as f64;
            ro.cs_build * (d.exp2() * (d + 1.0) + nn * nn * words(n))
                + ro.cs_shot * s * (d + nn) * words(n)
        }
        (Engine::StateVector, PlanRequest::Amplitudes(_)) => ro.lookup * am,
        (Engine::Sparse, PlanRequest::Amplitudes(_)) => ro.sparse_lookup * am,
        _ => 0.0,
    };
    evolve + read
}

/// A lower bound on [`predict_secs`] from the tier-0 features alone (the
/// model evaluated at the smallest work the engine can do: support 1,
/// active dimension 0, every MPS bond 1, a balanced HSF partition without
/// paths). Used to skip features of engines that cannot win.
pub fn lower_bound_secs(e: Engine, q: &QuickFeatures, req: &PlanRequest, m: &CostModel) -> f64 {
    if e == Engine::Tn {
        // every two-qubit gate tensor (16 entries) is contracted at least once
        let lc = (16.0 * q.g2.max(1) as f64).log10();
        let one = m.tn.secs(lc, 1);
        return match req {
            PlanRequest::Amplitudes(k) => one * (*k).max(1) as f64,
            PlanRequest::Expectation(_) => one,
            PlanRequest::Samples(_) => f64::INFINITY,
        };
    }
    let n = q.n;
    let g = q.gates.max(1) as f64;
    let mut f = PlanFeatures {
        base: Features {
            n,
            gates: q.gates,
            sv_l: g.log2() + n as f64,
            sparse_l: g.log2(),
            sup: 0,
            d: 0,
            dense_l: (q.rotations.max(1) as f64).log2(),
            hsf_na: n.div_ceil(2),
            hsf_nb: n / 2,
            hsf_keff: 0,
            ..Default::default()
        },
        mps_r: (m.mps_call_overhead * q.mps_steps as f64 + g)
            .max(1.0)
            .log2(),
        mps_bonds: vec![1; n.saturating_sub(1)],
        ..Default::default()
    };
    let half = n.div_ceil(2) as f64;
    // log2(G 2^half + 2^n) >= max(log2 G + half, n)
    f.base.hsf_l = (g.log2() + half).max(n as f64);
    predict_secs(e, &f, req, m)
}

fn applicable(e: Engine, f: &PlanFeatures, req: &PlanRequest, mem: u128) -> bool {
    let cap = (mem / 16).max(1).ilog2() as usize;
    let n = f.base.n;
    let amps = matches!(req, PlanRequest::Amplitudes(_));
    let indexed = !matches!(req, PlanRequest::Expectation(_));
    match e {
        Engine::StateVector => n <= cap,
        // amplitudes need only the two blocks in memory
        // the full 2^n output plus the two block registers must fit: at
        // n = cap the output alone fills the budget (measured: TooLarge at
        // n = 26, 1 GiB, after a speculative MPS run had been aborted for it)
        Engine::Hsf => {
            if amps {
                n <= 2 * cap && n < 64
            } else {
                n < cap
            }
        }
        Engine::Compressed => f.base.d <= cap.min(30) && !amps && (!indexed || n <= 128),
        // sparse and MPS check their memory at run time (and abort); the
        // bounds are too loose to exclude them up front.
        Engine::Sparse => n <= 64,
        Engine::Mps => !indexed || n <= 128,
        Engine::Tableau => f.clifford && !amps,
        Engine::Zero => false,
        // the tree is sliced to the budget; amplitudes are indexed by u128
        Engine::Tn => {
            f.tn.is_some() && !matches!(req, PlanRequest::Samples(_)) && (!amps || n <= 128)
        }
    }
}

/// Chooses the engine (nothing is simulated).
pub fn plan(c: &Circuit, req: &PlanRequest, cfg: &PlannerConfig) -> Result<Plan, SimError> {
    if cfg.tiered {
        plan_v2(c, req, cfg)
    } else {
        plan_v1(c, req, cfg)
    }
}

fn empty_plan(engine: Engine, ranked: Vec<(Engine, f64)>, features: PlanFeatures) -> Plan {
    Plan {
        engine,
        ranked,
        features,
        plan_secs: 0.0,
        solved: None,
        probe: None,
        stage_secs: [0.0; 4],
        cached: false,
    }
}

/// Planner v1 (research/simulability/planner.md §3), kept for A/B comparisons.
fn plan_v1(c: &Circuit, req: &PlanRequest, cfg: &PlannerConfig) -> Result<Plan, SimError> {
    let t0 = Instant::now();
    let n = c.num_qubits;
    let obs: Vec<usize> = match req {
        PlanRequest::Expectation(q) => q.clone(),
        _ => (0..n).collect(),
    };
    let clifford = c.gates().all(|g| g.is_clifford());
    if clifford
        && !(matches!(req, PlanRequest::Expectation(_)) && cfg.use_certificate)
        && !matches!(req, PlanRequest::Amplitudes(_))
    {
        // polynomial: nothing to compare
        let mut p = empty_plan(
            Engine::Tableau,
            vec![(Engine::Tableau, 0.0)],
            PlanFeatures {
                clifford,
                ..Default::default()
            },
        );
        p.plan_secs = t0.elapsed().as_secs_f64();
        return Ok(p);
    }
    // Stage 0: a state vector this cheap is not worth planning for.
    let sv_l = (c.num_gates().max(1) as f64).log2() + n as f64;
    let t_sv = (cfg.model.sv.a + cfg.model.sv.b * sv_l).exp2();
    if !clifford
        && n <= ((cfg.mem_bytes / 16).max(1).ilog2() as usize)
        && t_sv < cfg.sv_shortcut_secs
    {
        let mut p = empty_plan(
            Engine::StateVector,
            vec![(Engine::StateVector, t_sv)],
            PlanFeatures {
                clifford,
                ..Default::default()
            },
        );
        p.plan_secs = t0.elapsed().as_secs_f64();
        return Ok(p);
    }
    // Stage 1: O(gates · n) features (state vector, sparse, compressed
    // state, certificate).
    let mut f = PlanFeatures {
        base: simulability::features_for(c, false, &obs)?,
        mps_r: f64::INFINITY,
        mps_max_bond: 0,
        clifford,
        ..Default::default()
    };
    let predict = |e: Engine, f: &PlanFeatures| {
        let m = cfg.model.model(e).expect("state engine");
        (m.a + m.b * resource(e, f)).exp2()
    };
    let cheap = [Engine::StateVector, Engine::Sparse, Engine::Compressed]
        .iter()
        .filter(|&&e| applicable(e, &f, req, cfg.mem_bytes))
        .map(|&e| predict(e, &f))
        .fold(f64::INFINITY, f64::min);
    // Stage 2: MPS replay and HSF partition, only when they can pay off.
    let mut skip = Vec::new();
    if !clifford && cheap >= cfg.mps_feature_min_secs {
        let r = mps_cost::replay(c, BondSource::Bound(Estimator::Best))?;
        f.mps_r = mps_work_log2(&r.stats, &cfg.model);
        f.mps_max_bond = r.max_bond;
        f.mps_bonds = r.final_bonds;
    } else {
        skip.push(Engine::Mps);
    }
    // HSF only if it fits and nothing found so far is already cheap
    let so_far = if f.mps_r.is_finite() {
        cheap.min(predict(Engine::Mps, &f))
    } else {
        cheap
    };
    let hsf_fits = n <= (cfg.mem_bytes / 16).max(1).ilog2() as usize;
    if !clifford && hsf_fits && so_far >= cfg.hsf_feature_min_secs {
        simulability::add_hsf_features(c, &mut f.base)?;
    } else {
        skip.push(Engine::Hsf);
    }
    let mut ranked: Vec<(Engine, f64)> = STATE_ENGINES
        .iter()
        .filter(|&&e| !skip.contains(&e) && applicable(e, &f, req, cfg.mem_bytes))
        .map(|&e| (e, predict(e, &f)))
        .collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    let engine =
        if matches!(req, PlanRequest::Expectation(_)) && cfg.use_certificate && f.base.obs_zero {
            Engine::Zero
        } else if clifford && !matches!(req, PlanRequest::Amplitudes(_)) {
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
    let mut p = empty_plan(engine, ranked, f);
    probe_or_solve(c, req, cfg, &mut p)?;
    if matches!(p.engine, Engine::Zero | Engine::Tableau) {
        p.ranked.insert(0, (p.engine, 0.0));
    }
    p.plan_secs = t0.elapsed().as_secs_f64();
    Ok(p)
}

fn probe_or_solve(
    c: &Circuit,
    req: &PlanRequest,
    cfg: &PlannerConfig,
    p: &mut Plan,
) -> Result<(), SimError> {
    let (Some(cap), PlanRequest::Expectation(obs)) = (cfg.probe_cap, req) else {
        return Ok(());
    };
    let n = c.num_qubits;
    let best_t = p.ranked.first().map_or(0.0, |x| x.1);
    let budget = cfg.probe_frac * best_t;
    if matches!(p.engine, Engine::Zero | Engine::Tableau | Engine::Mps)
        || budget < cfg.min_deadline_secs
    {
        return Ok(());
    }
    let tp = Instant::now();
    let mut m = Mps::new(n, cap as usize);
    m.enable_trace();
    let mut finished = true;
    for g in c.gates() {
        m.apply_gate(g)?;
        if tp.elapsed().as_secs_f64() > budget {
            finished = false;
            break;
        }
    }
    let truncated = m.truncation_count() > 0;
    p.probe = Some((finished, truncated));
    if finished && !truncated {
        p.solved = Some(m.expectation_z_product(obs));
        p.engine = Engine::Mps;
    } else if finished {
        let r = mps_cost::replay(c, BondSource::ProbeExtrapolate(m.trace(), cap))?;
        p.features.mps_r = mps_work_log2(&r.stats, &cfg.model);
        let mm = cfg.model.mps;
        for x in p.ranked.iter_mut() {
            if x.0 == Engine::Mps {
                x.1 = (mm.a + mm.b * p.features.mps_r).exp2();
            }
        }
        p.ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
        p.engine = p.ranked[0].0;
    }
    Ok(())
}

static CACHE: Mutex<Option<HashMap<u64, Plan>>> = Mutex::new(None);
const CACHE_CAP: usize = 4096;

fn cache_key(q: &QuickFeatures, req: &PlanRequest, cfg: &PlannerConfig) -> u64 {
    let mut h = Fnv::new();
    h.u64(q.key);
    h.u64(cfg.fingerprint());
    match req {
        PlanRequest::Expectation(obs) => {
            h.u64(1);
            for &o in obs {
                h.u64(o as u64);
            }
        }
        // read-out costs scale with the request size: bucket by powers of 2
        PlanRequest::Amplitudes(m) => {
            h.u64(2);
            h.u64(u64::from(m.max(&1).ilog2()));
        }
        PlanRequest::Samples(s) => {
            h.u64(3);
            h.u64(u64::from(s.max(&1).ilog2()));
        }
    }
    h.0
}

/// Empties the plan cache.
pub fn clear_cache() {
    if let Ok(mut g) = CACHE.lock() {
        *g = None;
    }
}

fn cache_get(key: u64) -> Option<Plan> {
    CACHE.lock().ok()?.as_ref()?.get(&key).cloned()
}

fn cache_put(key: u64, p: &Plan) {
    // never reuse values (probe) or certificates (angle-specific merging)
    if p.solved.is_some() || p.engine == Engine::Zero {
        return;
    }
    if let Ok(mut g) = CACHE.lock() {
        let m = g.get_or_insert_with(HashMap::new);
        if m.len() >= CACHE_CAP {
            m.clear();
        }
        m.insert(key, p.clone());
    }
}

/// Tier-0 bounds written into [`Features`]: `sup ≤ min(n, branching)`,
/// `d_j ≤ min(n, j)`.
fn tier0_features(q: &QuickFeatures) -> PlanFeatures {
    let n = q.n;
    let g = q.gates.max(1) as f64;
    let mut base = Features {
        n,
        gates: q.gates,
        g2: q.g2,
        ..Default::default()
    };
    base.sv_l = g.log2() + n as f64;
    base.sup = q.branching.min(n);
    base.sparse_l = g.log2() + base.sup as f64;
    let r = q.rotations;
    base.rotations = r;
    base.d = r.min(n);
    // log2 Σ_{j=1}^{r} 2^{min(n, j)}
    base.dense_l = if r == 0 {
        0.0
    } else if r <= n {
        ((r as f64 + 1.0).exp2() - 2.0).log2()
    } else {
        ((n as f64 + 1.0).exp2() - 2.0 + (r - n) as f64 * (n as f64).exp2()).log2()
    };
    PlanFeatures {
        base,
        mps_r: f64::INFINITY,
        clifford: q.clifford,
        quick: q.clone(),
        tier: 0,
        ..Default::default()
    }
}

/// Tier 1b: the rotation-frame active-dimension profile.
fn frame_features(c: &Circuit, f: &mut PlanFeatures) -> Result<(), SimError> {
    let prof = crate::engines::adaptive::active_dimension_profile(c)?;
    f.base.rotations = prof.len();
    f.base.d = prof.last().copied().unwrap_or(0);
    f.base.dense_l = if prof.is_empty() {
        0.0
    } else {
        let m = prof.iter().copied().max().unwrap_or(0) as f64;
        m + prof
            .iter()
            .map(|&d| (d as f64 - m).exp2())
            .sum::<f64>()
            .log2()
    };
    Ok(())
}

fn rank(
    f: &PlanFeatures,
    req: &PlanRequest,
    cfg: &PlannerConfig,
    engines: &[Engine],
) -> Vec<(Engine, f64)> {
    let mut r: Vec<(Engine, f64)> = engines
        .iter()
        .filter(|&&e| applicable(e, f, req, cfg.mem_bytes))
        .map(|&e| (e, predict_secs(e, f, req, &cfg.model)))
        .collect();
    r.sort_by(|a, b| a.1.total_cmp(&b.1));
    r
}

fn plan_v2(c: &Circuit, req: &PlanRequest, cfg: &PlannerConfig) -> Result<Plan, SimError> {
    let t0 = Instant::now();
    let n = c.num_qubits;
    let q = quick_features(c);
    let key = cfg.cache.then(|| cache_key(&q, req, cfg));
    if let Some(k) = key {
        if let Some(mut p) = cache_get(k) {
            p.cached = true;
            p.plan_secs = t0.elapsed().as_secs_f64();
            return Ok(p);
        }
    }
    let mut stage = [0.0f64; 4];
    stage[0] = t0.elapsed().as_secs_f64();
    let is_exp = matches!(req, PlanRequest::Expectation(_));
    let amps = matches!(req, PlanRequest::Amplitudes(_));
    let cert_obs: Option<&[usize]> = match req {
        PlanRequest::Expectation(o) if cfg.use_certificate => Some(o),
        _ => None,
    };
    let finish = |mut p: Plan, stage: [f64; 4]| -> Plan {
        p.stage_secs = stage;
        p.plan_secs = t0.elapsed().as_secs_f64();
        if let Some(k) = key {
            if cfg.probe_cap.is_none() {
                cache_put(k, &p);
            }
        }
        p
    };
    let mut f = tier0_features(&q);
    if q.clifford && cert_obs.is_none() && !amps {
        let t = predict_secs(Engine::Tableau, &f, req, &cfg.model);
        return Ok(finish(
            empty_plan(Engine::Tableau, vec![(Engine::Tableau, t)], f),
            stage,
        ));
    }
    let fc = &cfg.feature_cost;
    // tuned offline on the read-out session (research/simulability/planner-v2.md §3)
    let cfg = &PlannerConfig {
        voi: if amps { cfg.voi_amplitudes } else { cfg.voi },
        ..*cfg
    };
    let (g, nn) = (q.gates as f64, n as f64);
    let cheap = [Engine::StateVector, Engine::Sparse, Engine::Compressed];
    let mut ranked = rank(&f, req, cfg, &cheap);
    let best = |r: &[(Engine, f64)]| r.first().map_or(f64::INFINITY, |x| x.1);
    // Tier 1a (affine support bound, sparse), the certificate and tier 1b
    // (rotation-frame d-profile, compressed state): each only if the engine
    // it informs could beat the best prediction by more than `voi` x its
    // cost (its lower bound from tier 0).
    let c1a = fc.support[0] + fc.support[1] * g;
    let c1b = fc.frame[0] + fc.frame[1] * g + fc.frame[2] * q.rotations as f64 * nn;
    let gain = |e: Engine, r: &[(Engine, f64)], f: &PlanFeatures| -> f64 {
        // applicability with the engine's most favourable size (d = 0)
        let mut fo = PlanFeatures {
            base: f.base.clone(),
            clifford: f.clifford,
            ..Default::default()
        };
        fo.base.d = 0;
        if applicable(e, &fo, req, cfg.mem_bytes) {
            best(r) - lower_bound_secs(e, &q, req, &cfg.model)
        } else {
            f64::NEG_INFINITY
        }
    };
    if gain(Engine::Sparse, &ranked, &f) > cfg.voi * c1a {
        let t1 = Instant::now();
        let gates: Vec<Gate> = c.gates().copied().collect();
        f.base.sup = simulability::support_bound(n, &gates);
        f.base.sparse_l = g.max(1.0).log2() + f.base.sup as f64;
        f.tier = 1;
        f.computed[0] = true;
        stage[1] += t1.elapsed().as_secs_f64();
        ranked = rank(&f, req, cfg, &cheap);
    }
    if let Some(obs) = cert_obs {
        if q.clifford || best(&ranked) > cfg.voi * c1b {
            let t1 = Instant::now();
            f.base.obs_zero = crate::engines::adaptive::z_product_vanishes(c, obs)?;
            f.computed[1] = true;
            stage[1] += t1.elapsed().as_secs_f64();
        }
    }
    if !f.base.obs_zero && !q.clifford && gain(Engine::Compressed, &ranked, &f) > cfg.voi * c1b {
        let t1 = Instant::now();
        frame_features(c, &mut f)?;
        f.tier = 1;
        f.computed[2] = true;
        stage[1] += t1.elapsed().as_secs_f64();
        ranked = rank(&f, req, cfg, &cheap);
    }
    if f.base.obs_zero && cert_obs.is_some() {
        let mut r = ranked.clone();
        r.insert(0, (Engine::Zero, 0.0));
        return Ok(finish(empty_plan(Engine::Zero, r, f), stage));
    }
    if q.clifford && !amps {
        let t = predict_secs(Engine::Tableau, &f, req, &cfg.model);
        let mut r = ranked.clone();
        r.insert(0, (Engine::Tableau, t));
        return Ok(finish(empty_plan(Engine::Tableau, r, f), stage));
    }
    let mut considered: Vec<Engine> = cheap.to_vec();
    // Tier 2: MPS replay, if MPS could beat the best prediction by more
    // than `voi` x the replay's cost.
    let c2 = fc.mps[0] + fc.mps[1] * q.mps_steps as f64 * nn;
    let mps_lb = lower_bound_secs(Engine::Mps, &q, req, &cfg.model);
    let indexed = !matches!(req, PlanRequest::Expectation(_));
    if (!indexed || n <= 128) && best(&ranked) - mps_lb > cfg.voi * c2 {
        let t2 = Instant::now();
        let r = mps_cost::replay(c, BondSource::Bound(Estimator::Best))?;
        f.mps_r = mps_work_log2(&r.stats, &cfg.model);
        f.mps_max_bond = r.max_bond;
        f.mps_bonds = r.final_bonds;
        f.tier = 2;
        f.computed[3] = true;
        stage[2] = t2.elapsed().as_secs_f64();
        considered.push(Engine::Mps);
        ranked = rank(&f, req, cfg, &considered);
    }
    // Tier 3: HSF. 3a prices it on the plain line split in O(gates) (the
    // engine then runs on that split, so the prediction is for the run that
    // happens); 3b refines with the Kernighan-Lin partition (~4 ms) only if
    // HSF could still win and is not already first.
    let hsf_lb = lower_bound_secs(Engine::Hsf, &q, req, &cfg.model);
    let mut probe_f = f.clone();
    probe_f.base.hsf_na = n / 2;
    let hsf_ok = n >= 2 && applicable(Engine::Hsf, &probe_f, req, cfg.mem_bytes);
    let c3a = fc.support[0] + fc.support[1] * g;
    if hsf_ok && best(&ranked) - hsf_lb > cfg.voi * c3a {
        let t3 = Instant::now();
        let split: Vec<bool> = (0..n).map(|i| i < n / 2).collect();
        simulability::hsf_split_features(c, &mut f.base, &split)?;
        // priced with the nominal path bits: the zero-path pruning estimate
        // (`keff`) was fitted on KL partitions and is far too optimistic on
        // a line split of a non-local circuit (measured: QAOA on random
        // graphs, n = 24, HSF on the line split 3.2 s vs 0.04 s with KL)
        f.base.hsf_keff = f.base.hsf_k;
        f.base.hsf_l = f.base.hsf_l0;
        f.hsf_split = Some(split);
        f.tier = 3;
        f.computed[4] = true;
        stage[3] += t3.elapsed().as_secs_f64();
        considered.push(Engine::Hsf);
        ranked = rank(&f, req, cfg, &considered);
    }
    // (also when HSF on the line split is already first: KL can still make
    // the run itself much cheaper, e.g. on random-matching circuits)
    let c3 = fc.hsf[0] + fc.hsf[1] * g * nn * nn;
    if hsf_ok && best(&ranked) - hsf_lb > cfg.voi * c3 {
        let t3 = Instant::now();
        let opts = HsfOptions {
            max_bytes: cfg.mem_bytes,
            ..HsfOptions::default()
        };
        let kl = crate::engines::hsf::auto_partition(c, &opts)?;
        simulability::hsf_split_features(c, &mut f.base, &kl)?;
        // the engine runs on this partition (no second KL at run time)
        f.hsf_split = Some(kl);
        f.tier = 3;
        f.computed[5] = true;
        stage[3] += t3.elapsed().as_secs_f64();
        if !considered.contains(&Engine::Hsf) {
            considered.push(Engine::Hsf);
        }
        ranked = rank(&f, req, cfg, &considered);
    }
    // Tier 4: tensor-network tree search (amplitudes and expectations), if
    // the engine could beat the best prediction by more than `voi` x the
    // search's cost. The engine then contracts along the planned tree.
    if cfg.tn && !matches!(req, PlanRequest::Samples(_)) && (!amps || n <= 128) {
        let tn_lb = lower_bound_secs(Engine::Tn, &q, req, &cfg.model);
        let c4 = fc.tn[0] + fc.tn[1] * g;
        if best(&ranked) - tn_lb > cfg.voi * c4 {
            let t4 = Instant::now();
            if let Some(tf) = tn_feature(c, req, cfg)? {
                f.tn = Some(tf);
                considered.push(Engine::Tn);
                ranked = rank(&f, req, cfg, &considered);
            }
            stage[3] += t4.elapsed().as_secs_f64();
        }
    }
    let Some(&(engine, _)) = ranked.first() else {
        return Err(SimError::TooLarge {
            what: "planner: no exact engine fits the budget",
            bytes: 16u128 << n.min(120),
            limit: cfg.mem_bytes,
        });
    };
    let _ = is_exp;
    let mut p = empty_plan(engine, ranked, f);
    probe_or_solve(c, req, cfg, &mut p)?;
    Ok(finish(p, stage))
}

/// Qubits that appear an odd number of times in a Z-product (`Z_q Z_q = I`).
fn odd_qubits(obs: &[usize]) -> Vec<usize> {
    let mut v: Vec<usize> = Vec::new();
    for &q in obs {
        if let Some(p) = v.iter().position(|&x| x == q) {
            v.swap_remove(p);
        } else {
            v.push(q);
        }
    }
    v.sort_unstable();
    v
}

fn tn_options(cfg: &PlannerConfig) -> tn::TnOptions {
    tn::TnOptions {
        max_bytes: cfg.mem_bytes,
        path: tn::PathOptions::quick(),
        ..tn::TnOptions::default()
    }
}

/// The circuit whose `<0|.|0>` amplitude the TN engine contracts for `req`
/// (`None` for samples).
fn tn_circuit(c: &Circuit, req: &PlanRequest) -> Result<Option<Circuit>, SimError> {
    Ok(match req {
        PlanRequest::Amplitudes(_) => Some(c.clone()),
        PlanRequest::Expectation(obs) => {
            let p: Vec<(usize, tn::Pauli)> = odd_qubits(obs)
                .into_iter()
                .map(|q| (q, tn::Pauli::Z))
                .collect();
            Some(tn::expectation_circuit(c, &p)?.0)
        }
        PlanRequest::Samples(_) => None,
    })
}

/// Tier 4: builds and simplifies the network of `req` (amplitude of
/// `|0^n>`, or the doubled light-cone network) and runs the quick search.
pub fn tn_feature(
    c: &Circuit,
    req: &PlanRequest,
    cfg: &PlannerConfig,
) -> Result<Option<TnFeature>, SimError> {
    let Some(d) = tn_circuit(c, req)? else {
        return Ok(None);
    };
    let mut nw = tn::Network::amplitude(&d, &vec![false; d.num_qubits], &[])?;
    let opts = tn_options(cfg);
    nw.simplify(&opts.simplify);
    let hg = tn::Hypergraph::from_network(&nw);
    let mut po = opts.path.clone();
    po.target_log2_size = Some(tn::default_target_log2(&opts));
    let path = tn::search(&hg, &po);
    Ok(Some(TnFeature {
        log10_cost: path.stats.log10_sliced_flops,
        log2_size: path.stats.log2_sliced_max_size,
        tensors: nw.tensors.len(),
        path: Arc::new(path),
    }))
}

/// `<x|d|0>` by tensor-network contraction, along `path` when it matches
/// the simplified network and fits the budget, else after a fresh search.
fn tn_contract(
    d: &Circuit,
    bits: &[bool],
    path: Option<&tn::Path>,
    mem_bytes: u128,
) -> Result<Complex64, SimError> {
    let cfg = PlannerConfig {
        mem_bytes,
        ..PlannerConfig::default()
    };
    let opts = tn_options(&cfg);
    let nw = tn::Network::amplitude(d, bits, &[])?;
    if let Some(p) = path {
        let mut sn = nw.clone();
        sn.simplify(&opts.simplify);
        if p.tree.n_leaves == sn.tensors.len() {
            let eo = tn::ExecOptions {
                max_bytes: mem_bytes,
                ..tn::ExecOptions::default()
            };
            let sliced: Vec<bool> = (0..sn.dims.len())
                .map(|i| p.sliced.get(i).copied().unwrap_or(false))
                .collect();
            match tn::contract::<Complex64>(&sn, &p.tree, &sliced, &eo) {
                Ok((v, _)) => return Ok(v[0]),
                Err(SimError::TooLarge { .. }) => {}
                Err(e) => return Err(e),
            }
        }
    }
    Ok(tn::run_network(nw, &opts)?.0[0])
}

/// A circuit prepared for tensor-network amplitudes (no state is built).
#[derive(Clone, Debug)]
pub struct TnPrepared {
    /// The circuit.
    pub circuit: Circuit,
    /// The tree planned for `<0^n|C|0^n>` (reused for every amplitude when
    /// the simplified network has the same tensors).
    pub path: Option<Arc<tn::Path>>,
    /// Memory budget of one contraction.
    pub mem_bytes: u128,
}

// ---------------------------------------------------------------------------
// Execution

/// What [`execute_expectation`] did.
#[derive(Clone, Debug)]
pub struct Execution {
    /// The expectation value `<Z_obs>`.
    pub value: f64,
    /// The engine that produced the value.
    pub engine: Engine,
    /// Engines aborted on the way (speculation).
    pub aborted: Vec<Engine>,
    /// Wall-clock seconds of execution (planning and the debug reference excluded).
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
    split: Option<&[bool]>,
    tnf: Option<&TnFeature>,
) -> Result<Outcome, SimError> {
    if e == Engine::Tn {
        let d = tn_circuit(c, &PlanRequest::Expectation(obs.to_vec()))?.expect("expectation");
        let v = tn_contract(
            &d,
            &vec![false; d.num_qubits],
            tnf.map(|t| &*t.path),
            cfg.mem_bytes,
        )?;
        return Ok(Outcome::Value(v.re));
    }
    if let (Engine::Hsf, Some(sp)) = (e, split) {
        // HSF on the split the plan priced, full output, parity
        let opts = HsfOptions {
            max_bytes: cfg.mem_bytes,
            ..HsfOptions::default()
        };
        let out = 16u128 << c.num_qubits.min(120);
        if out > cfg.mem_bytes {
            return Err(SimError::TooLarge {
                what: "hsf full output",
                bytes: out,
                limit: cfg.mem_bytes,
            });
        }
        let amps = HybridSchrodingerFeynman::new(c, sp, opts)?.state_vector()?;
        let mask = obs.iter().fold(0u64, |m, &q| m ^ (1u64 << q));
        return Ok(Outcome::Value(
            amps.iter()
                .enumerate()
                .map(|(x, a)| parity(x as u64, mask) * a.norm_sqr())
                .sum(),
        ));
    }
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
                s.iter().map(|(x, a)| parity(x, mask) * a.norm_sqr()).sum(),
            ))
        }
        // the compressed state runs exactly as the cost model measured it
        // (frame + dense register, always evolved: `Strategy::Auto`'s
        // run-time hand-over is not used, its cost is not predictable).
        Engine::Tableau | Engine::StateVector | Engine::Hsf | Engine::Compressed | Engine::Tn => {
            let r = simulability::run_engine_obs(e.name(), c, cfg.mem_bytes, obs)?;
            Ok(Outcome::Value(r.value))
        }
    }
}

/// The engines to try, the planned one first.
fn order_of(plan: &Plan) -> Vec<(Engine, f64)> {
    let mut order: Vec<(Engine, f64)> = plan.ranked.clone();
    if order.first().map(|x| x.0) != Some(plan.engine) {
        order.retain(|x| x.0 != plan.engine);
        order.insert(0, (plan.engine, 0.0));
    }
    order
}

fn deadline_for(order: &[(Engine, f64)], i: usize, cfg: &PlannerConfig) -> Option<f64> {
    let e = order[i].0;
    let speculative = matches!(e, Engine::Mps | Engine::Sparse) && cfg.speculate > 0.0;
    if !speculative {
        return None;
    }
    order
        .get(i + 1)
        .map(|&(_, t)| (cfg.speculate * t).max(cfg.min_deadline_secs))
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
    let order = order_of(plan);
    let mut result = plan.solved.map(|v| (v, Engine::Mps));
    for i in 0..order.len() {
        if result.is_some() {
            break;
        }
        let e = order[i].0;
        match run_one(
            e,
            c,
            obs,
            cfg,
            deadline_for(&order, i, cfg),
            plan.features.hsf_split.as_deref(),
            plan.features.tn.as_ref(),
        ) {
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
        let sv = reference_state(c)?;
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

fn reference_state(c: &Circuit) -> Result<StateVectorF64, SimError> {
    let mut sv = StateVectorF64::try_new(c.num_qubits)?;
    for g in c.gates() {
        sv.apply_gate(g)?;
    }
    Ok(sv)
}

/// Plans and runs `<Z_obs>` of `c|0^n>`.
pub fn expectation(c: &Circuit, obs: &[usize], cfg: &PlannerConfig) -> Result<Execution, SimError> {
    let p = plan(c, &PlanRequest::Expectation(obs.to_vec()), cfg)?;
    execute_expectation(&p, c, obs, cfg)
}

/// An engine's final state of `c|0^n>`, ready for read-out.
pub enum Prepared {
    /// Dense state vector.
    Sv(StateVectorF64),
    /// Sparse state.
    Sparse(SparseState),
    /// Matrix product state.
    Mps(Mps),
    /// The HSF set-up (partition, segments); the full output is computed
    /// on the first sampling call and kept.
    Hsf(Box<HybridSchrodingerFeynman>, Option<StateVectorF64>),
    /// The compressed state, turned into its sampler on the first
    /// sampling call.
    Compressed(Option<Box<CompressedState>>, Option<Box<Sampler>>),
    /// Stabilizer tableau.
    Tableau(Box<Tableau>),
    /// Tensor-network contraction (amplitudes only; no state).
    Tn(Box<TnPrepared>),
}

/// Evolves `c|0^n>` on `e`; `Ok(None)` if the deadline (seconds) or the
/// memory budget was hit (MPS, sparse) or the MPS truncated.
pub fn prepare(
    e: Engine,
    c: &Circuit,
    cfg: &PlannerConfig,
    deadline: Option<f64>,
) -> Result<Option<Prepared>, SimError> {
    prepare_split(e, c, cfg, deadline, None)
}

/// [`prepare`], with the HSF partition the plan priced (`None`: the
/// engine's Kernighan–Lin partition).
pub fn prepare_split(
    e: Engine,
    c: &Circuit,
    cfg: &PlannerConfig,
    deadline: Option<f64>,
    split: Option<&[bool]>,
) -> Result<Option<Prepared>, SimError> {
    let t0 = Instant::now();
    let over = |t0: &Instant| deadline.is_some_and(|d| t0.elapsed().as_secs_f64() > d);
    let n = c.num_qubits;
    let too_large = |what: &'static str, bytes: u128| SimError::TooLarge {
        what,
        bytes,
        limit: cfg.mem_bytes,
    };
    Ok(Some(match e {
        Engine::StateVector => {
            let bytes = 16u128 << n.min(120);
            if bytes > cfg.mem_bytes {
                return Err(too_large("state vector", bytes));
            }
            let mut sv = StateVectorF64::try_new(n)?;
            sv.apply_circuit_blocked(c, &BlockConfig::default())?;
            Prepared::Sv(sv)
        }
        Engine::Sparse => {
            if n > 64 {
                return Ok(None);
            }
            let max_nnz = (cfg.mem_bytes / 48) as usize;
            let mut s = SparseState::new(n);
            for g in c.gates() {
                s.apply_gate(g)?;
                if s.nnz() > max_nnz || over(&t0) {
                    return Ok(None);
                }
            }
            Prepared::Sparse(s)
        }
        Engine::Mps => {
            let mut m = Mps::new(n, 1 << 20);
            for g in c.gates() {
                m.apply_gate(g)?;
                if m.bytes() as u128 > cfg.mem_bytes / 4 || over(&t0) {
                    return Ok(None);
                }
            }
            if 1.0 - m.fidelity_estimate() > 1e-10 {
                return Ok(None);
            }
            Prepared::Mps(m)
        }
        Engine::Hsf => {
            let opts = HsfOptions {
                max_bytes: cfg.mem_bytes,
                ..HsfOptions::default()
            };
            let h = match split {
                Some(sp) => HybridSchrodingerFeynman::new(c, sp, opts)?,
                None => HybridSchrodingerFeynman::auto(c, opts)?,
            };
            Prepared::Hsf(Box::new(h), None)
        }
        Engine::Compressed => {
            let max_d = ((cfg.mem_bytes / 16).max(1).ilog2() as usize).min(30);
            Prepared::Compressed(Some(Box::new(CompressedState::new(c, max_d)?)), None)
        }
        Engine::Tableau => {
            let mut t = Tableau::try_new(n)?;
            for g in c.gates() {
                t.apply_gate(g)?;
            }
            Prepared::Tableau(Box::new(t))
        }
        Engine::Zero => {
            return Err(SimError::NotSupported {
                what: "planner: the Zero engine has no state",
            })
        }
        Engine::Tn => Prepared::Tn(Box::new(TnPrepared {
            circuit: c.clone(),
            path: None,
            mem_bytes: cfg.mem_bytes,
        })),
    }))
}

fn not_indexed(what: &'static str) -> SimError {
    SimError::NotSupported { what }
}

impl Prepared {
    /// One-off read-out preparation, done by the first sampling call
    /// anyway: MPS canonical form, HSF full output, compressed sampler.
    pub fn prepare_sampling(&mut self) -> Result<(), SimError> {
        match self {
            Prepared::Mps(m) => {
                m.canonicalize();
            }
            Prepared::Hsf(h, full) => {
                if full.is_none() {
                    *full = Some(StateVectorF64::from_amplitudes(h.state_vector()?));
                }
            }
            Prepared::Compressed(st, sm) => {
                if sm.is_none() {
                    let s = st.take().expect("compressed state");
                    *sm = Some(Box::new(s.sampler()));
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// `shots` samples of all qubits (bit `q` = qubit `q`, `n ≤ 128`);
    /// `Ok(None)` if `deadline` (an instant and seconds) passed (MPS).
    pub fn samples<R: Rng + ?Sized>(
        &mut self,
        shots: usize,
        rng: &mut R,
        deadline: Option<(Instant, f64)>,
    ) -> Result<Option<Vec<u128>>, SimError> {
        self.prepare_sampling()?;
        Ok(Some(match self {
            Prepared::Sv(sv) => sv
                .sample(shots, rng)
                .into_iter()
                .map(|x| x as u128)
                .collect(),
            Prepared::Sparse(s) => s.sample(shots, rng).into_iter().map(u128::from).collect(),
            Prepared::Mps(m) => {
                if m.num_qubits() > 128 {
                    return Err(not_indexed("planner samples need n <= 128"));
                }
                let mut out = Vec::with_capacity(shots);
                while out.len() < shots {
                    let k = (shots - out.len()).min(256);
                    out.extend(m.sample(k, rng));
                    if deadline.is_some_and(|(t, d)| t.elapsed().as_secs_f64() > d) {
                        return Ok(None);
                    }
                }
                out
            }
            Prepared::Hsf(_, full) => full
                .as_ref()
                .expect("prepared")
                .sample(shots, rng)
                .into_iter()
                .map(|x| x as u128)
                .collect(),
            Prepared::Compressed(_, sm) => {
                let sm = sm.as_ref().expect("prepared");
                if sm.num_qubits() > 128 {
                    return Err(not_indexed("planner samples need n <= 128"));
                }
                (0..shots)
                    .map(|_| {
                        let b = sm.sample_packed(rng);
                        u128::from(b[0]) | (u128::from(*b.get(1).unwrap_or(&0)) << 64)
                    })
                    .collect()
            }
            Prepared::Tn(_) => {
                return Err(not_indexed(
                    "planner: the tensor-network engine computes amplitudes, not samples",
                ))
            }
            Prepared::Tableau(t) => {
                if t.num_qubits() > 128 {
                    return Err(not_indexed("planner samples need n <= 128"));
                }
                t.sample(shots, rng)
                    .into_iter()
                    .map(|b| {
                        b.iter()
                            .enumerate()
                            .fold(0u128, |a, (q, &v)| a | (u128::from(v) << q))
                    })
                    .collect()
            }
        }))
    }

    /// Exact amplitudes `<x|ψ>` with the global phase.
    pub fn amplitudes(&mut self, xs: &[u128]) -> Result<Vec<Complex64>, SimError> {
        Ok(match self {
            Prepared::Sv(sv) => xs.iter().map(|&x| sv.amplitude(x as usize)).collect(),
            Prepared::Sparse(s) => xs.iter().map(|&x| s.amplitude(x as u64)).collect(),
            Prepared::Mps(m) => xs.iter().map(|&x| m.amplitude(x)).collect(),
            Prepared::Hsf(h, full) => match full {
                Some(sv) => xs.iter().map(|&x| sv.amplitude(x as usize)).collect(),
                None => {
                    let ix: Vec<usize> = xs.iter().map(|&x| x as usize).collect();
                    h.amplitudes(&ix)?
                }
            },
            Prepared::Tn(t) => {
                let n = t.circuit.num_qubits;
                xs.iter()
                    .map(|&x| {
                        tn_contract(
                            &t.circuit,
                            &tn::bits_of(x, n),
                            t.path.as_deref(),
                            t.mem_bytes,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
            Prepared::Compressed(..) | Prepared::Tableau(_) => {
                return Err(not_indexed(
                    "planner: compressed state / tableau amplitudes drop the global phase",
                ))
            }
        })
    }
}

/// What [`execute_samples`] did.
#[derive(Clone, Debug)]
pub struct SampleExecution {
    /// One basis index per shot (bit `q` = qubit `q`).
    pub samples: Vec<u128>,
    /// The engine that produced the samples.
    pub engine: Engine,
    /// Engines aborted on the way (speculation or budget errors).
    pub aborted: Vec<Engine>,
    /// Wall-clock seconds of execution (planning excluded).
    pub secs: f64,
    /// Seconds spent planning.
    pub plan_secs: f64,
}

/// Runs a [`Plan`] for `shots` samples of all qubits of `c|0^n>` (`n ≤
/// 128`). Speculative engines that are aborted fall back to the next one;
/// the samples then come from a different (equally exact) engine, so with
/// speculation on, a seeded `rng` reproduces the samples only when no
/// abort happens.
pub fn execute_samples<R: Rng + ?Sized>(
    plan: &Plan,
    c: &Circuit,
    shots: usize,
    rng: &mut R,
    cfg: &PlannerConfig,
) -> Result<SampleExecution, SimError> {
    let t0 = Instant::now();
    if c.num_qubits > 128 {
        return Err(not_indexed("planner samples need n <= 128"));
    }
    let order = order_of(plan);
    let mut aborted = Vec::new();
    for i in 0..order.len() {
        let e = order[i].0;
        if e == Engine::Zero {
            continue;
        }
        let dl = deadline_for(&order, i, cfg);
        let ts = Instant::now();
        let r =
            prepare_split(e, c, cfg, dl, plan.features.hsf_split.as_deref()).and_then(
                |p| match p {
                    None => Ok(None),
                    Some(mut p) => p.samples(shots, rng, dl.map(|d| (ts, d))),
                },
            );
        match r {
            Ok(Some(samples)) => {
                return Ok(SampleExecution {
                    samples,
                    engine: e,
                    aborted,
                    secs: t0.elapsed().as_secs_f64(),
                    plan_secs: plan.plan_secs,
                })
            }
            Ok(None) => aborted.push(e),
            Err(SimError::TooLarge { .. } | SimError::NotSupported { .. })
                if i + 1 < order.len() =>
            {
                aborted.push(e)
            }
            Err(err) => return Err(err),
        }
    }
    Err(SimError::TooLarge {
        what: "planner: every engine aborted",
        bytes: 0,
        limit: cfg.mem_bytes,
    })
}

/// What [`execute_amplitudes`] did.
#[derive(Clone, Debug)]
pub struct AmplitudeExecution {
    /// One amplitude per requested basis state, in request order.
    pub amplitudes: Vec<Complex64>,
    /// The engine that produced the amplitudes.
    pub engine: Engine,
    /// Engines aborted on the way (speculation or budget errors).
    pub aborted: Vec<Engine>,
    /// Wall-clock seconds of execution (planning excluded).
    pub secs: f64,
    /// Seconds spent planning.
    pub plan_secs: f64,
    /// Largest deviation from the reference state vector (debug mode).
    pub max_err: Option<f64>,
}

/// Runs a [`Plan`] for the amplitudes `<x|ψ>` (global phase included).
pub fn execute_amplitudes(
    plan: &Plan,
    c: &Circuit,
    xs: &[u128],
    cfg: &PlannerConfig,
) -> Result<AmplitudeExecution, SimError> {
    let t0 = Instant::now();
    let order = order_of(plan);
    let mut aborted = Vec::new();
    let mut out = None;
    for i in 0..order.len() {
        let e = order[i].0;
        if matches!(e, Engine::Zero | Engine::Tableau | Engine::Compressed) {
            continue;
        }
        let r = prepare_split(
            e,
            c,
            cfg,
            deadline_for(&order, i, cfg),
            plan.features.hsf_split.as_deref(),
        )
        .map(|p| match p {
            Some(Prepared::Tn(mut t)) => {
                t.path = plan.features.tn.as_ref().map(|f| f.path.clone());
                Some(Prepared::Tn(t))
            }
            other => other,
        })
        .and_then(|p| p.map(|mut p| p.amplitudes(xs)).transpose());
        match r {
            Ok(Some(a)) => {
                out = Some((a, e));
                break;
            }
            Ok(None) => aborted.push(e),
            Err(SimError::TooLarge { .. }) if i + 1 < order.len() => aborted.push(e),
            Err(err) => return Err(err),
        }
    }
    let (amplitudes, engine) = out.ok_or(SimError::TooLarge {
        what: "planner: every engine aborted",
        bytes: 0,
        limit: cfg.mem_bytes,
    })?;
    let secs = t0.elapsed().as_secs_f64();
    let mut max_err = None;
    if cfg.debug_reference && c.num_qubits <= cfg.debug_max_qubits {
        let sv = reference_state(c)?;
        let err = xs
            .iter()
            .zip(&amplitudes)
            .map(|(&x, a)| (sv.amplitude(x as usize) - a).norm())
            .fold(0.0, f64::max);
        assert!(
            err <= cfg.debug_tol,
            "planner debug: {engine:?} amplitudes differ by {err} (plan {:?})",
            plan.ranked
        );
        max_err = Some(err);
    }
    Ok(AmplitudeExecution {
        amplitudes,
        engine,
        aborted,
        secs,
        plan_secs: plan.plan_secs,
        max_err,
    })
}

/// Plans and draws `shots` samples of every qubit of `c|0^n>`.
pub fn samples<R: Rng + ?Sized>(
    c: &Circuit,
    shots: usize,
    rng: &mut R,
    cfg: &PlannerConfig,
) -> Result<SampleExecution, SimError> {
    let p = plan(c, &PlanRequest::Samples(shots), cfg)?;
    execute_samples(&p, c, shots, rng, cfg)
}

/// Plans and computes the amplitudes `<x|ψ>` of `c|0^n>`.
pub fn amplitudes(
    c: &Circuit,
    xs: &[u128],
    cfg: &PlannerConfig,
) -> Result<AmplitudeExecution, SimError> {
    let p = plan(c, &PlanRequest::Amplitudes(xs.len()), cfg)?;
    execute_amplitudes(&p, c, xs, cfg)
}
