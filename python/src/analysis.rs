//! Native half of `qsimlab.analysis` (phase 2).
//!
//! Bindings over the engine's structural analyses:
//!
//! * [`magic_profile`]: the magic atlas (`qsim_lab::magic_atlas::profile`):
//!   active dimension `d_k`, factored dimension `f_k`, stabilizer
//!   entanglement of the Clifford skeleton, affine support bound;
//! * [`state_magic`]: stabilizer nullity and stabilizer 2-Rényi entropy of a
//!   dense state (all `4^n` Pauli expectations, small `n`);
//! * [`branching_rank`]: the branching-rank simulator (`stab_rank`): the
//!   number of stabilizer terms after every gate;
//! * [`features`]: the simulability features (`simulability::features`);
//! * [`gaussian`], [`gaussian_z`]: the free-fermion detector report and the
//!   Gaussian engine's Z read-outs (`engines::gaussian`);
//! * [`monitored`]: exact monitored Clifford+T simulation with mid-circuit
//!   measurements (`monitored`): `d(t)`, Born probabilities, cut entropies.

use crate::circuit::PyCircuit;
use crate::errors::{map_sim_err, qerr_with, unsupported, value_err};
use crate::threads::heavy;
use num_complex::Complex64;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use qsim_lab::circuit::{Circuit, Op};
use qsim_lab::engines::gaussian::{
    detect as gaussian_detect, DetectOptions, GaussianOptions, GaussianReport, GaussianState,
    InteractionPolicy, Ordering,
};
use qsim_lab::engines::monitored::{ent, Cliff2, Mode, Monitored};
use qsim_lab::engines::stab_rank::RankState;
use qsim_lab::gate::Gate;
use qsim_lab::magic_atlas::{self, AtlasOptions};
use qsim_lab::simulability;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::FRAC_PI_2;
use std::sync::OnceLock;

/// The unitary part of a circuit (terminal measurements dropped); anything
/// else that is not a gate is refused.
fn unitary(circuit: &PyCircuit, what: &str) -> PyResult<Circuit> {
    let d = circuit.snapshot();
    let c = d.without_terminal_measurements();
    if let Some(op) = c.ops.iter().find(|o| !matches!(o, Op::Gate(_))) {
        return Err(unsupported(format!(
            "{what} needs a unitary circuit (terminal measurements are ignored); found {op:?}"
        )));
    }
    Ok(c)
}

// ---------------------------------------------------------------------------
// magic atlas

#[pyfunction]
#[pyo3(signature = (circuit, checkpoints=64, entanglement=true, cut=None, support=true, threads=None))]
fn magic_profile<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    checkpoints: usize,
    entanglement: bool,
    cut: Option<usize>,
    support: bool,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let c = unitary(circuit, "magic_profile")?;
    if let Some(k) = cut {
        if k > c.num_qubits {
            return Err(value_err(format!(
                "cut must be in [0, {}], got {k}",
                c.num_qubits
            )));
        }
    }
    let opts = AtlasOptions {
        checkpoints,
        entanglement,
        cut,
        support,
    };
    let p = heavy(py, threads, move || magic_atlas::profile(&c, &opts)).map_err(map_sim_err)?;
    let d = PyDict::new(py);
    d.set_item("num_qubits", p.n)?;
    d.set_item("gates", p.gates)?;
    d.set_item("lowered_gates", p.lowered)?;
    d.set_item("two_qubit_gates", p.two_qubit)?;
    d.set_item("toffolis", p.toffolis)?;
    d.set_item("rotations", p.rotations)?;
    d.set_item("t_count", p.t_count)?;
    d.set_item("d", p.d)?;
    d.set_item("f", p.f)?;
    d.set_item("log2_work", p.log2_work)?;
    d.set_item("log2_work_factored", p.log2_work_f)?;
    d.set_item("cut", p.cut)?;
    d.set_item("e_stab_max", p.e_stab_max)?;
    d.set_item("e_bound_max", p.e_bound_max)?;
    d.set_item("support_log2", p.support)?;
    d.set_item("seconds", p.secs)?;
    d.set_item("d_profile", PyArray1::from_vec(py, p.d_prof))?;
    d.set_item("f_profile", PyArray1::from_vec(py, p.f_prof))?;
    d.set_item("rotation_gate", PyArray1::from_vec(py, p.rot_gate))?;
    let cps = PyList::empty(py);
    for cp in &p.checkpoints {
        cps.append((cp.gate, cp.rotations, cp.t_count, cp.d, cp.f, cp.e_stab))?;
    }
    d.set_item("checkpoints", cps)?;
    Ok(d)
}

/// `(nullity, M2)` of a normalised dense state.
#[pyfunction]
#[pyo3(signature = (state, threads=None))]
fn state_magic(
    py: Python<'_>,
    state: PyReadonlyArray1<'_, Complex64>,
    threads: Option<usize>,
) -> PyResult<(f64, f64)> {
    let v = state.as_slice()?.to_vec();
    let len = v.len();
    if len == 0 || !len.is_power_of_two() {
        return Err(value_err(format!(
            "the state must have 2^n entries, got {len}"
        )));
    }
    let n = len.trailing_zeros();
    if n > 13 {
        return Err(value_err(format!(
            "stabilizer magic enumerates all 4^n Pauli expectations; n = {n} > 13"
        )));
    }
    let norm: f64 = v.iter().map(|a| a.norm_sqr()).sum();
    if (norm - 1.0).abs() > 1e-8 {
        return Err(value_err(format!(
            "the state is not normalised (‖ψ‖² = {norm})"
        )));
    }
    let m = heavy(py, threads, move || magic_atlas::state_magic(&v));
    Ok((m.nullity, m.m2))
}

// ---------------------------------------------------------------------------
// branching rank

#[pyfunction]
#[pyo3(signature = (circuit, max_terms=1usize << 16, pair_merge=6, state=false, threads=None))]
fn branching_rank<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    max_terms: usize,
    pair_merge: usize,
    state: bool,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let c = unitary(circuit, "branching_rank")?;
    let n = c.num_qubits;
    if state && n > 20 {
        return Err(value_err(format!(
            "state=True builds a dense vector; n = {n} > 20"
        )));
    }
    let (rs, ok, sv) = heavy(py, threads, move || {
        let mut rs = RankState::new(n);
        rs.max_terms = max_terms.max(1);
        rs.pair_merge_s = pair_merge;
        let ok = rs.run(&c);
        let sv = (state && ok).then(|| rs.to_statevector());
        (rs, ok, sv)
    });
    let st = &rs.stats;
    let d = PyDict::new(py);
    d.set_item("rank", rs.rank())?;
    d.set_item("max_rank", st.max_r)?;
    d.set_item("overflow", !ok)?;
    d.set_item(
        "trace",
        PyArray1::from_vec(py, st.r.iter().map(|&x| x as u64).collect::<Vec<u64>>()),
    )?;
    d.set_item("branch_events", st.branch_events)?;
    d.set_item("clifford_events", st.clifford_events)?;
    d.set_item("diagonal_events", st.diag_events)?;
    d.set_item("merges", st.merges)?;
    d.set_item("pair_merges", st.pair_merges)?;
    d.set_item("cancellations", st.cancellations)?;
    match sv {
        Some(v) => d.set_item("state", PyArray1::from_vec(py, v))?,
        None => d.set_item("state", py.None())?,
    }
    Ok(d)
}

// ---------------------------------------------------------------------------
// simulability features

#[pyfunction]
#[pyo3(signature = (circuit, hsf=true, threads=None))]
fn features<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    hsf: bool,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let c = unitary(circuit, "simulability")?;
    let f = heavy(py, threads, move || simulability::features(&c, hsf)).map_err(map_sim_err)?;
    let d = PyDict::new(py);
    macro_rules! put {
        ($($k:ident),*) => {$( d.set_item(stringify!($k), f.$k)?; )*};
    }
    put!(
        n,
        gates,
        g2,
        g3,
        depth2,
        t_count,
        rotations,
        d,
        dense_l,
        redundant,
        frame_l,
        obs_zero,
        chi_bits,
        mps_l,
        chi_bits0,
        mps_l0,
        hsf_k,
        hsf_na,
        hsf_nb,
        hsf_l,
        hsf_keff,
        hsf_l0,
        sup,
        sparse_l,
        sv_l,
        secs,
        gauss_fraction,
        gauss_residual,
        gauss_interaction,
        gauss_exact
    );
    Ok(d)
}

// ---------------------------------------------------------------------------
// free fermions

fn detect_opts(tol: f64, relabel_swaps: bool, reorder: bool) -> PyResult<DetectOptions> {
    if !(tol >= 0.0) {
        return Err(value_err(format!("tol must be >= 0, got {tol}")));
    }
    Ok(DetectOptions {
        tol,
        relabel_swaps,
        reorder,
    })
}

fn report_dict<'py>(py: Python<'py>, r: &GaussianReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("num_qubits", r.n)?;
    d.set_item("blocks", r.blocks)?;
    d.set_item("blocks_2q", r.blocks_2q)?;
    d.set_item("gaussian_blocks", r.gaussian_blocks)?;
    d.set_item("gaussian_fraction", r.gaussian_fraction)?;
    d.set_item("max_residual", r.max_residual)?;
    d.set_item("non_gaussian", r.non_gaussian)?;
    d.set_item("nonadjacent", r.nonadjacent)?;
    let ints: Vec<(usize, (usize, usize), (usize, usize), f64)> = r
        .interactions
        .iter()
        .map(|i| (i.block, i.wires, i.modes, i.g))
        .collect();
    d.set_item("interactions", ints)?;
    d.set_item("interaction_total", r.interaction_total)?;
    d.set_item("interaction_max", r.interaction_max)?;
    d.set_item("swaps_relabelled", r.swaps_relabelled)?;
    d.set_item(
        "ordering",
        match r.ordering {
            Some(Ordering::Identity) | None => "identity",
            Some(Ordering::Paths) => "paths",
            Some(Ordering::GreedyCover) => "greedy_cover",
        },
    )?;
    d.set_item("order", r.order.clone())?;
    d.set_item("paths", r.paths.clone())?;
    d.set_item("mode_of_qubit", r.mode_of_qubit.clone())?;
    d.set_item("number_conserving", r.number_conserving)?;
    d.set_item("free", r.free)?;
    d.set_item("exact", r.exact)?;
    d.set_item("seconds", r.secs)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (circuit, tol=1e-10, relabel_swaps=true, reorder=true, threads=None))]
fn gaussian<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    tol: f64,
    relabel_swaps: bool,
    reorder: bool,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let c = unitary(circuit, "gaussian")?;
    let opts = detect_opts(tol, relabel_swaps, reorder)?;
    let r = heavy(py, threads, move || {
        Ok::<_, qsim_lab::SimError>(gaussian_detect(&c, &opts))
    })
    .map_err(map_sim_err)?;
    report_dict(py, &r)
}

/// `<Z_q>` of every qubit and `<Z_i Z_j>` of the given pairs on the
/// Gaussian engine; `drop_interactions` sets the interaction phases to 0.
#[pyfunction]
#[pyo3(signature = (circuit, pairs=None, drop_interactions=false, tol=1e-10, threads=None))]
fn gaussian_z<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    pairs: Option<Vec<(usize, usize)>>,
    drop_interactions: bool,
    tol: f64,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let c = unitary(circuit, "gaussian_expectations")?;
    let n = c.num_qubits;
    let pairs = pairs.unwrap_or_default();
    for &(i, j) in &pairs {
        if i >= n || j >= n {
            return Err(value_err(format!(
                "pair ({i}, {j}) out of range for {n} qubits"
            )));
        }
    }
    let opts = GaussianOptions {
        detect: detect_opts(tol, true, true)?,
        interactions: if drop_interactions {
            InteractionPolicy::Drop
        } else {
            InteractionPolicy::Refuse
        },
        ..Default::default()
    };
    let (z, zz, r) = heavy(py, threads, move || {
        let (st, r) = GaussianState::from_circuit(&c, &opts)?;
        let z: Vec<f64> = (0..n)
            .map(|q| st.expectation_z(q))
            .collect::<Result<_, _>>()?;
        let zz: Vec<f64> = pairs
            .iter()
            .map(|&(i, j)| st.z_correlation(i, j))
            .collect::<Result<_, _>>()?;
        Ok::<_, qsim_lab::SimError>((z, zz, r))
    })
    .map_err(map_sim_err)?;
    let d = PyDict::new(py);
    d.set_item("z", PyArray1::from_vec(py, z))?;
    d.set_item("zz", PyArray1::from_vec(py, zz))?;
    d.set_item("report", report_dict(py, &r)?)?;
    Ok(d)
}

// ---------------------------------------------------------------------------
// monitored Clifford+T

fn cnot_elem() -> &'static Cliff2 {
    static C: OnceLock<Cliff2> = OnceLock::new();
    C.get_or_init(|| {
        Cliff2::group()
            .into_iter()
            .find(|c| c.word == [4u8])
            .expect("CNOT in the two-qubit Clifford group")
    })
}

struct MonRun {
    sim: Monitored,
    /// Physical Clifford gates applied to the frame, in time order.
    phys: Vec<Gate>,
    rng: StdRng,
}

enum MonErr {
    TooLarge(usize),
    Unsupported(String),
}

impl MonRun {
    fn h(&mut self, q: usize) {
        self.sim.cliff1(q, 0);
        self.phys.push(Gate::H(q));
    }
    fn s(&mut self, q: usize, k: usize) {
        for _ in 0..k % 4 {
            self.sim.cliff1(q, 2);
            self.phys.push(Gate::S(q));
        }
    }
    fn cx(&mut self, a: usize, b: usize) {
        self.sim.cliff2(cnot_elem(), a, b);
        self.phys.push(Gate::Cnot(a, b));
    }
    fn x(&mut self, q: usize) {
        self.h(q);
        self.s(q, 2);
        self.h(q);
    }
    fn rz(&mut self, q: usize, theta: f64) -> Result<(), MonErr> {
        let k = theta / FRAC_PI_2;
        if (k - k.round()).abs() < 1e-12 {
            self.s(q, k.round().rem_euclid(4.0) as usize);
            return Ok(());
        }
        self.sim.rz(q, theta).map_err(|e| MonErr::TooLarge(e.d))
    }

    fn gate(&mut self, g: &Gate) -> Result<(), MonErr> {
        use Gate::*;
        match *g {
            I(_) => {}
            H(q) => self.h(q),
            S(q) => self.s(q, 1),
            Sdg(q) => self.s(q, 3),
            Z(q) => self.s(q, 2),
            X(q) => self.x(q),
            Y(q) => {
                // Y = i X Z
                self.s(q, 2);
                self.x(q);
            }
            Cnot(a, b) => self.cx(a, b),
            Cz(a, b) => {
                self.h(b);
                self.cx(a, b);
                self.h(b);
            }
            Swap(a, b) => {
                self.cx(a, b);
                self.cx(b, a);
                self.cx(a, b);
            }
            T(q) => self.rz(q, std::f64::consts::FRAC_PI_4)?,
            Tdg(q) => self.rz(q, -std::f64::consts::FRAC_PI_4)?,
            Rz(q, t) | Phase(q, t) => self.rz(q, t)?,
            ref other => {
                let d = other.decompose_to_clifford_rz();
                if d.len() == 1 && d[0] == *other {
                    return Err(MonErr::Unsupported(format!("gate {other:?}")));
                }
                for h in &d {
                    self.gate(h)?;
                }
            }
        }
        Ok(())
    }

    fn pauli(&mut self, q: usize, which: u8) -> Result<(), MonErr> {
        match which {
            1 => self.gate(&Gate::X(q)),
            2 => self.gate(&Gate::Y(q)),
            3 => self.gate(&Gate::Z(q)),
            _ => Ok(()),
        }
    }

    fn op(&mut self, op: &Op) -> Result<(), MonErr> {
        match *op {
            Op::Gate(ref g) => self.gate(g)?,
            Op::Measure(q) => {
                self.sim.measure(q, &mut self.rng, None);
            }
            Op::Reset(q) => {
                let r = self.sim.measure(q, &mut self.rng, None);
                // the reset's measurement is not a record of the circuit
                self.sim.records.pop();
                if r.outcome {
                    self.x(q);
                }
            }
            Op::ClassicControlled {
                ref gate,
                meas_index,
                target_value,
            } => {
                let rec = self.sim.records.get(meas_index).ok_or_else(|| {
                    MonErr::Unsupported(format!(
                        "c_if on measurement {meas_index}, which has not happened yet"
                    ))
                })?;
                if rec.outcome == target_value {
                    self.gate(gate)?;
                }
            }
            Op::XFlip(q, p) => {
                if self.rng.random::<f64>() < p {
                    self.pauli(q, 1)?;
                }
            }
            Op::YFlip(q, p) => {
                if self.rng.random::<f64>() < p {
                    self.pauli(q, 2)?;
                }
            }
            Op::ZFlip(q, p) => {
                if self.rng.random::<f64>() < p {
                    self.pauli(q, 3)?;
                }
            }
            Op::Depolarize1q(q, p) => {
                if self.rng.random::<f64>() < p {
                    let w = self.rng.random_range(1..4u8);
                    self.pauli(q, w)?;
                }
            }
            Op::Depolarize2q(a, b, p) => {
                if self.rng.random::<f64>() < p {
                    let w = self.rng.random_range(1..16u8);
                    self.pauli(a, w & 3)?;
                    self.pauli(b, w >> 2)?;
                }
            }
        }
        Ok(())
    }
}

/// `(lower, upper, s2)` per cut, at one op index.
type CutEntropies = (usize, Vec<(f64, f64, Option<f64>)>);

struct MonOut {
    d: Vec<u32>,
    records: Vec<(usize, bool, f64, u8)>,
    /// (op index, per cut (lower, upper, s2))
    ent: Vec<CutEntropies>,
    stats: qsim_lab::engines::monitored::MonStats,
    state: Option<Vec<Complex64>>,
    final_d: usize,
}

#[pyfunction]
#[pyo3(signature = (circuit, seed=0, exact=true, max_d=24, cuts=None, entropy_every=0,
                    max_cost_log2=24, state=false, threads=None))]
#[allow(clippy::too_many_arguments)]
fn monitored<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    seed: u64,
    exact: bool,
    max_d: usize,
    cuts: Option<Vec<Vec<usize>>>,
    entropy_every: usize,
    max_cost_log2: u32,
    state: bool,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = circuit.snapshot();
    let c = data.circuit.clone();
    let n = c.num_qubits;
    if data.readout_error > 0.0 {
        return Err(unsupported(
            "monitored() does not model readout_error; use X_ERROR before the measurement",
        ));
    }
    if state && (!exact || n > 20) {
        return Err(value_err(
            "state=True needs exact=True and at most 20 qubits",
        ));
    }
    if max_d > 34 {
        return Err(value_err(format!("max_d must be ≤ 34, got {max_d}")));
    }
    let cuts: Vec<Vec<bool>> = cuts
        .unwrap_or_else(|| vec![(0..n / 2).collect()])
        .into_iter()
        .map(|qs| {
            let mut r = vec![false; n];
            for q in qs {
                if q >= n {
                    return Err(value_err(format!("cut qubit {q} out of range")));
                }
                r[q] = true;
            }
            Ok(r)
        })
        .collect::<PyResult<_>>()?;
    let res: Result<MonOut, MonErr> = heavy(py, threads, move || {
        let mode = if exact {
            Mode::Exact
        } else {
            Mode::DimensionOnly
        };
        let mut sim = Monitored::new(n, mode, max_d);
        if state {
            sim = sim.with_log();
        }
        let mut run = MonRun {
            sim,
            phys: Vec::new(),
            rng: StdRng::seed_from_u64(seed),
        };
        let mut d = Vec::with_capacity(c.ops.len());
        let mut entv = Vec::new();
        let entropies = |sim: &Monitored| {
            cuts.iter()
                .map(|r| {
                    let e = ent::cut_entropy(sim, r, max_cost_log2);
                    (e.lower, e.upper, e.s2)
                })
                .collect::<Vec<_>>()
        };
        let nops = c.ops.len();
        for (k, op) in c.ops.iter().enumerate() {
            run.op(op)?;
            d.push(run.sim.d() as u32);
            if entropy_every > 0 && (k + 1) % entropy_every == 0 && k + 1 < nops {
                entv.push((k, entropies(&run.sim)));
            }
        }
        entv.push((nops.saturating_sub(1), entropies(&run.sim)));
        let st = if state {
            Some(run.sim.to_statevector(&run.phys).amplitudes().to_vec())
        } else {
            None
        };
        Ok(MonOut {
            d,
            records: run
                .sim
                .records
                .iter()
                .map(|r| (r.qubit, r.outcome, r.prob, r.kind))
                .collect(),
            ent: entv,
            stats: run.sim.stats.clone(),
            state: st,
            final_d: run.sim.d(),
        })
    });
    let out = match res {
        Ok(o) => o,
        Err(MonErr::TooLarge(dd)) => {
            return Err(qerr_with(
                "ResourceLimitError",
                format!(
                    "the active register would grow to d = {dd} qubits (> max_d = {max_d}); \
                     raise max_d (memory 16·2^d bytes) or use exact=False for d(t) only"
                ),
                &[
                    ("needed", 16u128 << dd.min(100)),
                    ("limit", 16u128 << max_d),
                ],
            ))
        }
        Err(MonErr::Unsupported(s)) => return Err(unsupported(format!("monitored(): {s}"))),
    };
    let d = PyDict::new(py);
    d.set_item("d", PyArray1::from_vec(py, out.d))?;
    d.set_item("final_d", out.final_d)?;
    d.set_item(
        "qubits",
        PyArray1::from_vec(
            py,
            out.records.iter().map(|r| r.0 as u64).collect::<Vec<u64>>(),
        ),
    )?;
    d.set_item(
        "outcomes",
        PyArray1::from_vec(
            py,
            out.records.iter().map(|r| r.1 as u8).collect::<Vec<u8>>(),
        ),
    )?;
    d.set_item(
        "probabilities",
        PyArray1::from_vec(py, out.records.iter().map(|r| r.2).collect::<Vec<f64>>()),
    )?;
    d.set_item(
        "kinds",
        PyArray1::from_vec(py, out.records.iter().map(|r| r.3).collect::<Vec<u8>>()),
    )?;
    let el = PyList::empty(py);
    for (k, v) in &out.ent {
        el.append((*k, v.clone()))?;
    }
    d.set_item("entropies", el)?;
    let s = &out.stats;
    let sd = PyDict::new(py);
    sd.set_item("t_gates", s.t_gates)?;
    sd.set_item("t_activating", s.t_activating)?;
    sd.set_item("t_register", s.t_register)?;
    sd.set_item("measurements", s.meas)?;
    sd.set_item("meas_frame", s.meas_frame)?;
    sd.set_item("meas_register", s.meas_register)?;
    sd.set_item("meas_determined", s.meas_determined)?;
    sd.set_item("max_d", s.max_d)?;
    sd.set_item("element_ops", s.element_ops)?;
    d.set_item("stats", sd)?;
    match out.state {
        Some(v) => d.set_item("state", PyArray1::from_vec(py, v))?,
        None => d.set_item("state", py.None())?,
    }
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "native part of qsimlab.analysis (phase 2)")?;
    m.add_function(wrap_pyfunction!(magic_profile, m)?)?;
    m.add_function(wrap_pyfunction!(state_magic, m)?)?;
    m.add_function(wrap_pyfunction!(branching_rank, m)?)?;
    m.add_function(wrap_pyfunction!(features, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_z, m)?)?;
    m.add_function(wrap_pyfunction!(monitored, m)?)?;
    Ok(())
}
