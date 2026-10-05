//! Native half of `qsimlab.qec`: memory-experiment generators, detector
//! sampling (FastSampler / SymPhase), circuit-derived detector error models,
//! BP+OSD decoding, fused logical-error-rate runs and exact circuit distance.
//!
//! Everything heavy runs with the GIL released ([`crate::threads::heavy`]).
//! Shots are produced in chunks of [`CHUNK_SHOTS`]; chunk `k` draws from a
//! wyrand stream seeded by `splitmix64(seed, k)`, so the output for a given
//! seed does not depend on the thread count.

use crate::circuit::{CircuitData, PyCircuit};
use crate::errors::{map_sim_err, qerr, unsupported, value_err};
use crate::threads::heavy;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use qsim_lab::engines::stabilizer::fast_sampler::{FastSampler, WyRand};
use qsim_lab::engines::stabilizer::symphase::{SymPhaseSampler, VarDist};
use qsim_lab::qec::bposd::{BpOsd, DecodeStats, DemMatrix};
use qsim_lab::qec::color::{ColorCode, ColorNoise, ColorSchedule, MAX_STEP};
use qsim_lab::qec::distance::min_logical;
use qsim_lab::{Circuit, Gate, Op};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// 64-shot blocks per sampling chunk (FastSampler needs a power of two).
const CHUNK_BLOCKS: usize = 16;
/// Shots per chunk (the unit of RNG streams and of parallel work).
pub const CHUNK_SHOTS: usize = 64 * CHUNK_BLOCKS;
/// Chunks per wave of a logical-error-rate run (early stop is checked between
/// waves, so the result for a seed does not depend on the thread count).
const WAVE_CHUNKS: usize = 64;

// ------------------------------------------------------------------ noise

/// Per-location error probabilities of the built-in circuit noise models.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct NoiseParams {
    /// `DEPOLARIZE2` after every two-qubit gate.
    p2: f64,
    /// `DEPOLARIZE1` on every qubit idle during a gate moment.
    idle: f64,
    /// `DEPOLARIZE1` on every qubit idle during a measure/reset moment.
    idle_mr: f64,
    /// Readout flip of every measurement.
    meas: f64,
    /// Flip after every reset (`X_ERROR` after `R`, `Z_ERROR` after `RX`).
    reset: f64,
}

fn noise_params(model: &str, p: f64) -> PyResult<NoiseParams> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        return Err(value_err(format!(
            "noise strength p = {p} is not in [0, 1]"
        )));
    }
    let np = match model {
        "cnot" => NoiseParams {
            p2: p,
            ..Default::default()
        },
        "uniform" => NoiseParams {
            p2: p,
            idle: p,
            idle_mr: p,
            meas: p,
            reset: p,
        },
        "si1000" => {
            if 5.0 * p > 1.0 {
                return Err(value_err("si1000 needs p <= 0.2 (readout flip is 5p)"));
            }
            NoiseParams {
                p2: p,
                idle: p / 10.0,
                idle_mr: 2.0 * p,
                meas: 5.0 * p,
                reset: 2.0 * p,
            }
        }
        m => {
            return Err(value_err(format!(
                "unknown noise model '{m}' (expected 'cnot', 'uniform' or 'si1000')"
            )))
        }
    };
    Ok(np)
}

/// Builds a memory circuit moment by moment, writing the noise of
/// [`NoiseParams`] as explicit ops (the readout flip is the circuit's global
/// `readout_error`). Moments have the same shape as the colour-code circuits
/// of `qsim_lab::qec::color`: reset (`R`, `RX = R H`), CNOT layers,
/// measurement (`M`, `MX = H M H`).
struct Builder {
    n: usize,
    np: NoiseParams,
    ops: Vec<Op>,
    nmeas: usize,
}

impl Builder {
    fn new(n: usize, np: NoiseParams) -> Self {
        Builder {
            n,
            np,
            ops: Vec::new(),
            nmeas: 0,
        }
    }

    fn idle(&mut self, busy: &[bool], p: f64) {
        if p > 0.0 {
            for (q, &b) in busy.iter().enumerate() {
                if !b {
                    self.ops.push(Op::Depolarize1q(q, p));
                }
            }
        }
    }

    /// `(qubit, x_basis)`.
    fn reset(&mut self, qs: &[(usize, bool)]) {
        let mut busy = vec![false; self.n];
        for &(q, x) in qs {
            self.ops.push(Op::Reset(q));
            if x {
                self.ops.push(Op::Gate(Gate::H(q)));
            }
            if self.np.reset > 0.0 {
                self.ops.push(if x {
                    Op::ZFlip(q, self.np.reset)
                } else {
                    Op::XFlip(q, self.np.reset)
                });
            }
            busy[q] = true;
        }
        self.idle(&busy, self.np.idle_mr);
    }

    /// Returns the measurement record index of each qubit, in order.
    fn measure(&mut self, qs: &[(usize, bool)]) -> Vec<usize> {
        let mut busy = vec![false; self.n];
        let mut recs = Vec::with_capacity(qs.len());
        for &(q, x) in qs {
            if x {
                self.ops.push(Op::Gate(Gate::H(q)));
            }
            self.ops.push(Op::Measure(q));
            if x {
                self.ops.push(Op::Gate(Gate::H(q)));
            }
            recs.push(self.nmeas);
            self.nmeas += 1;
            busy[q] = true;
        }
        self.idle(&busy, self.np.idle_mr);
        recs
    }

    /// One CNOT layer `(control, target)`.
    fn cx(&mut self, pairs: &[(usize, usize)]) {
        let mut busy = vec![false; self.n];
        for &(c, t) in pairs {
            self.ops.push(Op::Gate(Gate::Cnot(c, t)));
            if self.np.p2 > 0.0 {
                self.ops.push(Op::Depolarize2q(c, t, self.np.p2));
            }
            busy[c] = true;
            busy[t] = true;
        }
        self.idle(&busy, self.np.idle);
    }

    fn finish(
        self,
        detectors: Vec<Vec<usize>>,
        observables: Vec<Vec<usize>>,
        has_repeats: bool,
    ) -> CircuitData {
        CircuitData {
            circuit: Circuit {
                num_qubits: self.n,
                ops: self.ops,
            },
            global_phase: 0.0,
            readout_error: self.np.meas,
            detectors,
            observables,
            has_repeats,
        }
    }
}

/// Layout returned to Python with every generated circuit.
#[derive(Default)]
struct Layout {
    data: Vec<usize>,
    ancillas: Vec<usize>,
    flags: Vec<usize>,
    /// `(x, y)` of every qubit.
    qubit_coords: Vec<(f64, f64)>,
    /// `(x, y, t)` of every detector.
    detector_coords: Vec<(f64, f64, f64)>,
    /// `'X'` / `'Z'` per detector (the stabilizer type it checks).
    detector_basis: Vec<char>,
    /// Whether each detector is a flag-qubit detector.
    flag_detector: Vec<bool>,
}

fn layout_dict<'py>(py: Python<'py>, l: Layout) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("data_qubits", l.data)?;
    d.set_item("ancilla_qubits", l.ancillas)?;
    d.set_item("flag_qubits", l.flags)?;
    d.set_item("qubit_coords", l.qubit_coords)?;
    d.set_item("detector_coords", l.detector_coords)?;
    d.set_item(
        "detector_basis",
        l.detector_basis
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>(),
    )?;
    d.set_item("flag_detectors", l.flag_detector)?;
    Ok(d)
}

fn check_rounds(rounds: usize) -> PyResult<()> {
    if rounds == 0 {
        return Err(value_err("rounds must be >= 1"));
    }
    Ok(())
}

fn parse_basis(basis: &str) -> PyResult<bool> {
    match basis {
        "Z" | "z" => Ok(false),
        "X" | "x" => Ok(true),
        b => Err(value_err(format!("basis must be 'X' or 'Z', not '{b}'"))),
    }
}

// ---------------------------------------------------------- surface code

/// Rotated surface code memory (Stim's `surface_code:rotated_memory_*`
/// layout and hook-safe CNOT order), auxiliaries reset/measured in their
/// stabilizer's basis.
fn surface_memory(
    d: usize,
    rounds: usize,
    x_basis: bool,
    np: NoiseParams,
) -> (CircuitData, Layout) {
    // data (2x+1, 2y+1), index x*d + y
    let mut coords: Vec<(f64, f64)> = Vec::new();
    for x in 0..d {
        for y in 0..d {
            coords.push(((2 * x + 1) as f64, (2 * y + 1) as f64));
        }
    }
    let nd = d * d;
    // auxiliaries at even sites: (x, y, is_x)
    let mut anc: Vec<(i64, i64, bool)> = Vec::new();
    for x in 0..=d {
        for y in 0..=d {
            let b1 = x == 0 || x == d;
            let b2 = y == 0 || y == d;
            let parity = (x % 2) != (y % 2);
            if (b1 && parity) || (b2 && !parity) {
                continue;
            }
            anc.push((2 * x as i64, 2 * y as i64, parity));
        }
    }
    for &(x, y, _) in &anc {
        coords.push((x as f64, y as f64));
    }
    let na = anc.len();
    let n = nd + na;
    let data_at = |x: i64, y: i64| -> Option<usize> {
        let lim = 2 * d as i64 - 1;
        if x < 1 || y < 1 || x > lim || y > lim || x % 2 == 0 || y % 2 == 0 {
            None
        } else {
            Some(((x - 1) / 2) as usize * d + ((y - 1) / 2) as usize)
        }
    };
    // hook errors run across the logical they could shorten: the last two
    // CNOTs of an X auxiliary (an X-error pair on the data) are a horizontal
    // pair (the X logical is a vertical column), those of a Z auxiliary a
    // vertical pair (the Z logical is a horizontal row)
    let x_order = [(1i64, 1i64), (-1, 1), (1, -1), (-1, -1)];
    let z_order = [(1i64, 1i64), (1, -1), (-1, 1), (-1, -1)];
    let layers: Vec<Vec<(usize, usize)>> = (0..4)
        .map(|k| {
            anc.iter()
                .enumerate()
                .filter_map(|(i, &(x, y, is_x))| {
                    let (dx, dy) = if is_x { x_order[k] } else { z_order[k] };
                    data_at(x + dx, y + dy).map(|q| {
                        let a = nd + i;
                        if is_x {
                            (a, q)
                        } else {
                            (q, a)
                        }
                    })
                })
                .collect()
        })
        .collect();
    let mut b = Builder::new(n, np);
    let aq: Vec<(usize, bool)> = anc.iter().enumerate().map(|(i, a)| (nd + i, a.2)).collect();
    let init: Vec<(usize, bool)> = (0..nd).map(|q| (q, x_basis)).chain(aq.clone()).collect();
    b.reset(&init);
    let mut recs: Vec<Vec<usize>> = Vec::new();
    for r in 0..rounds {
        if r > 0 {
            b.reset(&aq);
        }
        for l in &layers {
            b.cx(l);
        }
        recs.push(b.measure(&aq));
    }
    let data_rec = b.measure(&(0..nd).map(|q| (q, x_basis)).collect::<Vec<_>>());
    let mut layout = Layout {
        data: (0..nd).collect(),
        ancillas: (nd..n).collect(),
        qubit_coords: coords,
        ..Default::default()
    };
    let mut dets = Vec::new();
    let mut push = |v: Vec<usize>, i: usize, t: usize, layout: &mut Layout| {
        dets.push(v);
        layout
            .detector_coords
            .push((anc[i].0 as f64, anc[i].1 as f64, t as f64));
        layout.detector_basis.push(if anc[i].2 { 'X' } else { 'Z' });
        layout.flag_detector.push(false);
    };
    for r in 0..rounds {
        // own type first (deterministic from round 0), then the other type
        for own in [true, false] {
            if !own && r == 0 {
                continue;
            }
            for i in 0..na {
                if (anc[i].2 == x_basis) != own {
                    continue;
                }
                let mut v = vec![recs[r][i]];
                if r > 0 {
                    v.push(recs[r - 1][i]);
                }
                push(v, i, r, &mut layout);
            }
        }
    }
    for i in 0..na {
        if anc[i].2 != x_basis {
            continue;
        }
        let (x, y, _) = anc[i];
        let mut v: Vec<usize> = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            .iter()
            .filter_map(|&(dx, dy)| data_at(x + dx, y + dy))
            .map(|q| data_rec[q])
            .collect();
        v.sort_unstable();
        v.push(recs[rounds - 1][i]);
        push(v, i, rounds, &mut layout);
    }
    // Z logical: the y = 1 row; X logical: the x = 1 column
    let obs: Vec<usize> = (0..nd)
        .filter(|&q| if x_basis { q / d == 0 } else { q % d == 0 })
        .map(|q| data_rec[q])
        .collect();
    (b.finish(dets, vec![obs], rounds > 1), layout)
}

// ------------------------------------------------------- repetition code

/// Bit-flip repetition code memory: data `0..d`, auxiliary `d + j` checks
/// `Z_j Z_{j+1}`. Logical observable: data qubit `d - 1` (Stim's choice).
fn repetition_memory(d: usize, rounds: usize, np: NoiseParams) -> (CircuitData, Layout) {
    let n = 2 * d - 1;
    let mut b = Builder::new(n, np);
    let aq: Vec<(usize, bool)> = (0..d - 1).map(|j| (d + j, false)).collect();
    b.reset(&(0..n).map(|q| (q, false)).collect::<Vec<_>>());
    let l1: Vec<(usize, usize)> = (0..d - 1).map(|j| (j, d + j)).collect();
    let l2: Vec<(usize, usize)> = (0..d - 1).map(|j| (j + 1, d + j)).collect();
    let mut recs: Vec<Vec<usize>> = Vec::new();
    for r in 0..rounds {
        if r > 0 {
            b.reset(&aq);
        }
        b.cx(&l1);
        b.cx(&l2);
        recs.push(b.measure(&aq));
    }
    let data_rec = b.measure(&(0..d).map(|q| (q, false)).collect::<Vec<_>>());
    let mut layout = Layout {
        data: (0..d).collect(),
        ancillas: (d..n).collect(),
        qubit_coords: (0..d)
            .map(|q| (2.0 * q as f64, 0.0))
            .chain((0..d - 1).map(|j| (2.0 * j as f64 + 1.0, 0.0)))
            .collect(),
        ..Default::default()
    };
    let mut dets = Vec::new();
    for r in 0..=rounds {
        for j in 0..d - 1 {
            let v = if r == rounds {
                vec![data_rec[j], data_rec[j + 1], recs[r - 1][j]]
            } else if r == 0 {
                vec![recs[0][j]]
            } else {
                vec![recs[r][j], recs[r - 1][j]]
            };
            dets.push(v);
            layout
                .detector_coords
                .push((2.0 * j as f64 + 1.0, 0.0, r as f64));
            layout.detector_basis.push('Z');
            layout.flag_detector.push(false);
        }
    }
    (
        b.finish(dets, vec![vec![data_rec[d - 1]]], rounds > 1),
        layout,
    )
}

// ------------------------------------------------------------ colour code

/// Rescales the explicit noise of a colour circuit built with
/// `ColorNoise::Uniform(p)` to SI1000: two-qubit `p`, reset `2p`, idle `p/10`
/// in CNOT moments and `2p` in measure/reset moments (readout `5p` is set by
/// the caller). Moment kinds are read off the op stream: idle `DEPOLARIZE1`
/// runs follow their moment's ops, and a qubit repeating inside one idle run
/// starts a new (empty CNOT) moment.
fn rescale_si1000(ops: &mut [Op], n: usize, p: f64) {
    let mut mr = true;
    let mut seen = vec![false; n];
    let mut in_run = false;
    for op in ops.iter_mut() {
        match op {
            Op::Depolarize1q(q, pp) => {
                if !in_run {
                    seen.iter_mut().for_each(|s| *s = false);
                    in_run = true;
                }
                if seen[*q] {
                    // an all-idle moment: only CNOT layers can be empty
                    seen.iter_mut().for_each(|s| *s = false);
                    mr = false;
                }
                seen[*q] = true;
                *pp = if mr { 2.0 * p } else { p / 10.0 };
            }
            Op::Depolarize2q(_, _, pp) => {
                *pp = p;
                mr = false;
                in_run = false;
            }
            Op::XFlip(_, pp) | Op::ZFlip(_, pp) | Op::YFlip(_, pp) => {
                *pp = 2.0 * p;
                mr = true;
                in_run = false;
            }
            Op::Gate(Gate::Cnot(..)) => {
                mr = false;
                in_run = false;
            }
            _ => {
                mr = true;
                in_run = false;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn color_memory(
    d: usize,
    rounds: usize,
    x_basis: bool,
    schedule: ScheduleSpec,
    flags: Vec<bool>,
    model: &str,
    p: f64,
) -> PyResult<(CircuitData, Layout, Vec<[u8; 6]>)> {
    if d < 3 || d % 2 == 0 {
        return Err(value_err("colour code distance must be odd and >= 3"));
    }
    if d > 41 {
        return Err(value_err("colour code distance > 41 is not supported"));
    }
    noise_params(model, p)?; // validates model and p
    let cc = ColorCode::new(d);
    let np = cc.plaquettes.len();
    let s: ColorSchedule = match schedule {
        ScheduleSpec::Named(name) => match name.as_str() {
            "kf" => cc.uniform_schedule(qsim_lab::qec::color::KF_SCHEDULE),
            "tri" => cc.uniform_schedule([qsim_lab::qec::color::TRI_OPTIMAL; 3]),
            other => return Err(value_err(format!("unknown schedule '{other}'"))),
        },
        ScheduleSpec::Explicit(rows) => {
            if rows.len() != np {
                return Err(value_err(format!(
                    "schedule has {} rows, the d = {d} colour code has {np} plaquettes",
                    rows.len()
                )));
            }
            let mut s = ColorSchedule::new();
            for (pi, r) in rows.iter().enumerate() {
                if r.len() != 6 {
                    return Err(value_err(format!(
                        "schedule row {pi} has {} entries, expected 6 (positions a..f)",
                        r.len()
                    )));
                }
                let mut a = [0u8; 6];
                for k in 0..6 {
                    a[k] = u8::try_from(r[k])
                        .map_err(|_| value_err(format!("schedule step {} too large", r[k])))?;
                }
                s.push(a);
            }
            s
        }
    };
    let bad = cc.collisions(&s);
    if !bad.is_empty() {
        return Err(value_err(format!(
            "invalid schedule ({} problems), first: {}",
            bad.len(),
            bad[0]
        )));
    }
    let flagged: Vec<bool> = if flags.is_empty() {
        vec![false; np]
    } else if flags.len() != np {
        return Err(value_err(format!(
            "flags has {} entries, the d = {d} colour code has {np} plaquettes",
            flags.len()
        )));
    } else {
        flags
    };
    for (pi, &f) in flagged.iter().enumerate() {
        if f && cc.plaquettes[pi].weight() < 4 {
            return Err(value_err(format!(
                "plaquette {pi} has weight {} < 4: nothing to flag",
                cc.plaquettes[pi].weight()
            )));
        }
    }
    if cc.num_steps(&s) > MAX_STEP {
        return Err(value_err(format!("schedule steps exceed {MAX_STEP}")));
    }
    let noise = match model {
        "cnot" => ColorNoise::Cnot(p),
        _ => ColorNoise::Uniform(p),
    };
    let mut m = cc.memory_flagged(&s, &flagged, rounds, noise, x_basis);
    let n = m.circuit.num_qubits;
    let mut readout = m.noise.p_meas;
    if model == "si1000" && p > 0.0 {
        rescale_si1000(&mut m.circuit.ops, n, p);
        readout = 5.0 * p;
    }
    let nd = cc.data.len();
    let mut layout = Layout {
        data: (0..nd).collect(),
        ancillas: (nd..nd + np).collect(),
        flags: (nd + np..n).collect(),
        ..Default::default()
    };
    layout.qubit_coords = cc
        .data
        .iter()
        .map(|&(x, y)| (x as f64, y as f64))
        .chain(cc.plaquettes.iter().map(|p| (p.x as f64, p.y as f64)))
        .chain(
            cc.plaquettes
                .iter()
                .zip(&flagged)
                .filter(|(_, &f)| f)
                .map(|(p, _)| (p.x as f64 + 0.5, p.y as f64 + 0.5)),
        )
        .collect();
    for (k, &(pi, is_x, r)) in m.detector_info.iter().enumerate() {
        let pl = &cc.plaquettes[pi];
        let fl = m.flag_detector[k];
        let off = if fl { 0.5 } else { 0.0 };
        layout
            .detector_coords
            .push((pl.x as f64 + off, pl.y as f64 + off, r as f64));
        layout.detector_basis.push(if is_x { 'X' } else { 'Z' });
        layout.flag_detector.push(fl);
    }
    let data = CircuitData {
        circuit: m.circuit,
        global_phase: 0.0,
        readout_error: readout,
        detectors: m.detectors,
        observables: m.observables,
        has_repeats: rounds > 1,
    };
    Ok((data, layout, s))
}

enum ScheduleSpec {
    Named(String),
    Explicit(Vec<Vec<u32>>),
}

fn generated<'py>(
    py: Python<'py>,
    data: CircuitData,
    layout: Layout,
) -> PyResult<(PyCircuit, Bound<'py, PyDict>)> {
    Ok((PyCircuit::from_data(data), layout_dict(py, layout)?))
}

/// Rotated surface code memory experiment; returns `(core, layout dict)`.
#[pyfunction]
#[pyo3(signature = (d, rounds, basis="Z", p=0.0, noise="uniform"))]
fn surface_code_memory<'py>(
    py: Python<'py>,
    d: usize,
    rounds: usize,
    basis: &str,
    p: f64,
    noise: &str,
) -> PyResult<(PyCircuit, Bound<'py, PyDict>)> {
    if !(2..=101).contains(&d) {
        return Err(value_err("surface code distance must be in 2..=101"));
    }
    check_rounds(rounds)?;
    let x = parse_basis(basis)?;
    let np = noise_params(noise, p)?;
    let (data, layout) = heavy(py, Some(1), move || surface_memory(d, rounds, x, np));
    generated(py, data, layout)
}

/// Repetition code memory experiment; returns `(core, layout dict)`.
#[pyfunction]
#[pyo3(signature = (d, rounds, p=0.0, noise="uniform"))]
fn repetition_code_memory<'py>(
    py: Python<'py>,
    d: usize,
    rounds: usize,
    p: f64,
    noise: &str,
) -> PyResult<(PyCircuit, Bound<'py, PyDict>)> {
    if !(2..=100_000).contains(&d) {
        return Err(value_err("repetition code distance must be in 2..=100000"));
    }
    check_rounds(rounds)?;
    let np = noise_params(noise, p)?;
    let (data, layout) = heavy(py, Some(1), move || repetition_memory(d, rounds, np));
    generated(py, data, layout)
}

/// Triangular 6.6.6 colour code memory (Kishony–Fowler round structure);
/// `schedule`: `"kf"`, `"tri"` or a list of per-plaquette step rows.
/// Returns `(core, layout dict, schedule rows)`.
#[pyfunction]
#[pyo3(signature = (d, rounds, basis="Z", schedule=None, flags=Vec::new(), p=0.0, noise="cnot"))]
#[allow(clippy::too_many_arguments)]
fn color_code_memory<'py>(
    py: Python<'py>,
    d: usize,
    rounds: usize,
    basis: &str,
    schedule: Option<&Bound<'py, PyAny>>,
    flags: Vec<bool>,
    p: f64,
    noise: &str,
) -> PyResult<(PyCircuit, Bound<'py, PyDict>, Vec<Vec<u8>>)> {
    check_rounds(rounds)?;
    let x = parse_basis(basis)?;
    let spec = match schedule {
        None => ScheduleSpec::Named("kf".into()),
        Some(o) => match o.extract::<String>() {
            Ok(s) => ScheduleSpec::Named(s),
            Err(_) => ScheduleSpec::Explicit(o.extract::<Vec<Vec<u32>>>()?),
        },
    };
    let noise = noise.to_string();
    let (data, layout, s) = heavy(py, Some(1), move || {
        color_memory(d, rounds, x, spec, flags, &noise, p)
    })?;
    let (c, l) = generated(py, data, layout)?;
    Ok((c, l, s.iter().map(|r| r.to_vec()).collect()))
}

/// `(x, y, colour, data qubit per position a..f or -1)`.
type PlaquetteInfo = (i32, i32, u8, Vec<i64>);

/// Per-plaquette data of the d colour code: `[(x, y, colour, [data or -1; 6])]`
/// and whether each plaquette touches the boundary.
#[pyfunction]
fn color_code_plaquettes(d: usize) -> PyResult<(Vec<PlaquetteInfo>, Vec<bool>)> {
    if d < 3 || d % 2 == 0 || d > 41 {
        return Err(value_err("colour code distance must be odd, 3..=41"));
    }
    let cc = ColorCode::new(d);
    Ok((
        cc.plaquettes
            .iter()
            .map(|p| {
                (
                    p.x,
                    p.y,
                    p.color,
                    p.data.iter().map(|q| q.map_or(-1, |v| v as i64)).collect(),
                )
            })
            .collect(),
        cc.boundary_plaquettes(),
    ))
}

// ------------------------------------------------------------- sampling

/// The noise-aware detector sampler behind `DetectorSampler`.
enum Engine {
    Fast(FastSampler),
    Sym(SymPhaseSampler),
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The RNG of chunk `k` for `seed`.
fn chunk_rng(seed: u64, k: usize) -> WyRand {
    WyRand(splitmix64(
        splitmix64(seed) ^ (k as u64).wrapping_mul(0xd1b5_4a32_d192_ed03),
    ))
}

/// Scratch buffers of one worker.
struct Scratch {
    out: Vec<u64>,
    vals: Vec<u64>,
}

impl Engine {
    fn rows(&self) -> usize {
        match self {
            Engine::Fast(f) => f.rows(),
            Engine::Sym(s) => s.num_measurements(),
        }
    }
    fn stride(&self) -> usize {
        match self {
            Engine::Fast(f) => f.stride(),
            Engine::Sym(s) => s.num_measurements(),
        }
    }
    fn scratch(&self) -> Scratch {
        Scratch {
            out: vec![0u64; self.stride() * CHUNK_BLOCKS],
            vals: match self {
                Engine::Fast(_) => Vec::new(),
                Engine::Sym(s) => vec![0u64; s.num_vars()],
            },
        }
    }
    /// Samples chunk `k`: `CHUNK_BLOCKS` 64-shot blocks into `sc.out`
    /// (`out[b * stride + r]` bit `s` = row `r` of shot `64 b + s`).
    fn sample_chunk(&self, seed: u64, k: usize, sc: &mut Scratch) {
        let mut rng = chunk_rng(seed, k);
        match self {
            Engine::Fast(f) => f.sample_batch(&mut rng, &mut sc.out),
            Engine::Sym(s) => {
                let st = self.stride();
                for blk in sc.out.chunks_exact_mut(st.max(1)) {
                    if st > 0 {
                        s.sample_batch(&mut rng, &mut sc.vals, blk);
                    }
                }
            }
        }
    }
}

/// In place: `a[s]` bit `r` <- `a[r]` bit `s` (constant trip counts so the
/// compiler unrolls and vectorises the six swap stages).
#[inline]
fn transpose64(a: &mut [u64; 64]) {
    #[inline(always)]
    fn stage(a: &mut [u64; 64], j: usize, m: u64) {
        for base in (0..64).step_by(2 * j) {
            for k in base..base + j {
                let t = ((a[k] >> j) ^ a[k + j]) & m;
                a[k] ^= t << j;
                a[k + j] ^= t;
            }
        }
    }
    stage(a, 32, 0x0000_0000_ffff_ffff);
    stage(a, 16, 0x0000_ffff_0000_ffff);
    stage(a, 8, 0x00ff_00ff_00ff_00ff);
    stage(a, 4, 0x0f0f_0f0f_0f0f_0f0f);
    stage(a, 2, 0x3333_3333_3333_3333);
    stage(a, 1, 0x5555_5555_5555_5555);
}

/// `SPREAD[b]`: byte `b` as eight 0/1 bytes (little-endian bit order).
const SPREAD: [u64; 256] = {
    let mut t = [0u64; 256];
    let mut b = 0;
    while b < 256 {
        let mut v = 0u64;
        let mut i = 0;
        while i < 8 {
            if b >> i & 1 == 1 {
                v |= 1 << (8 * i);
            }
            i += 1;
        }
        t[b] = v;
        b += 1;
    }
    t
};

/// Writes rows `[r0, r1)` of the sampled blocks into per-shot output rows of
/// `width` entries (`packed`: bytes with 8 rows each, little-endian bit
/// order as Stim's `bit_packed`; else one 0/1 byte per row).
fn write_rows(
    blocks: &[u64],
    stride: usize,
    r0: usize,
    r1: usize,
    shots: usize,
    packed: bool,
    out: &mut [u8],
) {
    let nrows = r1 - r0;
    let width = if packed { nrows.div_ceil(8) } else { nrows };
    if width == 0 {
        return;
    }
    let mut a = [0u64; 64];
    for (b, blk) in blocks.chunks_exact(stride.max(1)).enumerate() {
        let s0 = 64 * b;
        if s0 >= shots {
            break;
        }
        let ns = (shots - s0).min(64);
        let dst = &mut out[s0 * width..(s0 + ns) * width];
        for g in 0..nrows.div_ceil(64) {
            let base = r0 + 64 * g;
            let cnt = (r1 - base).min(64);
            a[..cnt].copy_from_slice(&blk[base..base + cnt]);
            a[cnt..].iter_mut().for_each(|w| *w = 0);
            transpose64(&mut a);
            // entries of this group in a shot's row: [lo, hi)
            let lo = if packed { 8 * g } else { 64 * g };
            let hi = if packed {
                (lo + 8).min(width)
            } else {
                lo + cnt
            };
            for (s, row) in dst.chunks_exact_mut(width).enumerate() {
                let w = a[s];
                let seg = &mut row[lo..hi];
                if packed {
                    if seg.len() == 8 {
                        seg.copy_from_slice(&w.to_le_bytes());
                    } else {
                        seg.copy_from_slice(&w.to_le_bytes()[..seg.len()]);
                    }
                } else {
                    let bytes = w.to_le_bytes();
                    let mut chunks = seg.chunks_exact_mut(8);
                    for (c, &byte) in (&mut chunks).zip(bytes.iter()) {
                        c.copy_from_slice(&SPREAD[byte as usize].to_le_bytes());
                    }
                    let rem = chunks.into_remainder();
                    if !rem.is_empty() {
                        let k = cnt / 8;
                        let v = SPREAD[bytes[k] as usize].to_le_bytes();
                        let n = rem.len();
                        rem.copy_from_slice(&v[..n]);
                    }
                }
            }
        }
    }
}

/// A compiled detector sampler (see `qsimlab.qec.DetectorSampler`).
#[pyclass(name = "DetectorSamplerCore", module = "qsimlab._native.qec", frozen)]
pub struct NativeSampler {
    engine: Arc<Engine>,
    num_detectors: usize,
    num_observables: usize,
    engine_name: &'static str,
    note: String,
    compile_time: f64,
}

fn compile_parities(d: &CircuitData) -> Result<SymPhaseSampler, PyErr> {
    if !d.circuit.is_clifford() {
        return Err(unsupported(
            "detector sampling needs a Clifford circuit with Pauli noise \
             (non-Clifford gates found)",
        ));
    }
    if d.circuit
        .ops
        .iter()
        .any(|o| matches!(o, Op::ClassicControlled { .. }))
    {
        return Err(unsupported(
            "detector sampling does not support classically controlled gates",
        ));
    }
    let sets: Vec<Vec<usize>> = d
        .detectors
        .iter()
        .chain(d.observables.iter())
        .cloned()
        .collect();
    let s = SymPhaseSampler::new(&d.circuit, &d.noise_model()).map_err(map_sim_err)?;
    Ok(s.with_parities(&sets).relative_to_reference())
}

impl NativeSampler {
    fn build(d: &CircuitData, engine: &str) -> PyResult<NativeSampler> {
        let t0 = Instant::now();
        let s = compile_parities(d)?;
        let (eng, name, note) = match engine {
            "symphase" => (Engine::Sym(s), "symphase", String::new()),
            "fast" | "auto" => {
                let r =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| FastSampler::new(&s)));
                match r {
                    Ok(f) => (Engine::Fast(f), "fast", String::new()),
                    Err(_) if engine == "auto" => (
                        Engine::Sym(s),
                        "symphase",
                        "FastSampler rejected the circuit; used the SymPhase sampler \
                         (same distribution, slower)"
                            .to_string(),
                    ),
                    Err(_) => {
                        return Err(unsupported(
                            "FastSampler rejected the circuit; use engine='symphase' or 'auto'",
                        ))
                    }
                }
            }
            e => {
                return Err(value_err(format!(
                    "unknown sampler engine '{e}' (expected 'auto', 'fast' or 'symphase')"
                )))
            }
        };
        Ok(NativeSampler {
            engine: Arc::new(eng),
            num_detectors: d.detectors.len(),
            num_observables: d.observables.len(),
            engine_name: name,
            note,
            compile_time: t0.elapsed().as_secs_f64(),
        })
    }

    /// Samples `shots` into `(dets, obs)` byte buffers.
    fn run(&self, shots: usize, seed: u64, packed: bool) -> (Vec<u8>, Vec<u8>) {
        let nd = self.num_detectors;
        let no = self.num_observables;
        let wd = if packed { nd.div_ceil(8) } else { nd };
        let wo = if packed { no.div_ceil(8) } else { no };
        let mut dets = vec![0u8; shots * wd];
        let mut obs = vec![0u8; shots * wo];
        let eng = &*self.engine;
        let stride = eng.stride();
        let nchunks = shots.div_ceil(CHUNK_SHOTS);
        let dchunks: Vec<&mut [u8]> = if wd > 0 {
            dets.chunks_mut(CHUNK_SHOTS * wd).collect()
        } else {
            (0..nchunks).map(|_| &mut [][..]).collect()
        };
        let ochunks: Vec<&mut [u8]> = if wo > 0 {
            obs.chunks_mut(CHUNK_SHOTS * wo).collect()
        } else {
            (0..nchunks).map(|_| &mut [][..]).collect()
        };
        dchunks
            .into_par_iter()
            .zip(ochunks)
            .enumerate()
            .for_each_init(
                || eng.scratch(),
                |sc, (k, (dc, oc))| {
                    let ns = (shots - k * CHUNK_SHOTS).min(CHUNK_SHOTS);
                    eng.sample_chunk(seed, k, sc);
                    write_rows(&sc.out, stride, 0, nd, ns, packed, dc);
                    write_rows(&sc.out, stride, nd, nd + no, ns, packed, oc);
                },
            );
        (dets, obs)
    }
}

/// Raw pointer for disjoint parallel writes.
#[derive(Clone, Copy)]
struct SyncPtr(*mut u8);
// SAFETY: every task writes a disjoint set of bytes (its own chunk's columns).
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}

impl NativeSampler {
    /// Detector-major bit-packed output: `dets[r * w + j]` holds shots
    /// `8 j .. 8 j + 8` of row `r` (`w = ceil(shots / 8)`), bits past `shots`
    /// zero. No transposition: each 64-shot word is stored as 8 bytes.
    fn run_transposed(&self, shots: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
        let nd = self.num_detectors;
        let no = self.num_observables;
        let w = shots.div_ceil(8);
        let mut dets = vec![0u8; nd * w];
        let mut obs = vec![0u8; no * w];
        let eng = &*self.engine;
        let stride = eng.stride();
        let pd = SyncPtr(dets.as_mut_ptr());
        let po = SyncPtr(obs.as_mut_ptr());
        (0..shots.div_ceil(CHUNK_SHOTS))
            .into_par_iter()
            .for_each_init(
                || eng.scratch(),
                |sc, k| {
                    let (pd, po) = (pd, po);
                    let ns = (shots - k * CHUNK_SHOTS).min(CHUNK_SHOTS);
                    eng.sample_chunk(seed, k, sc);
                    let col = k * CHUNK_SHOTS / 8;
                    let nbytes = ns.div_ceil(8);
                    let mut buf = [0u8; CHUNK_SHOTS / 8];
                    // row-outer: each row's bytes of this chunk are contiguous
                    for r in 0..nd + no {
                        for (b, blk) in sc.out.chunks_exact(stride.max(1)).enumerate() {
                            let s0 = 64 * b;
                            if s0 >= ns {
                                break;
                            }
                            let nsb = (ns - s0).min(64);
                            let mask = if nsb == 64 { !0u64 } else { (1u64 << nsb) - 1 };
                            buf[8 * b..8 * b + 8].copy_from_slice(&(blk[r] & mask).to_le_bytes());
                        }
                        let (p, rr) = if r < nd { (pd.0, r) } else { (po.0, r - nd) };
                        // SAFETY: row rr, columns [col, col + nbytes) are inside
                        // the buffer and written only by this chunk.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                buf.as_ptr(),
                                p.add(rr * w + col),
                                nbytes,
                            );
                        }
                    }
                },
            );
        (dets, obs)
    }
}

#[pymethods]
impl NativeSampler {
    #[new]
    #[pyo3(signature = (circuit, engine="auto"))]
    fn py_new(py: Python<'_>, circuit: &PyCircuit, engine: &str) -> PyResult<Self> {
        let snap = circuit.snapshot();
        let engine = engine.to_string();
        heavy(py, Some(1), move || NativeSampler::build(&snap, &engine))
    }
    #[getter]
    fn num_detectors(&self) -> usize {
        self.num_detectors
    }
    #[getter]
    fn num_observables(&self) -> usize {
        self.num_observables
    }
    #[getter]
    fn engine(&self) -> &'static str {
        self.engine_name
    }
    #[getter]
    fn note(&self) -> String {
        self.note.clone()
    }
    #[getter]
    fn compile_time(&self) -> f64 {
        self.compile_time
    }
    /// Rows of the compiled sampler (detectors + observables).
    #[getter]
    fn rows(&self) -> usize {
        self.engine.rows()
    }

    /// Diagnostics: seconds to sample `shots` single-threaded without any
    /// output conversion (the Rust-bench equivalent), and with the shot-major
    /// conversion into a reused buffer.
    fn _bench(&self, py: Python<'_>, shots: usize, packed: bool) -> (f64, f64) {
        heavy(py, Some(1), || {
            let eng = &*self.engine;
            let stride = eng.stride();
            let nd = self.num_detectors;
            let no = self.num_observables;
            let mut sc = eng.scratch();
            let nchunks = shots.div_ceil(CHUNK_SHOTS);
            let t0 = Instant::now();
            for k in 0..nchunks {
                eng.sample_chunk(1, k, &mut sc);
            }
            let t_sample = t0.elapsed().as_secs_f64();
            let wd = if packed { nd.div_ceil(8) } else { nd };
            let wo = if packed { no.div_ceil(8) } else { no };
            let mut db = vec![0u8; CHUNK_SHOTS * wd];
            let mut ob = vec![0u8; CHUNK_SHOTS * wo];
            let t1 = Instant::now();
            for k in 0..nchunks {
                eng.sample_chunk(1, k, &mut sc);
                write_rows(&sc.out, stride, 0, nd, CHUNK_SHOTS, packed, &mut db);
                write_rows(&sc.out, stride, nd, nd + no, CHUNK_SHOTS, packed, &mut ob);
            }
            (t_sample, t1.elapsed().as_secs_f64())
        })
    }

    /// `(dets, obs, wall_time)`: `uint8` arrays `(shots, width)`, bit-packed
    /// (Stim's `bit_packed` layout) or one 0/1 byte per entry.
    /// With `transposed=True`: detector-major bit-packed `(rows, ceil(shots/8))`.
    #[pyo3(signature = (shots, seed, packed=false, threads=None, transposed=false))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        shots: usize,
        seed: u64,
        packed: bool,
        threads: Option<usize>,
        transposed: bool,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, f64)> {
        let nd = self.num_detectors;
        let no = self.num_observables;
        if transposed {
            let w = shots.div_ceil(8);
            if (w as u128) * ((nd + no) as u128) > 1u128 << 36 {
                return Err(qerr(
                    "ResourceLimitError",
                    "output too large; sample in chunks",
                ));
            }
            let t0 = Instant::now();
            let (d, o) = heavy(py, threads, || self.run_transposed(shots, seed));
            let wall = t0.elapsed().as_secs_f64();
            let d = PyArray1::from_vec(py, d).reshape([nd, w])?.into_any();
            let o = PyArray1::from_vec(py, o).reshape([no, w])?.into_any();
            return Ok((d, o, wall));
        }
        let wd = if packed { nd.div_ceil(8) } else { nd };
        let wo = if packed { no.div_ceil(8) } else { no };
        let bytes = shots as u128 * (wd + wo) as u128;
        if bytes > 1u128 << 36 {
            return Err(qerr(
                "ResourceLimitError",
                format!("{shots} shots need {bytes} bytes of output; sample in chunks"),
            ));
        }
        let t0 = Instant::now();
        let (d, o) = heavy(py, threads, || self.run(shots, seed, packed));
        let wall = t0.elapsed().as_secs_f64();
        let d = PyArray1::from_vec(py, d).reshape([shots, wd])?.into_any();
        let o = PyArray1::from_vec(py, o).reshape([shots, wo])?.into_any();
        Ok((d, o, wall))
    }
}

// --------------------------------------------------------------------- DEM

/// Probability of each of the `2^b - 1` independent components that
/// reproduce a uniform-over-non-identity channel of total probability `p` on
/// `Z_2^b` (Stim's exact conversion for `DEPOLARIZE1/2`).
fn independent_component(p: f64, b: u32) -> Option<f64> {
    if p == 0.0 {
        return Some(0.0);
    }
    let m = ((1u64 << b) - 1) as f64;
    let base = 1.0 - p * (m + 1.0) / m;
    if base < 0.0 {
        return None;
    }
    Some(0.5 * (1.0 - base.powf(1.0 / (1u64 << (b - 1)) as f64)))
}

/// One merged mechanism: sorted detectors, observable mask, probability.
type Mech = (Vec<u32>, u64, f64);

fn circuit_dem(d: &CircuitData) -> PyResult<Vec<Mech>> {
    let nd = d.detectors.len();
    let no = d.observables.len();
    if no > 64 {
        return Err(unsupported(
            "detector error models support at most 64 observables",
        ));
    }
    let s = compile_parities(d)?;
    let mut cols: Vec<Vec<u32>> = vec![Vec::new(); s.num_vars()];
    for j in 0..s.num_measurements() {
        for &v in s.row(j) {
            cols[v as usize].push(j as u32);
        }
    }
    let mut map: HashMap<(Vec<u32>, u64), f64> = HashMap::new();
    let mut sig: Vec<u32> = Vec::new();
    for g in s.groups() {
        let (bits, q) = match g.dist {
            VarDist::Coin => (1u32, 0.5),
            VarDist::Flip(p) => (1, p),
            VarDist::Depol1(p) => (2, p),
            VarDist::Depol2(p) => (4, p),
        };
        let q = if bits == 1 {
            q
        } else {
            independent_component(q, bits).ok_or_else(|| {
                value_err(format!(
                    "depolarizing probability {q} is too large for an error model"
                ))
            })?
        };
        for pat in 1u32..(1 << bits) {
            sig.clear();
            for k in 0..bits {
                if pat >> k & 1 == 1 {
                    for &r in &cols[g.first as usize + k as usize] {
                        match sig.binary_search(&r) {
                            Ok(i) => {
                                sig.remove(i);
                            }
                            Err(i) => sig.insert(i, r),
                        }
                    }
                }
            }
            if sig.is_empty() {
                continue;
            }
            if g.dist == VarDist::Coin {
                let r = sig[0] as usize;
                let what = if r < nd {
                    format!("detector {r}")
                } else {
                    format!("observable {}", r - nd)
                };
                return Err(unsupported(format!(
                    "{what} is not deterministic (it depends on a random measurement \
                     outcome); a detector error model needs deterministic detectors \
                     and observables"
                )));
            }
            if q == 0.0 {
                continue;
            }
            let mut obs = 0u64;
            let dets: Vec<u32> = sig
                .iter()
                .copied()
                .filter(|&r| {
                    if (r as usize) >= nd {
                        obs ^= 1 << (r as usize - nd);
                        false
                    } else {
                        true
                    }
                })
                .collect();
            let e = map.entry((dets, obs)).or_insert(0.0);
            *e = *e * (1.0 - q) + q * (1.0 - *e);
        }
    }
    let mut v: Vec<Mech> = map.into_iter().map(|((a, b), p)| (a, b, p)).collect();
    v.sort_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
    Ok(v)
}

fn mask_to_list(m: u64) -> Vec<usize> {
    (0..64).filter(|k| m >> k & 1 == 1).collect()
}

/// `(num_detectors, num_observables, [(p, detectors, observables)])`.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn detector_error_model(
    py: Python<'_>,
    circuit: &PyCircuit,
) -> PyResult<(usize, usize, Vec<(f64, Vec<u32>, Vec<usize>)>)> {
    let snap = circuit.snapshot();
    let nd = snap.detectors.len();
    let no = snap.observables.len();
    let v = heavy(py, Some(1), move || circuit_dem(&snap))?;
    Ok((
        nd,
        no,
        v.into_iter()
            .map(|(d, o, p)| (p, d, mask_to_list(o)))
            .collect(),
    ))
}

// ---------------------------------------------------------------- decoding

fn obs_mask(o: &[usize]) -> PyResult<u64> {
    let mut m = 0u64;
    for &k in o {
        if k >= 64 {
            return Err(unsupported("the decoders support at most 64 observables"));
        }
        m ^= 1 << k;
    }
    Ok(m)
}

/// A BP+OSD decoder for one detector error model.
#[pyclass(name = "BpOsdCore", module = "qsimlab._native.qec", frozen)]
pub struct BpOsdCore {
    dec: Arc<BpOsd>,
    num_detectors: usize,
    num_observables: usize,
}

#[pymethods]
impl BpOsdCore {
    #[new]
    #[pyo3(signature = (num_detectors, num_observables, errors, max_iter=50, ms_scale=0.625, osd_order=10))]
    fn py_new(
        num_detectors: usize,
        num_observables: usize,
        errors: Vec<(f64, Vec<u32>, Vec<usize>)>,
        max_iter: usize,
        ms_scale: f64,
        osd_order: usize,
    ) -> PyResult<Self> {
        if num_observables > 64 {
            return Err(unsupported("BP+OSD supports at most 64 observables"));
        }
        let mut cols = Vec::with_capacity(errors.len());
        let mut obs = Vec::with_capacity(errors.len());
        let mut ps = Vec::with_capacity(errors.len());
        for (p, mut d, o) in errors {
            if !(0.0..=1.0).contains(&p) {
                return Err(value_err(format!("error probability {p} not in [0, 1]")));
            }
            d.sort_unstable();
            d.dedup();
            if let Some(&x) = d.iter().find(|&&x| x as usize >= num_detectors) {
                return Err(value_err(format!(
                    "error mechanism names detector {x} of {num_detectors}"
                )));
            }
            if d.is_empty() || p == 0.0 {
                continue; // undetectable mechanisms cannot be decoded
            }
            cols.push(d);
            obs.push(obs_mask(&o)?);
            ps.push(p);
        }
        if !(ms_scale > 0.0 && ms_scale <= 1.0) {
            return Err(value_err("ms_scale must be in (0, 1]"));
        }
        let m = DemMatrix {
            num_detectors,
            cols,
            obs,
            p: ps,
        };
        Ok(BpOsdCore {
            dec: Arc::new(BpOsd::new(m, max_iter.max(1), ms_scale, osd_order)),
            num_detectors,
            num_observables,
        })
    }

    /// Decodes bit-packed syndromes `(shots, ceil(D/8))` (`uint8`); returns
    /// `(observable masks as uint64, bp_converged, osd_calls)`.
    #[pyo3(signature = (syndromes, threads=None))]
    fn decode_packed<'py>(
        &self,
        py: Python<'py>,
        syndromes: PyReadonlyArray2<'py, u8>,
        threads: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray1<u64>>, u64, u64)> {
        let shape = syndromes.shape();
        let (shots, w) = (shape[0], shape[1]);
        if w != self.num_detectors.div_ceil(8) {
            return Err(value_err(format!(
                "syndromes have {w} bytes per shot, expected {} for {} detectors",
                self.num_detectors.div_ceil(8),
                self.num_detectors
            )));
        }
        let data: Vec<u8> = syndromes.as_array().iter().copied().collect();
        let dec = Arc::clone(&self.dec);
        let nd = self.num_detectors;
        let (masks, st) = heavy(py, threads, move || decode_bytes(&dec, &data, shots, w, nd));
        Ok((PyArray1::from_vec(py, masks), st.bp_converged, st.osd_calls))
    }

    #[getter]
    fn num_detectors(&self) -> usize {
        self.num_detectors
    }
    #[getter]
    fn num_observables(&self) -> usize {
        self.num_observables
    }
    #[getter]
    fn num_mechanisms(&self) -> usize {
        self.dec.num_mechanisms()
    }
}

fn decode_bytes(
    dec: &BpOsd,
    data: &[u8],
    shots: usize,
    w: usize,
    nd: usize,
) -> (Vec<u64>, DecodeStats) {
    let per = 256usize;
    let parts: Vec<(Vec<u64>, DecodeStats)> = (0..shots.div_ceil(per))
        .into_par_iter()
        .map_init(
            || (dec.scratch(), Vec::<u32>::new()),
            |(sc, fired), c| {
                let mut st = DecodeStats::default();
                let lo = c * per;
                let hi = (lo + per).min(shots);
                let mut out = Vec::with_capacity(hi - lo);
                for s in lo..hi {
                    fired.clear();
                    for (b, &byte) in data[s * w..(s + 1) * w].iter().enumerate() {
                        let mut x = byte;
                        while x != 0 {
                            let i = 8 * b + x.trailing_zeros() as usize;
                            if i < nd {
                                fired.push(i as u32);
                            }
                            x &= x - 1;
                        }
                    }
                    out.push(dec.decode(fired, sc, &mut st));
                }
                (out, st)
            },
        )
        .collect();
    let mut masks = Vec::with_capacity(shots);
    let mut st = DecodeStats::default();
    for (m, s) in parts {
        masks.extend(m);
        st.bp_converged += s.bp_converged;
        st.osd_calls += s.osd_calls;
    }
    (masks, st)
}

/// Sample + decode in one GIL-free pass. Returns a dict with `errors`,
/// `shots`, `bp_converged`, `osd_calls`, `wall_time`.
#[pyfunction]
#[pyo3(signature = (sampler, decoder, shots, seed, max_errors=None, threads=None))]
fn sample_decode_count<'py>(
    py: Python<'py>,
    sampler: &NativeSampler,
    decoder: &BpOsdCore,
    shots: usize,
    seed: u64,
    max_errors: Option<u64>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    if decoder.num_detectors != sampler.num_detectors {
        return Err(value_err(format!(
            "decoder has {} detectors, circuit has {}",
            decoder.num_detectors, sampler.num_detectors
        )));
    }
    if sampler.num_observables > 64 {
        return Err(unsupported("at most 64 observables"));
    }
    let eng = Arc::clone(&sampler.engine);
    let dec = Arc::clone(&decoder.dec);
    let nd = sampler.num_detectors;
    let no = sampler.num_observables;
    let t0 = Instant::now();
    let (errs, done, st) = heavy(py, threads, move || {
        let stride = eng.stride();
        let nchunks = shots.div_ceil(CHUNK_SHOTS);
        let (mut errs, mut done) = (0u64, 0usize);
        let mut st = DecodeStats::default();
        let mut k0 = 0usize;
        while k0 < nchunks {
            let k1 = (k0 + WAVE_CHUNKS).min(nchunks);
            let r: Vec<(u64, usize, DecodeStats)> = (k0..k1)
                .into_par_iter()
                .map_init(
                    || (eng.scratch(), dec.scratch(), vec![Vec::<u32>::new(); 64]),
                    |(sc, ds, lists), k| {
                        let ns = (shots - k * CHUNK_SHOTS).min(CHUNK_SHOTS);
                        eng.sample_chunk(seed, k, sc);
                        let mut st = DecodeStats::default();
                        let mut e = 0u64;
                        for (b, blk) in sc.out.chunks_exact(stride.max(1)).enumerate() {
                            let s0 = 64 * b;
                            if s0 >= ns {
                                break;
                            }
                            let nsb = (ns - s0).min(64);
                            lists.iter_mut().for_each(|l| l.clear());
                            let mut omask = [0u64; 64];
                            for (r, &word) in blk[..nd + no].iter().enumerate() {
                                let mut x = word;
                                while x != 0 {
                                    let s = x.trailing_zeros() as usize;
                                    if r < nd {
                                        lists[s].push(r as u32);
                                    } else {
                                        omask[s] ^= 1 << (r - nd);
                                    }
                                    x &= x - 1;
                                }
                            }
                            for s in 0..nsb {
                                let pred = dec.decode(&lists[s], ds, &mut st);
                                if pred != omask[s] {
                                    e += 1;
                                }
                            }
                        }
                        (e, ns, st)
                    },
                )
                .collect();
            for (e, n, s) in r {
                errs += e;
                done += n;
                st.bp_converged += s.bp_converged;
                st.osd_calls += s.osd_calls;
            }
            k0 = k1;
            if max_errors.is_some_and(|m| errs >= m) {
                break;
            }
        }
        (errs, done, st)
    });
    let d = PyDict::new(py);
    d.set_item("errors", errs)?;
    d.set_item("shots", done)?;
    d.set_item("bp_converged", st.bp_converged)?;
    d.set_item("osd_calls", st.osd_calls)?;
    d.set_item("wall_time", t0.elapsed().as_secs_f64())?;
    Ok(d)
}

// ---------------------------------------------------------------- distance

/// Exact minimum-weight logical of a DEM (branch and bound). `errors` as in
/// [`detector_error_model`]; the search is restricted to `keep` detectors
/// when given (mechanisms are projected onto them and merged; the result is
/// then a lower bound, `certified` when the example lifts to the full DEM).
#[pyfunction]
#[pyo3(signature = (num_detectors, errors, observable=0, keep=None, max_weight=64,
                    count_cap=1_000_000, node_limit=None, timeout=None))]
#[allow(clippy::too_many_arguments)]
fn min_weight_logical<'py>(
    py: Python<'py>,
    num_detectors: usize,
    errors: Vec<(f64, Vec<u32>, Vec<usize>)>,
    observable: usize,
    keep: Option<Vec<usize>>,
    max_weight: usize,
    count_cap: u64,
    node_limit: Option<u64>,
    timeout: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut map = vec![u32::MAX; num_detectors];
    let nkeep = match &keep {
        None => {
            for (i, m) in map.iter_mut().enumerate() {
                *m = i as u32;
            }
            num_detectors
        }
        Some(k) => {
            for (j, &i) in k.iter().enumerate() {
                if i >= num_detectors {
                    return Err(value_err(format!("detector {i} out of range")));
                }
                map[i] = j as u32;
            }
            k.len()
        }
    };
    // project and merge: signature -> (index, pure, source mechanism)
    let mut index: HashMap<(Vec<u32>, bool), usize> = HashMap::new();
    let mut dets: Vec<Vec<u32>> = Vec::new();
    let mut obs: Vec<bool> = Vec::new();
    let mut pure: Vec<Option<usize>> = Vec::new();
    let mut any_source: Vec<usize> = Vec::new();
    for (j, (p, d, o)) in errors.iter().enumerate() {
        if *p == 0.0 {
            continue;
        }
        let mut zs: Vec<u32> = Vec::new();
        for &i in d {
            if i as usize >= num_detectors {
                return Err(value_err(format!("detector {i} out of range")));
            }
            let z = map[i as usize];
            if z != u32::MAX {
                zs.push(z);
            }
        }
        zs.sort_unstable();
        let ob = o.iter().filter(|&&k| k == observable).count() % 2 == 1;
        if zs.is_empty() && !ob {
            continue;
        }
        let is_pure = zs.len() == d.len();
        let k = *index.entry((zs.clone(), ob)).or_insert_with(|| {
            dets.push(zs);
            obs.push(ob);
            pure.push(None);
            any_source.push(j);
            dets.len() - 1
        });
        if is_pure && pure[k].is_none() {
            pure[k] = Some(j);
        }
    }
    let t0 = Instant::now();
    let nl = node_limit.unwrap_or(u64::MAX);
    // `searched`: the largest weight proven to have no logical (0: unknown)
    let (res, timed_out, searched, limit_hit) = heavy(py, Some(1), move || {
        let Some(tmax) = timeout else {
            let r = min_logical(nkeep, &dets, &obs, max_weight, count_cap, nl);
            let hit = r.weight.is_none() && r.nodes > nl;
            let searched = if hit { 0 } else { max_weight };
            return (r, false, searched, hit);
        };
        // iterative deepening; each weight gets the node budget the measured
        // node rate allows in the remaining time
        let mut used = 0u64;
        let mut w = 0usize;
        loop {
            w += 1;
            let el = t0.elapsed().as_secs_f64();
            let user_left = nl.saturating_sub(used);
            let time_left = if el > 0.05 && used > 0 {
                ((used as f64 / el) * (tmax - el).max(0.0)) as u64
            } else {
                u64::MAX
            };
            let budget = user_left.min(time_left);
            let mut r = min_logical(nkeep, &dets, &obs, w, count_cap, budget);
            used = used.saturating_add(r.nodes);
            let over = r.weight.is_none() && r.nodes > budget;
            let late =
                !over && r.weight.is_none() && w < max_weight && t0.elapsed().as_secs_f64() > tmax;
            if r.weight.is_some() || over || late || w >= max_weight {
                r.nodes = used;
                let user_hit = over && budget == user_left;
                let timed = (over && !user_hit) || late;
                let searched = if over {
                    w - 1
                } else if r.weight.is_some() {
                    0
                } else {
                    w
                };
                return (r, timed, searched, user_hit);
            }
        }
    });
    let d = PyDict::new(py);
    d.set_item("weight", res.weight)?;
    d.set_item("count", res.count)?;
    d.set_item("count_capped", res.count >= count_cap)?;
    d.set_item("nodes", res.nodes)?;
    d.set_item("timed_out", timed_out)?;
    d.set_item("node_limit_hit", limit_hit)?;
    d.set_item("searched_weight", searched)?;
    let lifted: Vec<Option<usize>> = res.example.iter().map(|&j| pure[j]).collect();
    let certified = res.weight.is_some() && lifted.iter().all(|x| x.is_some());
    d.set_item("certified", certified)?;
    d.set_item(
        "example",
        res.example
            .iter()
            .map(|&j| pure[j].unwrap_or(any_source[j]))
            .collect::<Vec<_>>(),
    )?;
    d.set_item("wall_time", t0.elapsed().as_secs_f64())?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "native part of qsimlab.qec")?;
    m.add("CHUNK_SHOTS", CHUNK_SHOTS)?;
    m.add_function(wrap_pyfunction!(surface_code_memory, m)?)?;
    m.add_function(wrap_pyfunction!(repetition_code_memory, m)?)?;
    m.add_function(wrap_pyfunction!(color_code_memory, m)?)?;
    m.add_function(wrap_pyfunction!(color_code_plaquettes, m)?)?;
    m.add_function(wrap_pyfunction!(detector_error_model, m)?)?;
    m.add_function(wrap_pyfunction!(sample_decode_count, m)?)?;
    m.add_function(wrap_pyfunction!(min_weight_logical, m)?)?;
    m.add_class::<NativeSampler>()?;
    m.add_class::<BpOsdCore>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn transpose64_is_a_transpose() {
        let mut a = [0u64; 64];
        let mut x = 0x1234_5678_9abc_def1u64;
        for w in a.iter_mut() {
            x = splitmix64(x);
            *w = x;
        }
        let orig = a;
        transpose64(&mut a);
        for r in 0..64 {
            for s in 0..64 {
                assert_eq!((a[s] >> r) & 1, (orig[r] >> s) & 1);
            }
        }
    }

    #[test]
    fn independent_components_reproduce_depolarizing() {
        // b = 2: Stim's formula
        let p: f64 = 0.01;
        let q = independent_component(p, 2).unwrap();
        assert!((q - 0.5 * (1.0 - (1.0 - 4.0 * p / 3.0).sqrt())).abs() < 1e-15);
        // b = 4: P(identity) of 15 independent components
        let q = independent_component(p, 4).unwrap();
        // identity iff the XOR of the fired components is 0; by Fourier:
        let chi = (1.0 - 2.0 * q).powi(8);
        let p_id = (1.0 + 15.0 * chi) / 16.0;
        assert!((p_id - (1.0 - p)).abs() < 1e-14);
    }

    #[test]
    fn noiseless_generators_have_quiet_detectors() {
        for (data, _) in [
            surface_memory(3, 3, false, NoiseParams::default()),
            surface_memory(3, 2, true, NoiseParams::default()),
            repetition_memory(5, 3, NoiseParams::default()),
        ] {
            let s = compile_parities(&data).unwrap();
            assert!(s.groups().is_empty() || s.num_vars() == 0);
            let raw = SymPhaseSampler::new(&data.circuit, &data.noise_model()).unwrap();
            let sets: Vec<Vec<usize>> = data
                .detectors
                .iter()
                .chain(&data.observables)
                .cloned()
                .collect();
            let w = raw.with_parities(&sets);
            assert!(w.reference().iter().all(|&b| !b), "noiseless parity 1");
        }
    }
}
