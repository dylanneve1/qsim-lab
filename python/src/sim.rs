//! `qsimlab._native.run` / `plan`: the simulation entry points behind
//! `qsimlab.simulate` (see python/API.md §3).
//!
//! Everything heavy happens in [`run_request`], which takes plain data and
//! runs with the GIL released; the `#[pyfunction]`s only convert.

use crate::circuit::{CircuitData, PyCircuit};
use crate::convert::{parse_bitstrings, parse_memory, parse_pauli, PauliTerm};
use crate::errors::{map_sim_err, unsupported, value_err};
use crate::threads::heavy;
use num_complex::{Complex32, Complex64};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use qsim_lab::compile::plan::Backend;
use qsim_lab::engines::blocked::BlockConfig;
use qsim_lab::engines::hsf::{HsfOptions, HybridSchrodingerFeynman};
use qsim_lab::engines::stabilizer::fast_sampler::FastSampler;
use qsim_lab::engines::stabilizer::symphase::SymPhaseSampler;
use qsim_lab::engines::statevector::{state_bytes, Real, StateVector, MAX_STATE_BYTES};
use qsim_lab::noise::NoiseModel;
use qsim_lab::pipeline::{self, Budget, Output, RepeatOptions, Request, SimOptions};
use qsim_lab::planner::{self, Engine, Plan, PlanFeatures, PlanRequest, PlannerConfig};
use qsim_lab::{Circuit, Gate, Mps, Op, SimError, Simulator, SparseState, Tableau};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::time::Instant;

/// Engines selectable with `engine=` and what each supports:
/// `(name, requests, description)`.
pub const ENGINES: &[(&str, &str, &str)] = &[
    (
        "auto",
        "statevector amplitudes samples expectation",
        "compile pipeline + Planner v2 per component",
    ),
    (
        "statevector",
        "statevector amplitudes samples expectation",
        "dense 2^n amplitudes, f64 or f32, any circuit",
    ),
    (
        "sparse",
        "statevector amplitudes samples expectation",
        "non-zero amplitudes only (n <= 64)",
    ),
    (
        "mps",
        "statevector amplitudes samples expectation",
        "exact matrix product state (refuses to truncate)",
    ),
    (
        "hsf",
        "statevector amplitudes samples expectation",
        "hybrid Schrodinger-Feynman path sum over a cut",
    ),
    (
        "compressed",
        "samples expectation",
        "Clifford frame + dense register on the active qubits (no global phase)",
    ),
    (
        "tableau",
        "samples expectation",
        "stabilizer tableau, Clifford circuits only, any size",
    ),
    (
        "gaussian",
        "samples expectation",
        "free-fermion (matchgate) circuits: Majorana covariance, O(gates n + n^3); Z products only",
    ),
    (
        "symphase",
        "samples",
        "batched noisy-Clifford sampler (measurements affine in noise variables)",
    ),
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EngineSel {
    Auto,
    Planner(Engine),
    Symphase,
}

pub fn parse_engine(s: &str) -> PyResult<EngineSel> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "auto" => EngineSel::Auto,
        "statevector" | "sv" => EngineSel::Planner(Engine::StateVector),
        "sparse" => EngineSel::Planner(Engine::Sparse),
        "mps" => EngineSel::Planner(Engine::Mps),
        "hsf" => EngineSel::Planner(Engine::Hsf),
        "compressed" | "cstate" => EngineSel::Planner(Engine::Compressed),
        "tableau" | "stabilizer" => EngineSel::Planner(Engine::Tableau),
        "gaussian" | "gauss" | "free_fermion" => EngineSel::Planner(Engine::Gaussian),
        "symphase" => EngineSel::Symphase,
        other => {
            return Err(value_err(format!(
                "unknown engine '{other}'; choose one of {}",
                ENGINES.iter().map(|e| e.0).collect::<Vec<_>>().join(", ")
            )))
        }
    })
}

/// Python name of a planner engine.
pub fn engine_name(e: Engine) -> &'static str {
    match e {
        Engine::Zero => "zero",
        Engine::Tableau => "tableau",
        Engine::StateVector => "statevector",
        Engine::Sparse => "sparse",
        Engine::Mps => "mps",
        Engine::Hsf => "hsf",
        Engine::Compressed => "compressed",
        Engine::Tn => "tn",
        Engine::Gaussian => "gaussian",
    }
}

/// Python name of a pipeline backend.
pub fn backend_name(b: Backend) -> &'static str {
    match b {
        Backend::Idle => "idle",
        Backend::Tableau => "tableau",
        Backend::StateVector => "statevector",
        Backend::Adaptive => "compressed",
        Backend::PauliPath => "pauli_path",
        Backend::Planned(e) => engine_name(e),
    }
}

#[derive(Clone, Debug)]
pub enum Req {
    State,
    Amps(Vec<u128>),
    Samples(usize),
    Expect(Vec<PauliTerm>),
}

#[derive(Clone, Debug)]
pub struct Opts {
    pub engine: EngineSel,
    pub f32: bool,
    pub seed: u64,
    pub mem: u128,
    pub noise: NoiseModel,
    pub repeat: Option<bool>,
    pub explain: bool,
}

#[derive(Clone, Debug)]
pub enum Payload {
    State64(Vec<Complex64>),
    State32(Vec<Complex32>),
    Amps(Vec<Complex64>),
    Bits {
        m: usize,
        shots: usize,
        data: Vec<u8>,
    },
    Values(Vec<f64>),
}

#[derive(Clone, Debug, Default)]
pub struct Explanation {
    pub engine: Option<String>,
    pub ranked: Vec<(String, f64)>,
    pub plan_secs: f64,
    pub cached: bool,
    pub features: Vec<(String, f64)>,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Out {
    pub payload: Payload,
    pub engine: String,
    pub components: Vec<(usize, usize, String)>,
    pub precision: &'static str,
    pub wall: f64,
    pub measured: Vec<usize>,
    pub explanation: Option<Explanation>,
}

fn not_supported(what: &'static str) -> SimError {
    SimError::NotSupported { what }
}

fn too_large(what: &'static str, bytes: u128, limit: u128) -> SimError {
    SimError::TooLarge { what, bytes, limit }
}

fn phase_factor(phi: f64) -> Complex64 {
    Complex64::from_polar(1.0, phi)
}

fn planner_cfg(mem: u128) -> PlannerConfig {
    PlannerConfig {
        mem_bytes: mem.min(MAX_STATE_BYTES),
        ..Default::default()
    }
}

/// A plan that runs exactly engine `e` (no speculation, no fallback).
fn forced_plan(e: Engine) -> Plan {
    Plan {
        engine: e,
        ranked: vec![(e, 0.0)],
        features: PlanFeatures::default(),
        plan_secs: 0.0,
        solved: None,
        probe: None,
        stage_secs: [0.0; 4],
        cached: false,
    }
}

/// The unitary part of a circuit for statevector/amplitude/expectation
/// requests: terminal measurements dropped, everything else must be a gate.
fn unitary_part(d: &CircuitData) -> Result<Circuit, SimError> {
    let c = d.without_terminal_measurements();
    if c.ops.iter().all(|o| matches!(o, Op::Gate(_))) {
        Ok(c)
    } else {
        Err(not_supported(
            "this request needs a circuit whose only non-gate ops are terminal measurements \
             (mid-circuit measurement, reset, conditionals and noise channels need samples())",
        ))
    }
}

fn dense_state<T: Real>(c: &Circuit, mem: u128) -> Result<StateVector<T>, SimError> {
    let bytes = state_bytes::<T>(c.num_qubits);
    if bytes > mem {
        return Err(too_large("state vector", bytes, mem));
    }
    let mut sv = StateVector::<T>::try_new(c.num_qubits)?;
    sv.apply_circuit_blocked(c, &BlockConfig::default())?;
    Ok(sv)
}

/// `<ψ|P|ψ>` on a dense vector: Σ_x conj(ψ[x^xm]) ψ[x] i^{nY} (-1)^{|x & zm|}.
fn pauli_expectation_dense<T: Real>(amps: &[num_complex::Complex<T>], t: &PauliTerm) -> f64 {
    let (xm, zm, ny) = t.masks();
    let (xm, zm) = (xm as usize, zm as usize);
    let s: Complex64 = amps
        .par_iter()
        .enumerate()
        .map(|(x, a)| {
            let a = Complex64::new(a.re.to_f64(), a.im.to_f64());
            let b = amps[x ^ xm];
            let b = Complex64::new(b.re.to_f64(), b.im.to_f64());
            let v = b.conj() * a;
            if (x & zm).count_ones() % 2 == 1 {
                -v
            } else {
                v
            }
        })
        .sum();
    let s = s * Complex64::i().powu(ny % 4);
    t.sign * s.re
}

/// The circuit with the basis change that turns `t` into a Z product, and
/// the Z-product support.
fn rotate_to_z(c: &Circuit, t: &PauliTerm) -> (Circuit, Vec<usize>) {
    let mut r = c.clone();
    for &(q, o) in &t.ops {
        match o {
            1 => {
                r.h(q);
            }
            2 => {
                r.sdg(q).h(q);
            }
            _ => {}
        }
    }
    (r, t.ops.iter().map(|&(q, _)| q).collect())
}

fn repeat_opts(d: &CircuitData, o: &Opts) -> SimOptions {
    let on = o.repeat.unwrap_or(d.has_repeats);
    SimOptions {
        repeat: on.then(RepeatOptions::default),
        planner_debug: false,
        ..Default::default()
    }
}

fn components(engines: &[(usize, usize, Backend)]) -> Vec<(usize, usize, String)> {
    engines
        .iter()
        .map(|&(n, g, b)| (n, g, backend_name(b).to_string()))
        .collect()
}

fn summarize(components: &[(usize, usize, String)], fallback: &str) -> String {
    let mut names: Vec<&str> = components
        .iter()
        .filter(|c| c.2 != "idle")
        .map(|c| c.2.as_str())
        .collect();
    names.dedup();
    names.sort_unstable();
    names.dedup();
    match names.len() {
        0 => fallback.to_string(),
        1 => names[0].to_string(),
        _ => "pipeline".to_string(),
    }
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

fn combine_flip(a: f64, b: f64) -> f64 {
    a * (1.0 - b) + b * (1.0 - a)
}

fn has_noise(n: &NoiseModel) -> bool {
    n.p_1q > 0.0 || n.p_2q > 0.0 || n.p_meas > 0.0 || n.p_reset > 0.0
}

/// splitmix64: independent per-chunk seeds, so chunked parallel sampling is
/// reproducible regardless of the thread count.
fn chunk_seed(seed: u64, chunk: u64) -> u64 {
    let mut z = seed ^ chunk.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

const SHOT_CHUNK: usize = 256;

/// Shot-by-shot simulation (`Circuit::run_noisy`) on fresh simulators,
/// chunked over threads with deterministic per-chunk seeds.
fn shot_loop<S, F>(
    c: &Circuit,
    noise: &NoiseModel,
    shots: usize,
    seed: u64,
    parallel: bool,
    make: F,
) -> Result<Vec<u8>, SimError>
where
    S: Simulator,
    F: Fn() -> Result<S, SimError> + Sync,
{
    let m = c.ops.iter().filter(|o| matches!(o, Op::Measure(_))).count();
    let chunks: Vec<usize> = (0..shots.div_ceil(SHOT_CHUNK)).collect();
    let run_chunk = |k: &usize| -> Result<Vec<u8>, SimError> {
        let mut rng = StdRng::seed_from_u64(chunk_seed(seed, *k as u64));
        let mut sim = make()?;
        let lo = k * SHOT_CHUNK;
        let hi = (lo + SHOT_CHUNK).min(shots);
        let mut out = Vec::with_capacity((hi - lo) * m);
        for _ in lo..hi {
            sim.reset_all()?;
            let bits = c.run_noisy(&mut sim, noise, &mut rng)?;
            out.extend(bits.iter().map(|&b| b as u8));
        }
        Ok(out)
    };
    let parts: Vec<Result<Vec<u8>, SimError>> = if parallel {
        chunks.par_iter().map(run_chunk).collect()
    } else {
        chunks.iter().map(run_chunk).collect()
    };
    let mut data = Vec::with_capacity(shots * m);
    for p in parts {
        data.extend(p?);
    }
    Ok(data)
}

fn symphase_samples(
    c: &Circuit,
    noise: &NoiseModel,
    shots: usize,
    seed: u64,
) -> Result<Vec<u8>, SimError> {
    let s = SymPhaseSampler::new(c, noise)?;
    let m = s.num_measurements();
    let mut data = vec![0u8; shots * m];
    if m == 0 || shots == 0 {
        return Ok(data);
    }
    let f = FastSampler::new(&s);
    let stride = f.stride();
    let words = 16usize; // 1024 shots per batch
    let mut buf = vec![0u64; stride * words];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut done = 0usize;
    while done < shots {
        f.sample_batch(&mut rng, &mut buf);
        for w in 0..words {
            let blk = &buf[w * stride..w * stride + m];
            for s_ in 0..64 {
                let shot = done + w * 64 + s_;
                if shot >= shots {
                    break;
                }
                let row = &mut data[shot * m..(shot + 1) * m];
                for (r, word) in blk.iter().enumerate() {
                    row[r] = ((word >> s_) & 1) as u8;
                }
            }
        }
        done += words * 64;
    }
    Ok(data)
}

fn measured_qubits(c: &Circuit) -> Vec<usize> {
    c.ops
        .iter()
        .filter_map(|o| match o {
            Op::Measure(q) => Some(*q),
            _ => None,
        })
        .collect()
}

fn index_samples_to_bits(samples: &[u128], measured: &[usize]) -> Vec<u8> {
    let mut data = Vec::with_capacity(samples.len() * measured.len());
    for &x in samples {
        for &q in measured {
            data.push(((x >> q) & 1) as u8);
        }
    }
    data
}

fn explain_plan(c: &Circuit, req: &PlanRequest, mem: u128, notes: Vec<String>) -> Explanation {
    let mut ex = Explanation {
        notes,
        ..Default::default()
    };
    let n = c.num_qubits;
    ex.features.push(("qubits".into(), n as f64));
    ex.features.push(("gates".into(), c.num_gates() as f64));
    ex.features.push(("t_count".into(), c.t_count() as f64));
    ex.features
        .push(("clifford".into(), f64::from(u8::from(c.is_clifford()))));
    match planner::plan(c, req, &planner_cfg(mem)) {
        Ok(p) => {
            ex.engine = Some(engine_name(p.engine).to_string());
            ex.ranked = p
                .ranked
                .iter()
                .map(|&(e, t)| (engine_name(e).to_string(), t))
                .collect();
            ex.plan_secs = p.plan_secs;
            ex.cached = p.cached;
            ex.features
                .push(("tier".into(), f64::from(p.features.tier)));
            if p.features.mps_max_bond > 0 {
                ex.features
                    .push(("mps_max_bond_bound".into(), p.features.mps_max_bond as f64));
            }
            ex.features
                .push(("active_dim".into(), p.features.base.d as f64));
            ex.features
                .push(("two_qubit_depth".into(), p.features.base.depth2 as f64));
        }
        Err(e) => ex.notes.push(format!("planner unavailable: {e}")),
    }
    ex
}

/// Runs one request. No Python objects: call it inside [`heavy`].
pub fn run_request(d: &CircuitData, req: &Req, o: &Opts) -> Result<Out, SimError> {
    let t0 = Instant::now();
    let mut out = match req {
        Req::State => run_state(d, o)?,
        Req::Amps(xs) => run_amps(d, xs, o)?,
        Req::Samples(shots) => run_samples(d, *shots, o)?,
        Req::Expect(terms) => run_expect(d, terms, o)?,
    };
    out.wall = t0.elapsed().as_secs_f64();
    Ok(out)
}

fn out(payload: Payload, engine: &str, components: Vec<(usize, usize, String)>, f32: bool) -> Out {
    Out {
        payload,
        engine: engine.to_string(),
        components,
        precision: if f32 { "f32" } else { "f64" },
        wall: 0.0,
        measured: Vec::new(),
        explanation: None,
    }
}

fn run_state(d: &CircuitData, o: &Opts) -> Result<Out, SimError> {
    let c = unitary_part(d)?;
    let n = c.num_qubits;
    let ph = phase_factor(d.global_phase);
    let out_bytes = state_bytes::<f64>(n);
    let comp = vec![(n, c.num_gates(), String::new())];
    let named = |name: &str| {
        let mut v = comp.clone();
        v[0].2 = name.to_string();
        v
    };
    let mut r = match o.engine {
        EngineSel::Auto | EngineSel::Planner(Engine::StateVector) => {
            if o.f32 {
                let sv = dense_state::<f32>(&c, o.mem)?;
                let ph32 = Complex32::new(ph.re as f32, ph.im as f32);
                let v: Vec<Complex32> = sv.amplitudes().par_iter().map(|a| a * ph32).collect();
                out(
                    Payload::State32(v),
                    "statevector",
                    named("statevector"),
                    true,
                )
            } else {
                let sv = dense_state::<f64>(&c, o.mem)?;
                let v: Vec<Complex64> = sv.amplitudes().par_iter().map(|a| a * ph).collect();
                out(
                    Payload::State64(v),
                    "statevector",
                    named("statevector"),
                    false,
                )
            }
        }
        EngineSel::Planner(e @ (Engine::Sparse | Engine::Mps | Engine::Hsf)) => {
            if out_bytes > o.mem {
                return Err(too_large("state vector output", out_bytes, o.mem));
            }
            let v: Vec<Complex64> = match e {
                Engine::Sparse => {
                    if n > 64 {
                        return Err(not_supported("the sparse engine handles at most 64 qubits"));
                    }
                    let mut s = SparseState::new(n);
                    for g in c.gates() {
                        s.apply_gate(g)?;
                    }
                    s.to_dense()
                }
                Engine::Mps => {
                    let mut m = Mps::new(n, 1 << 20);
                    for g in c.gates() {
                        m.apply_gate(g)?;
                    }
                    if 1.0 - m.fidelity_estimate() > 1e-10 {
                        return Err(too_large("mps: aborted (would truncate)", 0, o.mem));
                    }
                    (0..1u128 << n)
                        .into_par_iter()
                        .map(|x| m.amplitude(x))
                        .collect()
                }
                _ => {
                    let opts = HsfOptions {
                        max_bytes: o.mem,
                        ..HsfOptions::default()
                    };
                    HybridSchrodingerFeynman::auto(&c, opts)?.state_vector()?
                }
            };
            let v = v.into_par_iter().map(|a| a * ph).collect();
            out(
                Payload::State64(v),
                engine_name(e),
                named(engine_name(e)),
                false,
            )
        }
        _ => {
            return Err(not_supported(
                "statevector(): this engine drops the global phase or has no state; \
                 use auto, statevector, sparse, mps or hsf",
            ))
        }
    };
    if o.explain {
        r.explanation = Some(Explanation {
            engine: Some(r.engine.clone()),
            notes: vec!["full state requested: output is 2^n amplitudes".into()],
            features: vec![
                ("qubits".into(), n as f64),
                ("gates".into(), c.num_gates() as f64),
            ],
            ..Default::default()
        });
    }
    Ok(r)
}

fn run_amps(d: &CircuitData, xs: &[u128], o: &Opts) -> Result<Out, SimError> {
    let c = unitary_part(d)?;
    let n = c.num_qubits;
    let ph = phase_factor(d.global_phase);
    let mut r = match o.engine {
        EngineSel::Auto => {
            let sim = pipeline::simulate_with(
                &c,
                &Request::Amplitudes(xs.to_vec()),
                &Budget { mem_bytes: o.mem.min(MAX_STATE_BYTES) },
                &repeat_opts(d, o),
            )?;
            let Output::Amplitudes(a) = sim.output else { unreachable!() };
            let comps = components(&sim.engines);
            let eng = summarize(&comps, "idle");
            out(Payload::Amps(a), &eng, comps, false)
        }
        EngineSel::Planner(Engine::StateVector) if o.f32 => {
            let sv = dense_state::<f32>(&c, o.mem)?;
            let a = xs.iter().map(|&x| sv.amplitude(x as usize)).collect();
            out(Payload::Amps(a), "statevector", vec![(n, c.num_gates(), "statevector".into())], true)
        }
        EngineSel::Planner(e @ (Engine::StateVector | Engine::Sparse | Engine::Mps | Engine::Hsf)) => {
            let ex = planner::execute_amplitudes(&forced_plan(e), &c, xs, &planner_cfg(o.mem))?;
            out(
                Payload::Amps(ex.amplitudes),
                engine_name(ex.engine),
                vec![(n, c.num_gates(), engine_name(ex.engine).into())],
                false,
            )
        }
        _ => {
            return Err(not_supported(
                "amplitudes(): this engine drops the global phase; use auto, statevector, sparse, mps or hsf",
            ))
        }
    };
    if let Payload::Amps(a) = &mut r.payload {
        for v in a.iter_mut() {
            *v *= ph;
        }
    }
    if o.explain {
        r.explanation = Some(explain_plan(
            &c,
            &PlanRequest::Amplitudes(xs.len()),
            o.mem,
            vec!["auto: the pipeline plans each connected component separately".into()],
        ));
    }
    Ok(r)
}

fn run_samples(d: &CircuitData, shots: usize, o: &Opts) -> Result<Out, SimError> {
    let mut c = d.circuit.clone();
    let mut notes = Vec::new();
    if !c.ops.iter().any(|op| matches!(op, Op::Measure(_))) {
        c.measure_all();
        notes.push("no measurements: sampled every qubit".to_string());
    }
    require_clifford_for_tableau(&c, o)?;
    let measured = measured_qubits(&c);
    let m = measured.len();
    let n = c.num_qubits;
    let mut noise = o.noise;
    noise.p_meas = combine_flip(noise.p_meas, d.readout_error);
    let noisy = has_noise(&noise);
    let terminal = is_terminal_unitary(&c);
    let clifford = c.is_clifford();
    let small_sv = n <= 20;
    let sv_ok = |f32: bool| -> Result<(), SimError> {
        let b = if f32 {
            state_bytes::<f32>(n)
        } else {
            state_bytes::<f64>(n)
        };
        if b > o.mem {
            Err(too_large("state vector", b, o.mem))
        } else {
            Ok(())
        }
    };
    let sv_loop = |f32: bool| -> Result<Vec<u8>, SimError> {
        sv_ok(f32)?;
        if f32 {
            shot_loop(&c, &noise, shots, o.seed, small_sv, || {
                StateVector::<f32>::try_new(n)
            })
        } else {
            shot_loop(&c, &noise, shots, o.seed, small_sv, || {
                StateVector::<f64>::try_new(n)
            })
        }
    };
    let tab_loop = || shot_loop(&c, &noise, shots, o.seed, true, || Tableau::try_new(n));
    let single = |name: &str| vec![(n, c.num_gates(), name.to_string())];

    let mut r = match o.engine {
        EngineSel::Symphase => {
            let data = symphase_samples(&c, &noise, shots, o.seed)?;
            out(
                Payload::Bits { m, shots, data },
                "symphase",
                single("symphase"),
                false,
            )
        }
        EngineSel::Auto if clifford && (noisy || !terminal) => {
            match symphase_samples(&c, &noise, shots, o.seed) {
                Ok(data) => {
                    notes.push("noisy/mid-circuit Clifford circuit: symphase sampler".into());
                    out(
                        Payload::Bits { m, shots, data },
                        "symphase",
                        single("symphase"),
                        false,
                    )
                }
                Err(SimError::Unsupported { .. } | SimError::NotSupported { .. }) => {
                    notes.push("symphase refused the circuit: tableau shot by shot".into());
                    out(
                        Payload::Bits {
                            m,
                            shots,
                            data: tab_loop()?,
                        },
                        "tableau",
                        single("tableau"),
                        false,
                    )
                }
                Err(e) => return Err(e),
            }
        }
        EngineSel::Auto if noisy => {
            notes.push("noise model on a non-Clifford circuit: state vector shot by shot".into());
            out(
                Payload::Bits {
                    m,
                    shots,
                    data: sv_loop(o.f32)?,
                },
                "statevector",
                single("statevector"),
                o.f32,
            )
        }
        EngineSel::Auto => {
            let sim = pipeline::simulate_with(
                &c,
                &Request::Samples {
                    shots,
                    seed: o.seed,
                },
                &Budget {
                    mem_bytes: o.mem.min(MAX_STATE_BYTES),
                },
                &repeat_opts(d, o),
            )?;
            let Output::Samples(s) = sim.output else {
                unreachable!()
            };
            let mut data = Vec::with_capacity(shots * m);
            for rec in &s {
                data.extend(rec.iter().map(|&b| b as u8));
            }
            let comps = components(&sim.engines);
            let eng = summarize(&comps, "idle");
            out(Payload::Bits { m, shots, data }, &eng, comps, false)
        }
        EngineSel::Planner(Engine::StateVector) if noisy || !terminal || o.f32 => {
            if terminal && !noisy {
                // f32 terminal sampling: evolve once, sample many
                sv_ok(true)?;
                let gates = Circuit {
                    num_qubits: n,
                    ops: c
                        .ops
                        .iter()
                        .filter(|o| matches!(o, Op::Gate(_)))
                        .copied()
                        .collect(),
                };
                let sv = dense_state::<f32>(&gates, o.mem)?;
                let mut rng = StdRng::seed_from_u64(o.seed);
                let idx: Vec<u128> = sv
                    .sample(shots, &mut rng)
                    .into_iter()
                    .map(|x| x as u128)
                    .collect();
                let data = index_samples_to_bits(&idx, &measured);
                out(
                    Payload::Bits { m, shots, data },
                    "statevector",
                    single("statevector"),
                    true,
                )
            } else {
                out(
                    Payload::Bits {
                        m,
                        shots,
                        data: sv_loop(o.f32)?,
                    },
                    "statevector",
                    single("statevector"),
                    o.f32,
                )
            }
        }
        EngineSel::Planner(Engine::Tableau) if noisy || !terminal => out(
            Payload::Bits {
                m,
                shots,
                data: tab_loop()?,
            },
            "tableau",
            single("tableau"),
            false,
        ),
        EngineSel::Planner(_) if noisy || !terminal => {
            return Err(not_supported(
                "this engine samples terminal measurements of noiseless circuits only; \
                 use auto, statevector, tableau or symphase",
            ))
        }
        EngineSel::Planner(e) => {
            let gates = Circuit {
                num_qubits: n,
                ops: c
                    .ops
                    .iter()
                    .filter(|o| matches!(o, Op::Gate(_)))
                    .copied()
                    .collect(),
            };
            let mut rng = StdRng::seed_from_u64(o.seed);
            let ex = planner::execute_samples(
                &forced_plan(e),
                &gates,
                shots,
                &mut rng,
                &planner_cfg(o.mem),
            )?;
            let data = index_samples_to_bits(&ex.samples, &measured);
            out(
                Payload::Bits { m, shots, data },
                engine_name(ex.engine),
                single(engine_name(ex.engine)),
                false,
            )
        }
    };
    r.measured = measured;
    if o.explain {
        let gates = Circuit {
            num_qubits: n,
            ops: c
                .ops
                .iter()
                .filter(|o| matches!(o, Op::Gate(_)))
                .copied()
                .collect(),
        };
        let ex = if terminal && !noisy {
            explain_plan(&gates, &PlanRequest::Samples(shots), o.mem, notes)
        } else {
            Explanation {
                engine: Some(r.engine.clone()),
                notes,
                features: vec![
                    ("qubits".into(), n as f64),
                    ("gates".into(), c.num_gates() as f64),
                    ("clifford".into(), f64::from(u8::from(clifford))),
                ],
                ..Default::default()
            }
        };
        r.explanation = Some(ex);
    }
    Ok(r)
}

fn require_clifford_for_tableau(c: &Circuit, o: &Opts) -> Result<(), SimError> {
    if o.engine == EngineSel::Planner(Engine::Tableau) && !c.is_clifford() {
        return Err(not_supported("the tableau engine needs a Clifford circuit"));
    }
    Ok(())
}

fn run_expect(d: &CircuitData, terms: &[PauliTerm], o: &Opts) -> Result<Out, SimError> {
    let c = unitary_part(d)?;
    require_clifford_for_tableau(&c, o)?;
    let n = c.num_qubits;
    let nontrivial = terms.iter().filter(|t| !t.ops.is_empty()).count();
    let dense_all = match o.engine {
        EngineSel::Auto => nontrivial >= 4 && n <= 20,
        EngineSel::Planner(Engine::StateVector) => true,
        _ => false,
    };
    let mut values = Vec::with_capacity(terms.len());
    let mut comps: Vec<(usize, usize, String)> = Vec::new();
    let mut f32_used = false;
    if dense_all {
        if o.f32 {
            let sv = dense_state::<f32>(&c, o.mem)?;
            for t in terms {
                values.push(pauli_expectation_dense(sv.amplitudes(), t));
            }
            f32_used = true;
        } else {
            let sv = dense_state::<f64>(&c, o.mem)?;
            for t in terms {
                values.push(pauli_expectation_dense(sv.amplitudes(), t));
            }
        }
        comps.push((n, c.num_gates(), "statevector".into()));
    } else {
        for t in terms {
            if t.ops.is_empty() {
                values.push(t.sign);
                continue;
            }
            let (rc, support) = rotate_to_z(&c, t);
            let v = match o.engine {
                EngineSel::Auto => {
                    let sim = pipeline::simulate_with(
                        &rc,
                        &Request::Expectation(support),
                        &Budget {
                            mem_bytes: o.mem.min(MAX_STATE_BYTES),
                        },
                        &repeat_opts(d, o),
                    )?;
                    comps.extend(components(&sim.engines));
                    let Output::Expectation(v) = sim.output else {
                        unreachable!()
                    };
                    v
                }
                EngineSel::Planner(e) => {
                    let ex = planner::execute_expectation(
                        &forced_plan(e),
                        &rc,
                        &support,
                        &planner_cfg(o.mem),
                    )?;
                    comps.push((n, rc.num_gates(), engine_name(ex.engine).into()));
                    ex.value
                }
                EngineSel::Symphase => {
                    return Err(not_supported(
                        "symphase only samples; use auto or tableau for expectations",
                    ))
                }
            };
            values.push(t.sign * v);
        }
    }
    let eng = summarize(&comps, if nontrivial == 0 { "idle" } else { "zero" });
    let mut r = out(Payload::Values(values), &eng, comps, f32_used);
    if o.explain {
        let first = terms.iter().find(|t| !t.ops.is_empty());
        let (rc, support) = match first {
            Some(t) => rotate_to_z(&c, t),
            None => (c.clone(), Vec::new()),
        };
        let mut notes = vec![format!(
            "{} Pauli term(s); ranking shown for the first non-identity term",
            terms.len()
        )];
        if dense_all {
            notes.push(
                "several terms on a small circuit: one dense state, every term read off it".into(),
            );
        }
        r.explanation = Some(explain_plan(
            &rc,
            &PlanRequest::Expectation(support),
            o.mem,
            notes,
        ));
    }
    Ok(r)
}

// ---------------------------------------------------------------------------
// Python side

fn parse_request(kind: &str, payload: &Bound<'_, PyAny>, n: usize) -> PyResult<Req> {
    Ok(match kind {
        "statevector" => Req::State,
        "amplitudes" => Req::Amps(parse_bitstrings(payload, n)?),
        "samples" => {
            let shots: i64 = payload.extract()?;
            if shots < 0 {
                return Err(value_err("shots must be non-negative"));
            }
            Req::Samples(shots as usize)
        }
        "expectation" => {
            let mut terms = Vec::new();
            if let Ok(s) = payload.cast::<PyString>() {
                terms.push(parse_pauli(&s.extract::<String>()?, n)?);
            } else {
                for item in payload.try_iter()? {
                    let item = item?;
                    let s: String = item.extract().map_err(|_| {
                        value_err("expectation() takes a Pauli string or a list of them")
                    })?;
                    terms.push(parse_pauli(&s, n)?);
                }
            }
            Req::Expect(terms)
        }
        other => return Err(value_err(format!("unknown request kind '{other}'"))),
    })
}

fn explanation_dict<'py>(py: Python<'py>, e: &Explanation) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("engine", e.engine.clone())?;
    d.set_item("ranked", e.ranked.clone())?;
    d.set_item("plan_seconds", e.plan_secs)?;
    d.set_item("cached", e.cached)?;
    let f = PyDict::new(py);
    for (k, v) in &e.features {
        if v.fract() == 0.0 && v.abs() < 1e15 {
            f.set_item(k, *v as i64)?;
        } else {
            f.set_item(k, *v)?;
        }
    }
    d.set_item("features", f)?;
    d.set_item("notes", e.notes.clone())?;
    Ok(d)
}

fn noise_from_tuple(t: Option<(f64, f64, f64, f64)>) -> PyResult<NoiseModel> {
    let (p1, p2, pm, pr) = t.unwrap_or((0.0, 0.0, 0.0, 0.0));
    for p in [p1, p2, pm, pr] {
        if !(0.0..=1.0).contains(&p) {
            return Err(value_err(format!("noise probability {p} is not in [0, 1]")));
        }
    }
    Ok(NoiseModel {
        p_1q: p1,
        p_2q: p2,
        p_meas: pm,
        p_reset: pr,
    })
}

/// Runs a request; returns a dict the Python layer turns into a result
/// object. Arguments mirror `qsimlab.simulate`.
#[pyfunction]
#[pyo3(signature = (circuit, kind, payload, engine="auto", precision="f64", seed=None,
                    memory=None, threads=None, noise=None, repeat=None, explain=false))]
#[allow(clippy::too_many_arguments)]
pub fn run<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    kind: &str,
    payload: &Bound<'py, PyAny>,
    engine: &str,
    precision: &str,
    seed: Option<u64>,
    memory: Option<&Bound<'py, PyAny>>,
    threads: Option<usize>,
    noise: Option<(f64, f64, f64, f64)>,
    repeat: Option<bool>,
    explain: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let data = circuit.snapshot();
    let req = parse_request(kind, payload, data.circuit.num_qubits)?;
    let f32 = match precision {
        "f64" | "double" => false,
        "f32" | "single" => true,
        p => {
            return Err(value_err(format!(
                "precision must be 'f64' or 'f32', got '{p}'"
            )))
        }
    };
    let mem = match memory {
        Some(m) if !m.is_none() => parse_memory(m)?,
        _ => MAX_STATE_BYTES,
    };
    let seed = seed.unwrap_or_else(rand::random::<u64>);
    let opts = Opts {
        engine: parse_engine(engine)?,
        f32,
        seed,
        mem: mem.min(MAX_STATE_BYTES),
        noise: noise_from_tuple(noise)?,
        repeat,
        explain,
    };
    if matches!(req, Req::Samples(_)) {
        // nothing extra
    } else if has_noise(&opts.noise) {
        return Err(unsupported("noise= applies to samples() only"));
    }
    let r = heavy(py, threads, move || run_request(&data, &req, &opts)).map_err(map_sim_err)?;
    result_dict(py, r, seed)
}

fn result_dict<'py>(py: Python<'py>, r: Out, seed: u64) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match r.payload {
        Payload::State64(v) => d.set_item("data", PyArray1::from_vec(py, v))?,
        Payload::State32(v) => d.set_item("data", PyArray1::from_vec(py, v))?,
        Payload::Amps(v) => d.set_item("data", PyArray1::from_vec(py, v))?,
        Payload::Values(v) => d.set_item("data", PyArray1::from_vec(py, v))?,
        Payload::Bits { m, shots, data } => {
            let arr = PyArray1::from_vec(py, data).reshape([shots, m])?;
            d.set_item("data", arr)?;
            d.set_item("shots", shots)?;
        }
    }
    d.set_item("engine", r.engine)?;
    d.set_item("components", r.components)?;
    d.set_item("precision", r.precision)?;
    d.set_item("wall_time", r.wall)?;
    d.set_item("seed", seed)?;
    d.set_item("measured_qubits", r.measured)?;
    match &r.explanation {
        Some(e) => d.set_item("explanation", explanation_dict(py, e)?)?,
        None => d.set_item("explanation", py.None())?,
    }
    Ok(d)
}

/// Predicts without running: the planner's ranking for a request.
#[pyfunction]
#[pyo3(signature = (circuit, kind, payload, memory=None))]
pub fn plan<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    kind: &str,
    payload: &Bound<'py, PyAny>,
    memory: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = circuit.snapshot();
    let n = data.circuit.num_qubits;
    let req = parse_request(kind, payload, n)?;
    let mem = match memory {
        Some(m) if !m.is_none() => parse_memory(m)?,
        _ => MAX_STATE_BYTES,
    };
    let ex = heavy(py, Some(1), move || -> Result<Explanation, SimError> {
        Ok(match &req {
            Req::State => {
                let c = unitary_part(&data)?;
                explain_plan(
                    &c,
                    &PlanRequest::Amplitudes(1usize << n.min(40)),
                    mem,
                    vec!["full state requested: the state-vector engine runs".into()],
                )
            }
            Req::Amps(xs) => explain_plan(
                &unitary_part(&data)?,
                &PlanRequest::Amplitudes(xs.len()),
                mem,
                vec![],
            ),
            Req::Samples(s) => {
                let c = &data.circuit;
                if is_terminal_unitary(c) && data.readout_error == 0.0 {
                    let gates = Circuit {
                        num_qubits: n,
                        ops: c
                            .ops
                            .iter()
                            .filter(|o| matches!(o, Op::Gate(_)))
                            .copied()
                            .collect(),
                    };
                    explain_plan(&gates, &PlanRequest::Samples(*s), mem, vec![])
                } else {
                    let eng = if c.is_clifford() {
                        "symphase"
                    } else {
                        "statevector (shot by shot)"
                    };
                    Explanation {
                        engine: Some(eng.split(' ').next().unwrap().to_string()),
                        notes: vec![format!("mid-circuit operations or noise: {eng}")],
                        ..Default::default()
                    }
                }
            }
            Req::Expect(terms) => {
                let c = unitary_part(&data)?;
                let (rc, support) = match terms.iter().find(|t| !t.ops.is_empty()) {
                    Some(t) => rotate_to_z(&c, t),
                    None => (c.clone(), Vec::new()),
                };
                explain_plan(&rc, &PlanRequest::Expectation(support), mem, vec![])
            }
        })
    })
    .map_err(map_sim_err)?;
    explanation_dict(py, &ex)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(plan, m)?)?;
    let list = PyList::empty(m.py());
    for &(name, reqs, desc) in ENGINES {
        list.append((name, reqs, desc))?;
    }
    m.add("ENGINES", list)?;
    m.add("MAX_STATE_BYTES", MAX_STATE_BYTES)?;
    Ok(())
}

#[allow(dead_code)]
fn _assert_gate_used(_: Gate) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(engine: EngineSel) -> Opts {
        Opts {
            engine,
            f32: false,
            seed: 7,
            mem: 1 << 30,
            noise: NoiseModel::default(),
            repeat: None,
            explain: true,
        }
    }

    fn ghz(n: usize) -> CircuitData {
        let mut d = CircuitData::new(n);
        d.circuit.h(0);
        for q in 1..n {
            d.circuit.cnot(q - 1, q);
        }
        d
    }

    #[test]
    fn dense_pauli_expectation_matches_basis_rotation() {
        let mut d = CircuitData::new(3);
        d.circuit
            .h(0)
            .rx(1, 0.4)
            .cnot(0, 2)
            .ry(2, 1.1)
            .cphase(1, 2, 0.3)
            .t(0);
        let terms = [
            PauliTerm {
                sign: 1.0,
                ops: vec![(0, 1), (2, 3)],
            },
            PauliTerm {
                sign: -1.0,
                ops: vec![(1, 2)],
            },
            PauliTerm {
                sign: 1.0,
                ops: vec![(0, 2), (1, 1), (2, 2)],
            },
        ];
        let sv = dense_state::<f64>(&d.circuit, 1 << 30).unwrap();
        for t in &terms {
            let (rc, sup) = rotate_to_z(&d.circuit, t);
            let svr = dense_state::<f64>(&rc, 1 << 30).unwrap();
            let mask = sup.iter().fold(0usize, |m, &q| m | 1 << q);
            let z: f64 = svr
                .amplitudes()
                .iter()
                .enumerate()
                .map(|(x, a)| {
                    if (x & mask).count_ones() % 2 == 1 {
                        -a.norm_sqr()
                    } else {
                        a.norm_sqr()
                    }
                })
                .sum();
            let direct = pauli_expectation_dense(sv.amplitudes(), t);
            assert!(
                (direct - t.sign * z).abs() < 1e-12,
                "{t:?}: {direct} vs {}",
                t.sign * z
            );
        }
    }

    #[test]
    fn every_engine_agrees_on_ghz_amplitudes() {
        let d = ghz(6);
        let xs = vec![0u128, 63, 5];
        for e in [
            Engine::StateVector,
            Engine::Sparse,
            Engine::Mps,
            Engine::Hsf,
        ] {
            let r = run_request(&d, &Req::Amps(xs.clone()), &opts(EngineSel::Planner(e))).unwrap();
            let Payload::Amps(a) = r.payload else {
                panic!()
            };
            let h = std::f64::consts::FRAC_1_SQRT_2;
            assert!(
                (a[0].re - h).abs() < 1e-12 && (a[1].re - h).abs() < 1e-12 && a[2].norm() < 1e-12,
                "{e:?}"
            );
        }
        let r = run_request(&d, &Req::Amps(xs), &opts(EngineSel::Auto)).unwrap();
        assert!(r.explanation.unwrap().engine.is_some());
    }

    #[test]
    fn symphase_samples_are_correlated_like_ghz() {
        let mut d = ghz(5);
        d.circuit.measure_all();
        let r = run_request(&d, &Req::Samples(1000), &opts(EngineSel::Symphase)).unwrap();
        let Payload::Bits { m, data, .. } = r.payload else {
            panic!()
        };
        assert_eq!(m, 5);
        let ones: usize = data
            .chunks(5)
            .map(|s| {
                assert!(s.iter().all(|&b| b == s[0]));
                s[0] as usize
            })
            .sum();
        assert!((350..650).contains(&ones));
    }

    #[test]
    fn chunked_shot_loop_is_independent_of_parallelism() {
        let mut d = ghz(3);
        d.circuit.measure(0).reset(0).h(0).measure(0);
        let c = d.circuit.clone();
        let noise = NoiseModel::default();
        let a = shot_loop(&c, &noise, 700, 3, true, || Tableau::try_new(3)).unwrap();
        let b = shot_loop(&c, &noise, 700, 3, false, || Tableau::try_new(3)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn global_phase_multiplies_the_state() {
        let mut d = CircuitData::new(1);
        d.circuit.x(0);
        d.global_phase = std::f64::consts::FRAC_PI_2;
        let r = run_request(&d, &Req::State, &opts(EngineSel::Auto)).unwrap();
        let Payload::State64(v) = r.payload else {
            panic!()
        };
        assert!((v[1] - Complex64::i()).norm() < 1e-12);
    }
}
