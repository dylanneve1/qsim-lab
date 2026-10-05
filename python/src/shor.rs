//! Native half of `qsimlab.shor` (phase 2).
//!
//! Thin bindings over the engine's Shor code (`qsim_lab::shor`,
//! `shor_window`, `shor_superopt`, `shor_mbu`, `shor_ge`, `shor::noisy`):
//!
//! * [`factor`]: semiclassical order finding / Ekerå–Håstad runs with random
//!   (or given) bases, the same random stream as `qsim run shor --seed s`,
//!   with a memory guard predicted from the support law (research/shor/shor.md,
//!   research/theory/theory-shor.md T1) *before* anything is allocated;
//! * [`resource_counts`], [`oracle_circuit`], [`shor_circuit`]: circuits and
//!   whole-run gate counts without simulating;
//! * [`predict_support`]: the T1 support bounds `B_i` and the cost law;
//! * [`exact_distribution`]: the exact distribution of the measured integer
//!   (whole measurement tree, small N);
//! * [`noisy_trajectories`]: exact Monte-Carlo trajectories under Pauli
//!   noise (`shor::noisy`).
//!
//! The classical number theory used by the guard (Pollard–Brent factoring,
//! Carmichael function, multiplicative order) lives here too. It is only
//! used to *predict resources* and to grade noisy runs; the simulated
//! algorithm never sees it.

use crate::circuit::{CircuitData, PyCircuit};
use crate::errors::{qerr_with, unsupported, value_err};
use crate::threads::heavy;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use qsim_lab::algorithms::{gcd, pow_mod};
use qsim_lab::circuit::Circuit;
use qsim_lab::gate::Gate;
use qsim_lab::shor::noisy::{self, NoiseKind, NoisyCircuit};
use qsim_lab::shor::sliced::{self, SlicedState};
use qsim_lab::shor::{self, fused, Instance, Oracle, OrderFindingState, SemiRun};
use qsim_lab::shor_ge::{self, GeOpts, GeRun};
use qsim_lab::shor_mbu::{MbuCounts, MbuOpts};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::f64::consts::PI;
use std::time::Instant;

// ---------------------------------------------------------------------------
// classical number theory (resource prediction only)

fn mulm(a: u64, b: u64, m: u64) -> u64 {
    (u128::from(a) * u128::from(b) % u128::from(m)) as u64
}

/// Deterministic Miller–Rabin for 64-bit integers.
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n % p == 0 {
            return n == p;
        }
    }
    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;
    'witness: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mulm(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// A non-trivial factor of the odd composite `n` (Pollard–Brent).
fn rho(n: u64) -> u64 {
    if n % 2 == 0 {
        return 2;
    }
    let mut c = 1u64;
    loop {
        let f = |x: u64| (mulm(x, x, n) + c) % n;
        let (mut x, mut y, mut d) = (2u64, 2u64, 1u64);
        let mut q = 1u64;
        let mut iters = 0u64;
        while d == 1 {
            x = f(x);
            y = f(f(y));
            q = mulm(q, x.abs_diff(y), n);
            iters += 1;
            if iters % 64 == 0 || q == 0 {
                d = gcd(if q == 0 { x.abs_diff(y) } else { q }, n);
                if q == 0 && d == 1 {
                    d = n;
                }
            }
        }
        if d != n && d != 1 {
            return d;
        }
        c += 1;
    }
}

/// Prime factorisation `[(p, k)]`, sorted.
pub fn factorize(n: u64) -> Vec<(u64, u32)> {
    fn rec(n: u64, out: &mut Vec<u64>) {
        if n == 1 {
            return;
        }
        if is_prime(n) {
            out.push(n);
            return;
        }
        let d = rho(n);
        rec(d, out);
        rec(n / d, out);
    }
    let mut ps = Vec::new();
    let mut m = n;
    for p in [2u64, 3, 5, 7, 11, 13] {
        while m % p == 0 {
            ps.push(p);
            m /= p;
        }
    }
    rec(m, &mut ps);
    ps.sort_unstable();
    let mut out: Vec<(u64, u32)> = Vec::new();
    for p in ps {
        match out.last_mut() {
            Some((q, k)) if *q == p => *k += 1,
            _ => out.push((p, 1)),
        }
    }
    out
}

/// Carmichael's function λ(n).
pub fn carmichael(n: u64) -> u64 {
    let mut l = 1u64;
    for (p, k) in factorize(n) {
        let lp = if p == 2 {
            match k {
                1 => 1,
                2 => 2,
                _ => 1 << (k - 2),
            }
        } else {
            p.pow(k - 1) * (p - 1)
        };
        l = l / gcd(l, lp) * lp;
    }
    l
}

/// The multiplicative order of `a` mod `n` (`gcd(a, n) = 1`).
pub fn order(a: u64, n: u64) -> u64 {
    let mut r = carmichael(n);
    for (q, _) in factorize(r) {
        while r % q == 0 && pow_mod(a, r / q, n) == 1 {
            r /= q;
        }
    }
    r
}

/// T1 (research/theory/theory-shor.md): the support of the work register before
/// round `i` is at most `B_i = min(2^i, r / gcd(r, 2^(t−i)))`.
pub fn support_bounds(r: u64, t: usize) -> Vec<u64> {
    let nu = r.trailing_zeros() as usize;
    (0..t)
        .map(|i| {
            let big = if i >= 64 { u64::MAX } else { 1u64 << i };
            let g = 1u64 << nu.min(t - i);
            big.min(r / g)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// oracle kinds

#[derive(Clone, Copy, Debug)]
enum Kind {
    Shor(Oracle),
    /// Gidney–Ekerå exponent-windowed engine; `eh` = Ekerå–Håstad schedule.
    Ge {
        eh: bool,
        o: GeOpts,
    },
}

pub const KINDS: &[&str] = &[
    "permutation",
    "beauregard",
    "ripple",
    "windowed",
    "windowed-opt",
    "windowed-mbu-lookup",
    "windowed-mbu",
    "ge",
    "eh",
];

fn parse_kind(kind: &str, window: Option<usize>, ewindow: Option<usize>) -> PyResult<Kind> {
    let k = kind.to_ascii_lowercase().replace('_', "-");
    let w = window.unwrap_or(4);
    if !(1..=10).contains(&w) {
        return Err(value_err(format!("window must be in 1..=10, got {w}")));
    }
    Ok(match k.as_str() {
        "permutation" => Kind::Shor(Oracle::Permutation),
        "beauregard" => Kind::Shor(Oracle::Beauregard),
        "ripple" => Kind::Shor(Oracle::Ripple),
        "windowed" => Kind::Shor(Oracle::Windowed(w)),
        "windowed-opt" => Kind::Shor(Oracle::WindowedOpt(w)),
        "windowed-mbu-lookup" => Kind::Shor(Oracle::WindowedMbuLookup(w)),
        "windowed-mbu" => Kind::Shor(Oracle::WindowedMbu(w)),
        "ge" | "eh" => {
            let we = ewindow.unwrap_or(2);
            if !(1..=6).contains(&we) {
                return Err(value_err(format!(
                    "exponent_window must be in 1..=6, got {we}"
                )));
            }
            Kind::Ge {
                eh: k == "eh",
                o: GeOpts {
                    we,
                    wm: window.unwrap_or(3),
                    mbu: MbuOpts::LOOKUPS,
                    coset: 0,
                },
            }
        }
        _ => {
            return Err(value_err(format!(
                "unknown oracle '{kind}'; known: {}",
                KINDS.join(", ")
            )))
        }
    })
}

fn kind_name(k: &Kind) -> String {
    match k {
        Kind::Shor(o) => match o {
            Oracle::Permutation => "permutation".into(),
            Oracle::Beauregard => "beauregard".into(),
            Oracle::Ripple => "ripple".into(),
            Oracle::Windowed(w) => format!("windowed(w={w})"),
            Oracle::WindowedOpt(w) => format!("windowed-opt(w={w})"),
            Oracle::WindowedMbuLookup(w) => format!("windowed-mbu-lookup(w={w})"),
            Oracle::WindowedMbu(w) => format!("windowed-mbu(w={w})"),
        },
        Kind::Ge { eh, o } => format!(
            "{}(w_e={}, w_m={})",
            if *eh { "eh" } else { "ge" },
            o.we,
            o.wm
        ),
    }
}

/// Checks `N` is something Shor's algorithm applies to.
fn check_modulus(n: u64) -> PyResult<()> {
    if n < 15 {
        return Err(value_err(format!(
            "N = {n} is too small; the smallest N Shor's algorithm applies to is 15"
        )));
    }
    if n % 2 == 0 {
        return Err(value_err(format!(
            "N = {n} is even (N = 2 × {}); Shor's algorithm needs an odd composite",
            n / 2
        )));
    }
    if n >= 1 << 62 {
        return Err(value_err(format!(
            "N = {n} has more than 62 bits; the order-finding engines take N < 2^62"
        )));
    }
    let f = factorize(n);
    if f.len() == 1 {
        let (p, k) = f[0];
        return Err(value_err(if k == 1 {
            format!("N = {n} is prime; nothing to factor")
        } else {
            format!(
                "N = {n} = {p}^{k} is a prime power; Shor's algorithm needs at least two \
                 distinct prime factors (prime powers are found classically by integer roots)"
            )
        }));
    }
    Ok(())
}

fn check_base(n: u64, a: u64) -> PyResult<()> {
    if a < 2 || a >= n - 1 {
        return Err(value_err(format!("base must be in [2, N − 2], got {a}")));
    }
    if gcd(a, n) != 1 {
        return Err(value_err(format!(
            "base {a} shares the factor {} with N = {n} (gcd > 1): no quantum step needed",
            gcd(a, n)
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// memory prediction

/// Which state backs a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Eng {
    Sliced,
    FusedDense,
    FusedSparse,
    Dense,
    Sparse,
    Ge,
}

fn eng_name(e: Eng) -> &'static str {
    match e {
        Eng::Sliced => "sliced",
        Eng::FusedDense => "fused-dense",
        Eng::FusedSparse => "fused-sparse",
        Eng::Dense => "statevector",
        Eng::Sparse => "sparse",
        Eng::Ge => "ge-windowed",
    }
}

/// Peak support of the work register over the stored rounds:
/// `max_{i<t} B_i` (= `max(r_odd, r/2)` for even `r`, `r` for odd `r`).
fn peak_support(r: u64, t: usize) -> u64 {
    support_bounds(r, t).into_iter().max().unwrap_or(1)
}

/// Bytes per stored support element, measured peak RSS / peak support in
/// research/shor/shor.md ("≈ 33 B (f32) and ≈ 51 B (f64)").
fn sliced_bytes_per_elem(f32: bool) -> u128 {
    if f32 {
        33
    } else {
        51
    }
}

/// Predicted peak bytes of one run.
fn predict_bytes(kind: &Kind, eng: Eng, m: usize, r: u64, f32: bool) -> u128 {
    let t = 2 * m;
    let elem: u128 = if f32 { 8 } else { 16 };
    let peak = u128::from(peak_support(r, t));
    match eng {
        Eng::Sliced => peak * sliced_bytes_per_elem(f32),
        // ψ plus the buffer for Uψ, 2^m amplitudes each
        Eng::FusedDense => 2 * (1u128 << m.min(100)) * elem,
        // hash-map entries (key, amplitude, slack); the Beauregard
        // accumulator lives in Fourier space, so its sparse state is dense
        // (research/shor/shor.md, lever 3), and a ripple round holds both
        // control branches
        Eng::FusedSparse => peak * 64,
        Eng::Sparse => match kind {
            Kind::Shor(Oracle::Beauregard) => (1u128 << (2 * m + 3).min(120)) * 64,
            _ => 2 * peak * 64,
        },
        Eng::Dense => {
            let qubits = match kind {
                Kind::Shor(Oracle::Beauregard) => 2 * m + 3,
                _ => m + 1,
            };
            (1u128 << qubits.min(120)) * elem
        }
        // research/shor/ge-shor.md §5: a `w_e` window holds 2^{w_e} (key,
        // amplitude) branches per stored value, the stored support is at most
        // `r`, and the input and output arrays coexist (factor 2).
        Eng::Ge => {
            let we = match kind {
                Kind::Ge { o, .. } => o.we,
                _ => 1,
            };
            2 * (u128::from(r) << we) * (elem + 8)
        }
    }
}

fn choose_engine(kind: &Kind, engine: &str, m: usize, r: u64, f32: bool) -> PyResult<Eng> {
    let e = engine.to_ascii_lowercase();
    let k = match kind {
        Kind::Ge { .. } => {
            return match e.as_str() {
                "auto" | "sliced" | "ge" => Ok(Eng::Ge),
                _ => Err(value_err(format!(
                    "engine '{engine}' cannot run the {} schedule (use 'auto')",
                    kind_name(kind)
                ))),
            };
        }
        Kind::Shor(o) => *o,
    };
    let reversible = !matches!(k, Oracle::Permutation | Oracle::Beauregard);
    Ok(match (e.as_str(), k) {
        ("auto", Oracle::Permutation) => {
            let dense = predict_bytes(kind, Eng::FusedDense, m, r, f32);
            let sparse = predict_bytes(kind, Eng::FusedSparse, m, r, f32);
            if sparse < dense {
                Eng::FusedSparse
            } else {
                Eng::FusedDense
            }
        }
        ("auto", Oracle::Beauregard) => Eng::Dense,
        ("auto" | "sliced", _) if reversible => Eng::Sliced,
        ("dense" | "statevector", Oracle::Permutation) => Eng::FusedDense,
        ("sparse", Oracle::Permutation) => Eng::FusedSparse,
        ("dense" | "statevector", Oracle::Beauregard) => Eng::Dense,
        ("sparse", Oracle::Beauregard | Oracle::Ripple) => {
            if 3 * m + 4 > 64 {
                return Err(value_err(format!(
                    "the sparse state keys basis states in 64 bits; the {} circuit for an \
                     {m}-bit N needs more qubits (use engine='auto')",
                    kind_name(kind)
                )));
            }
            Eng::Sparse
        }
        _ => {
            return Err(value_err(format!(
                "engine '{engine}' cannot run the {} oracle; use 'auto' \
                 (permutation: 'dense' | 'sparse'; beauregard: 'dense' | 'sparse'; \
                 ripple: 'sliced' | 'sparse'; windowed*: 'sliced')",
                kind_name(kind)
            )))
        }
    })
}

// ---------------------------------------------------------------------------
// runs

/// `shor::run_semiclassical`, returning the final state as well (for the
/// per-round traces). Same random draws, same gate bookkeeping.
fn run_keep<S: OrderFindingState, R: Rng + ?Sized>(
    inst: &Instance,
    mut s: S,
    rng: &mut R,
) -> (SemiRun, S) {
    let mut y = 0u128;
    let (mut peak_stored, mut peak_bytes) = (s.stored(), s.bytes());
    let (mut total_gates, mut toffoli_gates, mut measurements) = (0usize, 0usize, 0usize);
    for i in 0..inst.t {
        let k = inst.t - 1 - i;
        let mult = inst.mults[k];
        match inst.oracle {
            Oracle::Permutation => {}
            Oracle::Beauregard => {
                let lay = inst.layout();
                let c = qsim_lab::shor_arith::controlled_ua(&lay, 0, mult, inst.n_mod);
                total_gates += c.ops.len() + 2 + usize::from(y != 0);
            }
            _ => {
                let (ops, _, _) = sliced::oracle_ops(inst, mult);
                let c = MbuCounts::of(&ops);
                total_gates += c.total + 2 + usize::from(y != 0);
                toffoli_gates += c.toffoli;
                measurements += c.meas;
            }
        }
        s.round(inst, i, y);
        peak_stored = peak_stored.max(s.stored());
        peak_bytes = peak_bytes.max(s.bytes());
        let p1 = s.prob_one(0);
        let bit = rng.random::<f64>() < p1;
        s.collapse(0, bit);
        s.reset_control(bit);
        if bit {
            y |= 1 << i;
            if !matches!(inst.oracle, Oracle::Permutation) {
                total_gates += 1;
            }
        }
    }
    peak_stored = peak_stored.max(s.stored());
    let (order, factor) = shor::postprocess(inst.n_mod, inst.a, y, inst.t as u32);
    let run = SemiRun {
        a: inst.a,
        measured: y,
        order,
        factor,
        qubits: inst.qubits(),
        peak_stored,
        peak_bytes,
        total_gates,
        toffoli_gates,
        work_ops: s.work_ops(),
        measurements,
    };
    (run, s)
}

/// Everything one attempt reports.
#[derive(Clone, Debug, Default)]
struct Attempt {
    a: u64,
    /// Shor: one integer; GE/EH: one per exponent register.
    measured: Vec<u128>,
    order: Option<u64>,
    factors: Option<(u64, u64)>,
    qubits: usize,
    total_gates: usize,
    toffoli_gates: usize,
    measurements: usize,
    peak_support: usize,
    peak_bytes: usize,
    work_ops: u128,
    support_trace: Option<Vec<usize>>,
    p1_trace: Option<Vec<f64>>,
    true_order: u64,
    predicted_peak_support: u64,
    predicted_bytes: u128,
    engine: &'static str,
    secs: f64,
}

fn split(n: u64, f: u64) -> (u64, u64) {
    (f.min(n / f), f.max(n / f))
}

#[allow(clippy::too_many_arguments)]
fn attempt(
    n: u64,
    a: u64,
    kind: &Kind,
    eng: Eng,
    f32: bool,
    trace: bool,
    r: u64,
    pred: u128,
    rng: &mut StdRng,
) -> Attempt {
    let t0 = Instant::now();
    let m = shor::work_bits(n);
    let mut at = Attempt {
        a,
        true_order: r,
        predicted_peak_support: peak_support(r, 2 * m),
        predicted_bytes: pred,
        engine: eng_name(eng),
        ..Default::default()
    };
    match *kind {
        Kind::Shor(o) => {
            let inst = Instance::new(n, a, o);
            let run = match (eng, f32) {
                (Eng::Sliced, false) => {
                    let (run, s) = run_keep(&inst, SlicedState::<f64>::new(&inst), rng);
                    if trace {
                        at.support_trace = Some(s.support_trace.clone());
                        at.p1_trace = Some(s.p1_trace.clone());
                    }
                    run
                }
                (Eng::Sliced, true) => {
                    let (run, s) = run_keep(&inst, SlicedState::<f32>::new(&inst), rng);
                    if trace {
                        at.support_trace = Some(s.support_trace.clone());
                        at.p1_trace = Some(s.p1_trace.clone());
                    }
                    run
                }
                (Eng::FusedDense, false) => {
                    run_keep(&inst, fused::FusedDense::<f64>::new(&inst), rng).0
                }
                (Eng::FusedDense, true) => {
                    run_keep(&inst, fused::FusedDense::<f32>::new(&inst), rng).0
                }
                (Eng::FusedSparse, _) => run_keep(&inst, fused::FusedSparse::new(&inst), rng).0,
                (Eng::Dense, false) => run_keep(&inst, shor::dense_initial::<f64>(&inst), rng).0,
                (Eng::Dense, true) => run_keep(&inst, shor::dense_initial::<f32>(&inst), rng).0,
                (Eng::Sparse, _) => run_keep(&inst, shor::sparse_initial(&inst), rng).0,
                (Eng::Ge, _) => unreachable!(),
            };
            at.measured = vec![run.measured];
            at.order = run.order;
            at.factors = run.factor.map(|f| split(n, f));
            at.qubits = run.qubits;
            at.total_gates = run.total_gates;
            at.toffoli_gates = run.toffoli_gates;
            at.measurements = run.measurements;
            at.peak_support = run.peak_stored;
            at.peak_bytes = run.peak_bytes;
            at.work_ops = run.work_ops;
        }
        Kind::Ge { eh, o } => {
            let mut draw = || rng.random::<f64>();
            let (run, order, factors): (GeRun, Option<u64>, Option<(u64, u64)>) = if eh {
                let (run, f) = if f32 {
                    shor_ge::eh_run::<f32>(n, a, &o, &mut draw)
                } else {
                    shor_ge::eh_run::<f64>(n, a, &o, &mut draw)
                };
                (run, None, f)
            } else {
                let (run, order, f) = if f32 {
                    shor_ge::shor_run::<f32>(n, a, &o, &mut draw)
                } else {
                    shor_ge::shor_run::<f64>(n, a, &o, &mut draw)
                };
                (run, order, f.map(|f| split(n, f)))
            };
            at.measured = run.y.clone();
            at.order = order;
            at.factors = factors;
            at.qubits = run.qubits;
            at.total_gates = run.counts.total;
            at.toffoli_gates = run.counts.toffoli;
            at.measurements = run.counts.meas;
            at.peak_support = run.peak;
            at.peak_bytes = run.peak_branches;
            at.work_ops = run.gate_branch_ops;
        }
    }
    at.secs = t0.elapsed().as_secs_f64();
    at
}

fn u128_to_py<'py>(py: Python<'py>, v: u128) -> PyResult<Bound<'py, PyAny>> {
    Ok(v.into_pyobject(py)?.into_any())
}

fn attempt_dict<'py>(py: Python<'py>, at: &Attempt) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("base", at.a)?;
    let ms = PyList::empty(py);
    for &y in &at.measured {
        ms.append(u128_to_py(py, y)?)?;
    }
    d.set_item("measured", ms)?;
    d.set_item("order", at.order)?;
    d.set_item("factors", at.factors)?;
    d.set_item("qubits", at.qubits)?;
    d.set_item("total_gates", at.total_gates)?;
    d.set_item("toffoli_gates", at.toffoli_gates)?;
    d.set_item("measurements", at.measurements)?;
    d.set_item("peak_support", at.peak_support)?;
    d.set_item("peak_amplitude_bytes", at.peak_bytes)?;
    d.set_item("gate_branch_ops", u128_to_py(py, at.work_ops)?)?;
    d.set_item("true_order", at.true_order)?;
    d.set_item("predicted_peak_support", at.predicted_peak_support)?;
    d.set_item("predicted_bytes", u128_to_py(py, at.predicted_bytes)?)?;
    d.set_item("engine", at.engine)?;
    d.set_item("wall_time", at.secs)?;
    match &at.support_trace {
        Some(s) => d.set_item(
            "support_trace",
            PyArray1::from_vec(py, s.iter().map(|&x| x as u64).collect::<Vec<u64>>()),
        )?,
        None => d.set_item("support_trace", py.None())?,
    }
    match &at.p1_trace {
        Some(s) => d.set_item("p1_trace", PyArray1::from_vec(py, s.clone()))?,
        None => d.set_item("p1_trace", py.None())?,
    }
    Ok(d)
}

/// Every attempt and the factors found, if any.
type Attempts = (Vec<Attempt>, Option<(u64, u64)>);

enum FactorErr {
    Budget {
        a: u64,
        r: u64,
        needed: u128,
        limit: u128,
        engine: &'static str,
        peak: u64,
    },
    Value(String),
}

/// Factors `N`: up to `tries` order-finding (or EH) runs.
#[pyfunction]
#[pyo3(signature = (n, kind, window=None, exponent_window=None, base=None, f32=false, seed=0,
                    tries=10, budget=1u128 << 35, engine="auto", trace=true, threads=None))]
#[allow(clippy::too_many_arguments)]
fn factor<'py>(
    py: Python<'py>,
    n: u64,
    kind: &str,
    window: Option<usize>,
    exponent_window: Option<usize>,
    base: Option<u64>,
    f32: bool,
    seed: u64,
    tries: usize,
    budget: u128,
    engine: &str,
    trace: bool,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    check_modulus(n)?;
    let kind = parse_kind(kind, window, exponent_window)?;
    if let Some(a) = base {
        check_base(n, a)?;
    }
    if tries == 0 {
        return Err(value_err("tries must be ≥ 1"));
    }
    if let Kind::Shor(Oracle::Beauregard) = kind {
        if f32 && engine == "sparse" {
            return Err(value_err("the sparse engine computes in f64 only"));
        }
    }
    if let Kind::Ge { eh: true, .. } = kind {
        let half = shor_ge::eh_m(n) as u32;
        if let Some(&(p, _)) = factorize(n).iter().find(|&&(p, _)| p >= 1u64 << half) {
            return Err(value_err(format!(
                "the Ekerå–Håstad schedule needs a balanced N (every prime factor below \
                 2^{half}, so that d = (p + q − 2)/2 has at most {half} bits); N = {n} has the \
                 factor {p} (checked classically). Use oracle='ge' or a Shor oracle"
            )));
        }
    }
    let m = shor::work_bits(n);
    // engine choice may depend on r (permutation auto); check it parses
    choose_engine(&kind, engine, m, 2, f32)?;
    let engine = engine.to_string();
    let t0 = Instant::now();
    let res: Result<Attempts, FactorErr> = heavy(py, threads, move || {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut runs = Vec::new();
        for _ in 0..tries {
            let a = match base {
                Some(a) => a,
                None => {
                    let a = rng.random_range(2..n - 1);
                    if gcd(a, n) > 1 {
                        continue; // lucky classical guess; skip so the quantum part runs
                    }
                    a
                }
            };
            let r = order(a, n);
            let eng = choose_engine(&kind, &engine, m, r, f32)
                .map_err(|e| FactorErr::Value(e.to_string()))?;
            let pred = predict_bytes(&kind, eng, m, r, f32);
            if pred > budget {
                return Err(FactorErr::Budget {
                    a,
                    r,
                    needed: pred,
                    limit: budget,
                    engine: eng_name(eng),
                    peak: peak_support(r, 2 * m),
                });
            }
            let at = attempt(n, a, &kind, eng, f32, trace, r, pred, &mut rng);
            let f = at.factors;
            runs.push(at);
            if f.is_some() {
                return Ok((runs, f));
            }
        }
        Ok((runs, None))
    });
    let (runs, found) = match res {
        Ok(x) => x,
        Err(FactorErr::Value(s)) => return Err(value_err(s)),
        Err(FactorErr::Budget {
            a,
            r,
            needed,
            limit,
            engine,
            peak,
        }) => {
            return Err(qerr_with(
                "ResourceLimitError",
                format!(
                    "refusing to run: N = {n} with base a = {a} has multiplicative order r = {r} \
                     (ν₂(r) = {}), so the {engine} engine's peak support is ≈ {peak} branches and \
                     it would need ≈ {:.3} GB, over the budget of {:.3} GB (prediction from the \
                     support law, research/theory/theory-shor.md T1; the order was computed classically \
                     for this check only). Raise budget=, pass another base=, use f32 precision, \
                     or a smaller N.",
                    r.trailing_zeros(),
                    needed as f64 / 1e9,
                    limit as f64 / 1e9
                ),
                &[("needed", needed), ("limit", limit)],
            ))
        }
    };
    let d = PyDict::new(py);
    d.set_item("n", n)?;
    d.set_item("oracle", kind_name(&kind))?;
    d.set_item("factors", found)?;
    let lst = PyList::empty(py);
    for at in &runs {
        lst.append(attempt_dict(py, at)?)?;
    }
    d.set_item("runs", lst)?;
    d.set_item("seed", seed)?;
    d.set_item("precision", if f32 { "f32" } else { "f64" })?;
    d.set_item("wall_time", t0.elapsed().as_secs_f64())?;
    Ok(d)
}

// ---------------------------------------------------------------------------
// counts and circuits

fn default_base(n: u64, base: Option<u64>) -> PyResult<u64> {
    match base {
        Some(a) => {
            check_base(n, a)?;
            Ok(a)
        }
        None => Ok((2..n - 1).find(|&a| gcd(a, n) == 1).unwrap()),
    }
}

/// Whole-run counts of every controlled-`U` block (and the control
/// overhead) without simulating.
#[pyfunction]
#[pyo3(signature = (n, kind, window=None, exponent_window=None, base=None, per_round=false, threads=None))]
#[allow(clippy::too_many_arguments)]
fn resource_counts<'py>(
    py: Python<'py>,
    n: u64,
    kind: &str,
    window: Option<usize>,
    exponent_window: Option<usize>,
    base: Option<u64>,
    per_round: bool,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    check_modulus(n)?;
    let kind = parse_kind(kind, window, exponent_window)?;
    let a = default_base(n, base)?;
    let d = PyDict::new(py);
    d.set_item("n", n)?;
    d.set_item("base", a)?;
    d.set_item("oracle", kind_name(&kind))?;
    match kind {
        Kind::Shor(o) => {
            let inst = Instance::new(n, a, o);
            let rounds: Vec<MbuCounts> = heavy(py, threads, || {
                (0..inst.t)
                    .into_par_iter()
                    .map(|i| {
                        let mult = inst.mults[inst.t - 1 - i];
                        match o {
                            Oracle::Permutation => MbuCounts::default(),
                            Oracle::Beauregard => {
                                let c =
                                    qsim_lab::shor_arith::controlled_ua(&inst.layout(), 0, mult, n);
                                let mut k = MbuCounts::default();
                                for g in c.gates() {
                                    k.total += 1;
                                    match g {
                                        Gate::Ccx(..) => k.toffoli += 1,
                                        Gate::Cnot(..) => k.cnot += 1,
                                        Gate::X(_) => k.x += 1,
                                        _ => {}
                                    }
                                }
                                k
                            }
                            _ => MbuCounts::of(&sliced::oracle_ops(&inst, mult).0),
                        }
                    })
                    .collect()
            });
            let mut tot = MbuCounts::default();
            for r in &rounds {
                tot.add(r);
            }
            let gate_level = !matches!(o, Oracle::Permutation);
            d.set_item("qubits", inst.qubits())?;
            d.set_item("rounds", inst.t)?;
            d.set_item("exponent_bits", inst.t)?;
            d.set_item("gate_level", gate_level)?;
            d.set_item("oracle_gates", tot.total)?;
            d.set_item("toffoli", tot.toffoli)?;
            d.set_item("cnot", tot.cnot)?;
            d.set_item("x", tot.x)?;
            d.set_item("measurements", tot.meas)?;
            d.set_item("fixups", tot.fixup)?;
            if per_round {
                d.set_item(
                    "per_round_gates",
                    PyArray1::from_vec(py, rounds.iter().map(|c| c.total as u64).collect()),
                )?;
                d.set_item(
                    "per_round_toffoli",
                    PyArray1::from_vec(py, rounds.iter().map(|c| c.toffoli as u64).collect()),
                )?;
            }
        }
        Kind::Ge { eh, o } => {
            let regs = if eh {
                shor_ge::eh_regs(n, a)
            } else {
                shor_ge::shor_regs(n, a)
            };
            let bits: usize = regs.iter().map(|r| r.len).sum();
            let (c, steps, nq) = heavy(py, threads, || shor_ge::schedule_counts(n, &regs, &o));
            d.set_item("qubits", nq)?;
            d.set_item("rounds", bits)?;
            d.set_item("exponent_bits", bits)?;
            d.set_item("gate_level", true)?;
            d.set_item("oracle_gates", c.total)?;
            d.set_item("toffoli", c.toffoli)?;
            d.set_item("cnot", c.cnot)?;
            d.set_item("x", c.x)?;
            d.set_item("measurements", c.meas)?;
            d.set_item("fixups", c.fixup)?;
            d.set_item("slice_steps", steps)?;
        }
    }
    Ok(d)
}

fn reversible_block(inst: &Instance, mult: u64) -> PyResult<(Circuit, Vec<usize>)> {
    match inst.oracle {
        Oracle::Beauregard => {
            let lay = inst.layout();
            let c = qsim_lab::shor_arith::controlled_ua(&lay, 0, mult, inst.n_mod);
            Ok((c, (1..=inst.m).collect()))
        }
        Oracle::Ripple | Oracle::Windowed(_) | Oracle::WindowedOpt(_) => {
            let (c, io) = sliced::oracle_block(inst, mult);
            Ok((c, io.x))
        }
        Oracle::Permutation => Err(unsupported(
            "the permutation oracle is a classical lookup table, not a gate-level circuit; \
             use 'beauregard', 'ripple', 'windowed' or 'windowed-opt'",
        )),
        Oracle::WindowedMbu(_) | Oracle::WindowedMbuLookup(_) => Err(unsupported(
            "measurement-based uncomputation needs classical parity feed-forward over several \
             X-basis measurements, which a Circuit (single-bit c_if) cannot express; use \
             resource_counts() for these oracles, or 'windowed-opt' for the unitary circuit",
        )),
    }
}

fn wrap(py: Python<'_>, c: Circuit) -> PyResult<Py<PyCircuit>> {
    Py::new(
        py,
        PyCircuit::from_data(CircuitData {
            circuit: c,
            ..Default::default()
        }),
    )
}

/// The controlled-`U_a` block (`|c>|x>|0> -> |c>|a^c x mod N>|0>`) as a
/// circuit, with its control and work qubits.
#[pyfunction]
#[pyo3(signature = (n, a, kind, window=None))]
fn oracle_circuit<'py>(
    py: Python<'py>,
    n: u64,
    a: u64,
    kind: &str,
    window: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    if n < 3 || n % 2 == 0 {
        return Err(value_err(format!("N must be odd and ≥ 3, got {n}")));
    }
    if a < 1 || a >= n || gcd(a, n) != 1 {
        return Err(value_err(format!(
            "a must be in [1, N) and coprime to N = {n}, got {a}"
        )));
    }
    let Kind::Shor(o) = parse_kind(kind, window, None)? else {
        return Err(unsupported(
            "the ge/eh schedules are windowed over the exponent; oracle_circuit builds one \
             controlled-U_a block of the Shor oracles",
        ));
    };
    let inst = Instance::new(n, a, o);
    let (c, work) = reversible_block(&inst, a % n)?;
    let d = PyDict::new(py);
    d.set_item("control", 0)?;
    d.set_item("work", work)?;
    d.set_item("num_qubits", c.num_qubits)?;
    d.set_item("circuit", wrap(py, c)?)?;
    Ok(d)
}

/// The whole semiclassical order-finding circuit (one recycled control,
/// classically controlled phase corrections; measurement `i` is bit `i`
/// of the measured integer).
#[pyfunction]
#[pyo3(signature = (n, a, kind, window=None))]
fn shor_circuit(
    py: Python<'_>,
    n: u64,
    a: u64,
    kind: &str,
    window: Option<usize>,
) -> PyResult<Py<PyCircuit>> {
    check_modulus(n)?;
    check_base(n, a)?;
    let Kind::Shor(o) = parse_kind(kind, window, None)? else {
        return Err(unsupported(
            "shor_circuit builds the Shor oracles' circuits; the ge/eh schedules are not \
             exported as circuits",
        ));
    };
    let inst = Instance::new(n, a, o);
    let mut c = Circuit::new(inst.qubits());
    c.x(1); // work register |1>
    for i in 0..inst.t {
        let k = inst.t - 1 - i;
        let (blk, _) = reversible_block(&inst, inst.mults[k])?;
        if i > 0 {
            c.c_if(i - 1, Gate::X(0));
        }
        c.h(0);
        c.append(&blk);
        for l in 0..i {
            c.c_if(l, Gate::Phase(0, -PI / ((1u64 << (i - l)) as f64)));
        }
        c.h(0);
        c.measure(0);
    }
    wrap(py, c)
}

/// The exact distribution of the measured `2n`-bit integer, by walking the
/// whole measurement tree (small N only).
#[pyfunction]
#[pyo3(signature = (n, a, kind, window=None, prune=0.0, threads=None))]
fn exact_distribution<'py>(
    py: Python<'py>,
    n: u64,
    a: u64,
    kind: &str,
    window: Option<usize>,
    prune: f64,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    check_modulus(n)?;
    check_base(n, a)?;
    let Kind::Shor(o) = parse_kind(kind, window, None)? else {
        return Err(unsupported(
            "exact_distribution supports the Shor oracles (not ge/eh)",
        ));
    };
    let m = shor::work_bits(n);
    // the tree has 2^(2n) leaves and every node holds a copy of the state
    let max_m = match o {
        Oracle::Permutation => 10,
        Oracle::Beauregard => 6,
        _ => 8,
    };
    if m > max_m {
        return Err(value_err(format!(
            "exact_distribution walks all 2^(2n) measurement outcomes; N = {n} has n = {m} bits, \
             the limit for this oracle is {max_m}"
        )));
    }
    let inst = Instance::new(n, a, o);
    let v = heavy(py, threads, || match o {
        Oracle::Permutation | Oracle::Beauregard => {
            shor::semiclassical_distribution(&inst, shor::dense_initial::<f64>(&inst), prune)
        }
        _ => shor::semiclassical_distribution(&inst, SlicedState::<f64>::new(&inst), prune),
    });
    Ok(PyArray1::from_vec(py, v))
}

/// The T1 support bounds and the cost law for `(N, a)`.
#[pyfunction]
#[pyo3(signature = (n, a, kind="windowed-opt", window=None, exponent_window=None, f32=false))]
fn predict_support<'py>(
    py: Python<'py>,
    n: u64,
    a: u64,
    kind: &str,
    window: Option<usize>,
    exponent_window: Option<usize>,
    f32: bool,
) -> PyResult<Bound<'py, PyDict>> {
    check_modulus(n)?;
    check_base(n, a)?;
    let kind = parse_kind(kind, window, exponent_window)?;
    let m = shor::work_bits(n);
    let t = 2 * m;
    let r = order(a, n);
    let b = support_bounds(r, t);
    let d = PyDict::new(py);
    d.set_item("order", r)?;
    d.set_item("nu", r.trailing_zeros())?;
    d.set_item("rounds", t)?;
    d.set_item("peak", peak_support(r, t))?;
    let sum: u128 = b.iter().map(|&x| u128::from(x)).sum();
    d.set_item("sum", u128_to_py(py, sum)?)?;
    let eng = choose_engine(&kind, "auto", m, r, f32)?;
    d.set_item("engine", eng_name(eng))?;
    d.set_item(
        "predicted_bytes",
        u128_to_py(py, predict_bytes(&kind, eng, m, r, f32))?,
    )?;
    d.set_item("bounds", PyArray1::from_vec(py, b))?;
    Ok(d)
}

/// `(order of a, λ(n), [(prime, exponent)])`.
type NumberTheory = (Option<u64>, u64, Vec<(u64, u32)>);

/// Classical helpers: `(order, carmichael, factorisation)`.
#[pyfunction]
fn number_theory(n: u64, a: Option<u64>) -> PyResult<NumberTheory> {
    if n < 2 {
        return Err(value_err("n must be ≥ 2"));
    }
    let ord = match a {
        Some(a) => {
            if gcd(a, n) != 1 {
                return Err(value_err(format!("{a} is not coprime to {n}")));
            }
            Some(order(a % n, n))
        }
        None => None,
    };
    Ok((ord, carmichael(n), factorize(n)))
}

// ---------------------------------------------------------------------------
// noisy trajectories

fn mix(seed: u64, i: u64) -> u64 {
    let mut z = seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `|y/2^t − s/r| < 1/(2r²)` for the nearest `s` (the textbook "good y").
fn peak_ok(y: u128, r: u64, t: usize) -> bool {
    let r = u128::from(r);
    let big = 1u128 << t;
    let s = (y * r + big / 2) / big;
    let num = (y * r).abs_diff(s * big); // |y r − s 2^t|
                                         // |y/2^t − s/r| = num / (r 2^t) < 1/(2r²)  ⇔  2 r num < 2^t
    2 * r * num < big
}

/// Exact Monte-Carlo trajectories of the gate-level circuit under Pauli
/// noise (`shor::noisy`). `faults`: fixed fault count per trajectory
/// (stratified) instead of the rate `p`.
#[pyfunction]
#[pyo3(signature = (n, a, p, noise="depolarizing", trajectories=100, kind="windowed", window=None,
                    seed=0, faults=None, cap=1usize << 26, reset_ancillas=false, threads=None))]
#[allow(clippy::too_many_arguments)]
fn noisy_trajectories<'py>(
    py: Python<'py>,
    n: u64,
    a: u64,
    p: f64,
    noise: &str,
    trajectories: usize,
    kind: &str,
    window: Option<usize>,
    seed: u64,
    faults: Option<usize>,
    cap: usize,
    reset_ancillas: bool,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    check_modulus(n)?;
    check_base(n, a)?;
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        return Err(value_err(format!("p must be a probability, got {p}")));
    }
    let nk = NoiseKind::parse(&noise.to_ascii_lowercase()).ok_or_else(|| {
        value_err(format!(
            "unknown noise '{noise}'; known: depolarizing (depol), bitflip, phaseflip"
        ))
    })?;
    let Kind::Shor(o) = parse_kind(kind, window, None)? else {
        return Err(unsupported("noisy trajectories run the Shor oracles"));
    };
    if !matches!(
        o,
        Oracle::Ripple | Oracle::Windowed(_) | Oracle::WindowedOpt(_)
    ) {
        return Err(unsupported(format!(
            "the noisy engine supports the reversible oracles 'ripple', 'windowed' and \
             'windowed-opt', not {}",
            kind_name(&Kind::Shor(o))
        )));
    }
    let inst = Instance::new(n, a, o);
    let qubits = inst.qubits();
    if qubits > 129 {
        return Err(value_err(format!(
            "the noisy engine keys the non-control qubits in 128 bits; this circuit has {} qubits",
            inst.qubits()
        )));
    }
    let r = order(a, n);
    let t = inst.t;
    let out = heavy(py, threads, move || {
        let nc = NoisyCircuit::new(&inst, nk);
        let locs = nc.num_locations();
        let res: Vec<(noisy::Trajectory, usize)> = (0..trajectories)
            .into_par_iter()
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(mix(seed, i as u64));
                let fs = match faults {
                    Some(k) => nc.sample_k(k, &mut rng),
                    None => nc.sample_p(p, &mut rng),
                };
                let mut tr =
                    noisy::run_trajectory_opts::<f64, _>(&nc, &fs, cap, reset_ancillas, &mut rng);
                tr.support_trace = Vec::new();
                (tr, fs.len())
            })
            .collect();
        (res, locs)
    });
    let (res, locs) = out;
    let d = PyDict::new(py);
    d.set_item("locations", locs)?;
    d.set_item("order", r)?;
    d.set_item("qubits", qubits)?;
    let ms = PyList::empty(py);
    let mut factor_ok = Vec::with_capacity(res.len());
    let mut order_ok = Vec::with_capacity(res.len());
    let mut pk = Vec::with_capacity(res.len());
    let mut capped = Vec::with_capacity(res.len());
    let mut nf = Vec::with_capacity(res.len());
    for (tr, k) in &res {
        match tr.measured {
            Some(y) => ms.append(u128_to_py(py, y)?)?,
            None => ms.append(py.None())?,
        }
        factor_ok.push(tr.factor.is_some());
        order_ok.push(tr.order == Some(r));
        pk.push(tr.measured.is_some_and(|y| peak_ok(y, r, t)));
        capped.push(tr.capped.is_some());
        nf.push(*k as u64);
    }
    d.set_item("measured", ms)?;
    d.set_item("factor_ok", PyArray1::from_vec(py, factor_ok))?;
    d.set_item("order_ok", PyArray1::from_vec(py, order_ok))?;
    d.set_item("peak_ok", PyArray1::from_vec(py, pk))?;
    d.set_item("capped", PyArray1::from_vec(py, capped))?;
    d.set_item("faults", PyArray1::from_vec(py, nf))?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "native part of qsimlab.shor (phase 2)")?;
    m.add("ORACLES", KINDS.to_vec())?;
    m.add_function(wrap_pyfunction!(factor, m)?)?;
    m.add_function(wrap_pyfunction!(resource_counts, m)?)?;
    m.add_function(wrap_pyfunction!(oracle_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(shor_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(exact_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(predict_support, m)?)?;
    m.add_function(wrap_pyfunction!(number_theory, m)?)?;
    m.add_function(wrap_pyfunction!(noisy_trajectories, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order_brute(a: u64, n: u64) -> u64 {
        let mut x = a % n;
        let mut r = 1;
        while x != 1 {
            x = mulm(x, a, n);
            r += 1;
        }
        r
    }

    #[test]
    fn number_theory_matches_brute_force() {
        for n in [15u64, 21, 35, 143, 1003, 4087, 65_535, 1_005_973] {
            for a in 2..40u64 {
                if gcd(a, n) == 1 {
                    assert_eq!(order(a, n), order_brute(a, n), "a={a} n={n}");
                }
            }
        }
        assert_eq!(factorize(1_537_596_787), vec![(29_287, 1), (52_501, 1)]);
        assert_eq!(
            factorize(3_384_163_410_217_561),
            vec![(41_134_921, 1), (82_269_841, 1)]
        );
        assert!(is_prime(4_294_967_291) && !is_prime(4_294_967_297));
        assert_eq!(carmichael(1_537_596_787), 256_252_500);
    }

    #[test]
    fn peak_support_is_the_closed_form() {
        for r in [1u64, 2, 12, 60, 41_832, 110_806_800, 256_252_500, 2_538_720] {
            let t = 2 * (64 - r.leading_zeros() as usize) + 4;
            let nu = r.trailing_zeros();
            let ro = r >> nu;
            let want = if nu == 0 { r } else { ro.max(r / 2) };
            assert_eq!(peak_support(r, t), want, "r={r}");
        }
    }

    #[test]
    fn peak_metric() {
        // N = 15, r = 4, t = 8: y = 64 = 2^8/4 is exactly s/r = 1/4
        assert!(peak_ok(64, 4, 8));
        assert!(!peak_ok(10, 4, 8));
    }
}
