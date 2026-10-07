//! Phase diagram of exact simulability (research/simulability/simulability.md).
//!
//! Three pieces, used by `examples/simulability.rs` and the Python driver in
//! `research/data/simulability/`:
//!
//! * [`build`]: four parameterised circuit families that sweep across the
//!   regimes of the exact engines (Clifford+T, generic brickwork,
//!   reversible arithmetic, QAOA).
//! * [`features`]: cheap statistics, computed in about O(gates · n) without
//!   simulating, one "resource" per engine: the active-dimension profile of
//!   the rotation frame (compressed state), a crossing-count bound on the MPS
//!   bond, the KL cut of the HSF partition, an affine bound on the support
//!   size (sparse state), and the plain `n` / gate count (state vector).
//! * [`run_engine`]: every exact engine, all answering the same request,
//!   `<ψ| Z^{⊗n} |ψ>` for `|ψ> = U|0^n>` (a global observable, so no engine
//!   gets a light-cone shortcut), under a common memory budget.
//!
//! Everything here is exact; an engine that would have to truncate reports
//! an error instead of a value.

use crate::circuit::{Circuit, Op, SimError};
use crate::engines::adaptive::{self, AdaptiveOptions, Strategy};
use crate::engines::blocked::BlockConfig;
use crate::engines::hsf::{self, HsfOptions, HybridSchrodingerFeynman};
use crate::engines::mps::Mps;
use crate::engines::pauli_frame::FrameOptions;
use crate::engines::pauli_path::PauliSum;
use crate::engines::sparse::SparseState;
use crate::engines::statevector::StateVectorF64;
use crate::gate::Gate;
use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::time::Instant;

/// Engines [`run_engine`] knows, in a fixed order.
///
/// `cstate` is the compressed Schrödinger state ([`adaptive::CompressedState`]:
/// Clifford frame + dense register of the `d` active qubits), always evolved
/// in full. `dense`, `frame` and `auto` are the three policies of
/// [`adaptive::expectation`], which start in the Heisenberg picture and can
/// finish early when the observable is pruned away (observable engines).
pub const ENGINES: [&str; 9] = [
    "sv", "sparse", "mps", "hsf", "tableau", "cstate", "frame", "dense", "auto",
];

/// Parsed `family:key=value,key=value` specification.
#[derive(Clone, Debug)]
pub struct Spec {
    /// Circuit family name (the part before `:`).
    pub family: String,
    /// Numeric parameters, by key.
    pub params: BTreeMap<String, f64>,
}

impl Spec {
    /// Parses `family:key=value,...`; an empty parameter list is allowed.
    ///  Fails on a parameter without `=` or with a non-numeric value.
    pub fn parse(s: &str) -> Result<Spec, String> {
        let (family, rest) = s.split_once(':').unwrap_or((s, ""));
        let mut params = BTreeMap::new();
        for kv in rest.split(',').filter(|x| !x.is_empty()) {
            let (k, v) = kv
                .split_once('=')
                .ok_or_else(|| format!("bad parameter {kv:?}"))?;
            let v: f64 = v.parse().map_err(|_| format!("bad value in {kv:?}"))?;
            params.insert(k.to_string(), v);
        }
        Ok(Spec {
            family: family.to_string(),
            params,
        })
    }

    fn get(&self, k: &str, default: f64) -> f64 {
        *self.params.get(k).unwrap_or(&default)
    }

    fn geti(&self, k: &str, default: usize) -> usize {
        self.get(k, default as f64).round() as usize
    }
}

/// Builds the circuit for a [`Spec`] (deterministic in `seed`).
///
/// * `ct:n,L,t,nn` — `L` layers of random single-qubit Cliffords
///   (`I, H, S, HS`) and a CNOT layer (nearest-neighbour brickwork if
///   `nn=1`, else a random perfect matching), with `t` T gates at random
///   (layer, qubit) slots.
/// * `brick:n,D,nn` — `D` layers of Haar-ish random `U` on every qubit and
///   a CZ layer (`nn=1` brickwork, else random matching), then a final `U`
///   layer.
/// * `arith:bits,h,reps` — Cuccaro ripple-carry adders on two `bits`-bit
///   registers: `h` qubits of each register in `|+>`, the rest a random
///   classical value; `reps` additions alternating `b += a` / `a += b`.
///   Pure permutation after the Hadamards.
/// * `qaoa:n,p,deg,nn` — QAOA on a graph with `n·deg/2` edges (ring
///   neighbourhood if `nn=1`, else uniformly random), `p` rounds of
///   `exp(-iγ ZZ)` per edge (CNOT·Rz·CNOT) and `Rx(2β)` mixers.
pub fn build(spec: &Spec, seed: u64) -> Result<Circuit, String> {
    let mut rng = StdRng::seed_from_u64(seed);
    match spec.family.as_str() {
        "ct" => {
            let n = spec.geti("n", 16);
            let layers = spec.geti("L", 4).max(1);
            let t = spec.geti("t", 0);
            let nn = spec.geti("nn", 1) == 1;
            Ok(clifford_t(n, layers, t, nn, &mut rng))
        }
        "brick" => {
            let n = spec.geti("n", 16);
            let depth = spec.geti("D", 4);
            let nn = spec.geti("nn", 1) == 1;
            Ok(brick(n, depth, nn, &mut rng))
        }
        "arith" => {
            let bits = spec.geti("bits", 8).max(1);
            let h = spec.geti("h", 2).min(bits);
            let reps = spec.geti("reps", 1);
            Ok(arith(bits, h, reps, &mut rng))
        }
        "qaoa" => {
            let n = spec.geti("n", 16);
            let p = spec.geti("p", 1);
            let deg = spec.geti("deg", 3);
            let nn = spec.geti("nn", 0) == 1;
            Ok(qaoa(n, p, deg, nn, &mut rng))
        }
        "hea" => {
            let n = spec.geti("n", 16);
            let depth = spec.geti("D", 4);
            Ok(hea(n, depth, &mut rng))
        }
        "qft" => {
            let n = spec.geti("n", 12);
            let h = spec.geti("h", 0).min(n);
            Ok(qft(n, h, &mut rng))
        }
        f => Err(format!("unknown family {f:?}")),
    }
}

/// Hardware-efficient ansatz: `D` layers of `Ry Rz` on every qubit and a
/// sequential CNOT ladder `0→1→…→n−1` (added for the planner study as a
/// held-out family).
fn hea<R: Rng>(n: usize, depth: usize, rng: &mut R) -> Circuit {
    let mut c = Circuit::new(n);
    for _ in 0..depth {
        for q in 0..n {
            c.ry(q, rng.random_range(0.0..2.0 * PI));
            c.rz(q, rng.random_range(0.0..2.0 * PI));
        }
        for q in 0..n.saturating_sub(1) {
            c.cnot(q, q + 1);
        }
    }
    for q in 0..n {
        c.ry(q, rng.random_range(0.0..2.0 * PI));
    }
    c
}

/// QFT (no final SWAPs) of a state with `h` qubits in `|+>` and the rest a
/// random basis state (held-out family for the planner study).
fn qft<R: Rng>(n: usize, h: usize, rng: &mut R) -> Circuit {
    let mut c = Circuit::new(n);
    for q in 0..n {
        if q < h {
            c.h(q);
        } else if rng.random_bool(0.5) {
            c.x(q);
        }
    }
    for i in (0..n).rev() {
        c.h(i);
        for j in (0..i).rev() {
            c.cphase(j, i, PI / (1u64 << (i - j)) as f64);
        }
    }
    c
}

fn matching<R: Rng>(n: usize, layer: usize, nn: bool, rng: &mut R) -> Vec<(usize, usize)> {
    if nn {
        (layer % 2..n.saturating_sub(1))
            .step_by(2)
            .map(|i| (i, i + 1))
            .collect()
    } else {
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.random_range(0..=i);
            perm.swap(i, j);
        }
        perm.chunks_exact(2).map(|c| (c[0], c[1])).collect()
    }
}

fn clifford_t<R: Rng>(n: usize, layers: usize, t: usize, nn: bool, rng: &mut R) -> Circuit {
    // T slots: distinct (layer, qubit) pairs while possible.
    let slots = layers * n;
    let mut chosen = vec![0usize; slots];
    if t <= slots {
        let mut idx: Vec<usize> = (0..slots).collect();
        for i in 0..t {
            let j = rng.random_range(i..slots);
            idx.swap(i, j);
            chosen[idx[i]] += 1;
        }
    } else {
        for _ in 0..t {
            chosen[rng.random_range(0..slots)] += 1;
        }
    }
    let mut c = Circuit::new(n);
    for l in 0..layers {
        for q in 0..n {
            match rng.random_range(0..4) {
                0 => {}
                1 => {
                    c.h(q);
                }
                2 => {
                    c.s(q);
                }
                _ => {
                    c.h(q).s(q);
                }
            }
            for _ in 0..chosen[l * n + q] {
                // H T keeps consecutive T gates on a wire from fusing to S.
                c.t(q).h(q);
            }
        }
        for (a, b) in matching(n, l, nn, rng) {
            if rng.random_bool(0.5) {
                c.cnot(a, b);
            } else {
                c.cnot(b, a);
            }
        }
    }
    c
}

fn random_u<R: Rng>(c: &mut Circuit, q: usize, rng: &mut R) {
    // Haar measure on SU(2): θ with density sin θ / 2.
    let th = (1.0 - 2.0 * rng.random::<f64>()).acos();
    c.u(
        q,
        th,
        rng.random_range(0.0..2.0 * PI),
        rng.random_range(0.0..2.0 * PI),
    );
}

fn brick<R: Rng>(n: usize, depth: usize, nn: bool, rng: &mut R) -> Circuit {
    let mut c = Circuit::new(n);
    for l in 0..depth {
        for q in 0..n {
            random_u(&mut c, q, rng);
        }
        for (a, b) in matching(n, l, nn, rng) {
            c.cz(a, b);
        }
    }
    for q in 0..n {
        random_u(&mut c, q, rng);
    }
    c
}

/// Cuccaro adder `y += x` (mod 2^bits) with carry-in ancilla `anc`.
fn cuccaro(c: &mut Circuit, x: &[usize], y: &[usize], anc: usize) {
    let bits = x.len();
    let carry = |i: usize| if i == 0 { anc } else { x[i - 1] };
    for i in 0..bits {
        // MAJ(carry, y_i, x_i)
        let (p, q, r) = (carry(i), y[i], x[i]);
        c.cnot(r, q).cnot(r, p).ccx(p, q, r);
    }
    for i in (0..bits).rev() {
        // UMA(carry, y_i, x_i)
        let (p, q, r) = (carry(i), y[i], x[i]);
        c.ccx(p, q, r).cnot(r, p).cnot(p, q);
    }
}

fn arith<R: Rng>(bits: usize, h: usize, reps: usize, rng: &mut R) -> Circuit {
    let n = 2 * bits + 1;
    let a: Vec<usize> = (0..bits).map(|i| 1 + 2 * i).collect();
    let b: Vec<usize> = (0..bits).map(|i| 2 + 2 * i).collect();
    let mut c = Circuit::new(n);
    for reg in [&a, &b] {
        for (i, &q) in reg.iter().enumerate() {
            if i < h {
                c.h(q);
            } else if rng.random_bool(0.5) {
                c.x(q);
            }
        }
    }
    for r in 0..reps {
        if r % 2 == 0 {
            cuccaro(&mut c, &a, &b, 0);
        } else {
            cuccaro(&mut c, &b, &a, 0);
        }
    }
    c
}

fn qaoa<R: Rng>(n: usize, p: usize, deg: usize, nn: bool, rng: &mut R) -> Circuit {
    let mut edges = Vec::new();
    if nn {
        for i in 0..n {
            for k in 1..=deg.div_ceil(2) {
                if i + k < n || n > 2 * k {
                    let j = (i + k) % n;
                    if i != j {
                        edges.push((i.min(j), i.max(j)));
                    }
                }
            }
        }
        edges.sort();
        edges.dedup();
    } else {
        let m = n * deg / 2;
        let mut seen = std::collections::BTreeSet::new();
        while seen.len() < m.min(n * (n - 1) / 2) {
            let a = rng.random_range(0..n);
            let b = rng.random_range(0..n);
            if a != b {
                seen.insert((a.min(b), a.max(b)));
            }
        }
        edges = seen.into_iter().collect();
    }
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    for _ in 0..p {
        let gamma = rng.random_range(0.1..1.4);
        let beta = rng.random_range(0.1..1.4);
        for &(a, b) in &edges {
            c.cnot(a, b).rz(b, 2.0 * gamma).cnot(a, b);
        }
        for q in 0..n {
            c.rx(q, 2.0 * beta);
        }
    }
    c
}

/// The Z-product observable of a request: `all` (global parity, the
/// default), `mid2` (`Z_{n/2-1} Z_{n/2}`) or `mid4` (`Z_{n/2-2} … Z_{n/2+1}`).
pub fn observable_qubits(name: &str, n: usize) -> Result<Vec<usize>, String> {
    let m = n / 2;
    match name {
        "all" => Ok((0..n).collect()),
        "mid2" if n >= 2 => Ok(vec![m - 1, m]),
        "mid4" if n >= 4 => Ok((m - 2..m + 2).collect()),
        _ => Err(format!("unknown observable {name:?} for n = {n}")),
    }
}

// ---------------------------------------------------------------------------
// Features.

/// Cheap circuit statistics. Every `*_l` field is a log2 work estimate for
/// one engine; the remaining fields are the raw ingredients.
#[derive(Clone, Debug, Default)]
pub struct Features {
    /// Number of qubits.
    pub n: usize,
    /// Unitary gates.
    pub gates: usize,
    /// Two-qubit gates.
    pub g2: usize,
    /// Three-qubit gates.
    pub g3: usize,
    /// Two-qubit-gate depth (ASAP layers counting only multi-qubit gates).
    pub depth2: usize,
    /// T/T† count after decomposition (Toffoli = 7).
    pub t_count: usize,
    /// Non-Clifford rotations seen by the rotation frame (after merging
    /// half-π multiples).
    pub rotations: usize,
    /// Final active dimension `d_m`.
    pub d: usize,
    /// `log2 Σ_j 2^{d_j}`: exact amplitude updates of the compressed state.
    pub dense_l: f64,
    /// Rotations that do not grow the x-span (each can double the
    /// Heisenberg term count).
    pub redundant: usize,
    /// Pauli-path proxy: `log2 m + min(redundant, 2 d)`... see code.
    pub frame_l: f64,
    /// `<Z^{⊗n}>` is provably 0 by the x-span lemma (O(gates·n) check);
    /// the Heisenberg engines then finish without propagating anything.
    pub obs_zero: bool,
    /// Max over line cuts of the crossing-count bound on log2 χ.
    pub chi_bits: usize,
    /// log2 Σ_gates Σ_{cuts swept} χ_cut(t)^3 with the time-resolved bound.
    pub mps_l: f64,
    /// The same two without the support cap (pure crossing count).
    pub chi_bits0: usize,
    /// The MPS work estimate matching `chi_bits0`.
    pub mps_l0: f64,
    /// HSF: log2 paths of the KL partition and block sizes.
    pub hsf_k: u32,
    /// Qubits on side A of the partition.
    pub hsf_na: usize,
    /// Qubits on side B of the partition.
    pub hsf_nb: usize,
    /// log2 HSF work (path evolutions plus output accumulation) with `hsf_keff` path bits.
    pub hsf_l: f64,
    /// Path bits after exact zero-path pruning is accounted for: a cut gate
    /// whose diagonal-side qubit is still in a definite Z state on every
    /// path (no branching gate since its last projection) adds no paths.
    pub hsf_keff: u32,
    /// `hsf_l` with the nominal `hsf_k`.
    pub hsf_l0: f64,
    /// Affine upper bound on log2 of the support size of the final state.
    pub sup: usize,
    /// log2 sparse-state work: `log2 gates + sup`.
    pub sparse_l: f64,
    /// log2 state-vector work: `log2 gates + n`.
    pub sv_l: f64,
    /// Seconds spent computing these features.
    pub secs: f64,
    /// Seconds spent on the rotation-frame part.
    pub secs_frame: f64,
    /// Seconds spent on the HSF features (0 if not computed).
    pub secs_hsf: f64,
    /// Free-fermion detector ([`crate::engines::gaussian::detect`]):
    /// fraction of the fused blocks that are Gaussian.
    pub gauss_fraction: f64,
    /// Largest residual of a non-interaction block (0: exactly Gaussian up
    /// to interaction phases).
    pub gauss_residual: f64,
    /// `Σ |g|` over the diagonal interaction phases `exp(i g n_a n_b)`.
    pub gauss_interaction: f64,
    /// The circuit is exactly Gaussian (the free-fermion engine applies).
    pub gauss_exact: bool,
}

fn log2sum(xs: impl Iterator<Item = f64>) -> f64 {
    // log2 Σ 2^x, stable.
    let v: Vec<f64> = xs.collect();
    if v.is_empty() {
        return 0.0;
    }
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    m + v.iter().map(|x| (x - m).exp2()).sum::<f64>().log2()
}

fn gate_list(c: &Circuit) -> Vec<Gate> {
    c.ops
        .iter()
        .filter_map(|op| match op {
            Op::Gate(g) => Some(*g),
            _ => None,
        })
        .collect()
}

/// log2 of the operator-Schmidt rank of a two-qubit gate (across its two
/// qubits).
fn schmidt_bits(g: &Gate) -> usize {
    match g {
        Gate::Cnot(..) | Gate::Cz(..) | Gate::CPhase(..) => 1,
        Gate::Swap(..) | Gate::ISwap(..) | Gate::ISwapdg(..) => 2,
        _ => 2,
    }
}

/// Computes [`Features`]. `with_hsf=false` skips the KL partition (the
/// most expensive feature on large circuits).
pub fn features(c: &Circuit, with_hsf: bool) -> Result<Features, SimError> {
    let all: Vec<usize> = (0..c.num_qubits).collect();
    features_for(c, with_hsf, &all)
}

/// [`features`] for the observable `Z_{obs}` (only `obs_zero` and the
/// frame estimate depend on it).
pub fn features_for(c: &Circuit, with_hsf: bool, obs: &[usize]) -> Result<Features, SimError> {
    let t0 = Instant::now();
    let n = c.num_qubits;
    let gates = gate_list(c);
    let mut f = Features {
        n,
        gates: gates.len(),
        ..Default::default()
    };
    // Gate classes, T count, 2q depth.
    let mut level = vec![0usize; n];
    for g in &gates {
        let qs = g.qubits();
        match qs.len() {
            2 => f.g2 += 1,
            3 => f.g3 += 1,
            _ => {}
        }
        if qs.len() >= 2 {
            let l = qs.iter().map(|&q| level[q]).max().unwrap_or(0) + 1;
            for &q in &qs {
                level[q] = l;
            }
        }
        for h in g.decompose_to_clifford_rz() {
            if matches!(h, Gate::T(_) | Gate::Tdg(_)) {
                f.t_count += 1;
            }
        }
    }
    f.depth2 = level.iter().copied().max().unwrap_or(0);

    // Rotation frame: active-dimension profile.
    let tf = Instant::now();
    let prof = adaptive::active_dimension_profile(c)?;
    f.rotations = prof.len();
    f.d = prof.last().copied().unwrap_or(0);
    f.dense_l = log2sum(prof.iter().map(|&d| d as f64));
    let mut prev = 0;
    for &d in &prof {
        if d == prev {
            f.redundant += 1;
        }
        prev = d;
    }
    // Heisenberg term count: span-growing rotations are free (pruned), each
    // redundant one can double the count, capped by 4^d strings. Cost is
    // ~ Σ_j terms_j; use m · 2^{min(redundant/2, 2d)} (growth prior 0.5 per
    // redundant rotation, as in adaptive::AdaptiveOptions).
    f.frame_l =
        (prof.len().max(1) as f64).log2() + (0.5 * f.redundant as f64).min(2.0 * f.d as f64);
    f.obs_zero = adaptive::z_product_vanishes(c, obs)?;
    if f.obs_zero {
        f.frame_l = (f.gates.max(1) as f64).log2();
    }
    f.secs_frame = tf.elapsed().as_secs_f64();

    // MPS: crossing-count bound per line cut, time resolved, capped by the
    // cut size and by the support: a Schmidt rank never exceeds the number
    // of non-zero amplitudes, which is at most 2^(branching gates so far).
    let ncut = n.saturating_sub(1);
    let cap: Vec<usize> = (0..ncut).map(|i| (i + 1).min(n - 1 - i)).collect();
    let mut bits = vec![0usize; ncut];
    let mut bits0 = vec![0usize; ncut];
    let mut nbranch = 0usize;
    let mut mps_terms: Vec<f64> = Vec::new();
    let mut mps_terms0: Vec<f64> = Vec::new();
    for g in &gates {
        let parts = if g.arity() == 3 {
            g.decompose_to_clifford_rz()
        } else {
            vec![*g]
        };
        // A Toffoli is a permutation: its Clifford+T expansion (which the
        // MPS engine applies) raises the support by at most one H inside.
        let (toffoli, sup_cap) = (g.arity() == 3, nbranch + usize::from(g.arity() == 3));
        for h in parts {
            let qs = h.qubits();
            if qs.len() == 1 {
                if !toffoli && is_branching(&h) {
                    nbranch += 1;
                }
                // single-site update ~ χ^2
                let q = qs[0];
                let nb = [q.checked_sub(1), (q + 1 < n).then_some(q)];
                let b = nb.iter().flatten().map(|&i| bits[i]).max().unwrap_or(0);
                let b0 = nb.iter().flatten().map(|&i| bits0[i]).max().unwrap_or(0);
                mps_terms.push(2.0 * b as f64);
                mps_terms0.push(2.0 * b0 as f64);
                continue;
            }
            let (lo, hi) = (qs[0].min(qs[1]), qs[0].max(qs[1]));
            let r = schmidt_bits(&h);
            for i in lo..hi {
                bits[i] = (bits[i] + r).min(cap[i]).min(nbranch.max(sup_cap));
                bits0[i] = (bits0[i] + r).min(cap[i]);
                // Each adjacent application on cut i costs ~ χ_i^3. A
                // non-adjacent gate is routed by SWAPs (there and back):
                // on every intermediate cut two SWAPs, during which the cut
                // carries one extra qubit (χ ≤ 2 χ_i), i.e. 2 · (2χ)^3.
                let extra: f64 = if i > lo { 4.0 } else { 0.0 };
                mps_terms.push(3.0 * bits[i] as f64 + extra);
                mps_terms0.push(3.0 * bits0[i] as f64 + extra);
            }
        }
    }
    f.chi_bits = bits.iter().copied().max().unwrap_or(0);
    f.mps_l = log2sum(mps_terms.into_iter());
    f.chi_bits0 = bits0.iter().copied().max().unwrap_or(0);
    f.mps_l0 = log2sum(mps_terms0.into_iter());

    // HSF: KL partition + its path count.
    if with_hsf {
        add_hsf_features(c, &mut f)?;
    }

    // Sparse: affine support bound. Each wire is a constant, an affine
    // function of the free variables, or an opaque (non-affine) function
    // of them; only branching (non-monomial) one-qubit gates create
    // variables.
    f.sup = support_bound(n, &gates);
    f.sparse_l = (f.gates.max(1) as f64).log2() + f.sup as f64;
    f.sv_l = (f.gates.max(1) as f64).log2() + n as f64;

    // Free fermions: block fusion and the matchgate test, O(gates).
    let gr = crate::engines::gaussian::detect(c, &Default::default());
    f.gauss_fraction = gr.gaussian_fraction;
    f.gauss_residual = gr.max_residual;
    f.gauss_interaction = gr.interaction_total;
    f.gauss_exact = gr.exact;
    f.secs = t0.elapsed().as_secs_f64();
    Ok(f)
}

/// Fills the HSF fields of [`Features`] (KL partition, path counts,
/// `hsf_l`); the most expensive feature, so the planner computes it only
/// when HSF could matter.
pub fn add_hsf_features(c: &Circuit, f: &mut Features) -> Result<(), SimError> {
    let n = c.num_qubits;
    if n < 2 {
        return Ok(());
    }
    let th = Instant::now();
    let opts = HsfOptions::default();
    let in_a = hsf::auto_partition(c, &opts)?;
    hsf_split_features(c, f, &in_a)?;
    f.secs_hsf = th.elapsed().as_secs_f64();
    Ok(())
}

/// The HSF fields of [`Features`] for a given partition `in_a` (O(gates);
/// [`add_hsf_features`] uses the Kernighan–Lin partition, the planner also
/// prices the plain line split `[0, n/2) | [n/2, n)` this way).
pub fn hsf_split_features(c: &Circuit, f: &mut Features, in_a: &[bool]) -> Result<(), SimError> {
    let n = c.num_qubits;
    let gates = gate_list(c);
    let opts = HsfOptions::default();
    let in_a = in_a.to_vec();
    f.hsf_k = hsf::cut_bits(c, &in_a, &opts)?;
    f.hsf_na = in_a.iter().filter(|&&x| x).count();
    f.hsf_nb = n - f.hsf_na;
    f.hsf_keff = effective_cut_bits(n, &gates, &in_a);
    let g = gates.len().max(1) as f64;
    let big = f.hsf_na.max(f.hsf_nb) as f64;
    // paths × (block evolutions) + GEMM accumulation of the 2^n output
    let cost = |k: f64| log2sum([k + g.log2() + big, k + n as f64].into_iter());
    f.hsf_l = cost(f.hsf_keff as f64);
    f.hsf_l0 = cost(f.hsf_k as f64);
    Ok(())
}

/// Path bits of an HSF partition once exact zero-path pruning is taken
/// into account (see [`Features::hsf_keff`]). A qubit is *definite* while it
/// holds a computational-basis value on every path: initially, after a
/// projection, and through diagonal gates, X/Y, and CNOTs from definite
/// controls. A cut CNOT/CZ/CPhase branches on a Z projector of one of its
/// diagonal-side qubits; if that qubit is definite only one branch survives.
/// Other crossing gates (SWAP-like, generic) always count their full rank.
/// This is a heuristic estimate of the engine's live path count, not a
/// bound (pruning also happens for reasons it does not model).
pub fn effective_cut_bits(n: usize, gates: &[Gate], in_a: &[bool]) -> u32 {
    let mut definite = vec![true; n];
    let mut bits = 0u32;
    let crosses = |a: usize, b: usize| in_a[a] != in_a[b];
    for g in gates {
        let parts = if g.arity() == 3 {
            g.decompose_to_clifford_rz()
        } else {
            vec![*g]
        };
        for h in parts {
            match h {
                Gate::Cnot(c, t) => {
                    if crosses(c, t) {
                        if !definite[c] {
                            bits += 1;
                            definite[c] = true;
                        }
                    } else if !definite[c] {
                        definite[t] = false;
                    }
                }
                Gate::Cz(a, b) | Gate::CPhase(a, b, _) => {
                    if crosses(a, b) && !definite[a] && !definite[b] {
                        bits += 1;
                        definite[a] = true;
                    }
                }
                Gate::Swap(a, b) | Gate::ISwap(a, b) | Gate::ISwapdg(a, b) => {
                    if crosses(a, b) {
                        bits += 2;
                    }
                    definite.swap(a, b);
                    if !matches!(h, Gate::Swap(..)) {
                        definite[a] = false;
                        definite[b] = false;
                    }
                }
                ref x if x.arity() == 1 => {
                    if is_branching(x) {
                        definite[x.qubits()[0]] = false;
                    }
                }
                _ => {
                    let qs = h.qubits();
                    if crosses(qs[0], qs[1]) {
                        bits += 2;
                    }
                    definite[qs[0]] = false;
                    definite[qs[1]] = false;
                }
            }
        }
    }
    bits
}

#[derive(Clone)]
enum Wire {
    Const,
    Affine(Vec<u64>),
    Opaque,
}

fn is_branching(g: &Gate) -> bool {
    g.arity() == 1
        && g.diagonal_1q().is_none()
        && !matches!(g, Gate::X(_) | Gate::Y(_) | Gate::I(_))
}

/// Upper bound on log2 |supp U|0^n>| (see [`features`]).
pub fn support_bound(n: usize, gates: &[Gate]) -> usize {
    let nvars = gates.iter().filter(|g| is_branching(g)).count();
    let w = nvars.div_ceil(64).max(1);
    let mut wire = vec![Wire::Const; n];
    let mut next = 0usize;
    let xor = |a: &Wire, b: &Wire| -> Wire {
        match (a, b) {
            (Wire::Const, x) | (x, Wire::Const) => x.clone(),
            (Wire::Affine(u), Wire::Affine(v)) => {
                let s: Vec<u64> = u.iter().zip(v).map(|(x, y)| x ^ y).collect();
                if s.iter().all(|&x| x == 0) {
                    Wire::Const
                } else {
                    Wire::Affine(s)
                }
            }
            _ => Wire::Opaque,
        }
    };
    for g in gates {
        match *g {
            Gate::Cnot(c, t) => wire[t] = xor(&wire[t], &wire[c]),
            Gate::Swap(a, b) | Gate::ISwap(a, b) | Gate::ISwapdg(a, b) => wire.swap(a, b),
            Gate::Ccx(a, b, t) => {
                let ca = matches!(wire[a], Wire::Const);
                let cb = matches!(wire[b], Wire::Const);
                if ca && cb {
                    // X or nothing
                } else {
                    // One constant control: CNOT from the other control *or
                    // nothing*, and `Wire::Const` does not record which.
                    // Neither `t` nor `t ^ other` is a safe affine form for
                    // both cases (the old rule kept `t` constant when the
                    // two forms cancelled, which under-counts: audit §16),
                    // so the target becomes opaque. Two non-constant
                    // controls: AND of two forms, opaque as well.
                    wire[t] = Wire::Opaque;
                }
            }
            ref h if is_branching(h) => {
                let q = h.qubits()[0];
                let mut v = vec![0u64; w];
                v[next / 64] |= 1 << (next % 64);
                next += 1;
                wire[q] = Wire::Affine(v);
            }
            _ => {} // diagonal or X/Y: support unchanged
        }
    }
    // rank of the affine wires + opaque wires (opaque wires are functions of
    // the variables, so the total is also capped by the variable count).
    let mut basis: Vec<Vec<u64>> = Vec::new();
    let mut opaque = 0usize;
    for wv in &wire {
        match wv {
            Wire::Const => {}
            Wire::Opaque => opaque += 1,
            Wire::Affine(v) => {
                let mut v = v.clone();
                for b in &basis {
                    let p = lead(b);
                    if v[p / 64] >> (p % 64) & 1 == 1 {
                        for (x, y) in v.iter_mut().zip(b) {
                            *x ^= y;
                        }
                    }
                }
                if v.iter().any(|&x| x != 0) {
                    basis.push(v);
                }
            }
        }
    }
    (basis.len() + opaque).min(nvars).min(n)
}

fn lead(v: &[u64]) -> usize {
    for (i, &x) in v.iter().enumerate() {
        if x != 0 {
            return i * 64 + x.trailing_zeros() as usize;
        }
    }
    usize::MAX
}

// ---------------------------------------------------------------------------
// Engines.

/// Outcome of one engine run.
#[derive(Clone, Debug, Default)]
pub struct EngineRun {
    /// The computed expectation value.
    pub value: f64,
    /// Seconds in the engine (circuit construction excluded).
    pub secs: f64,
    /// Engine-specific size: SV amplitudes, sparse peak nnz, MPS max bond,
    /// HSF paths, frame peak terms, dense active qubits.
    pub size: f64,
    /// Engine-specific diagnostics (free-form `key=value` text, may be empty).
    pub note: String,
}

fn parity_dense(amps: &[Complex64], mask: u128) -> f64 {
    amps.iter()
        .enumerate()
        .map(|(x, a)| {
            if (x as u128 & mask).count_ones() & 1 == 1 {
                -a.norm_sqr()
            } else {
                a.norm_sqr()
            }
        })
        .sum()
}

/// Runs `engine` on `<Z^{⊗n}>` of `c|0^n>` within `mem_bytes`.
pub fn run_engine(engine: &str, c: &Circuit, mem_bytes: u128) -> Result<EngineRun, SimError> {
    let all: Vec<usize> = (0..c.num_qubits).collect();
    run_engine_obs(engine, c, mem_bytes, &all)
}

/// Runs `engine` on `<Z_{obs}>` (product of `Z` on the listed qubits).
pub fn run_engine_obs(
    engine: &str,
    c: &Circuit,
    mem_bytes: u128,
    obs: &[usize],
) -> Result<EngineRun, SimError> {
    let n = c.num_qubits;
    let too_large = |what: &'static str, bytes: u128| SimError::TooLarge {
        what,
        bytes,
        limit: mem_bytes,
    };
    let all: Vec<usize> = obs.to_vec();
    let mask: u128 = obs.iter().fold(0u128, |m, &q| m ^ (1u128 << q));
    let t0 = Instant::now();
    let mut run = EngineRun::default();
    match engine {
        "gauss" => {
            let opts = crate::engines::gaussian::GaussianOptions {
                max_bytes: mem_bytes,
                ..Default::default()
            };
            run.value = crate::engines::gaussian::expectation_z_product(c, obs, &opts)?;
            run.size = (4 * n * n) as f64;
        }
        "sv" => {
            let bytes = 16u128 << n;
            if bytes > mem_bytes {
                return Err(too_large("state vector", bytes));
            }
            let mut sv = StateVectorF64::try_new(n)?;
            sv.apply_circuit_blocked(c, &BlockConfig::default())?;
            run.value = parity_dense(sv.amplitudes(), mask);
            run.size = (1u128 << n) as f64;
        }
        "sparse" => {
            if n > 64 {
                return Err(too_large("sparse (n > 64)", 0));
            }
            // ~48 bytes per entry in the hash map (key, value, control, load)
            let max_nnz = (mem_bytes / 48) as usize;
            let mut s = SparseState::new(n);
            for g in c.gates() {
                s.apply_gate(g)?;
                if s.nnz() > max_nnz {
                    return Err(too_large("sparse state", s.nnz() as u128 * 48));
                }
            }
            run.value = s
                .iter()
                .map(|(x, a)| {
                    if (x as u128 & mask).count_ones() & 1 == 1 {
                        -a.norm_sqr()
                    } else {
                        a.norm_sqr()
                    }
                })
                .sum();
            run.size = s.peak_nnz() as f64;
        }
        "mps" | "mpsb" => {
            let mut m = Mps::new(n, 1 << 20);
            if engine == "mpsb" {
                // bound-capped: drop singular values beyond the rigorous
                // Schmidt-rank bound (numerical noise only).
                let b = crate::engines::mps_cost::replay_traced(
                    c,
                    crate::engines::mps_cost::BondSource::Bound(
                        crate::engines::mps_cost::Estimator::Best,
                    ),
                )?;
                m.set_step_caps(b.trace);
            }
            for g in c.gates() {
                m.apply_gate(g)?;
                if m.bytes() as u128 > mem_bytes / 4 {
                    return Err(too_large("mps", m.bytes() as u128));
                }
            }
            // The SVD cutoff (relative weight 1e-14 per singular value) is
            // numerical noise; anything that discards more is a truncation.
            if 1.0 - m.fidelity_estimate() > 1e-10 {
                return Err(SimError::TooLarge {
                    what: "mps (truncated)",
                    bytes: 0,
                    limit: 0,
                });
            }
            run.value = m.expectation_z_product(&all);
            run.size = m.max_bond_dim() as f64;
        }
        "hsf" => {
            let opts = HsfOptions {
                max_bytes: mem_bytes,
                ..HsfOptions::default()
            };
            let h = HybridSchrodingerFeynman::auto(c, opts)?;
            let out = 16u128 << n;
            if out > mem_bytes {
                return Err(too_large("hsf full output", out));
            }
            let amps = h.state_vector()?;
            run.value = parity_dense(&amps, mask);
            run.size = h.num_paths() as f64;
        }
        "tableau" => {
            if !c.gates().all(|g| g.is_clifford()) {
                return Err(SimError::TooLarge {
                    what: "tableau (non-Clifford circuit)",
                    bytes: 0,
                    limit: 0,
                });
            }
            // Heisenberg picture: <0|C† P C|0>. `conjugate_by_clifford(X)`
            // maps P -> X P X†, so conjugate by the inverse circuit C†
            // (reversed, inverted gates). The first version conjugated by C
            // itself, which only happened to agree on <Z^n> (caught by the
            // local-observable sweep).
            let mut inv = Circuit::new(n);
            for g in c.gates().collect::<Vec<_>>().into_iter().rev() {
                inv.gate(g.inverse());
            }
            let mut p = PauliSum::z_product(n, &all);
            p.conjugate_by_clifford(&inv)?;
            run.value = p.expectation_zero_state();
        }
        "cstate" => {
            let max_d = ((mem_bytes / 16).max(1).ilog2() as usize).min(30);
            let st = adaptive::CompressedState::new(c, max_d)?;
            run.value = st.expectation(&PauliSum::z_product(n, &all));
            run.size = st.active_qubits() as f64;
        }
        "frame" | "dense" | "auto" | "auto0" => {
            let max_d = ((mem_bytes / 16).max(1).ilog2() as usize).min(30);
            let strategy = match engine {
                "frame" => Strategy::Frame,
                "dense" => Strategy::Dense,
                _ => Strategy::Auto,
            };
            let opt = AdaptiveOptions {
                // QSIM_EXPLORE_FRAC: ablation of Auto's exploration budget
                // `auto0`: Auto without the v2 exploration past a switch
                // (the round-4 behaviour), for A/B timings
                explore_frac: if engine == "auto0" {
                    0.0
                } else {
                    AdaptiveOptions::default().explore_frac
                },
                strategy,
                max_dense_qubits: max_d,
                frame: FrameOptions {
                    // ~ (2 W words + coefficient + hash overhead) per term
                    max_terms: (mem_bytes / 64) as usize,
                    ..FrameOptions::default()
                },
                ..AdaptiveOptions::default()
            };
            let obs = PauliSum::z_product(n, &all);
            let r = adaptive::expectation(c, &obs, &opt)?;
            run.value = r.value;
            run.size = match engine {
                "frame" => r.frame_stats.peak_terms as f64,
                _ => r.dense_qubits as f64,
            };
            run.note = format!(
                "switched_at={:?} dense_qubits={} peak_terms={} term_visits={} dense_ops={} frame_secs={:.6} restarted={}",
                r.switched_at,
                r.dense_qubits,
                r.frame_stats.peak_terms,
                r.frame_stats.term_visits,
                r.dense_ops,
                r.frame_secs,
                r.restarted
            );
        }
        "plan" | "planx" | "planp" => {
            // Planner v0 end to end (planning + speculation included);
            // `planx` without the vanishing certificate (state engines only).
            let cfg = crate::planner::PlannerConfig {
                mem_bytes,
                use_certificate: engine == "plan",
                probe_cap: (engine == "planp").then_some(16),
                ..Default::default()
            };
            let r = crate::planner::expectation(c, &all, &cfg)?;
            run.value = r.value;
            run.note = format!(
                "engine={} aborted={:?} plan_secs={:.6}",
                r.engine.name(),
                r.aborted.iter().map(|e| e.name()).collect::<Vec<_>>(),
                r.plan_secs
            );
        }
        e => {
            return Err(SimError::TooLarge {
                what: "unknown engine",
                bytes: e.len() as u128,
                limit: 0,
            })
        }
    }
    run.secs = t0.elapsed().as_secs_f64();
    Ok(run)
}
