//! Compiled execution plans: optimise once, then run every independent
//! piece of the circuit on the cheapest exact backend.

use super::analysis::{
    apply_classical, clifford_prefix, components, eliminate_swaps, light_cone, restrict,
    split_monomial_suffix, suffix_inputs, terminal_measurements,
};
use super::peephole::optimize;
use super::stabsv::clifford_statevector;
use super::stateprop::propagate;
use super::{require_unitary, validate};
use crate::adaptive::{expectation as adaptive_expectation, AdaptiveOptions, CompressedState};
use crate::circuit::{Circuit, Op, SimError, Simulator};
use crate::gate::{is_multiple_of_half_pi, Gate};
use crate::pauli_path::{self, PauliSum, DEFAULT_MAX_TERMS};
use crate::stabilizer::Tableau;
use crate::statevector::{state_bytes, Real, StateVector, MAX_STATE_BYTES};
use num_complex::{Complex, Complex64};
use rand::Rng;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Which passes to run (all on by default; switch off for ablations).
#[derive(Clone, Copy, Debug)]
pub struct PlanOptions {
    pub peephole: bool,
    /// Phase folding (merge Z-rotations on equal parities) after the first
    /// peephole. Off by default.
    pub phase_fold: bool,
    pub light_cone: bool,
    pub suffix: bool,
    pub split: bool,
    pub clifford_prefix: bool,
    /// Turn SWAPs into wire relabelling.
    pub swap_elim: bool,
    /// Simplify gates acting on known single-qubit stabilizer states.
    pub state_prop: bool,
    /// Pick tableau / Pauli paths where they are cheaper; otherwise every
    /// component runs on the state vector.
    pub dispatch: bool,
    /// Compressed-state (rotation frame) engine for Clifford+T components
    /// that would otherwise need a dense state vector: `None` (the
    /// default here) never picks it; [`crate::pipeline`] enables it.
    pub adaptive: Option<AdaptiveRule>,
}

/// When a component that would run on a dense state vector runs on the
/// compressed state of [`crate::adaptive`] instead: the component has at
/// least `min_qubits` qubits and its active register `d`
/// ([`crate::adaptive::active_dimension`]) satisfies
/// `d <= max_active` and `d + margin <= n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptiveRule {
    pub min_qubits: usize,
    pub margin: usize,
    pub max_active: usize,
}

impl Default for AdaptiveRule {
    fn default() -> Self {
        AdaptiveRule {
            min_qubits: 12,
            margin: 1,
            max_active: 26,
        }
    }
}

impl AdaptiveRule {
    /// Active register size if the rule accepts the unitary circuit `c`.
    pub fn accepts(&self, c: &Circuit) -> Option<usize> {
        let n = c.num_qubits;
        if n < self.min_qubits || n > 512 {
            return None;
        }
        let d = crate::adaptive::active_dimension(c).ok()?;
        (d <= self.max_active && d + self.margin <= n).then_some(d)
    }
}

impl Default for PlanOptions {
    fn default() -> Self {
        PlanOptions {
            peephole: true,
            phase_fold: false,
            light_cone: true,
            suffix: true,
            split: true,
            clifford_prefix: true,
            swap_elim: true,
            state_prop: true,
            dispatch: true,
            adaptive: None,
        }
    }
}

impl PlanOptions {
    /// Every pass off: the plan is the original circuit on one state vector.
    pub fn none() -> Self {
        PlanOptions {
            peephole: false,
            phase_fold: false,
            light_cone: false,
            suffix: false,
            split: false,
            clifford_prefix: false,
            swap_elim: false,
            state_prop: false,
            dispatch: false,
            adaptive: None,
        }
    }
}

/// The backend chosen for one component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// No gates: the component stays in `|0...0>`.
    Idle,
    Tableau,
    StateVector,
    /// Compressed state of a Clifford+T component (see [`AdaptiveRule`]);
    /// terminal sampling and expectation values only.
    Adaptive,
    /// Exact marginal over the needed qubits from Pauli-path expectation
    /// values (terminal measurements only).
    PauliPath,
}

/// One independent part of the circuit.
#[derive(Clone, Debug)]
pub struct Component {
    /// Global qubit of each local qubit.
    pub qubits: Vec<usize>,
    /// The ops on these qubits, relabelled to `0..qubits.len()`.
    pub circuit: Circuit,
    pub backend: Backend,
    /// Local qubits whose final bit is needed (terminal plans).
    pub needed: Vec<usize>,
}

/// What the compiler did, for reporting.
#[derive(Clone, Debug, Default)]
pub struct CompileStats {
    pub gates_in: usize,
    pub gates_after_peephole: usize,
    /// After SWAP elimination, state propagation and a second peephole.
    pub gates_after_state_prop: usize,
    pub gates_after_light_cone: usize,
    /// Permutation gates moved into classical post-processing.
    pub suffix_gates: usize,
    /// Gates simulated in total (over all components).
    pub gates_simulated: usize,
    /// `(qubits, gates, backend)` per component with gates.
    pub components: Vec<(usize, usize, Backend)>,
}

impl CompileStats {
    /// Largest number of qubits any one state-vector component needs.
    pub fn max_sv_qubits(&self) -> usize {
        self.components
            .iter()
            .filter(|c| c.2 == Backend::StateVector)
            .map(|c| c.0)
            .max()
            .unwrap_or(0)
    }
}

#[derive(Clone, Debug)]
enum Kind {
    /// All measurements at the end: sample final bit strings.
    Terminal { meas: Vec<usize>, suffix: Vec<Gate> },
    /// Mid-circuit measurements: run shot by shot; `src[i] = (component,
    /// index of the measurement within that component)` for outcome `i`.
    MidCircuit { src: Vec<(usize, usize)> },
}

/// A circuit compiled for drawing measurement samples.
#[derive(Clone, Debug)]
pub struct SamplingPlan {
    n: usize,
    kind: Kind,
    comps: Vec<Component>,
    opts: PlanOptions,
    pub stats: CompileStats,
}

/// Cost model used by the dispatcher (rough operation counts).
fn non_clifford_count(c: &Circuit) -> usize {
    c.gates()
        .flat_map(|g| g.decompose_to_clifford_rz())
        .filter(|g| match *g {
            Gate::T(_) | Gate::Tdg(_) => true,
            Gate::Phase(_, t) | Gate::Rz(_, t) => !is_multiple_of_half_pi(t),
            _ => false,
        })
        .count()
}

fn choose_backend(c: &Circuit, needed: usize, terminal: bool, opts: &PlanOptions) -> Backend {
    let n = c.num_qubits;
    if c.num_gates() == 0 {
        return Backend::Idle;
    }
    if !opts.dispatch {
        return Backend::StateVector;
    }
    if c.is_clifford() {
        return Backend::Tableau;
    }
    if terminal && needed <= 12 {
        // Pauli paths: 2^needed observables, each up to 2^t terms, each term
        // costing ~n/64 words per gate. State vector: 2^n per gate.
        let t = non_clifford_count(c).min(60) as u32;
        let words = n.div_ceil(64).max(1) as f64;
        let pauli = (needed as f64).exp2() * (t as f64).exp2() * words;
        let sv = (n.min(60) as f64).exp2();
        if pauli < sv / 4.0 && t <= 22 {
            return Backend::PauliPath;
        }
    }
    if terminal {
        if let Some(rule) = &opts.adaptive {
            if rule.accepts(c).is_some() {
                return Backend::Adaptive;
            }
        }
    }
    Backend::StateVector
}

/// Prepares the final state of a unitary component on the state vector,
/// absorbing its causal Clifford prefix with one stabilizer-to-vector pass.
pub fn prepare_statevector<T: Real>(
    c: &Circuit,
    use_prefix: bool,
) -> Result<StateVector<T>, SimError> {
    let n = c.num_qubits;
    let bytes = state_bytes::<T>(n);
    if n >= 63 || bytes > MAX_STATE_BYTES {
        return Err(SimError::TooLarge {
            what: "state vector",
            bytes,
            limit: MAX_STATE_BYTES,
        });
    }
    if use_prefix && n <= 64 {
        let (prefix, rest) = clifford_prefix(c);
        if prefix.num_gates() > 0 {
            let mut s = clifford_statevector::<T>(&prefix);
            apply_ops(&mut s, &rest)?;
            return Ok(s);
        }
    }
    let mut s = StateVector::<T>::try_new(n)?;
    apply_ops(&mut s, c)?;
    Ok(s)
}

fn apply_ops<T: Real>(s: &mut StateVector<T>, c: &Circuit) -> Result<(), SimError> {
    require_unitary(c, "preparing a state vector of a non-unitary circuit")?;
    // One batch: the cache-blocked executor fuses and reorders it.
    let gates: Vec<Gate> = c.gates().copied().collect();
    Simulator::apply_gates(s, &gates)
}

/// The state-independent and state-aware rewrites shared by every plan:
/// peephole, SWAP elimination, state propagation (keeping the final state
/// of the logical qubits `keep`), peephole again. Returns the circuit on
/// wires, the global phase it dropped and `wire_of` (logical -> wire).
fn front_end(
    c: &Circuit,
    keep: &[usize],
    opts: &PlanOptions,
    stats: &mut CompileStats,
) -> (Circuit, f64, Vec<usize>) {
    let n = c.num_qubits;
    stats.gates_in = c.num_gates();
    let mut phase = 0.0;
    let mut c = if opts.peephole {
        let o = optimize(c);
        phase += o.global_phase;
        o.circuit
    } else {
        c.clone()
    };
    if opts.phase_fold {
        let o = super::phasefold::phase_fold(&c);
        phase += o.global_phase;
        c = o.circuit;
        if opts.peephole {
            let o = optimize(&c);
            phase += o.global_phase;
            c = o.circuit;
        }
    }
    stats.gates_after_peephole = c.num_gates();
    let mut wire_of: Vec<usize> = (0..n).collect();
    if opts.swap_elim {
        let (d, w) = eliminate_swaps(&c);
        c = d;
        wire_of = w;
    }
    if opts.state_prop {
        let keep_w: Vec<usize> = keep.iter().map(|&q| wire_of[q]).collect();
        let (d, ph) = propagate(&c, &keep_w);
        phase += ph;
        c = d;
        if opts.peephole {
            let o = optimize(&c);
            phase += o.global_phase;
            c = o.circuit;
        }
    }
    stats.gates_after_state_prop = c.num_gates();
    (c, phase, wire_of)
}

/// Compiles a circuit for sampling its measurement outcomes.
///
/// Any circuit [`Circuit::run`] accepts is supported. Unitary circuits with
/// terminal measurements get the full treatment (classical suffix,
/// per-component backend choice, Pauli paths). Circuits with mid-circuit
/// measurements, resets, classically controlled gates or noise channels
/// are split into independent components (classical control joins the
/// reading and the measured qubits) and every component is run shot by
/// shot with [`Circuit::run`], after the passes that are exact for them.
///
/// Fails with the error `Circuit::run` would report for invalid qubit or
/// classical-bit indices.
pub fn compile_sampling(c: &Circuit, opts: PlanOptions) -> Result<SamplingPlan, SimError> {
    validate(c)?;
    let n = c.num_qubits;
    let mut stats = CompileStats::default();
    // Global phase and final wire positions are irrelevant for sampling:
    // measurements were relabelled along with the gates.
    let (c, _, _) = front_end(c, &[], &opts, &mut stats);
    let group = |body: &Circuit| -> Vec<Vec<usize>> {
        if opts.split {
            components(body)
        } else {
            vec![(0..n).collect()]
        }
    };
    if let Some((body, meas)) = terminal_measurements(&c) {
        let (body, suffix) = if opts.suffix {
            split_monomial_suffix(&body)
        } else {
            (body, Vec::new())
        };
        stats.suffix_gates = suffix.len();
        let need = suffix_inputs(n, &meas, &suffix);
        let body = if opts.light_cone {
            let outs: Vec<usize> = (0..n).filter(|&q| need[q]).collect();
            light_cone(&body, &outs)
        } else {
            body
        };
        stats.gates_after_light_cone = body.num_gates();
        let mut comps = Vec::new();
        for qs in group(&body) {
            let needed: Vec<usize> = (0..qs.len()).filter(|&i| need[qs[i]]).collect();
            if needed.is_empty() && opts.light_cone {
                continue;
            }
            let circuit = restrict(&body, &qs);
            let backend = choose_backend(&circuit, needed.len(), true, &opts);
            comps.push(Component {
                qubits: qs,
                circuit,
                backend,
                needed,
            });
        }
        fill_stats(&mut stats, &comps);
        return Ok(SamplingPlan {
            n,
            kind: Kind::Terminal { meas, suffix },
            comps,
            opts,
            stats,
        });
    }
    let c = if opts.light_cone {
        light_cone(&c, &[])
    } else {
        c
    };
    stats.gates_after_light_cone = c.num_gates();
    let groups = group(&c);
    let mut comp_of = vec![0; n];
    for (i, qs) in groups.iter().enumerate() {
        for &q in qs {
            comp_of[q] = i;
        }
    }
    let mut count = vec![0; groups.len()];
    let mut src = Vec::new();
    for op in &c.ops {
        if let Op::Measure(q) = op {
            let ci = comp_of[*q];
            src.push((ci, count[ci]));
            count[ci] += 1;
        }
    }
    let comps: Vec<Component> = groups
        .into_iter()
        .map(|qs| {
            let circuit = restrict(&c, &qs);
            // Idle only if nothing but measurements happen: a reset or a
            // noise channel without any gate still needs simulating.
            let idle = circuit.ops.iter().all(|o| matches!(o, Op::Measure(_)));
            let backend = if idle {
                Backend::Idle
            } else if opts.dispatch && circuit.is_clifford() {
                Backend::Tableau
            } else {
                Backend::StateVector
            };
            Component {
                qubits: qs,
                circuit,
                backend,
                needed: Vec::new(),
            }
        })
        .collect();
    fill_stats(&mut stats, &comps);
    Ok(SamplingPlan {
        n,
        kind: Kind::MidCircuit { src },
        comps,
        opts,
        stats,
    })
}

fn fill_stats(stats: &mut CompileStats, comps: &[Component]) {
    stats.gates_simulated = comps.iter().map(|c| c.circuit.num_gates()).sum();
    stats.components = comps
        .iter()
        .filter(|c| c.circuit.num_gates() > 0)
        .map(|c| (c.qubits.len(), c.circuit.num_gates(), c.backend))
        .collect();
}

/// Draws `shots` samples of a component's needed final bits (terminal
/// plans); each sample lists the bits in `comp.needed` order.
fn sample_component<T: Real, R: Rng>(
    comp: &Component,
    shots: usize,
    use_prefix: bool,
    adaptive_max_active: usize,
    rng: &mut R,
) -> Result<Vec<Vec<bool>>, SimError> {
    let k = comp.needed.len();
    Ok(match comp.backend {
        Backend::Idle => vec![vec![false; k]; shots],
        Backend::Tableau => {
            let mut t = Tableau::try_new(comp.qubits.len())?;
            for g in comp.circuit.gates() {
                t.apply_gate(g)?;
            }
            t.sample(shots, rng)
                .into_iter()
                .map(|b| comp.needed.iter().map(|&q| b[q]).collect())
                .collect()
        }
        Backend::StateVector => {
            let s = prepare_statevector::<T>(&comp.circuit, use_prefix)?;
            s.sample(shots, rng)
                .into_iter()
                .map(|x| comp.needed.iter().map(|&q| (x >> q) & 1 == 1).collect())
                .collect()
        }
        Backend::Adaptive => {
            let max_active = adaptive_max_active;
            let sampler = CompressedState::new(&comp.circuit, max_active)?.sampler();
            (0..shots)
                .map(|_| {
                    let b = sampler.sample_packed(rng);
                    comp.needed
                        .iter()
                        .map(|&q| (b[q / 64] >> (q % 64)) & 1 == 1)
                        .collect()
                })
                .collect()
        }
        Backend::PauliPath => {
            let p = pauli_path::marginal_distribution(&comp.circuit, &comp.needed)?;
            let cdf: Vec<f64> = p
                .iter()
                .scan(0.0, |acc, &x| {
                    *acc += x.max(0.0);
                    Some(*acc)
                })
                .collect();
            let total = *cdf.last().expect("nonempty");
            (0..shots)
                .map(|_| {
                    let r = rng.random::<f64>() * total;
                    let b = cdf.partition_point(|&c| c <= r).min(cdf.len() - 1);
                    (0..k).map(|i| (b >> i) & 1 == 1).collect()
                })
                .collect()
        }
    })
}

impl SamplingPlan {
    pub fn components(&self) -> &[Component] {
        &self.comps
    }

    /// Draws `shots` outcome records (one bool per measurement, in program
    /// order), with the same distribution as running the original circuit.
    pub fn sample<T: Real, R: Rng>(
        &self,
        shots: usize,
        rng: &mut R,
    ) -> Result<Vec<Vec<bool>>, SimError> {
        match &self.kind {
            Kind::Terminal { meas, suffix } => {
                let mut bits = vec![vec![false; self.n]; shots];
                for comp in &self.comps {
                    let max_active = self
                        .opts
                        .adaptive
                        .map_or(AdaptiveRule::default().max_active, |r| r.max_active);
                    let s = sample_component::<T, R>(
                        comp,
                        shots,
                        self.opts.clifford_prefix,
                        max_active,
                        rng,
                    )?;
                    for (row, sb) in bits.iter_mut().zip(s) {
                        for (&lq, b) in comp.needed.iter().zip(sb) {
                            row[comp.qubits[lq]] = b;
                        }
                    }
                }
                Ok(bits
                    .into_iter()
                    .map(|mut row| {
                        apply_classical(&mut row, suffix);
                        meas.iter().map(|&q| row[q]).collect()
                    })
                    .collect())
            }
            Kind::MidCircuit { src } => {
                // Prepare each component's measurement-free Clifford prefix
                // once; every shot starts from a copy.
                let mut per_comp: Vec<Vec<Vec<bool>>> = Vec::new();
                for comp in &self.comps {
                    per_comp.push(run_component_shots::<T, R>(
                        comp,
                        shots,
                        self.opts.clifford_prefix,
                        rng,
                    )?);
                }
                Ok((0..shots)
                    .map(|s| src.iter().map(|&(c, i)| per_comp[c][s][i]).collect())
                    .collect())
            }
        }
    }

    /// The exact distribution of outcome records (for testing; exponential
    /// in the component sizes, and every component is simulated on a
    /// state vector here).
    pub fn exact_distribution(&self) -> BTreeMap<Vec<bool>, f64> {
        match &self.kind {
            Kind::Terminal { meas, suffix } => {
                let mut dist: BTreeMap<Vec<bool>, f64> = BTreeMap::new();
                dist.insert(vec![false; self.n], 1.0);
                for comp in &self.comps {
                    let s = prepare_statevector::<f64>(&comp.circuit, self.opts.clifford_prefix)
                        .expect("small");
                    let p = s.probabilities();
                    let mut next = BTreeMap::new();
                    for (bits, w) in &dist {
                        for (x, &px) in p.iter().enumerate() {
                            if px < 1e-15 {
                                continue;
                            }
                            let mut b = bits.clone();
                            for &lq in &comp.needed {
                                b[comp.qubits[lq]] = (x >> lq) & 1 == 1;
                            }
                            *next.entry(b).or_insert(0.0) += w * px;
                        }
                    }
                    dist = next;
                }
                let mut out = BTreeMap::new();
                for (mut b, w) in dist {
                    apply_classical(&mut b, suffix);
                    let key: Vec<bool> = meas.iter().map(|&q| b[q]).collect();
                    *out.entry(key).or_insert(0.0) += w;
                }
                out
            }
            Kind::MidCircuit { src } => {
                let per: Vec<BTreeMap<Vec<bool>, f64>> = self
                    .comps
                    .iter()
                    .map(|c| exact_outcome_distribution(&c.circuit))
                    .collect();
                let mut joint: Vec<(Vec<Vec<bool>>, f64)> = vec![(Vec::new(), 1.0)];
                for d in &per {
                    let mut next = Vec::new();
                    for (rec, w) in &joint {
                        for (o, p) in d {
                            let mut r = rec.clone();
                            r.push(o.clone());
                            next.push((r, w * p));
                        }
                    }
                    joint = next;
                }
                let mut out = BTreeMap::new();
                for (rec, w) in joint {
                    let key: Vec<bool> = src.iter().map(|&(c, i)| rec[c][i]).collect();
                    *out.entry(key).or_insert(0.0) += w;
                }
                out
            }
        }
    }
}

fn run_component_shots<T: Real, R: Rng>(
    comp: &Component,
    shots: usize,
    use_prefix: bool,
    rng: &mut R,
) -> Result<Vec<Vec<bool>>, SimError> {
    let has_meas = comp.circuit.ops.iter().any(|o| matches!(o, Op::Measure(_)));
    if !has_meas {
        return Ok(vec![Vec::new(); shots]);
    }
    let nq = comp.circuit.num_qubits;
    match comp.backend {
        Backend::Idle => {
            // Only measurements of |0...0>: every outcome is 0.
            let k = comp.circuit.ops.len();
            Ok(vec![vec![false; k]; shots])
        }
        Backend::Tableau => {
            let mut out = Vec::with_capacity(shots);
            for _ in 0..shots {
                let mut t = Tableau::try_new(nq)?;
                out.push(comp.circuit.run(&mut t, rng)?);
            }
            Ok(out)
        }
        _ => {
            let (prefix, rest) = if use_prefix {
                clifford_prefix(&comp.circuit)
            } else {
                (Circuit::new(nq), comp.circuit.clone())
            };
            let start = prepare_statevector::<T>(&prefix, true)?;
            let mut out = Vec::with_capacity(shots);
            for _ in 0..shots {
                let mut s = start.clone();
                out.push(rest.run(&mut s, rng)?);
            }
            Ok(out)
        }
    }
}

/// Exact distribution of the outcome records of a circuit, by branching
/// on every measurement, reset and noise channel of a state vector (tests
/// and small circuits only: the number of branches is exponential).
///
/// Follows the semantics of [`Circuit::run`]: a reset measures and flips
/// back to `|0>` without recording, `XFlip(q, p)` applies `X` with
/// probability `p`, `Depolarize1q` applies each of `X, Y, Z` with
/// probability `p/3`, `Depolarize2q` each of the 15 non-identity Pauli
/// pairs with probability `p/15`, and a classically controlled gate is
/// applied when the recorded outcome matches.
pub fn exact_outcome_distribution(c: &Circuit) -> BTreeMap<Vec<bool>, f64> {
    fn branch(
        c: &Circuit,
        i: usize,
        s: &StateVector<f64>,
        p: f64,
        rec: &mut Vec<bool>,
        out: &mut BTreeMap<Vec<bool>, f64>,
        cases: &[(f64, &[Gate])],
    ) {
        for &(w, gates) in cases {
            if w * p < 1e-300 || w <= 0.0 {
                continue;
            }
            let mut t = s.clone();
            for g in gates {
                t.apply_gate(g).expect("valid");
            }
            go(c, i + 1, t, p * w, rec, out);
        }
    }
    fn go(
        c: &Circuit,
        i: usize,
        s: StateVector<f64>,
        p: f64,
        rec: &mut Vec<bool>,
        out: &mut BTreeMap<Vec<bool>, f64>,
    ) {
        let mut s = s;
        let mut i = i;
        let pauli = |k: usize, q: usize| match k {
            1 => Some(Gate::X(q)),
            2 => Some(Gate::Y(q)),
            3 => Some(Gate::Z(q)),
            _ => None,
        };
        while i < c.ops.len() {
            match c.ops[i] {
                Op::Gate(g) => s.apply_gate(&g).expect("valid"),
                Op::ClassicControlled {
                    gate,
                    meas_index,
                    target_value,
                } => {
                    if rec[meas_index] == target_value {
                        s.apply_gate(&gate).expect("valid");
                    }
                }
                Op::Measure(q) | Op::Reset(q) => {
                    let record = matches!(c.ops[i], Op::Measure(_));
                    let p1 = s.prob_one(q);
                    for (outcome, po) in [(false, 1.0 - p1), (true, p1)] {
                        if po < 1e-14 {
                            continue;
                        }
                        let mut t = s.clone();
                        t.collapse(q, outcome);
                        if record {
                            rec.push(outcome);
                        } else if outcome {
                            t.apply_gate(&Gate::X(q)).expect("valid");
                        }
                        go(c, i + 1, t, p * po, rec, out);
                        if record {
                            rec.pop();
                        }
                    }
                    return;
                }
                Op::XFlip(q, pf) | Op::YFlip(q, pf) | Op::ZFlip(q, pf) => {
                    let e = match c.ops[i] {
                        Op::XFlip(..) => Gate::X(q),
                        Op::YFlip(..) => Gate::Y(q),
                        _ => Gate::Z(q),
                    };
                    let pf = pf.clamp(0.0, 1.0);
                    branch(c, i, &s, p, rec, out, &[(1.0 - pf, &[]), (pf, &[e])]);
                    return;
                }
                Op::Depolarize1q(q, pd) => {
                    let pd = pd.clamp(0.0, 1.0);
                    let (x, y, z) = ([Gate::X(q)], [Gate::Y(q)], [Gate::Z(q)]);
                    let cases: [(f64, &[Gate]); 4] = [
                        (1.0 - pd, &[]),
                        (pd / 3.0, &x),
                        (pd / 3.0, &y),
                        (pd / 3.0, &z),
                    ];
                    branch(c, i, &s, p, rec, out, &cases);
                    return;
                }
                Op::Depolarize2q(a, b, pd) => {
                    let pd = pd.clamp(0.0, 1.0);
                    let errs: Vec<Vec<Gate>> = (1..16)
                        .map(|k| pauli(k / 4, a).into_iter().chain(pauli(k % 4, b)).collect())
                        .collect();
                    let mut cases: Vec<(f64, &[Gate])> = vec![(1.0 - pd, &[])];
                    cases.extend(errs.iter().map(|e| (pd / 15.0, &e[..])));
                    branch(c, i, &s, p, rec, out, &cases);
                    return;
                }
            }
            i += 1;
        }
        *out.entry(rec.clone()).or_insert(0.0) += p;
    }
    let mut out = BTreeMap::new();
    go(
        c,
        0,
        StateVector::new(c.num_qubits),
        1.0,
        &mut Vec::new(),
        &mut out,
    );
    out
}

/// A unitary circuit compiled for producing its final state.
#[derive(Clone, Debug)]
pub struct UnitaryPlan {
    n: usize,
    /// `U = e^{i global_phase} (U_1 ⊗ U_2 ⊗ ...)`.
    pub global_phase: f64,
    comps: Vec<Component>,
    use_prefix: bool,
    pub stats: CompileStats,
}

/// Compiles a unitary circuit for computing its state vector or individual
/// amplitudes. Refuses ([`SimError::NotSupported`]) circuits with
/// measurements, resets, classically controlled gates or noise channels,
/// which have no single final state.
pub fn compile_unitary(c: &Circuit, opts: PlanOptions) -> Result<UnitaryPlan, SimError> {
    validate(c)?;
    require_unitary(
        c,
        "compile_unitary needs a unitary circuit (no measurements, resets, \
         classical control or noise); use compile_sampling",
    )?;
    let mut stats = CompileStats::default();
    let all: Vec<usize> = (0..c.num_qubits).collect();
    let (c, global_phase, wire_of) = front_end(c, &all, &opts, &mut stats);
    let mut logical_of = vec![0; c.num_qubits];
    for (q, &w) in wire_of.iter().enumerate() {
        logical_of[w] = q;
    }
    stats.gates_after_light_cone = c.num_gates();
    let groups = if opts.split {
        components(&c)
    } else {
        vec![(0..c.num_qubits).collect()]
    };
    let comps: Vec<Component> = groups
        .into_iter()
        .map(|qs| {
            let circuit = restrict(&c, &qs);
            let backend = if circuit.num_gates() == 0 {
                Backend::Idle
            } else {
                Backend::StateVector
            };
            Component {
                needed: (0..qs.len()).collect(),
                qubits: qs.iter().map(|&w| logical_of[w]).collect(),
                circuit,
                backend,
            }
        })
        .collect();
    fill_stats(&mut stats, &comps);
    Ok(UnitaryPlan {
        n: c.num_qubits,
        global_phase,
        comps,
        use_prefix: opts.clifford_prefix,
        stats,
    })
}

/// Component states of a [`UnitaryPlan`]; enough to evaluate any
/// amplitude without ever building the full vector.
pub struct FactoredState<T: Real> {
    n: usize,
    phase: Complex64,
    parts: Vec<(Vec<usize>, Option<StateVector<T>>)>,
}

impl UnitaryPlan {
    pub fn components(&self) -> &[Component] {
        &self.comps
    }

    /// Simulates every component (idle ones cost nothing).
    pub fn factored<T: Real>(&self) -> Result<FactoredState<T>, SimError> {
        let parts = self
            .comps
            .iter()
            .map(|c| {
                Ok((
                    c.qubits.clone(),
                    match c.backend {
                        Backend::Idle => None,
                        _ => Some(prepare_statevector::<T>(&c.circuit, self.use_prefix)?),
                    },
                ))
            })
            .collect::<Result<_, SimError>>()?;
        Ok(FactoredState {
            n: self.n,
            phase: Complex64::from_polar(1.0, self.global_phase),
            parts,
        })
    }

    /// The full state vector.
    pub fn statevector<T: Real>(&self) -> Result<StateVector<T>, SimError> {
        self.factored::<T>()?.to_statevector()
    }
}

impl<T: Real> FactoredState<T> {
    /// `<x|ψ>` for a basis state `x` (bit `q` = qubit `q`).
    pub fn amplitude(&self, x: u128) -> Complex64 {
        let mut a = self.phase;
        for (qs, s) in &self.parts {
            let local = qs.iter().enumerate().fold(0usize, |acc, (i, &q)| {
                acc | ((((x >> q) & 1) as usize) << i)
            });
            match s {
                None => {
                    if local != 0 {
                        return Complex64::new(0.0, 0.0);
                    }
                }
                Some(s) => a *= s.amplitude(local),
            }
        }
        a
    }

    /// Expands the tensor product into one vector.
    pub fn to_statevector(&self) -> Result<StateVector<T>, SimError> {
        let n = self.n;
        let bytes = state_bytes::<T>(n);
        if n >= 63 || bytes > MAX_STATE_BYTES {
            return Err(SimError::TooLarge {
                what: "state vector",
                bytes,
                limit: MAX_STATE_BYTES,
            });
        }
        // Idle qubits must be 0: mask of those, then for the rest gather
        // local indices with a low/high split of the global index.
        let mut idle_mask = 0usize;
        let mut live: Vec<(&[usize], &StateVector<T>)> = Vec::new();
        for (qs, s) in &self.parts {
            match s {
                None => idle_mask |= qs.iter().fold(0, |m, &q| m | (1 << q)),
                Some(s) => live.push((qs, s)),
            }
        }
        let low = n.min(12);
        let len = 1usize << n;
        let chunk = 1usize << low;
        // Per component: local index contributed by the low bits.
        let low_tab: Vec<Vec<usize>> = live
            .iter()
            .map(|(qs, _)| {
                (0..chunk)
                    .map(|v| {
                        qs.iter().enumerate().fold(0, |acc, (i, &q)| {
                            if q < low {
                                acc | (((v >> q) & 1) << i)
                            } else {
                                acc
                            }
                        })
                    })
                    .collect()
            })
            .collect();
        let ph = Complex::new(T::from_f64(self.phase.re), T::from_f64(self.phase.im));
        let zero = Complex::new(T::zero(), T::zero());
        let mut amps: Vec<Complex<T>> = (0..len).into_par_iter().map(|_| zero).collect();
        amps.par_chunks_mut(chunk).enumerate().for_each(|(h, out)| {
            let base = h << low;
            if base & idle_mask & !(chunk - 1) != 0 {
                return;
            }
            let hi: Vec<usize> = live
                .iter()
                .map(|(qs, _)| {
                    qs.iter().enumerate().fold(0, |acc, (i, &q)| {
                        if q >= low {
                            acc | (((base >> q) & 1) << i)
                        } else {
                            acc
                        }
                    })
                })
                .collect();
            for (v, o) in out.iter_mut().enumerate() {
                if v & idle_mask != 0 {
                    continue;
                }
                let mut a = ph;
                for (ci, (_, s)) in live.iter().enumerate() {
                    a = a * s.amplitudes()[hi[ci] | low_tab[ci][v]];
                }
                *o = a;
            }
        });
        Ok(StateVector::from_amplitudes(amps))
    }
}

/// Exact `<Z_{q1} Z_{q2} ...>` after a unitary circuit, using the light
/// cone of the observable, the component factorisation (the expectation of
/// a product over independent components is the product of expectations)
/// and the cheaper of Pauli paths and the state vector per component.
///
/// Refuses ([`SimError::NotSupported`]) non-unitary circuits.
pub fn expectation_z_product(
    c: &Circuit,
    qubits: &[usize],
    opts: PlanOptions,
) -> Result<f64, SimError> {
    validate(c)?;
    require_unitary(
        c,
        "expectation_z_product needs a unitary circuit (no measurements, resets, \
         classical control or noise)",
    )?;
    for &q in qubits {
        if q >= c.num_qubits {
            return Err(SimError::QubitOutOfRange {
                qubit: q,
                num_qubits: c.num_qubits,
            });
        }
    }
    let mut stats = CompileStats::default();
    let (c, _, wire_of) = front_end(c, qubits, &opts, &mut stats);
    let qubits: Vec<usize> = qubits.iter().map(|&q| wire_of[q]).collect();
    let qubits = &qubits[..];
    let c = if opts.light_cone {
        light_cone(&c, qubits)
    } else {
        c
    };
    let groups = if opts.split {
        components(&c)
    } else {
        vec![(0..c.num_qubits).collect()]
    };
    let mut value = 1.0;
    for qs in groups {
        let local: Vec<usize> = (0..qs.len()).filter(|&i| qubits.contains(&qs[i])).collect();
        if local.is_empty() {
            continue;
        }
        let sub = restrict(&c, &qs);
        if sub.num_gates() == 0 {
            continue; // <0|Z..Z|0> = 1
        }
        let n = sub.num_qubits;
        let t = non_clifford_count(&sub) as u32;
        let use_pauli = opts.dispatch
            && t <= 24
            && (t as f64).exp2() * (n.div_ceil(64) as f64) < (n.min(60) as f64).exp2() / 4.0;
        let adaptive = if opts.dispatch && !use_pauli && !sub.is_clifford() {
            opts.adaptive.filter(|r| r.accepts(&sub).is_some())
        } else {
            None
        };
        let v = if use_pauli || (opts.dispatch && sub.is_clifford()) {
            let obs = PauliSum::z_product(n, &local);
            pauli_path::expectation(&sub, &obs, DEFAULT_MAX_TERMS)?.0
        } else if let Some(rule) = adaptive {
            let obs = PauliSum::z_product(n, &local);
            let ao = AdaptiveOptions {
                max_dense_qubits: rule.max_active,
                ..AdaptiveOptions::default()
            };
            adaptive_expectation(&sub, &obs, &ao)?.value
        } else {
            let s = prepare_statevector::<f64>(&sub, opts.clifford_prefix)?;
            let mask = local.iter().fold(0usize, |m, &q| m | (1 << q));
            s.amplitudes()
                .par_iter()
                .enumerate()
                .map(|(i, a)| {
                    let p = a.norm_sqr();
                    if (i & mask).count_ones() % 2 == 1 {
                        -p
                    } else {
                        p
                    }
                })
                .sum()
        };
        value *= v;
    }
    Ok(value)
}
