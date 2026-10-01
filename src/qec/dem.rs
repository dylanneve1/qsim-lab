//! Circuit-derived detector error models (DEMs) for Clifford circuits with
//! stochastic Pauli noise.
//!
//! A *detector* is a parity of measurement records that is deterministic in
//! the absence of noise; an *observable* is a parity of measurement records
//! that encodes the logical outcome. Because every noise channel in
//! [`NoiseModel`] is a stochastic Pauli channel and the circuit is Clifford,
//! the effect of a fault on the detectors and the observable is a fixed
//! GF(2)-linear function of the Pauli that was inserted. This module
//! computes that function exactly, for **every** noise location the
//! `NoiseModel` acts on and **every** Pauli it can insert there.
//!
//! # Noise locations (mirrors [`Circuit::run_noisy`] exactly)
//!
//! | location                         | outcomes (mutually exclusive)                     |
//! |----------------------------------|---------------------------------------------------|
//! | after every 1-qubit gate         | `X`, `Y`, `Z`, each with probability `p_1q / 3`   |
//! | after every 2-qubit gate         | the 15 non-identity Paulis on `(a, b)`, each `p_2q / 15` |
//! | at every measurement             | classical flip of that record, probability `p_meas` |
//! | after every reset                | `X` on the reset qubit, probability `p_reset`     |
//!
//! Three-qubit gates receive no noise in `run_noisy`, so they are not fault
//! locations here either.
//!
//! # Two independent propagation algorithms
//!
//! * [`CircuitFaults::from_circuit`] uses a single **backward** sweep: it
//!   tracks, for every qubit, which detectors/observable an `X` (resp. `Z`)
//!   error inserted *at that point* would flip (bit-packed over detectors).
//!   Rules, read right-to-left: measuring qubit `q` (Z basis) adds the
//!   record's detector set to `q`'s X-sensitivity; reset clears both
//!   sensitivities of `q`; `H` swaps X/Z sensitivity; `CNOT(c, t)` sets
//!   `sx[c] ^= sx[t]` and `sz[t] ^= sz[c]`; `S`/`S†` set `sx ^= sz`;
//!   `CZ(a, b)` sets `sx[a] ^= sz[b]`, `sx[b] ^= sz[a]`; `SWAP` swaps.
//!   Cost: one pass over the circuit, independent of the number of faults.
//! * [`propagate_forward`] is the textbook **forward** Pauli-frame
//!   simulation of a single fault: `H` swaps X↔Z, `CNOT` copies X from
//!   control to target and Z from target to control, a Z-basis measurement
//!   record is flipped iff the frame has an X component on that qubit, and
//!   reset clears the frame on that qubit. It is quadratic when used for all
//!   faults and exists as an independent cross-check (see the tests, which
//!   compare both on every fault of the surface and repetition circuits).
//!
//! # Exact sampling
//!
//! [`DemSampler`] samples **exactly the same distribution** over
//! (detector events, observable flip) as running the circuit through
//! `run_noisy` on the tableau: each location fires independently with its
//! probability and, when it fires, picks one of its outcomes uniformly — the
//! same mutually exclusive choice `run_noisy` makes. The outcomes' detector
//! signatures are XOR-ed. (No independent-mechanism approximation and no
//! merging is used for sampling; merging into [`ErrorMechanism`]s is only
//! offered for inspection and for building decoding graphs.) Firing
//! locations are drawn with geometric skipping, which is exact for i.i.d.
//! Bernoulli trials up to floating-point rounding of `ln`.

use crate::circuit::{Circuit, Op};
use crate::gate::Gate;
use crate::noise::NoiseModel;
use rand::{Rng, RngCore};
use std::collections::HashMap;

/// Single-qubit Pauli (non-identity).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pauli {
    X,
    Y,
    Z,
}

impl Pauli {
    /// `1 -> X`, `2 -> Y`, `3 -> Z`, `0 -> None` (matches the index
    /// convention of [`crate::noise::sample_depolarizing_2q`]).
    pub fn from_index(i: usize) -> Option<Pauli> {
        match i {
            0 => None,
            1 => Some(Pauli::X),
            2 => Some(Pauli::Y),
            3 => Some(Pauli::Z),
            _ => panic!("invalid Pauli index {i}"),
        }
    }

    fn has_x(self) -> bool {
        matches!(self, Pauli::X | Pauli::Y)
    }

    fn has_z(self) -> bool {
        matches!(self, Pauli::Y | Pauli::Z)
    }
}

/// The effect of one fault outcome: the detectors it flips (sorted) and
/// whether it flips the logical observable.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Signature {
    pub detectors: Vec<usize>,
    pub flips_logical: bool,
}

impl Signature {
    /// True if the fault has no effect on detectors or observable.
    pub fn is_trivial(&self) -> bool {
        self.detectors.is_empty() && !self.flips_logical
    }
}

/// Which `NoiseModel` parameter drives a fault location.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FaultKind {
    /// Depolarizing after a 1-qubit gate (`p_1q`, outcomes X, Y, Z).
    Gate1q,
    /// Depolarizing after a 2-qubit gate (`p_2q`, 15 outcomes).
    Gate2q,
    /// Classical readout flip (`p_meas`, 1 outcome).
    Readout,
    /// X after reset (`p_reset`, 1 outcome).
    Reset,
}

impl FaultKind {
    /// Total firing probability of a location of this kind.
    pub fn probability(self, noise: &NoiseModel) -> f64 {
        match self {
            FaultKind::Gate1q => noise.p_1q,
            FaultKind::Gate2q => noise.p_2q,
            FaultKind::Readout => noise.p_meas,
            FaultKind::Reset => noise.p_reset,
        }
    }
}

/// One noise location of the circuit with all of its (equiprobable,
/// mutually exclusive) outcomes.
#[derive(Clone, Debug)]
pub struct FaultLocation {
    /// Index of the op in `circuit.ops` the noise is attached to.
    pub op_index: usize,
    pub kind: FaultKind,
    /// Outcomes, each chosen with probability `1 / outcomes.len()` when the
    /// location fires. Order: 1q `[X, Y, Z]`; 2q index `k = 1..16` with
    /// Pauli `k / 4` on the first qubit and `k % 4` on the second.
    pub outcomes: Vec<Signature>,
}

/// An independent error mechanism of a merged detector error model.
#[derive(Clone, Debug)]
pub struct ErrorMechanism {
    pub detectors: Vec<usize>,
    pub flips_logical: bool,
    pub probability: f64,
}

/// Errors from DEM construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DemError {
    /// The circuit contains an op the frame propagation does not support
    /// (non-Clifford gate, classical control, or an explicit noise channel).
    Unsupported { op_index: usize, what: String },
    /// A detector or observable refers to a measurement that does not exist.
    BadRecord { record: usize, available: usize },
}

impl std::fmt::Display for DemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DemError::Unsupported { op_index, what } => {
                write!(f, "op {op_index}: unsupported in DEM construction: {what}")
            }
            DemError::BadRecord { record, available } => {
                write!(f, "measurement record {record} out of range ({available})")
            }
        }
    }
}

impl std::error::Error for DemError {}

/// Every fault location of a circuit with its detector signatures.
#[derive(Clone, Debug)]
pub struct CircuitFaults {
    pub num_detectors: usize,
    pub locations: Vec<FaultLocation>,
}

/// Bit-packed set over `num_detectors + 1` bits (last bit = observable).
type Bits = Vec<u64>;

fn xor_into(dst: &mut [u64], src: &[u64]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d ^= *s;
    }
}

fn bits_to_signature(bits: &[u64], num_detectors: usize) -> Signature {
    let mut detectors = Vec::new();
    for (w, &word) in bits.iter().enumerate() {
        let mut m = word;
        while m != 0 {
            let b = m.trailing_zeros() as usize;
            m &= m - 1;
            let idx = w * 64 + b;
            if idx < num_detectors {
                detectors.push(idx);
            }
        }
    }
    let obs = num_detectors;
    Signature {
        detectors,
        flips_logical: (bits[obs / 64] >> (obs % 64)) & 1 == 1,
    }
}

/// For each measurement record, the bitset of detectors (and observable) it
/// participates in.
fn record_bits(
    num_records: usize,
    detectors: &[Vec<usize>],
    observable: &[usize],
) -> Result<Vec<Bits>, DemError> {
    let words = (detectors.len() + 1).div_ceil(64);
    let mut rec = vec![vec![0u64; words]; num_records];
    let mut set = |r: usize, bit: usize| -> Result<(), DemError> {
        if r >= num_records {
            return Err(DemError::BadRecord {
                record: r,
                available: num_records,
            });
        }
        rec[r][bit / 64] ^= 1u64 << (bit % 64);
        Ok(())
    };
    for (di, recs) in detectors.iter().enumerate() {
        for &r in recs {
            set(r, di)?;
        }
    }
    for &r in observable {
        set(r, detectors.len())?;
    }
    Ok(rec)
}

fn unsupported(op_index: usize, what: impl Into<String>) -> DemError {
    DemError::Unsupported {
        op_index,
        what: what.into(),
    }
}

/// Combine the X/Z sensitivities of the qubits a Pauli acts on.
fn pauli_bits(sx: &[Bits], sz: &[Bits], paulis: &[(usize, Pauli)], words: usize) -> Bits {
    let mut out = vec![0u64; words];
    for &(q, p) in paulis {
        if p.has_x() {
            xor_into(&mut out, &sx[q]);
        }
        if p.has_z() {
            xor_into(&mut out, &sz[q]);
        }
    }
    out
}

/// The Paulis of 2-qubit depolarizing outcome `k` in `1..16` on `(a, b)`.
pub fn two_qubit_outcome(k: usize, a: usize, b: usize) -> Vec<(usize, Pauli)> {
    let mut v = Vec::with_capacity(2);
    if let Some(p) = Pauli::from_index(k / 4) {
        v.push((a, p));
    }
    if let Some(p) = Pauli::from_index(k % 4) {
        v.push((b, p));
    }
    v
}

impl CircuitFaults {
    /// Enumerates every fault location of `circuit` and computes the
    /// signature of every outcome with one backward sensitivity sweep.
    ///
    /// `detectors[i]` lists the measurement-record indices whose parity is
    /// detector `i`; `observable` lists the records whose parity is the
    /// logical observable. Detectors must be deterministic without noise;
    /// this is not checked here (the surface-code tests check it).
    pub fn from_circuit(
        circuit: &Circuit,
        detectors: &[Vec<usize>],
        observable: &[usize],
    ) -> Result<Self, DemError> {
        let nq = circuit.num_qubits;
        let nd = detectors.len();
        let words = (nd + 1).div_ceil(64);
        let num_records = circuit
            .ops
            .iter()
            .filter(|o| matches!(o, Op::Measure(_)))
            .count();
        let rec = record_bits(num_records, detectors, observable)?;

        let mut sx: Vec<Bits> = vec![vec![0u64; words]; nq];
        let mut sz: Vec<Bits> = vec![vec![0u64; words]; nq];
        let mut next_record = num_records;
        let mut locations = Vec::new();

        for (i, op) in circuit.ops.iter().enumerate().rev() {
            // `sx`/`sz` currently describe a Pauli inserted right AFTER op i.
            match op {
                Op::Gate(g) => {
                    let qs = g.qubits();
                    match qs.len() {
                        1 => {
                            let q = qs[0];
                            let outcomes = [Pauli::X, Pauli::Y, Pauli::Z]
                                .iter()
                                .map(|&p| {
                                    bits_to_signature(&pauli_bits(&sx, &sz, &[(q, p)], words), nd)
                                })
                                .collect();
                            locations.push(FaultLocation {
                                op_index: i,
                                kind: FaultKind::Gate1q,
                                outcomes,
                            });
                        }
                        2 => {
                            let (a, b) = (qs[0], qs[1]);
                            let outcomes = (1..16)
                                .map(|k| {
                                    let ps = two_qubit_outcome(k, a, b);
                                    bits_to_signature(&pauli_bits(&sx, &sz, &ps, words), nd)
                                })
                                .collect();
                            locations.push(FaultLocation {
                                op_index: i,
                                kind: FaultKind::Gate2q,
                                outcomes,
                            });
                        }
                        _ => {} // run_noisy applies no noise to 3-qubit gates
                    }
                    backward_gate(&mut sx, &mut sz, g).map_err(|w| unsupported(i, w))?;
                }
                Op::Measure(q) => {
                    next_record -= 1;
                    let r = &rec[next_record];
                    locations.push(FaultLocation {
                        op_index: i,
                        kind: FaultKind::Readout,
                        outcomes: vec![bits_to_signature(r, nd)],
                    });
                    xor_into(&mut sx[*q], r);
                }
                Op::Reset(q) => {
                    locations.push(FaultLocation {
                        op_index: i,
                        kind: FaultKind::Reset,
                        outcomes: vec![bits_to_signature(&sx[*q], nd)],
                    });
                    sx[*q].fill(0);
                    sz[*q].fill(0);
                }
                Op::ClassicControlled { .. } => {
                    return Err(unsupported(i, "classically controlled gate"))
                }
                _ => return Err(unsupported(i, "explicit noise channel")),
            }
        }
        locations.reverse();
        Ok(Self {
            num_detectors: nd,
            locations,
        })
    }

    /// Merges all outcomes into independent mechanisms keyed by signature,
    /// combining probabilities with `p ⊕ q = p(1-q) + q(1-p)`. Trivial
    /// signatures are dropped.
    ///
    /// This treats the mutually exclusive outcomes of one location as
    /// independent, which is accurate to `O(p^2)` per location; it is used
    /// for inspection and decoding-graph construction, never for sampling.
    pub fn merged_mechanisms(&self, noise: &NoiseModel) -> Vec<ErrorMechanism> {
        let mut map: HashMap<Signature, f64> = HashMap::new();
        for loc in &self.locations {
            let p = loc.kind.probability(noise) / loc.outcomes.len() as f64;
            if p <= 0.0 {
                continue;
            }
            for s in &loc.outcomes {
                if s.is_trivial() {
                    continue;
                }
                let e = map.entry(s.clone()).or_insert(0.0);
                *e = *e * (1.0 - p) + p * (1.0 - *e);
            }
        }
        let mut v: Vec<ErrorMechanism> = map
            .into_iter()
            .map(|(s, probability)| ErrorMechanism {
                detectors: s.detectors,
                flips_logical: s.flips_logical,
                probability,
            })
            .collect();
        v.sort_by(|a, b| (&a.detectors, a.flips_logical).cmp(&(&b.detectors, b.flips_logical)));
        v
    }
}

/// Backward conjugation: turn "sensitivity of a Pauli after `g`" into
/// "sensitivity of a Pauli before `g`".
fn backward_gate(sx: &mut [Bits], sz: &mut [Bits], g: &Gate) -> Result<(), String> {
    match *g {
        Gate::H(q) => std::mem::swap(&mut sx[q], &mut sz[q]),
        Gate::X(_) | Gate::Y(_) | Gate::Z(_) => {}
        Gate::S(q) | Gate::Sdg(q) => {
            // X -> ±Y = X·Z, Z -> Z
            let z = sz[q].clone();
            xor_into(&mut sx[q], &z);
        }
        Gate::Cnot(c, t) => {
            // X_c -> X_c X_t ; Z_t -> Z_c Z_t
            let xt = sx[t].clone();
            xor_into(&mut sx[c], &xt);
            let zc = sz[c].clone();
            xor_into(&mut sz[t], &zc);
        }
        Gate::Cz(a, b) => {
            // X_a -> X_a Z_b ; X_b -> Z_a X_b
            let zb = sz[b].clone();
            let za = sz[a].clone();
            xor_into(&mut sx[a], &zb);
            xor_into(&mut sx[b], &za);
        }
        Gate::Swap(a, b) => {
            sx.swap(a, b);
            sz.swap(a, b);
        }
        ref other => return Err(format!("non-Clifford or unsupported gate {other:?}")),
    }
    Ok(())
}

/// Forward Pauli-frame propagation of one fault (reference implementation).
///
/// The fault is inserted at location `(op_index, kind)` exactly as
/// `run_noisy` would insert it: after a gate (`Gate1q`/`Gate2q`, with
/// `paulis` on the gate's qubits), as a classical flip of the record
/// produced by the measurement at `op_index` (`Readout`, `paulis` ignored),
/// or after the reset at `op_index` (`Reset`, `paulis` is normally
/// `[(q, X)]`). Returns the flipped detectors and observable.
pub fn propagate_forward(
    circuit: &Circuit,
    op_index: usize,
    kind: FaultKind,
    paulis: &[(usize, Pauli)],
    detectors: &[Vec<usize>],
    observable: &[usize],
) -> Result<Signature, DemError> {
    let nq = circuit.num_qubits;
    let mut fx = vec![false; nq];
    let mut fz = vec![false; nq];
    let mut flipped: Vec<bool> = Vec::new();
    let mut record = 0usize;
    for (i, op) in circuit.ops.iter().enumerate() {
        if i < op_index {
            if matches!(op, Op::Measure(_)) {
                flipped.push(false);
                record += 1;
            }
            continue;
        }
        if i == op_index {
            match (op, kind) {
                (Op::Measure(_), FaultKind::Readout) => {
                    flipped.push(true);
                    record += 1;
                    continue;
                }
                (Op::Gate(_), FaultKind::Gate1q | FaultKind::Gate2q)
                | (Op::Reset(_), FaultKind::Reset) => {
                    // The noise comes AFTER the op: the op itself acts on the
                    // empty frame (a no-op), then the Pauli is inserted.
                    for &(q, p) in paulis {
                        fx[q] ^= p.has_x();
                        fz[q] ^= p.has_z();
                    }
                    continue;
                }
                _ => return Err(unsupported(i, "fault kind does not match op")),
            }
        }
        match op {
            Op::Gate(g) => match *g {
                Gate::H(q) => std::mem::swap(&mut fx[q], &mut fz[q]),
                Gate::X(_) | Gate::Y(_) | Gate::Z(_) => {}
                Gate::S(q) | Gate::Sdg(q) => fz[q] ^= fx[q],
                Gate::Cnot(c, t) => {
                    fx[t] ^= fx[c];
                    fz[c] ^= fz[t];
                }
                Gate::Cz(a, b) => {
                    let (xa, xb) = (fx[a], fx[b]);
                    fz[b] ^= xa;
                    fz[a] ^= xb;
                }
                Gate::Swap(a, b) => {
                    fx.swap(a, b);
                    fz.swap(a, b);
                }
                ref other => return Err(unsupported(i, format!("{other:?}"))),
            },
            Op::Measure(q) => {
                flipped.push(fx[*q]);
                record += 1;
            }
            Op::Reset(q) => {
                fx[*q] = false;
                fz[*q] = false;
            }
            _ => return Err(unsupported(i, "classical control / explicit noise")),
        }
    }
    let _ = record;
    let parity = |recs: &[usize]| -> Result<bool, DemError> {
        let mut v = false;
        for &r in recs {
            v ^= *flipped.get(r).ok_or(DemError::BadRecord {
                record: r,
                available: flipped.len(),
            })?;
        }
        Ok(v)
    };
    let mut dets = Vec::new();
    for (i, recs) in detectors.iter().enumerate() {
        if parity(recs)? {
            dets.push(i);
        }
    }
    Ok(Signature {
        detectors: dets,
        flips_logical: parity(observable)?,
    })
}

/// Exact sampler of (detector events, observable flip) for a fixed noise
/// model; see the module docs.
#[derive(Clone, Debug)]
pub struct DemSampler {
    num_detectors: usize,
    /// One group per distinct firing probability: `(p, ln(1 - p), location indices)`.
    groups: Vec<(f64, f64, Vec<usize>)>,
    outcomes: Vec<Vec<Signature>>,
}

impl DemSampler {
    pub fn new(faults: &CircuitFaults, noise: &NoiseModel) -> Self {
        let mut by_kind: HashMap<FaultKind, Vec<usize>> = HashMap::new();
        for (i, loc) in faults.locations.iter().enumerate() {
            by_kind.entry(loc.kind).or_default().push(i);
        }
        let mut groups = Vec::new();
        for kind in [
            FaultKind::Gate1q,
            FaultKind::Gate2q,
            FaultKind::Readout,
            FaultKind::Reset,
        ] {
            let p = kind.probability(noise);
            if p <= 0.0 {
                continue;
            }
            if let Some(locs) = by_kind.remove(&kind) {
                groups.push((p.min(1.0), (1.0 - p.min(1.0)).ln(), locs));
            }
        }
        Self {
            num_detectors: faults.num_detectors,
            groups,
            outcomes: faults
                .locations
                .iter()
                .map(|l| l.outcomes.clone())
                .collect(),
        }
    }

    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    /// Samples one shot. `flags` is scratch space (resized as needed);
    /// detector events are written to `defects` (sorted ascending).
    /// Returns whether the observable flipped.
    pub fn sample_into<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        flags: &mut Vec<bool>,
        defects: &mut Vec<usize>,
    ) -> bool {
        flags.clear();
        flags.resize(self.num_detectors, false);
        let mut logical = false;
        for (p, ln_q, locs) in &self.groups {
            let n = locs.len();
            let mut fire = |idx: usize, rng: &mut R| {
                let outs = &self.outcomes[locs[idx]];
                let s = if outs.len() == 1 {
                    &outs[0]
                } else {
                    &outs[rng.random_range(0..outs.len())]
                };
                for &d in &s.detectors {
                    flags[d] ^= true;
                }
                logical ^= s.flips_logical;
            };
            if *p >= 1.0 {
                for i in 0..n {
                    fire(i, rng);
                }
                continue;
            }
            // Geometric skipping: number of non-firing trials before the
            // next firing one is floor(ln U / ln(1-p)), U ~ Uniform(0, 1].
            let mut i = 0usize;
            loop {
                let u: f64 = 1.0 - rng.random::<f64>();
                let skip = (u.ln() / ln_q).floor();
                if !(skip < (n - i) as f64) {
                    break;
                }
                i += skip as usize;
                fire(i, rng);
                i += 1;
                if i >= n {
                    break;
                }
            }
        }
        defects.clear();
        defects.extend(
            flags
                .iter()
                .enumerate()
                .filter_map(|(d, &f)| if f { Some(d) } else { None }),
        );
        logical
    }

    /// Convenience wrapper around [`DemSampler::sample_into`].
    pub fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> (Vec<usize>, bool) {
        let mut flags = Vec::new();
        let mut defects = Vec::new();
        let l = self.sample_into(rng, &mut flags, &mut defects);
        (defects, l)
    }
}

/// How a decoding graph was derived from a circuit's faults.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GraphReport {
    /// Distinct non-trivial signatures over all fault outcomes.
    pub distinct_signatures: usize,
    /// Signatures with 1 or 2 detectors (become edges).
    pub graphlike_signatures: usize,
    /// Signatures with 3+ detectors (hyperedges).
    pub hyperedge_signatures: usize,
    /// Hyperedges that split into existing edges with a consistent logical flag.
    pub hyperedges_decomposable: usize,
    /// Edges for which faults with BOTH logical flags exist; the flag of the
    /// more probable side (under the reference noise) is kept.
    pub logical_flag_conflicts: usize,
    /// Faults that flip the observable but no detector (must be 0 for a
    /// code of distance >= 2).
    pub undetectable_logical: usize,
    /// Edges in the resulting graph.
    pub edges: usize,
}

/// Builds a Union-Find decoding graph from the circuit's own faults.
///
/// Nodes are the detectors plus one boundary node (`num_detectors`). Every
/// fault outcome whose signature has exactly two detectors becomes an edge
/// between them; exactly one detector becomes an edge to the boundary node.
/// Duplicate edges are merged. The logical flag of an edge is the
/// observable flip of the faults that produce it; if faults with both flags
/// produce the same edge (this happens for boundary edges, where one
/// boundary node stands for both the logical and the non-logical side), the
/// flag carrying more probability under `reference` is kept and the
/// conflict is counted.
///
/// Signatures with three or more detectors (e.g. a `Y` or two-qubit fault
/// that is simultaneously a data error and a measurement error) do not
/// become edges: the Union-Find decoder only handles graphs. They are still
/// sampled exactly; the decoder sees them as their decomposition into
/// existing edges, which `GraphReport::hyperedges_decomposable` checks.
/// The Union-Find decoder ignores edge weights, so probabilities are used
/// only to resolve flag conflicts.
pub fn decoding_graph_from_faults(
    faults: &CircuitFaults,
    reference: &NoiseModel,
) -> (crate::qec::decoder::DecodingGraph, GraphReport) {
    use crate::qec::decoder::DecodingGraph;
    let nd = faults.num_detectors;
    let b = nd;
    let mut report = GraphReport::default();
    let mut seen: HashMap<Signature, ()> = HashMap::new();
    // (u, v) with u < v (v = b for boundary) -> [weight without flip, weight with flip]
    let mut edges: HashMap<(usize, usize), [f64; 2]> = HashMap::new();
    let mut hyper: Vec<Signature> = Vec::new();
    for loc in &faults.locations {
        let p = loc.kind.probability(reference) / loc.outcomes.len() as f64;
        for s in &loc.outcomes {
            if s.is_trivial() {
                continue;
            }
            if seen.insert(s.clone(), ()).is_none() {
                report.distinct_signatures += 1;
                match s.detectors.len() {
                    0 => report.undetectable_logical += 1,
                    1 | 2 => report.graphlike_signatures += 1,
                    _ => {
                        report.hyperedge_signatures += 1;
                        hyper.push(s.clone());
                    }
                }
            }
            let key = match s.detectors.as_slice() {
                [u] => (*u, b),
                [u, v] => (*u, *v),
                _ => continue,
            };
            edges.entry(key).or_insert([0.0; 2])[s.flips_logical as usize] += p.max(1e-300);
        }
    }
    let mut graph = DecodingGraph::new(nd + 1, b);
    let mut keys: Vec<_> = edges.keys().copied().collect();
    keys.sort_unstable();
    let mut flag_of: HashMap<(usize, usize), bool> = HashMap::new();
    for k in keys {
        let w = edges[&k];
        if w[0] > 0.0 && w[1] > 0.0 {
            report.logical_flag_conflicts += 1;
        }
        let flag = w[1] > w[0];
        flag_of.insert(k, flag);
        graph.add_edge(k.0, k.1, flag, 1);
    }
    report.edges = graph.edges.len();
    for s in &hyper {
        if decomposes(&s.detectors, s.flips_logical, b, &flag_of) {
            report.hyperedges_decomposable += 1;
        }
    }
    (graph, report)
}

/// Can `dets` be split into edges of `flag_of` (pairs, or singletons to the
/// boundary) whose logical flags XOR to `logical`?
fn decomposes(
    dets: &[usize],
    logical: bool,
    b: usize,
    flag_of: &HashMap<(usize, usize), bool>,
) -> bool {
    let Some((&first, rest)) = dets.split_first() else {
        return !logical;
    };
    if let Some(&f) = flag_of.get(&(first, b)) {
        if decomposes(rest, logical ^ f, b, flag_of) {
            return true;
        }
    }
    for (i, &v) in rest.iter().enumerate() {
        if let Some(&f) = flag_of.get(&(first, v)) {
            let mut r: Vec<usize> = rest.to_vec();
            r.remove(i);
            if decomposes(&r, logical ^ f, b, flag_of) {
                return true;
            }
        }
    }
    false
}
