//! Exhaustive search over CNOT orderings for rotated surface-code syndrome extraction.
//!
//! For a weight-4 plaquette there are 4! = 24 orderings of the four data-qubit
//! neighbours (NW, NE, SW, SE). This module searches all 24 × 24 = 576
//! combinations of Z-check and X-check orderings, filters by circuit distance
//! (must equal d), and measures logical error rate at several bias levels.
//!
//! # Bias model
//!
//! Under Z-biased noise the CNOT depolarizing channel no longer has equal X, Y,
//! and Z components. We model the bias by scaling the error probabilities while
//! keeping the total 2-qubit error rate p_2q fixed:
//!
//! * 8 XY-type two-qubit Paulis (XI, YI, IX, IY, XX, XY, YX, YY): each gets p_x
//! * 7 Z-type two-qubit Paulis (ZI, IZ, ZZ, XZ, ZX, YZ, ZY): each gets p_z = η·p_x
//! * 8·p_x + 7·η·p_x = p_2q  →  p_x = p_2q / (8 + 7η)
//!
//! For 1-qubit gates: X, Y → p_1q/(2+η); Z → η·p_x_1q.
//! Measurement and reset errors are X-type; they keep their nominal rate.

use crate::circuit::Circuit;
use crate::noise::NoiseModel;
use crate::qec::decoder::{DecodingGraph, UnionFindDecoder};
use crate::qec::dem::{
    decoding_graph_from_faults, weighted_decoding_graph_from_faults, CircuitFaults, DemSampler,
    FaultKind, Signature,
};
use crate::qec::surface::SurfaceCode;
use rand::{Rng, RngCore};

// ──────────────────────────────────────────────────────────
// Permutation and Schedule types
// ──────────────────────────────────────────────────────────

/// The 4 compass positions of a plaquette's data-qubit neighbours.
///
/// Indices: NW=0, NE=1, SW=2, SE=3 in the dual-grid (face-centred) layout.
/// A `Permutation` is the 4-tuple of compass indices in application order.
/// Boundary plaquettes (with fewer than 4 neighbours) use only the entries
/// that name actual neighbours, in the order they appear in the permutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Permutation(pub [usize; 4]);

impl Permutation {
    /// All 24 permutations of [0, 1, 2, 3], sorted lexicographically.
    pub fn all() -> Vec<Self> {
        let mut result = Vec::with_capacity(24);
        let mut a = [0usize, 1, 2, 3];
        heap_permutations(&mut a, 4, &mut result);
        result.sort_by_key(|p| p.0);
        result
    }

    /// The standard Z-check order: NW, NE, SW, SE (= [0,1,2,3]).
    pub fn standard_z() -> Self {
        Self([0, 1, 2, 3])
    }

    /// The standard X-check order: NW, SW, NE, SE (= [0,2,1,3]).
    /// Hook pair is NE–SE (vertical), perpendicular to the X logical.
    pub fn standard_x() -> Self {
        Self([0, 2, 1, 3])
    }
}

fn heap_permutations(a: &mut [usize; 4], k: usize, out: &mut Vec<Permutation>) {
    if k == 1 {
        out.push(Permutation(*a));
        return;
    }
    for i in 0..k {
        heap_permutations(a, k - 1, out);
        if k % 2 == 0 {
            a.swap(i, k - 1);
        } else {
            a.swap(0, k - 1);
        }
    }
}

/// A schedule for the rotated surface code: one permutation per check type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Schedule {
    /// CNOT application order for Z-checks (4-qubit plaquettes).
    pub z_perm: Permutation,
    /// CNOT application order for X-checks (4-qubit plaquettes).
    pub x_perm: Permutation,
    /// Whether CNOT layers are interleaved (depth-4) or sequential (depth-8).
    pub interleaved: bool,
}

impl Schedule {
    /// The standard hook-safe sequential schedule from qec.md §3.
    pub fn standard() -> Self {
        Self {
            z_perm: Permutation::standard_z(),
            x_perm: Permutation::standard_x(),
            interleaved: false,
        }
    }

    /// The standard hook-safe interleaved schedule (depth-4).
    pub fn standard_interleaved() -> Self {
        Self {
            z_perm: Permutation::standard_z(),
            x_perm: Permutation::standard_x(),
            interleaved: true,
        }
    }

    /// Check whether the schedule has no simultaneous qubit collisions
    /// when Z and X syndrome extraction are interleaved in 4 steps.
    ///
    /// Compass indices: NW=0, NE=1, SW=2, SE=3.
    /// Diagonal 1: {0, 3} (NW, SE). Diagonal 2: {1, 2} (NE, SW).
    /// To avoid two CNOTs touching the same data qubit at the same clock cycle,
    /// at every step t in 0..4, z_perm[t] and x_perm[t] must either both belong
    /// to Diagonal 1 or both belong to Diagonal 2.
    pub fn is_collision_free(&self) -> bool {
        for t in 0..4 {
            let z_in_diag1 = self.z_perm.0[t] == 0 || self.z_perm.0[t] == 3;
            let x_in_diag1 = self.x_perm.0[t] == 0 || self.x_perm.0[t] == 3;
            if z_in_diag1 != x_in_diag1 {
                return false;
            }
        }
        true
    }
}

// ──────────────────────────────────────────────────────────
// ScheduledSurfaceCode
// ──────────────────────────────────────────────────────────

/// Stabiliser face (plaquette) with its CNOT application order.
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct ScheduledFace {
    r: usize,
    c: usize,
    is_z: bool,
    data_qubits: Vec<usize>,
}

/// A rotated surface code memory experiment with a custom CNOT schedule.
#[derive(Clone, Debug)]
pub struct ScheduledSurfaceCode {
    pub d: usize,
    pub rounds: usize,
    pub schedule: Schedule,
    z_stabilizers: Vec<ScheduledFace>,
    x_stabilizers: Vec<ScheduledFace>,
    pub decoder: UnionFindDecoder,
    pub faults: CircuitFaults,
}

impl ScheduledSurfaceCode {
    /// Build a surface code with the given CNOT schedule (unweighted decoder).
    pub fn new(d: usize, rounds: usize, schedule: Schedule) -> Self {
        assert!(d >= 3 && d % 2 == 1);
        assert!(rounds >= 1);

        let (z_stabs, x_stabs) = Self::generate_stabilizers(d, &schedule);

        let mut sc = Self {
            d,
            rounds,
            schedule,
            z_stabilizers: z_stabs,
            x_stabilizers: x_stabs,
            decoder: UnionFindDecoder::new(DecodingGraph::new(1, 0)),
            faults: CircuitFaults {
                num_detectors: 0,
                locations: Vec::new(),
            },
        };

        let circ = sc.build_circuit();
        let det_recs = sc.detector_records();
        let obs_recs = sc.observable_records();
        sc.faults = CircuitFaults::from_circuit(&circ, &det_recs, &obs_recs)
            .expect("surface code circuit is Clifford");

        let noise_ref = NoiseModel::uniform(1e-3);
        let (graph, _) = decoding_graph_from_faults(&sc.faults, &noise_ref);
        sc.decoder = UnionFindDecoder::new(graph);
        sc
    }

    /// Same but with a weighted decoder tuned for `noise`.
    pub fn new_weighted(d: usize, rounds: usize, schedule: Schedule, noise: &NoiseModel) -> Self {
        let mut sc = Self::new(d, rounds, schedule);
        let (graph, _) = weighted_decoding_graph_from_faults(&sc.faults, noise, 100);
        sc.decoder = UnionFindDecoder::new(graph);
        sc
    }

    /// Generate stabiliser faces with the schedule's CNOT ordering applied.
    fn generate_stabilizers(
        d: usize,
        schedule: &Schedule,
    ) -> (Vec<ScheduledFace>, Vec<ScheduledFace>) {
        let mut z_stabs = Vec::new();
        let mut x_stabs = Vec::new();

        for r in 0..=d {
            for c in 0..=d {
                let is_z = (r + c) % 2 == 0;

                // Compass neighbours: NW=0, NE=1, SW=2, SE=3.
                let mut compass: [Option<usize>; 4] = [None; 4];
                if r > 0 && c > 0 {
                    compass[0] = Some(SurfaceCode::data_idx(d, r - 1, c - 1));
                }
                if r > 0 && c < d {
                    compass[1] = Some(SurfaceCode::data_idx(d, r - 1, c));
                }
                if r < d && c > 0 {
                    compass[2] = Some(SurfaceCode::data_idx(d, r, c - 1));
                }
                if r < d && c < d {
                    compass[3] = Some(SurfaceCode::data_idx(d, r, c));
                }

                let present_count = compass.iter().filter(|x| x.is_some()).count();
                if present_count < 2 {
                    continue;
                }

                // Boundary validity.
                if r == 0 && !is_z {
                    continue;
                }
                if r == d && !is_z {
                    continue;
                }
                if c == 0 && is_z {
                    continue;
                }
                if c == d && is_z {
                    continue;
                }

                let perm = if is_z {
                    &schedule.z_perm.0
                } else {
                    &schedule.x_perm.0
                };
                let data_qubits: Vec<usize> = perm.iter().filter_map(|&ci| compass[ci]).collect();

                let face = ScheduledFace {
                    r,
                    c,
                    is_z,
                    data_qubits,
                };
                if is_z {
                    z_stabs.push(face);
                } else {
                    x_stabs.push(face);
                }
            }
        }

        assert_eq!(z_stabs.len(), (d * d - 1) / 2);
        assert_eq!(x_stabs.len(), (d * d - 1) / 2);
        (z_stabs, x_stabs)
    }

    /// Build the full syndrome extraction circuit.
    pub fn build_circuit(&self) -> Circuit {
        let d = self.d;
        let rounds = self.rounds;
        let num_z = self.z_stabilizers.len();
        let num_x = self.x_stabilizers.len();
        let total_q = SurfaceCode::total_qubits(d);
        let mut circ = Circuit::new(total_q);

        for r in 0..rounds {
            if r > 0 {
                for k in 0..num_z {
                    circ.reset(SurfaceCode::z_ancilla_idx(d, k));
                }
                for k in 0..num_x {
                    circ.reset(SurfaceCode::x_ancilla_idx(d, k));
                }
            }
            for k in 0..num_x {
                circ.h(SurfaceCode::x_ancilla_idx(d, k));
            }
            if self.schedule.interleaved {
                for step in 0..4 {
                    for (k, stab) in self.z_stabilizers.iter().enumerate() {
                        let a = SurfaceCode::z_ancilla_idx(d, k);
                        if step < stab.data_qubits.len() {
                            circ.cnot(stab.data_qubits[step], a);
                        }
                    }
                    for (k, stab) in self.x_stabilizers.iter().enumerate() {
                        let a = SurfaceCode::x_ancilla_idx(d, k);
                        if step < stab.data_qubits.len() {
                            circ.cnot(a, stab.data_qubits[step]);
                        }
                    }
                }
            } else {
                for (k, stab) in self.z_stabilizers.iter().enumerate() {
                    let a = SurfaceCode::z_ancilla_idx(d, k);
                    for &dq in &stab.data_qubits {
                        circ.cnot(dq, a);
                    }
                }
                for (k, stab) in self.x_stabilizers.iter().enumerate() {
                    let a = SurfaceCode::x_ancilla_idx(d, k);
                    for &dq in &stab.data_qubits {
                        circ.cnot(a, dq);
                    }
                }
            }
            for k in 0..num_x {
                circ.h(SurfaceCode::x_ancilla_idx(d, k));
            }
            for k in 0..num_z {
                circ.measure(SurfaceCode::z_ancilla_idx(d, k));
            }
            for k in 0..num_x {
                circ.measure(SurfaceCode::x_ancilla_idx(d, k));
            }
        }
        for dq in 0..d * d {
            circ.measure(dq);
        }
        circ
    }

    fn records_per_round(&self) -> usize {
        self.z_stabilizers.len() + self.x_stabilizers.len()
    }

    pub fn detector_records(&self) -> Vec<Vec<usize>> {
        let num_z = self.z_stabilizers.len();
        let apr = self.records_per_round();
        let data_base = self.rounds * apr;
        let mut dets = Vec::new();
        for r in 0..self.rounds {
            for k in 0..num_z {
                let mut v = vec![r * apr + k];
                if r > 0 {
                    v.push((r - 1) * apr + k);
                }
                dets.push(v);
            }
        }
        for (k, stab) in self.z_stabilizers.iter().enumerate() {
            let mut v: Vec<usize> = stab.data_qubits.iter().map(|&q| data_base + q).collect();
            v.push((self.rounds - 1) * apr + k);
            dets.push(v);
        }
        dets
    }

    pub fn observable_records(&self) -> Vec<usize> {
        let data_base = self.rounds * self.records_per_round();
        (0..self.d)
            .map(|r| data_base + SurfaceCode::data_idx(self.d, r, 0))
            .collect()
    }

    /// Graph-like circuit distance (BFS on the decoding graph).
    pub fn circuit_distance(&self) -> Option<usize> {
        self.decoder.graph.min_logical_weight()
    }

    /// Graph-like circuit distance and multiplicity of minimum-weight logical paths.
    pub fn circuit_distance_and_mechanisms(&self) -> Option<(usize, usize)> {
        self.decoder.graph.min_logical_weight_and_count()
    }

    /// Logical error rate under uniform depolarizing noise using the DEM sampler.
    pub fn logical_error_rate<R: RngCore>(
        &self,
        noise: &NoiseModel,
        shots: usize,
        rng: &mut R,
    ) -> f64 {
        let sampler = DemSampler::new(&self.faults, noise);
        let mut flags = Vec::new();
        let mut defects = Vec::new();
        let mut errors = 0usize;
        for _ in 0..shots {
            let raw = sampler.sample_into(rng, &mut flags, &mut defects);
            if raw ^ self.decoder.decode(&defects) {
                errors += 1;
            }
        }
        errors as f64 / shots.max(1) as f64
    }

    /// Logical error rate under Z-biased noise (bias η = p_Z/p_X).
    pub fn logical_error_rate_biased<R: RngCore>(
        &self,
        noise: &NoiseModel,
        eta: f64,
        shots: usize,
        rng: &mut R,
    ) -> f64 {
        let sampler = BiasedDemSamplerV2::new(&self.faults, noise, eta);
        let mut flags = Vec::new();
        let mut defects = Vec::new();
        let mut errors = 0usize;
        for _ in 0..shots {
            let raw = sampler.sample_into(rng, &mut flags, &mut defects);
            if raw ^ self.decoder.decode(&defects) {
                errors += 1;
            }
        }
        errors as f64 / shots.max(1) as f64
    }
}

// ──────────────────────────────────────────────────────────
// Biased DEM sampler
// ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct BiasedEntry {
    /// Total probability this location fires.
    p_total: f64,
    /// Cumulative probability CDF for outcome selection (conditional on firing).
    cum_weights: Vec<f64>,
    /// The Signature for each outcome (parallel to cum_weights).
    sigs: Vec<Signature>,
}

/// DEM sampler with Z-biased Pauli noise (simple per-location loop).
///
/// No geometric skipping: O(n_locs) per shot. For d=3 there are ~145 fault
/// locations, so ~145 uniform draws per shot. At 200k shots this is ~29M
/// draws — still very fast compared to tableau simulation.
#[derive(Clone, Debug)]
pub struct BiasedDemSamplerV2 {
    num_detectors: usize,
    entries: Vec<BiasedEntry>,
}

impl BiasedDemSamplerV2 {
    /// Bias η = p_Z / p_X. η = 1 recovers uniform depolarizing.
    pub fn new(faults: &CircuitFaults, noise: &NoiseModel, eta: f64) -> Self {
        let mut entries = Vec::with_capacity(faults.locations.len());

        for loc in &faults.locations {
            // Per-outcome probabilities (before normalisation).
            let probs: Vec<f64> = match loc.kind {
                FaultKind::Gate2q => {
                    // 15 two-qubit Paulis. Outcome k in 1..=15 matches two_qubit_outcome(k, a, b):
                    //   a_idx = k / 4   (0=I, 1=X, 2=Y, 3=Z on qubit a)
                    //   b_idx = k % 4   (0=I, 1=X, 2=Y, 3=Z on qubit b)
                    // Z-containing: a_idx==3 or b_idx==3 → weight η.
                    // Exactly 7 outcomes contain Z, 8 contain only X/Y.
                    let p_x = noise.p_2q / (8.0 + 7.0 * eta);
                    let p_z = eta * p_x;
                    (1usize..=15)
                        .map(|k| {
                            let a_idx = k / 4;
                            let b_idx = k % 4;
                            if a_idx == 3 || b_idx == 3 {
                                p_z
                            } else {
                                p_x
                            }
                        })
                        .collect()
                }
                FaultKind::Gate1q => {
                    // 3 outcomes: X, Y, Z.
                    let p_x1 = noise.p_1q / (2.0 + eta);
                    let p_z1 = eta * p_x1;
                    vec![p_x1, p_x1, p_z1]
                }
                FaultKind::Readout => vec![noise.p_meas],
                FaultKind::Reset => vec![noise.p_reset],
            };

            let p_total: f64 = probs.iter().sum::<f64>().min(1.0);
            if p_total <= 0.0 {
                continue;
            }

            // Normalise to build CDF for conditional outcome selection.
            let raw_sum: f64 = probs.iter().sum();
            let mut cum = Vec::with_capacity(probs.len());
            let mut acc = 0.0f64;
            for &p in &probs {
                acc += p / raw_sum;
                cum.push(acc.min(1.0));
            }

            // Signatures from the fault location; must align 1:1 with probs.
            let sigs: Vec<Signature> = loc.outcomes.clone();

            if sigs.len() != probs.len() {
                // Misalignment: fall back to equal-weight selection.
                let n = sigs.len();
                let cum_uniform: Vec<f64> = (1..=n).map(|i| i as f64 / n as f64).collect();
                entries.push(BiasedEntry {
                    p_total: p_total.min(1.0),
                    cum_weights: cum_uniform,
                    sigs,
                });
                continue;
            }

            entries.push(BiasedEntry {
                p_total,
                cum_weights: cum,
                sigs,
            });
        }

        Self {
            num_detectors: faults.num_detectors,
            entries,
        }
    }

    /// Sample one shot. Returns whether the observable flipped.
    pub fn sample_into<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        flags: &mut Vec<bool>,
        defects: &mut Vec<usize>,
    ) -> bool {
        flags.clear();
        flags.resize(self.num_detectors, false);
        let mut logical = false;

        for entry in &self.entries {
            let u: f64 = rng.random();
            if u >= entry.p_total {
                continue;
            }
            let v: f64 = rng.random();
            let idx = entry
                .cum_weights
                .iter()
                .position(|&c| v < c)
                .unwrap_or(entry.sigs.len() - 1);
            if idx < entry.sigs.len() {
                let sig = &entry.sigs[idx];
                for &d in &sig.detectors {
                    if d < self.num_detectors {
                        flags[d] ^= true;
                    }
                }
                logical ^= sig.flips_logical;
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
}

// ──────────────────────────────────────────────────────────
// Exhaustive search
// ──────────────────────────────────────────────────────────

/// Result of testing one schedule.
#[derive(Clone, Debug)]
pub struct ScheduleResult {
    pub schedule: Schedule,
    /// Graph-like circuit distance.
    pub circuit_distance: usize,
    /// Number of minimum-weight logical failure mechanisms in the decoding graph.
    pub min_weight_mechanisms: usize,
    /// Logical error rate under unbiased noise (η = 1).
    pub p_l_eta1: f64,
    /// Logical error rate under Z-biased noise (η = 10).
    pub p_l_eta10: f64,
    /// Logical error rate under Z-biased noise (η = 100).
    pub p_l_eta100: f64,
    pub shots_per_condition: usize,
    pub p_phys: f64,
}

/// Wilson score interval (95% confidence): returns `(center, lower, upper)`.
pub fn wilson_score_interval(successes: usize, total: usize) -> (f64, f64, f64) {
    if total == 0 {
        return (0.0, 0.0, 0.0);
    }
    let z = 1.959964f64; // 95% confidence
    let z2 = z * z;
    let n = total as f64;
    let p_hat = successes as f64 / n;
    let denom = 1.0 + z2 / n;
    let center = (p_hat + z2 / (2.0 * n)) / denom;
    let half_width = (z / denom)
        * ((p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
            .max(0.0)
            .sqrt());
    let lower = (center - half_width).max(0.0);
    let upper = (center + half_width).min(1.0);
    (center, lower, upper)
}

/// Filter all 576 schedule pairs, keeping only those whose circuit distance equals `d`.
/// Returns `(schedule, distance, min_weight_mechanisms)`.
pub fn filter_schedules_by_distance(
    d: usize,
    rounds: usize,
    interleaved: bool,
) -> Vec<(Schedule, usize, usize)> {
    let all_perms = Permutation::all();
    let mut survivors = Vec::new();

    for z_perm in &all_perms {
        for x_perm in &all_perms {
            let schedule = Schedule {
                z_perm: *z_perm,
                x_perm: *x_perm,
                interleaved,
            };
            let sc = ScheduledSurfaceCode::new(d, rounds, schedule.clone());
            if let Some((dist, count)) = sc.circuit_distance_and_mechanisms() {
                if dist == d {
                    survivors.push((schedule, dist, count));
                }
            }
        }
    }
    survivors
}

/// Search all schedule combinations for code distance `d`.
///
/// Filters by circuit distance (must equal `d`), then scores survivors by
/// logical error rate at η = 1, 10, 100 using `shots_per_condition` shots each.
pub fn exhaustive_search<R: RngCore>(
    d: usize,
    rounds: usize,
    p_phys: f64,
    shots_per_condition: usize,
    interleaved: bool,
    rng: &mut R,
) -> Vec<ScheduleResult> {
    let survivors = filter_schedules_by_distance(d, rounds, interleaved);
    let noise = NoiseModel::circuit_level(p_phys, p_phys);
    let mut results = Vec::with_capacity(survivors.len());

    for (schedule, dist, count) in survivors {
        let sc = ScheduledSurfaceCode::new(d, rounds, schedule.clone());
        let p_l_eta1 = score_biased(&sc, &noise, 1.0, shots_per_condition, rng);
        let p_l_eta10 = score_biased(&sc, &noise, 10.0, shots_per_condition, rng);
        let p_l_eta100 = score_biased(&sc, &noise, 100.0, shots_per_condition, rng);

        results.push(ScheduleResult {
            schedule,
            circuit_distance: dist,
            min_weight_mechanisms: count,
            p_l_eta1,
            p_l_eta10,
            p_l_eta100,
            shots_per_condition,
            p_phys,
        });
    }

    results
}

fn score_biased<R: RngCore>(
    sc: &ScheduledSurfaceCode,
    noise: &NoiseModel,
    eta: f64,
    shots: usize,
    rng: &mut R,
) -> f64 {
    let sampler = BiasedDemSamplerV2::new(&sc.faults, noise, eta);
    let mut flags = Vec::new();
    let mut defects = Vec::new();
    let mut errors = 0usize;
    for _ in 0..shots {
        let raw = sampler.sample_into(rng, &mut flags, &mut defects);
        if raw ^ sc.decoder.decode(&defects) {
            errors += 1;
        }
    }
    errors as f64 / shots.max(1) as f64
}

// ──────────────────────────────────────────────────────────
// Fault-injection distance check
// ──────────────────────────────────────────────────────────

/// Verify that every single-fault signature is correctable by the decoder.
///
/// Returns `true` if all single-fault outcomes have 0 or 1–2 detector triggers
/// and the decoder correctly handles each, and no outcome flips the logical
/// without triggering any detector (undetectable logical).
pub fn fault_injection_distance_ok(sc: &ScheduledSurfaceCode) -> bool {
    for loc in &sc.faults.locations {
        for sig in &loc.outcomes {
            // Undetectable logical: definite failure.
            if sig.detectors.is_empty() && sig.flips_logical {
                return false;
            }
            // Single fault with 1 or 2 detectors: check decoder.
            if sig.detectors.len() <= 2 {
                let correction = sc.decoder.decode(&sig.detectors);
                if correction ^ sig.flips_logical {
                    return false;
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn permutation_all_24_distinct() {
        let all = Permutation::all();
        assert_eq!(all.len(), 24);
        let mut set = std::collections::HashSet::new();
        for p in &all {
            assert!(set.insert(*p), "duplicate permutation: {:?}", p);
            let mut sorted = p.0;
            sorted.sort();
            assert_eq!(sorted, [0, 1, 2, 3]);
        }
    }

    #[test]
    fn collision_free_schedule_count() {
        let all = Permutation::all();
        let mut count = 0;
        for z in &all {
            for x in &all {
                let s = Schedule {
                    z_perm: *z,
                    x_perm: *x,
                    interleaved: true,
                };
                if s.is_collision_free() {
                    count += 1;
                }
            }
        }
        // Each of the 24 z permutations has exactly 2! * 2! = 4 compatible x permutations = 96
        assert_eq!(count, 96);
        assert!(Schedule::standard().is_collision_free());
        assert!(Schedule::standard_interleaved().is_collision_free());
    }

    #[test]
    fn standard_schedule_d3_has_distance_3() {
        let sc = ScheduledSurfaceCode::new(3, 3, Schedule::standard());
        let (dist, count) = sc.circuit_distance_and_mechanisms().unwrap();
        assert_eq!(dist, 3);
        assert!(count > 0);
        assert!(fault_injection_distance_ok(&sc));
    }

    #[test]
    fn unswapped_x_schedule_fails_fault_injection() {
        // [0,1,2,3] for both Z and X was the bug found in qec.md §3 where
        // hook errors reduced distance because the hook pair is horizontal.
        let bad_schedule = Schedule {
            z_perm: Permutation([0, 1, 2, 3]),
            x_perm: Permutation([0, 1, 2, 3]),
            interleaved: false,
        };
        let sc = ScheduledSurfaceCode::new(3, 3, bad_schedule);
        assert!(!fault_injection_distance_ok(&sc));
    }

    #[test]
    fn filter_schedules_finds_survivors() {
        let survivors = filter_schedules_by_distance(3, 3, false);
        assert!(!survivors.is_empty());
        // Verify standard schedule is among the survivors
        assert!(survivors.iter().any(|(s, d, _)| {
            *d == 3
                && s.z_perm == Permutation::standard_z()
                && s.x_perm == Permutation::standard_x()
        }));
    }

    #[test]
    fn biased_dem_sampler_eta1_matches_order_of_magnitude() {
        let mut rng = StdRng::seed_from_u64(42);
        let sc = ScheduledSurfaceCode::new(3, 3, Schedule::standard());
        let noise = NoiseModel::circuit_level(0.005, 0.005);
        let p_dem = sc.logical_error_rate(&noise, 2000, &mut rng);
        let p_biased_1 = sc.logical_error_rate_biased(&noise, 1.0, 2000, &mut rng);
        assert!(p_dem > 0.0 && p_biased_1 > 0.0);
        assert!((p_dem - p_biased_1).abs() < 0.03);
    }
}
