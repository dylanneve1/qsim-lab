//! Rotated surface code memory experiment, layout, and threshold benchmarks.
//!
//! Implements the standard rotated planar surface code (Horsman et al. 2012, Fowler et al.).
//! For code distance `d`, there are `d^2` data qubits and `d^2 - 1` syndrome ancillas
//! (`(d^2 - 1) / 2` Z-type and `(d^2 - 1) / 2` X-type), for a total of `2d^2 - 1` physical qubits.
//!
//! A logical qubit `|0_L>` is stored and protected across `rounds = d` syndrome extraction
//! cycles with circuit-level depolarizing noise and readout errors. Syndromes are extracted
//! on the stabilizer tableau and decoded with the [`UnionFindDecoder`].
//!
//! # Detector Error Model
//!
//! The DEM is derived from the actual circuit by Pauli frame propagation. For every
//! noise location in `build_circuit()` and every Pauli the `NoiseModel` can insert there,
//! we propagate that Pauli through the remaining Clifford operations (H swaps X↔Z,
//! CNOT copies X forward and Z backward, measurement records the X-component,
//! reset clears the frame). We record exactly which detectors fire and whether
//! the logical observable flips. Identical detector signatures are merged by
//! probability XOR: `p ⊕ q = p(1-q) + q(1-p)`.
//!
//! For Z-memory with uniform depolarizing noise, only X and Y errors on data qubits
//! affect the Z-type detectors (Z errors commute with the Z-stabilizer measurements).
//! Y = XZ produces both an X-component (hitting Z-detectors) and a Z-component.
//! Our DEM tracks the X-component's effect on Z-detectors and the logical-Z observable.
//! The Z-component is invisible to Z-type detectors and does not flip Z_L, so for
//! pure Z-memory decoding on the Z-graph, the Y error's effect is identical to an
//! X error for detector/logical purposes. We document this explicitly.

use crate::circuit::{Circuit, Op};
use crate::gate::Gate;
use crate::noise::NoiseModel;
use crate::qec::decoder::{DecodingGraph, UnionFindDecoder};
use crate::qec::repetition::MemoryExperimentResult;
use crate::stabilizer::Tableau;
use rand::{Rng, RngCore};
use std::collections::HashMap;

/// A stabilizer face on the dual grid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StabilizerFace {
    pub r: usize,
    pub c: usize,
    pub is_z: bool,
    /// Indices of neighboring data qubits in `0..d^2`.
    pub data_qubits: Vec<usize>,
}

/// An error mechanism in the detector error model.
#[derive(Clone, Debug)]
pub struct ErrorMechanism {
    /// Detectors flipped by this error (sorted).
    pub detectors: Vec<usize>,
    /// Whether this error flips the logical Z observable.
    pub flips_logical: bool,
    /// Probability of this mechanism firing.
    pub probability: f64,
}

/// Pauli frame: one X-bit and one Z-bit per qubit.
/// Y is represented as both X and Z set.
#[derive(Clone)]
struct PauliFrame {
    x: Vec<bool>,
    z: Vec<bool>,
}

impl PauliFrame {
    fn new(n: usize) -> Self {
        Self {
            x: vec![false; n],
            z: vec![false; n],
        }
    }

    /// Apply Hadamard: swaps X ↔ Z on qubit q.
    fn h(&mut self, q: usize) {
        std::mem::swap(&mut self.x[q], &mut self.z[q]);
    }

    /// Apply CNOT(control, target):
    /// X on control propagates to target (XOR).
    /// Z on target propagates to control (XOR).
    fn cnot(&mut self, c: usize, t: usize) {
        let xc = self.x[c];
        self.x[t] ^= xc;
        let zt = self.z[t];
        self.z[c] ^= zt;
    }

    /// Measure qubit q in Z basis: returns whether the X-component flips
    /// the measurement result. Does NOT modify the frame (measurement doesn't
    /// change a Pauli frame since it's a classical operation on outcomes).
    fn measure_x_flip(&self, q: usize) -> bool {
        self.x[q]
    }

    /// Reset qubit q: clears both X and Z components (the qubit is reinitialized).
    fn reset(&mut self, q: usize) {
        self.x[q] = false;
        self.z[q] = false;
    }

    fn apply_x(&mut self, q: usize) {
        self.x[q] = true;
    }

    fn apply_z(&mut self, q: usize) {
        self.z[q] = true;
    }

    fn apply_y(&mut self, q: usize) {
        self.x[q] = true;
        self.z[q] = true;
    }
}

/// A rotated surface code memory experiment.
#[derive(Clone, Debug)]
pub struct SurfaceCode {
    pub d: usize,
    pub rounds: usize,
    pub z_stabilizers: Vec<StabilizerFace>,
    pub x_stabilizers: Vec<StabilizerFace>,
    pub decoder: UnionFindDecoder,
    pub error_mechanisms: Vec<ErrorMechanism>,
}

impl SurfaceCode {
    /// Creates a rotated surface code for distance `d` (must be odd >= 3) and `rounds`.
    pub fn new(d: usize, rounds: usize) -> Self {
        Self::with_noise(d, rounds, &NoiseModel::uniform(1.0))
    }

    /// Creates a rotated surface code with a specific noise model for DEM construction.
    pub fn with_noise(d: usize, rounds: usize, noise: &NoiseModel) -> Self {
        assert!(d >= 3 && d % 2 == 1, "distance must be an odd integer >= 3");
        assert!(rounds >= 1, "rounds must be >= 1");

        let (z_stabilizers, x_stabilizers) = Self::generate_stabilizers(d);
        let graph = Self::build_z_decoding_graph(d, rounds, &z_stabilizers);
        let decoder = UnionFindDecoder::new(graph);

        // Build the circuit to derive error mechanisms
        let tmp = Self {
            d,
            rounds,
            z_stabilizers: z_stabilizers.clone(),
            x_stabilizers: x_stabilizers.clone(),
            decoder: decoder.clone(),
            error_mechanisms: Vec::new(),
        };
        let error_mechanisms = tmp.build_circuit_derived_dem(noise);

        Self {
            d,
            rounds,
            z_stabilizers,
            x_stabilizers,
            decoder,
            error_mechanisms,
        }
    }

    /// Number of data qubits: `d^2`.
    #[inline]
    pub fn num_data_qubits(d: usize) -> usize {
        d * d
    }

    /// Number of syndrome ancillas: `d^2 - 1`.
    #[inline]
    pub fn num_ancillas(d: usize) -> usize {
        d * d - 1
    }

    /// Total physical qubits: `2d^2 - 1`.
    #[inline]
    pub fn total_qubits(d: usize) -> usize {
        2 * d * d - 1
    }

    /// Index of data qubit at row `r` and column `c`.
    #[inline]
    pub fn data_idx(d: usize, r: usize, c: usize) -> usize {
        r * d + c
    }

    /// Coordinates `(r, c)` of data qubit `idx`.
    #[inline]
    pub fn data_coords(d: usize, idx: usize) -> (usize, usize) {
        (idx / d, idx % d)
    }

    /// Ancilla qubit index for Z-stabilizer `k`: `d^2 + k`.
    #[inline]
    pub fn z_ancilla_idx(d: usize, k: usize) -> usize {
        d * d + k
    }

    /// Ancilla qubit index for X-stabilizer `k`: `d^2 + num_z + k`.
    #[inline]
    pub fn x_ancilla_idx(d: usize, k: usize) -> usize {
        d * d + (d * d - 1) / 2 + k
    }

    /// Generates all Z and X stabilizer faces for distance `d`.
    pub fn generate_stabilizers(d: usize) -> (Vec<StabilizerFace>, Vec<StabilizerFace>) {
        let mut z_stabs = Vec::new();
        let mut x_stabs = Vec::new();

        for r in 0..=d {
            for c in 0..=d {
                let is_z = (r + c) % 2 == 0;

                // Collect data qubit neighbors around face (r, c)
                let mut data = Vec::with_capacity(4);
                if r > 0 && c > 0 {
                    data.push(Self::data_idx(d, r - 1, c - 1)); // NW
                }
                if r > 0 && c < d {
                    data.push(Self::data_idx(d, r - 1, c)); // NE
                }
                if r < d && c > 0 {
                    data.push(Self::data_idx(d, r, c - 1)); // SW
                }
                if r < d && c < d {
                    data.push(Self::data_idx(d, r, c)); // SE
                }

                if data.len() < 2 {
                    continue;
                }

                // Check boundary validity
                if r == 0 && !is_z {
                    continue; // top is Z boundary
                }
                if r == d && !is_z {
                    continue; // bottom is Z boundary
                }
                if c == 0 && is_z {
                    continue; // left is X boundary
                }
                if c == d && is_z {
                    continue; // right is X boundary
                }

                let face = StabilizerFace {
                    r,
                    c,
                    is_z,
                    data_qubits: data,
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

    /// Builds the 3D spacetime decoding graph for Z-stabilizers.
    pub fn build_z_decoding_graph(
        d: usize,
        rounds: usize,
        z_stabs: &[StabilizerFace],
    ) -> DecodingGraph {
        let num_z = z_stabs.len();
        let num_detectors = (rounds + 1) * num_z;
        let boundary_node = num_detectors;
        let mut graph = DecodingGraph::new(num_detectors + 1, boundary_node);

        // Precompute which Z-stabilizers touch each data qubit
        let mut data_to_z = vec![Vec::new(); d * d];
        for (k, z) in z_stabs.iter().enumerate() {
            for &dq in &z.data_qubits {
                data_to_z[dq].push(k);
            }
        }

        for r in 0..=rounds {
            // Space-like edges corresponding to data qubit errors
            for (dq, touching) in data_to_z.iter().enumerate().take(d * d) {
                let (_, c) = Self::data_coords(d, dq);
                let is_logical = c == 0; // data qubits on column 0 form logical Z_L

                match touching.len() {
                    1 => {
                        // Data qubit on boundary touches 1 Z-stabilizer
                        let u = r * num_z + touching[0];
                        graph.add_edge(u, boundary_node, is_logical, 1);
                    }
                    2 => {
                        // Internal data qubit connects two neighboring Z-stabilizers
                        let u = r * num_z + touching[0];
                        let v = r * num_z + touching[1];
                        graph.add_edge(u, v, is_logical, 1);
                    }
                    _ => {}
                }
            }

            // Time-like edges corresponding to ancilla measurement errors
            if r < rounds {
                for k in 0..num_z {
                    let u = r * num_z + k;
                    let v = (r + 1) * num_z + k;
                    graph.add_edge(u, v, false, 1);
                }
            }
        }

        graph
    }

    /// Builds the full quantum circuit for syndrome extraction.
    pub fn build_circuit(&self) -> Circuit {
        let d = self.d;
        let rounds = self.rounds;
        let num_z = self.z_stabilizers.len();
        let num_x = self.x_stabilizers.len();
        let total_q = Self::total_qubits(d);

        let mut c = Circuit::new(total_q);

        for r in 0..rounds {
            // Reset all ancillas
            if r > 0 {
                for k in 0..num_z {
                    c.reset(Self::z_ancilla_idx(d, k));
                }
                for k in 0..num_x {
                    c.reset(Self::x_ancilla_idx(d, k));
                }
            }

            // H on X-ancillas
            for k in 0..num_x {
                c.h(Self::x_ancilla_idx(d, k));
            }

            // CNOT interactions with data qubits
            // Z-checks: CNOT(data, ancilla)
            for (k, stab) in self.z_stabilizers.iter().enumerate() {
                let a = Self::z_ancilla_idx(d, k);
                for &dq in &stab.data_qubits {
                    c.cnot(dq, a);
                }
            }
            // X-checks: CNOT(ancilla, data)
            for (k, stab) in self.x_stabilizers.iter().enumerate() {
                let a = Self::x_ancilla_idx(d, k);
                for &dq in &stab.data_qubits {
                    c.cnot(a, dq);
                }
            }

            // H on X-ancillas before measurement
            for k in 0..num_x {
                c.h(Self::x_ancilla_idx(d, k));
            }

            // Measure all Z-ancillas, then X-ancillas
            for k in 0..num_z {
                c.measure(Self::z_ancilla_idx(d, k));
            }
            for k in 0..num_x {
                c.measure(Self::x_ancilla_idx(d, k));
            }
        }

        // Final round: measure all data qubits in Z basis
        for dq in 0..d * d {
            c.measure(dq);
        }

        c
    }

    /// Builds a circuit-derived detector error model via Pauli frame propagation.
    ///
    /// For every noise location in the circuit (1q gate, 2q gate, measurement,
    /// reset) and every Pauli error that noise model can insert, we:
    /// 1. Initialize a Pauli frame with that error.
    /// 2. Propagate it through the remaining Clifford circuit.
    /// 3. Record which Z-detectors fire and whether logical-Z flips.
    /// 4. Merge mechanisms with identical signatures by probability XOR.
    fn build_circuit_derived_dem(&self, noise: &NoiseModel) -> Vec<ErrorMechanism> {
        let d = self.d;
        let num_z = self.z_stabilizers.len();
        let num_x = self.x_stabilizers.len();
        let total_q = Self::total_qubits(d);
        let circuit = self.build_circuit();

        // For Z-memory we care about Z-type detectors.
        // Detector indexing: round r, Z-stabilizer k → detector = r * num_z + k
        // Total detectors: (rounds + 1) * num_z
        //   rounds 0..rounds-1 from syndrome measurements
        //   round `rounds` from final data qubit measurements

        // Build the map from measurement index (in circuit output) to detector indices.
        // Measurement layout per round: num_z Z-ancilla measurements, then num_x X-ancilla measurements.
        // Final: d^2 data qubit measurements.
        let _num_detectors = (self.rounds + 1) * num_z;

        // We need to know the Z-stabilizer → measurement index mapping.
        // Within each round r: Z-ancilla k measured at circuit output index r*(num_z+num_x) + k.
        let anc_per_round = num_z + num_x;

        // Precompute detector information:
        // For Z-type detector at round r, stabilizer k:
        //   if r == 0: fires when measurement r,k is 1 (compared to ideal 0)
        //   if 0 < r < rounds: fires when measurement r,k differs from r-1,k
        //   if r == rounds: fires when the Z-parity of data qubits for stab k
        //                   from final measurements differs from measurement rounds-1,k

        // For a given Pauli frame error at circuit position `op_idx`, we propagate
        // through ops[op_idx+1..] and record which measurement outcomes are flipped.
        // Then we compute which detectors fire.

        // Key: (sorted detector indices, flips_logical)
        // Value: merged probability
        let mut mechanism_map: HashMap<(Vec<usize>, bool), f64> = HashMap::new();

        let ops = &circuit.ops;
        let num_ops = ops.len();

        // Iterate over every op position and inject possible errors
        for op_idx in 0..num_ops {
            let op = &ops[op_idx];
            // Determine what errors this location can produce and their probabilities
            let errors = self.errors_at_location(op, noise);

            for (pauli_init, prob) in errors {
                if prob <= 0.0 {
                    continue;
                }

                // Initialize frame with this error
                let mut frame = PauliFrame::new(total_q);
                for &(q, p) in &pauli_init {
                    match p {
                        PauliType::X => frame.apply_x(q),
                        PauliType::Y => frame.apply_y(q),
                        PauliType::Z => frame.apply_z(q),
                    }
                }

                // Propagate through remaining circuit ops
                let mut meas_flips = Vec::new(); // which measurement indices are flipped
                let mut meas_counter = 0usize;

                // Count measurements before this op
                for prev_op in &ops[..op_idx] {
                    if matches!(prev_op, Op::Measure(_)) {
                        meas_counter += 1;
                    }
                }

                // Also need to handle: if this op is a Measure, the error happened
                // DURING the measurement (readout flip), so the frame doesn't change
                // but the measurement outcome is flipped.
                // If this op is a Reset, the error happened AFTER reset.

                for remaining_op in &ops[op_idx..] {
                    match remaining_op {
                        Op::Gate(g) => {
                            self.propagate_gate(&mut frame, g);
                        }
                        Op::Measure(q) => {
                            if frame.measure_x_flip(*q) {
                                meas_flips.push(meas_counter);
                            }
                            meas_counter += 1;
                        }
                        Op::Reset(q) => {
                            frame.reset(*q);
                        }
                        _ => {}
                    }
                }

                // Now compute which Z-detectors fire from flipped measurements
                let (detectors, flips_logical) =
                    self.compute_z_detectors_from_meas_flips(&meas_flips, d, num_z, num_x);

                if detectors.is_empty() && !flips_logical {
                    // This error is undetectable and doesn't flip logical - skip
                    continue;
                }

                // Merge: p_xor = p(1-q) + q(1-p)
                let key = (detectors, flips_logical);
                let entry = mechanism_map.entry(key).or_insert(0.0);
                *entry = *entry * (1.0 - prob) + prob * (1.0 - *entry);
            }
        }

        // Convert map to vec
        mechanism_map
            .into_iter()
            .filter(|(_, p)| *p > 1e-15)
            .map(|((detectors, flips_logical), probability)| ErrorMechanism {
                detectors,
                flips_logical,
                probability,
            })
            .collect()
    }

    /// Propagate a Clifford gate through the Pauli frame (Heisenberg picture).
    fn propagate_gate(&self, frame: &mut PauliFrame, g: &Gate) {
        match *g {
            Gate::H(q) => frame.h(q),
            Gate::Cnot(c, t) => frame.cnot(c, t),
            Gate::X(q) | Gate::Y(q) | Gate::Z(q) => {
                // Pauli gates: conjugation is trivial (may pick up phase, which
                // we don't track since we only care about commutation with Z-basis
                // measurements). X commutes with X, anticommutes with Z, etc.
                // Actually for Pauli frame tracking, Pauli gates don't change
                // the X/Z components - they may flip signs but signs don't matter
                // for detector computation.
            }
            Gate::S(q) => {
                // S: X → Y (X gets Z component), Z → Z
                // In frame: x[q] stays, z[q] ^= x[q]
                let xq = frame.x[q];
                frame.z[q] ^= xq;
            }
            Gate::Sdg(q) => {
                // S†: X → -Y, Z → Z. Same effect on unsigned Pauli frame.
                let xq = frame.x[q];
                frame.z[q] ^= xq;
            }
            Gate::Cz(a, b) => {
                // CZ = (I⊗H) CNOT (I⊗H). Effect: X_a → X_a Z_b, X_b → Z_a X_b,
                // Z_a → Z_a, Z_b → Z_b
                let xa = frame.x[a];
                let xb = frame.x[b];
                frame.z[b] ^= xa;
                frame.z[a] ^= xb;
            }
            Gate::Swap(a, b) => {
                frame.x.swap(a, b);
                frame.z.swap(a, b);
            }
            _ => {
                // For non-Clifford gates in the surface code circuit (there should be none),
                // we can't propagate exactly. Since the surface code circuit only uses
                // H, CNOT, Measure, Reset, this shouldn't happen.
            }
        }
    }

    /// Determine what errors can occur at a given circuit operation.
    /// Returns pairs of (error_paulis, probability).
    fn errors_at_location(
        &self,
        op: &Op,
        noise: &NoiseModel,
    ) -> Vec<(Vec<(usize, PauliType)>, f64)> {
        let mut errors = Vec::new();

        match op {
            Op::Gate(g) => {
                let qs = g.qubits();
                match qs.len() {
                    1 if noise.p_1q > 0.0 => {
                        let q = qs[0];
                        let p = noise.p_1q / 3.0;
                        errors.push((vec![(q, PauliType::X)], p));
                        errors.push((vec![(q, PauliType::Y)], p));
                        errors.push((vec![(q, PauliType::Z)], p));
                    }
                    2 if noise.p_2q > 0.0 => {
                        let (a, b) = (qs[0], qs[1]);
                        let p = noise.p_2q / 15.0;
                        // 15 non-trivial 2-qubit Paulis
                        for pa in 0..4u8 {
                            for pb in 0..4u8 {
                                if pa == 0 && pb == 0 {
                                    continue;
                                }
                                let mut paulis = Vec::new();
                                if pa > 0 {
                                    paulis.push((a, PauliType::from_idx(pa)));
                                }
                                if pb > 0 {
                                    paulis.push((b, PauliType::from_idx(pb)));
                                }
                                errors.push((paulis, p));
                            }
                        }
                    }
                    _ => {}
                }
            }
            Op::Measure(q) => {
                // Readout error: the measurement outcome is flipped.
                // We model this as an X error right before measurement
                // (since X flips the Z-basis measurement outcome).
                if noise.p_meas > 0.0 {
                    errors.push((vec![(*q, PauliType::X)], noise.p_meas));
                }
            }
            Op::Reset(q) => {
                // Reset error: X flip after reset.
                if noise.p_reset > 0.0 {
                    errors.push((vec![(*q, PauliType::X)], noise.p_reset));
                }
            }
            _ => {}
        }

        errors
    }

    /// Given a set of flipped measurement indices, compute which Z-detectors fire
    /// and whether the logical observable is flipped.
    fn compute_z_detectors_from_meas_flips(
        &self,
        meas_flips: &[usize],
        d: usize,
        num_z: usize,
        num_x: usize,
    ) -> (Vec<usize>, bool) {
        let rounds = self.rounds;
        let anc_per_round = num_z + num_x;

        // Build a set for fast lookup
        let flip_set: std::collections::HashSet<usize> = meas_flips.iter().copied().collect();

        let mut detectors = Vec::new();

        // Syndrome measurement detectors (rounds 0..rounds)
        for r in 0..rounds {
            for k in 0..num_z {
                let meas_idx = r * anc_per_round + k;
                let curr_flipped = flip_set.contains(&meas_idx);

                if r == 0 {
                    // Detector fires if this measurement differs from ideal (0)
                    if curr_flipped {
                        detectors.push(k); // detector index = 0 * num_z + k
                    }
                } else {
                    // Detector fires if this measurement differs from previous
                    let prev_meas_idx = (r - 1) * anc_per_round + k;
                    let prev_flipped = flip_set.contains(&prev_meas_idx);
                    if curr_flipped != prev_flipped {
                        detectors.push(r * num_z + k);
                    }
                }
            }
        }

        // Final round detector from data qubit measurements
        let data_meas_base = rounds * anc_per_round;
        for (k, stab) in self.z_stabilizers.iter().enumerate() {
            // Compute parity of data qubits for this stabilizer
            let mut parity_flipped = false;
            for &dq in &stab.data_qubits {
                let meas_idx = data_meas_base + dq;
                if flip_set.contains(&meas_idx) {
                    parity_flipped = !parity_flipped;
                }
            }
            // Compare with last syndrome measurement
            let last_meas_idx = (rounds - 1) * anc_per_round + k;
            let last_flipped = flip_set.contains(&last_meas_idx);
            if parity_flipped != last_flipped {
                detectors.push(rounds * num_z + k);
            }
        }

        // Logical Z observable: product of Z on column 0
        let mut flips_logical = false;
        for r in 0..d {
            let dq = Self::data_idx(d, r, 0);
            let meas_idx = data_meas_base + dq;
            if flip_set.contains(&meas_idx) {
                flips_logical = !flips_logical;
            }
        }

        detectors.sort();
        (detectors, flips_logical)
    }

    /// Extracts Z-detector defect node indices and the raw logical measurement from raw circuit output.
    pub fn extract_z_defects(&self, raw_bits: &[bool]) -> (Vec<usize>, bool) {
        let d = self.d;
        let rounds = self.rounds;
        let num_z = self.z_stabilizers.len();
        let num_x = self.x_stabilizers.len();
        let anc_per_round = num_z + num_x;

        let mut defects = Vec::new();

        // Round 0 defects
        for (k, &bit) in raw_bits[..num_z].iter().enumerate() {
            if bit {
                defects.push(k);
            }
        }

        // Intermediate rounds
        for r in 1..rounds {
            let curr_base = r * anc_per_round;
            let prev_base = (r - 1) * anc_per_round;
            for k in 0..num_z {
                let m_curr = raw_bits[curr_base + k];
                let m_prev = raw_bits[prev_base + k];
                if m_curr != m_prev {
                    defects.push(r * num_z + k);
                }
            }
        }

        // Final round from data qubit measurements
        let data_base = rounds * anc_per_round;
        let data_bits = &raw_bits[data_base..data_base + d * d];

        let prev_base = (rounds - 1) * anc_per_round;
        for (k, stab) in self.z_stabilizers.iter().enumerate() {
            let mut parity = false;
            for &dq in &stab.data_qubits {
                parity ^= data_bits[dq];
            }
            let m_prev = raw_bits[prev_base + k];
            if parity != m_prev {
                defects.push(rounds * num_z + k);
            }
        }

        // Logical Z_L is the product of Z on column 0: dq = (r, 0)
        let mut raw_logical = false;
        for r in 0..d {
            raw_logical ^= data_bits[Self::data_idx(d, r, 0)];
        }

        (defects, raw_logical)
    }

    /// Fast detector sampling using the circuit-derived DEM.
    ///
    /// Each error mechanism fires independently with its probability.
    /// We XOR the resulting detector flips and logical flips, then decode.
    pub fn run_experiment_fast<R: RngCore>(
        &self,
        _noise: &NoiseModel,
        shots: usize,
        rng: &mut R,
    ) -> MemoryExperimentResult {
        let mut logical_errors = 0;

        let num_detectors = self.decoder.graph.num_nodes;
        let mut defect_flags = vec![false; num_detectors];

        for _ in 0..shots {
            defect_flags.fill(false);
            let mut true_logical_flip = false;

            for em in &self.error_mechanisms {
                if em.probability > 0.0 && rng.random::<f64>() < em.probability {
                    for &det in &em.detectors {
                        defect_flags[det] = !defect_flags[det];
                    }
                    if em.flips_logical {
                        true_logical_flip = !true_logical_flip;
                    }
                }
            }

            let mut active_defects = Vec::new();
            for (det, &fired) in defect_flags.iter().enumerate() {
                if fired {
                    active_defects.push(det);
                }
            }

            let predicted_flip = self.decoder.decode(&active_defects);
            if predicted_flip != true_logical_flip {
                logical_errors += 1;
            }
        }

        let logical_error_rate = logical_errors as f64 / shots as f64;
        MemoryExperimentResult {
            distance: self.d,
            rounds: self.rounds,
            physical_p: self.error_mechanisms.iter().map(|m| m.probability).sum::<f64>()
                / self.error_mechanisms.len().max(1) as f64
                * 3.0, // approximate; not used in critical path
            shots,
            logical_errors,
            logical_error_rate,
        }
    }

    /// Full stabilizer tableau simulation of the surface code memory experiment.
    pub fn run_experiment_tableau<R: RngCore>(
        &self,
        noise: &NoiseModel,
        shots: usize,
        rng: &mut R,
    ) -> MemoryExperimentResult {
        let circuit = self.build_circuit();
        let total_q = Self::total_qubits(self.d);
        let mut logical_errors = 0;

        for _ in 0..shots {
            let mut tab = Tableau::new(total_q);
            let raw_bits = circuit
                .run_noisy(&mut tab, noise, rng)
                .expect("tableau simulation failed");
            let (defects, raw_logical) = self.extract_z_defects(&raw_bits);
            let predicted_flip = self.decoder.decode(&defects);
            let corrected_logical = raw_logical ^ predicted_flip;
            if corrected_logical {
                logical_errors += 1;
            }
        }

        let logical_error_rate = logical_errors as f64 / shots as f64;
        MemoryExperimentResult {
            distance: self.d,
            rounds: self.rounds,
            physical_p: noise.p_1q,
            shots,
            logical_errors,
            logical_error_rate,
        }
    }

    /// Default memory experiment: uses full tableau for small shot counts (<= 300) and
    /// fast detector sampling for large statistical sweeps.
    pub fn run_experiment<R: RngCore>(
        &self,
        noise: &NoiseModel,
        shots: usize,
        rng: &mut R,
    ) -> MemoryExperimentResult {
        if shots <= 300 && self.d <= 5 {
            self.run_experiment_tableau(noise, shots, rng)
        } else {
            self.run_experiment_fast(noise, shots, rng)
        }
    }

    /// Compute per-detector firing rates from DEM sampling.
    pub fn dem_detector_rates<R: RngCore>(
        &self,
        shots: usize,
        rng: &mut R,
    ) -> Vec<f64> {
        let num_z = self.z_stabilizers.len();
        let num_detectors = (self.rounds + 1) * num_z;
        let mut fire_counts = vec![0usize; num_detectors];
        let mut defect_flags = vec![false; num_detectors];

        for _ in 0..shots {
            defect_flags.fill(false);
            for em in &self.error_mechanisms {
                if em.probability > 0.0 && rng.random::<f64>() < em.probability {
                    for &det in &em.detectors {
                        if det < num_detectors {
                            defect_flags[det] = !defect_flags[det];
                        }
                    }
                }
            }
            for (det, &fired) in defect_flags.iter().enumerate() {
                if fired {
                    fire_counts[det] += 1;
                }
            }
        }

        fire_counts.iter().map(|&c| c as f64 / shots as f64).collect()
    }

    /// Compute per-detector firing rates from tableau simulation.
    pub fn tableau_detector_rates<R: RngCore>(
        &self,
        noise: &NoiseModel,
        shots: usize,
        rng: &mut R,
    ) -> Vec<f64> {
        let num_z = self.z_stabilizers.len();
        let num_detectors = (self.rounds + 1) * num_z;
        let mut fire_counts = vec![0usize; num_detectors];
        let circuit = self.build_circuit();
        let total_q = Self::total_qubits(self.d);

        for _ in 0..shots {
            let mut tab = Tableau::new(total_q);
            let raw_bits = circuit
                .run_noisy(&mut tab, noise, rng)
                .expect("tableau simulation failed");
            let (defects, _) = self.extract_z_defects(&raw_bits);
            for d in defects {
                if d < num_detectors {
                    fire_counts[d] += 1;
                }
            }
        }

        fire_counts.iter().map(|&c| c as f64 / shots as f64).collect()
    }
}

/// Pauli error type used in DEM construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PauliType {
    X,
    Y,
    Z,
}

impl PauliType {
    fn from_idx(i: u8) -> Self {
        match i {
            1 => PauliType::X,
            2 => PauliType::Y,
            3 => PauliType::Z,
            _ => panic!("invalid Pauli index"),
        }
    }
}
