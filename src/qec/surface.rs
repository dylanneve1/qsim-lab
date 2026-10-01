//! Rotated surface code memory experiment, layout, and threshold benchmarks.
//!
//! Implements the standard rotated planar surface code (Horsman et al. 2012, Fowler et al.).
//! For code distance `d`, there are `d^2` data qubits and `d^2 - 1` syndrome ancillas
//! (`(d^2 - 1) / 2` Z-type and `(d^2 - 1) / 2` X-type), for a total of `2d^2 - 1` physical qubits.
//!
//! A logical qubit `|0_L>` is stored and protected across `rounds = d` syndrome extraction
//! cycles with circuit-level depolarizing noise and readout errors. Syndromes are extracted
//! on the stabilizer tableau and decoded with the [`UnionFindDecoder`].

use crate::circuit::Circuit;
use crate::noise::NoiseModel;
use crate::qec::decoder::{DecodingGraph, UnionFindDecoder};
use crate::qec::repetition::MemoryExperimentResult;
use crate::stabilizer::Tableau;
use rand::{Rng, RngCore};

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
    /// Detectors flipped by this error.
    pub detectors: Vec<usize>,
    /// Whether this error flips the logical Z observable.
    pub flips_logical: bool,
    /// Probability scaling factor relative to base p.
    pub p_factor: f64,
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
        assert!(d >= 3 && d % 2 == 1, "distance must be an odd integer >= 3");
        assert!(rounds >= 1, "rounds must be >= 1");

        let (z_stabilizers, x_stabilizers) = Self::generate_stabilizers(d);
        let graph = Self::build_z_decoding_graph(d, rounds, &z_stabilizers);
        let decoder = UnionFindDecoder::new(graph);
        let error_mechanisms = Self::build_error_mechanisms(d, rounds, &z_stabilizers);

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
                        graph.add_edge(u, v, false, 1);
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

    /// Builds the list of independent error mechanisms for fast detector sampling.
    pub fn build_error_mechanisms(
        d: usize,
        rounds: usize,
        z_stabs: &[StabilizerFace],
    ) -> Vec<ErrorMechanism> {
        let num_z = z_stabs.len();
        let mut mechanisms = Vec::new();

        let mut data_to_z = vec![Vec::new(); d * d];
        for (k, z) in z_stabs.iter().enumerate() {
            for &dq in &z.data_qubits {
                data_to_z[dq].push(k);
            }
        }

        for r in 0..=rounds {
            // Data qubit X errors (bit flips)
            for (dq, touching) in data_to_z.iter().enumerate().take(d * d) {
                let (_, c) = Self::data_coords(d, dq);
                let flips_logical = c == 0;

                let mut detectors = Vec::new();
                for &k in touching {
                    detectors.push(r * num_z + k);
                }

                mechanisms.push(ErrorMechanism {
                    detectors,
                    flips_logical,
                    p_factor: 1.0,
                });
            }

            // Ancilla readout measurement errors
            if r < rounds {
                for k in 0..num_z {
                    let detectors = vec![r * num_z + k, (r + 1) * num_z + k];
                    mechanisms.push(ErrorMechanism {
                        detectors,
                        flips_logical: false,
                        p_factor: 1.0,
                    });
                }
            }
        }

        mechanisms
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

    /// Fast detector sampling (Pauli frame / DEM sampling).
    pub fn run_experiment_fast<R: RngCore>(
        &self,
        noise: &NoiseModel,
        shots: usize,
        rng: &mut R,
    ) -> MemoryExperimentResult {
        let p = noise.p_1q;
        let mut logical_errors = 0;

        let num_detectors = self.decoder.graph.num_nodes;
        let mut defect_flags = vec![false; num_detectors];

        for _ in 0..shots {
            defect_flags.fill(false);
            let mut true_logical_flip = false;

            for em in &self.error_mechanisms {
                let err_p = (p * em.p_factor).min(1.0);
                if err_p > 0.0 && rng.random::<f64>() < err_p {
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
            physical_p: p,
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
}
