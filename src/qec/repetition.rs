//! 1D Bit-flip repetition code memory experiment.
//!
//! A distance-`d` repetition code protects one logical qubit against bit flips (`X` errors)
//! using `d` data qubits and `d - 1` syndrome ancillas. Each syndrome ancilla measures the
//! weight-2 stabilizer `Z_j Z_{j+1}` across repeated syndrome extraction rounds.
//!
//! At the end of `T` rounds, all data qubits are measured in the computational basis.
//! The detector events (changes in syndrome across rounds) are decoded using the
//! [`UnionFindDecoder`] to correct data errors and recover the initial logical state `|0_L>`.

use crate::circuit::Circuit;
use crate::engines::stabilizer::Tableau;
use crate::noise::NoiseModel;
use crate::qec::decoder::{DecodingGraph, UnionFindDecoder};
use rand::RngCore;

/// A bit-flip repetition code experiment.
#[derive(Clone, Debug)]
pub struct RepetitionCode {
    pub d: usize,
    pub rounds: usize,
    pub decoder: UnionFindDecoder,
}

/// Results of a repetition code memory experiment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MemoryExperimentResult {
    pub distance: usize,
    pub rounds: usize,
    pub physical_p: f64,
    pub shots: usize,
    pub logical_errors: usize,
    pub logical_error_rate: f64,
}

impl MemoryExperimentResult {
    /// Wilson score 95% confidence interval for the logical error rate.
    /// Returns `(lower, upper)` as fractions in `[0, 1]`.
    pub fn wilson_ci_95(&self) -> (f64, f64) {
        let n = self.shots as f64;
        let p_hat = self.logical_error_rate;
        let z = 1.96; // 95% CI z-score
        let z2 = z * z;
        let denom = 1.0 + z2 / n;
        let center = (p_hat + z2 / (2.0 * n)) / denom;
        let half_width = z * (p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)).sqrt() / denom;
        let lo = (center - half_width).max(0.0);
        let hi = (center + half_width).min(1.0);
        (lo, hi)
    }
}

impl RepetitionCode {
    /// Creates a new repetition code of distance `d` with `rounds` syndrome extraction rounds.
    pub fn new(d: usize, rounds: usize) -> Self {
        assert!(d >= 2, "distance must be at least 2");
        assert!(rounds >= 1, "rounds must be at least 1");
        let graph = Self::build_decoding_graph(d, rounds);
        let decoder = UnionFindDecoder::new(graph);
        Self { d, rounds, decoder }
    }

    /// Data qubit indices are `0..d`.
    #[inline]
    pub fn data_qubit(i: usize) -> usize {
        i
    }

    /// Ancilla qubit indices are `d..2d-1`.
    #[inline]
    pub fn ancilla_qubit(d: usize, j: usize) -> usize {
        d + j
    }

    /// Total number of qubits: `2d - 1`.
    #[inline]
    pub fn num_qubits(d: usize) -> usize {
        2 * d - 1
    }

    /// Node index in the decoding graph for round `r` and syndrome ancilla `j`.
    #[inline]
    pub fn node_id(d: usize, r: usize, j: usize) -> usize {
        r * (d - 1) + j
    }

    /// Total detector nodes in the decoding graph: `(rounds + 1) * (d - 1)`.
    #[inline]
    pub fn num_detector_nodes(d: usize, rounds: usize) -> usize {
        (rounds + 1) * (d - 1)
    }

    /// Builds the decoding spacetime graph.
    pub fn build_decoding_graph(d: usize, rounds: usize) -> DecodingGraph {
        let num_detectors = Self::num_detector_nodes(d, rounds);
        let boundary_node = num_detectors; // last node is virtual open boundary
        let mut graph = DecodingGraph::new(num_detectors + 1, boundary_node);

        for r in 0..=rounds {
            // Space-like edges between neighboring syndrome detectors
            // Left boundary edge to detector 0 (represents error on data qubit 0 -> flips logical!)
            let left_det = Self::node_id(d, r, 0);
            graph.add_edge(left_det, boundary_node, true, 1);

            for j in 0..d - 2 {
                let u = Self::node_id(d, r, j);
                let v = Self::node_id(d, r, j + 1);
                // Internal data qubit error (does not flip logical observable Z_0)
                graph.add_edge(u, v, false, 1);
            }

            // Right boundary edge from detector d - 2 (represents error on data qubit d - 1)
            let right_det = Self::node_id(d, r, d - 2);
            graph.add_edge(right_det, boundary_node, false, 1);

            // Time-like edges between consecutive rounds (measurement readout errors)
            if r < rounds {
                for j in 0..d - 1 {
                    let u = Self::node_id(d, r, j);
                    let v = Self::node_id(d, r + 1, j);
                    graph.add_edge(u, v, false, 1);
                }
            }
        }

        graph
    }

    /// Constructs the quantum circuit for the repetition code memory experiment.
    pub fn build_circuit(&self) -> Circuit {
        let d = self.d;
        let rounds = self.rounds;
        let n = Self::num_qubits(d);
        let mut c = Circuit::new(n);

        // State starts in |0...0> = |0_L>
        for r in 0..rounds {
            // Reset all ancillas if r > 0 (they start in |0> for r == 0)
            if r > 0 {
                for j in 0..d - 1 {
                    c.reset(Self::ancilla_qubit(d, j));
                }
            }
            // Entangle with data qubits to measure Z_j Z_{j+1}
            for j in 0..d - 1 {
                let a = Self::ancilla_qubit(d, j);
                let q0 = Self::data_qubit(j);
                let q1 = Self::data_qubit(j + 1);
                c.cnot(q0, a);
                c.cnot(q1, a);
            }
            // Measure ancillas
            for j in 0..d - 1 {
                c.measure(Self::ancilla_qubit(d, j));
            }
        }

        // Final round: measure all data qubits in computational basis
        for i in 0..d {
            c.measure(Self::data_qubit(i));
        }

        c
    }

    /// Extracts detector defect node indices and the raw logical measurement from raw circuit output.
    pub fn extract_defects(&self, raw_bits: &[bool]) -> (Vec<usize>, bool) {
        let d = self.d;
        let rounds = self.rounds;
        let mut defects = Vec::new();

        // Ancilla measurements: rounds * (d - 1) bits
        let anc_bits = &raw_bits[0..rounds * (d - 1)];
        let data_bits = &raw_bits[rounds * (d - 1)..];

        // Round 0 defects
        for (j, &m0) in anc_bits[..d - 1].iter().enumerate() {
            if m0 {
                defects.push(Self::node_id(d, 0, j));
            }
        }

        // Intermediate rounds
        for r in 1..rounds {
            for j in 0..d - 1 {
                let m_curr = anc_bits[r * (d - 1) + j];
                let m_prev = anc_bits[(r - 1) * (d - 1) + j];
                if m_curr != m_prev {
                    defects.push(Self::node_id(d, r, j));
                }
            }
        }

        // Final round from data qubit measurements
        for j in 0..d - 1 {
            let m_data = data_bits[j] != data_bits[j + 1];
            let m_prev = anc_bits[(rounds - 1) * (d - 1) + j];
            if m_data != m_prev {
                defects.push(Self::node_id(d, rounds, j));
            }
        }

        // Logical observable: Z_L is measured by data qubit 0
        let raw_logical = data_bits[0];
        (defects, raw_logical)
    }

    /// Runs a Monte Carlo memory experiment on the stabilizer tableau backend.
    pub fn run_experiment<R: RngCore>(
        &self,
        noise: &NoiseModel,
        shots: usize,
        rng: &mut R,
    ) -> MemoryExperimentResult {
        let circuit = self.build_circuit();
        let n = Self::num_qubits(self.d);
        let mut logical_errors = 0;

        for _ in 0..shots {
            let mut tab = Tableau::new(n);
            let raw_bits = circuit
                .run_noisy(&mut tab, noise, rng)
                .expect("simulation failed");
            let (defects, raw_logical) = self.extract_defects(&raw_bits);
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
}
