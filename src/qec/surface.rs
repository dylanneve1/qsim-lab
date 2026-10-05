//! Rotated surface code memory experiment, layout, and threshold benchmarks.
//!
//! Implements the standard rotated planar surface code (Horsman et al. 2012, Fowler et al.).
//! For code distance `d`, there are `d^2` data qubits and `d^2 - 1` syndrome ancillas
//! (`(d^2 - 1) / 2` Z-type and `(d^2 - 1) / 2` X-type), for a total of `2d^2 - 1` physical qubits.
//!
//! A logical qubit `|0_L>` is stored for `rounds` syndrome-extraction cycles under
//! circuit-level noise ([`NoiseModel`]: depolarizing after every gate, readout flips,
//! reset flips) and then all data qubits are measured in the Z basis. Only the Z-type
//! detectors are decoded (a Z-basis memory experiment is sensitive to X errors only).
//!
//! # Detectors and observable
//!
//! Detector `r * num_z + k` (for Z-stabilizer `k`):
//! * `r = 0`: the round-0 measurement of ancilla `k` (deterministically 0 without noise);
//! * `0 < r < rounds`: round `r` XOR round `r - 1` of ancilla `k`;
//! * `r = rounds`: parity of the final data measurements in stabilizer `k`, XOR the last
//!   ancilla measurement.
//!
//! The observable is the parity of the final data measurements on column 0 (`Z_L`).
//!
//! # Sampling methods
//!
//! [`SamplingMethod::Tableau`] runs the full circuit through the CHP tableau with
//! [`Circuit::run_noisy`]. [`SamplingMethod::DetectorErrorModel`] samples the same
//! distribution from the circuit-derived fault list in [`crate::qec::dem`] (every noise
//! location and every Pauli the noise model can insert, propagated through the circuit).
//! The two are checked against each other per detector and on the decoded logical error
//! rate (`tests/qec_dem_audit.rs`). There is no silent switch between them.
//!
//! # Decoding graph
//!
//! [`SurfaceCode::new`] builds the Union-Find graph from the circuit's faults
//! ([`crate::qec::dem::decoding_graph_from_faults`]), so it contains the diagonal
//! (space-time) edges produced by faults in the middle of the CNOT schedule.
//! [`SurfaceCode::with_phenomenological_decoder`] keeps the older hand-built graph
//! (space edges per round and time edges only) for comparison.

use crate::circuit::Circuit;
use crate::engines::stabilizer::Tableau;
use crate::noise::NoiseModel;
use crate::qec::decoder::{DecodingGraph, UnionFindDecoder};
use crate::qec::dem::{
    decoding_graph_from_faults, weighted_decoding_graph_from_faults, CircuitFaults, DemSampler,
    ErrorMechanism, GraphReport,
};
use crate::qec::repetition::MemoryExperimentResult;
use rand::RngCore;

/// A stabilizer face on the dual grid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StabilizerFace {
    /// Row of the face on the `(d+1) × (d+1)` dual grid; its data neighbours are
    /// the qubits at rows `r-1`/`r` and columns `c-1`/`c`.
    pub r: usize,
    /// Column of the face on the dual grid.
    pub c: usize,
    /// `true` for a Z-type stabilizer (`(r + c)` even), `false` for X-type.
    pub is_z: bool,
    /// Indices of neighboring data qubits in `0..d^2`, in the order their
    /// CNOTs are applied.
    pub data_qubits: Vec<usize>,
}

/// How to sample shots of the memory experiment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingMethod {
    /// Full circuit simulation on the stabilizer tableau (`O(n^2)` per measurement).
    Tableau,
    /// Exact sampling from the circuit-derived fault list (see [`crate::qec::dem`]).
    DetectorErrorModel,
}

/// A rotated surface code memory experiment.
#[derive(Clone, Debug)]
pub struct SurfaceCode {
    /// Code distance (odd, ≥ 3).
    pub d: usize,
    /// Number of syndrome-extraction rounds before the final data measurement.
    pub rounds: usize,
    /// Z-type stabilizers, in ancilla order (index `k` = Z-ancilla / detector `k`).
    pub z_stabilizers: Vec<StabilizerFace>,
    /// X-type stabilizers (measured but not decoded in the Z-basis memory).
    pub x_stabilizers: Vec<StabilizerFace>,
    /// Union-Find decoder for the Z-type detectors.
    pub decoder: UnionFindDecoder,
    /// Every noise location of `build_circuit()` with the detector/observable
    /// signature of every Pauli the noise model can insert there.
    pub faults: CircuitFaults,
    /// How the decoding graph was derived (all zeros for the phenomenological graph).
    pub graph_report: GraphReport,
}

impl SurfaceCode {
    /// Creates a rotated surface code for distance `d` (odd, >= 3) and `rounds >= 1`,
    /// decoded on a graph derived from the circuit's own faults.
    pub fn new(d: usize, rounds: usize) -> Self {
        let mut sc = Self::skeleton(d, rounds);
        // Uniform noise only sets the relative weights used to break logical-flag ties.
        let (graph, report) = decoding_graph_from_faults(&sc.faults, &NoiseModel::uniform(1e-3));
        sc.decoder = UnionFindDecoder::new(graph);
        sc.graph_report = report;
        sc
    }

    /// Same code and circuit, decoded by a **weighted** Union-Find on the
    /// circuit-derived graph: edge lengths are quantised log-likelihood ratios
    /// `ln((1-p_e)/p_e)` of the edge probabilities under `noise` (see
    /// [`crate::qec::dem::weighted_decoding_graph_from_faults`]; the most likely
    /// edge has length `resolution`).
    pub fn new_weighted(d: usize, rounds: usize, noise: &NoiseModel, resolution: usize) -> Self {
        let mut sc = Self::skeleton(d, rounds);
        let (graph, report) = weighted_decoding_graph_from_faults(&sc.faults, noise, resolution);
        sc.decoder = UnionFindDecoder::new(graph);
        sc.graph_report = report;
        sc
    }

    /// Same code and circuit, but decoded on the hand-built phenomenological graph
    /// ([`SurfaceCode::build_z_decoding_graph`]): one space edge per data qubit per
    /// round and one time edge per check, no diagonal edges.
    pub fn with_phenomenological_decoder(d: usize, rounds: usize) -> Self {
        Self::skeleton(d, rounds)
    }

    fn skeleton(d: usize, rounds: usize) -> Self {
        assert!(d >= 3 && d % 2 == 1, "distance must be an odd integer >= 3");
        assert!(rounds >= 1, "rounds must be >= 1");
        let (z_stabilizers, x_stabilizers) = Self::generate_stabilizers(d);
        let graph = Self::build_z_decoding_graph(d, rounds, &z_stabilizers);
        let mut sc = Self {
            d,
            rounds,
            z_stabilizers,
            x_stabilizers,
            decoder: UnionFindDecoder::new(graph),
            faults: CircuitFaults {
                num_detectors: 0,
                locations: Vec::new(),
            },
            graph_report: GraphReport::default(),
        };
        sc.faults = CircuitFaults::from_circuit(
            &sc.build_circuit(),
            &sc.detector_records(),
            &sc.observable_records(),
        )
        .expect("surface code circuit is Clifford with NoiseModel-only noise");
        sc
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

                // CNOT order. Z checks: NW, NE, SW, SE. X checks: NW, SW, NE, SE.
                // An X fault on an X-check ancilla after two of its four CNOTs spreads
                // to the last two data qubits ("hook" error). With the X-check order
                // above that pair is vertical, i.e. perpendicular to the horizontal
                // X-type logical, so a hook costs at most one unit of distance
                // ([`UnionFindDecoder`] graph distance stays `d`). With NW, NE, SW, SE
                // the hook pair is horizontal and the circuit distance for this
                // Z-memory experiment drops to about `(d + 1) / 2` (measured: 3 at
                // d = 5). Z-check hooks are Z errors, invisible to Z memory.
                if !is_z && data.len() == 4 {
                    data.swap(1, 2);
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

    /// Measurement records per syndrome round (Z ancillas first, then X ancillas).
    fn records_per_round(&self) -> usize {
        self.z_stabilizers.len() + self.x_stabilizers.len()
    }

    /// For each Z-detector (index `r * num_z + k`), the measurement records whose
    /// parity it is. See the module docs.
    pub fn detector_records(&self) -> Vec<Vec<usize>> {
        let num_z = self.z_stabilizers.len();
        let apr = self.records_per_round();
        let data_base = self.rounds * apr;
        let mut dets = Vec::with_capacity((self.rounds + 1) * num_z);
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

    /// Measurement records whose parity is the logical `Z_L` (data column 0).
    pub fn observable_records(&self) -> Vec<usize> {
        let data_base = self.rounds * self.records_per_round();
        (0..self.d)
            .map(|r| data_base + Self::data_idx(self.d, r, 0))
            .collect()
    }

    /// Extracts Z-detector defect node indices and the raw logical measurement from raw
    /// circuit output, using exactly the definitions of [`Self::detector_records`].
    pub fn extract_z_defects(&self, raw_bits: &[bool]) -> (Vec<usize>, bool) {
        let parity = |recs: &[usize]| recs.iter().fold(false, |a, &r| a ^ raw_bits[r]);
        let defects = self
            .detector_records()
            .iter()
            .enumerate()
            .filter(|(_, recs)| parity(recs))
            .map(|(i, _)| i)
            .collect();
        (defects, parity(&self.observable_records()))
    }

    /// Exact detector sampler for `noise` (see [`crate::qec::dem`]).
    pub fn dem_sampler(&self, noise: &NoiseModel) -> DemSampler {
        DemSampler::new(&self.faults, noise)
    }

    /// The merged detector error model for `noise`: one independent mechanism per
    /// distinct signature (for inspection; sampling uses [`Self::dem_sampler`]).
    pub fn detector_error_model(&self, noise: &NoiseModel) -> Vec<ErrorMechanism> {
        self.faults.merged_mechanisms(noise)
    }

    /// Runs the memory experiment with an explicitly chosen sampling method.
    pub fn run_experiment<R: RngCore>(
        &self,
        noise: &NoiseModel,
        shots: usize,
        method: SamplingMethod,
        rng: &mut R,
    ) -> MemoryExperimentResult {
        let mut logical_errors = 0;
        self.for_each_shot(noise, shots, method, rng, |defects, raw_logical| {
            if raw_logical ^ self.decoder.decode(defects) {
                logical_errors += 1;
            }
        });
        MemoryExperimentResult {
            distance: self.d,
            rounds: self.rounds,
            physical_p: noise.p_2q,
            shots,
            logical_errors,
            logical_error_rate: logical_errors as f64 / shots.max(1) as f64,
        }
    }

    /// Full stabilizer-tableau memory experiment.
    pub fn run_experiment_tableau<R: RngCore>(
        &self,
        noise: &NoiseModel,
        shots: usize,
        rng: &mut R,
    ) -> MemoryExperimentResult {
        self.run_experiment(noise, shots, SamplingMethod::Tableau, rng)
    }

    /// Memory experiment sampled from the circuit-derived detector error model.
    pub fn run_experiment_dem<R: RngCore>(
        &self,
        noise: &NoiseModel,
        shots: usize,
        rng: &mut R,
    ) -> MemoryExperimentResult {
        self.run_experiment(noise, shots, SamplingMethod::DetectorErrorModel, rng)
    }

    /// Calls `f(defects, raw_logical_flip)` for each of `shots` sampled shots.
    pub fn for_each_shot<R: RngCore, F: FnMut(&[usize], bool)>(
        &self,
        noise: &NoiseModel,
        shots: usize,
        method: SamplingMethod,
        rng: &mut R,
        mut f: F,
    ) {
        match method {
            SamplingMethod::Tableau => {
                let circuit = self.build_circuit();
                let nq = Self::total_qubits(self.d);
                for _ in 0..shots {
                    let mut tab = Tableau::new(nq);
                    let bits = circuit
                        .run_noisy(&mut tab, noise, rng)
                        .expect("tableau simulation failed");
                    let (defects, raw) = self.extract_z_defects(&bits);
                    f(&defects, raw);
                }
            }
            SamplingMethod::DetectorErrorModel => {
                let sampler = self.dem_sampler(noise);
                let mut flags = Vec::new();
                let mut defects = Vec::new();
                for _ in 0..shots {
                    let raw = sampler.sample_into(rng, &mut flags, &mut defects);
                    f(&defects, raw);
                }
            }
        }
    }

    /// Per-detector firing rates under `method`.
    pub fn detector_rates<R: RngCore>(
        &self,
        noise: &NoiseModel,
        shots: usize,
        method: SamplingMethod,
        rng: &mut R,
    ) -> Vec<f64> {
        let mut counts = vec![0usize; self.faults.num_detectors];
        self.for_each_shot(noise, shots, method, rng, |defects, _| {
            for &d in defects {
                counts[d] += 1;
            }
        });
        counts
            .iter()
            .map(|&c| c as f64 / shots.max(1) as f64)
            .collect()
    }
}
