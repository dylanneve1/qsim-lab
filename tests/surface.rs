//! Tests for the rotated surface code memory experiment, its circuit-derived
//! detector error model and the Union-Find decoding graph.

use qsim_lab::circuit::Op;
use qsim_lab::noise::NoiseModel;
use qsim_lab::qec::dem::{propagate_forward, two_qubit_outcome, FaultKind, Pauli};
use qsim_lab::qec::{SamplingMethod, SurfaceCode};
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn surface_code_noiseless_has_zero_detectors_and_logical_errors() {
    let mut rng = StdRng::seed_from_u64(1);
    for d in [3, 5] {
        let sc = SurfaceCode::new(d, d);
        for method in [SamplingMethod::DetectorErrorModel, SamplingMethod::Tableau] {
            let shots = if method == SamplingMethod::Tableau {
                20
            } else {
                300
            };
            sc.for_each_shot(&NoiseModel::none(), shots, method, &mut rng, |defects, raw| {
                assert!(
                    defects.is_empty() && !raw,
                    "d={d} {method:?}: noiseless shot had defects {defects:?} / logical {raw}"
                );
            });
        }
    }
}

/// The backward sensitivity sweep and the forward Pauli-frame simulation are
/// independent implementations; they must agree on every outcome of every
/// fault location.
#[test]
fn backward_sweep_matches_forward_pauli_frames_on_every_fault() {
    for (d, rounds) in [(3, 3), (5, 2)] {
        let sc = SurfaceCode::new(d, rounds);
        let circuit = sc.build_circuit();
        let dets = sc.detector_records();
        let obs = sc.observable_records();
        let mut n_checked = 0;
        for loc in &sc.faults.locations {
            let op = &circuit.ops[loc.op_index];
            for (j, sig) in loc.outcomes.iter().enumerate() {
                let paulis: Vec<(usize, Pauli)> = match (loc.kind, op) {
                    (FaultKind::Gate1q, Op::Gate(g)) => {
                        vec![(g.qubits()[0], [Pauli::X, Pauli::Y, Pauli::Z][j])]
                    }
                    (FaultKind::Gate2q, Op::Gate(g)) => {
                        let q = g.qubits();
                        two_qubit_outcome(j + 1, q[0], q[1])
                    }
                    (FaultKind::Reset, Op::Reset(q)) => vec![(*q, Pauli::X)],
                    (FaultKind::Readout, Op::Measure(_)) => vec![],
                    other => panic!("unexpected location {other:?}"),
                };
                let fwd =
                    propagate_forward(&circuit, loc.op_index, loc.kind, &paulis, &dets, &obs)
                        .unwrap();
                assert_eq!(&fwd, sig, "d={d} op {} outcome {j}", loc.op_index);
                n_checked += 1;
            }
        }
        assert!(n_checked > 100);
    }
}

/// Every location `run_noisy` puts noise on is enumerated, with the right
/// number of outcomes.
#[test]
fn fault_locations_cover_every_noisy_op() {
    let sc = SurfaceCode::new(3, 3);
    let circuit = sc.build_circuit();
    let mut expected = Vec::new();
    for (i, op) in circuit.ops.iter().enumerate() {
        match op {
            Op::Gate(g) if g.qubits().len() == 1 => expected.push((i, FaultKind::Gate1q, 3)),
            Op::Gate(g) if g.qubits().len() == 2 => expected.push((i, FaultKind::Gate2q, 15)),
            Op::Measure(_) => expected.push((i, FaultKind::Readout, 1)),
            Op::Reset(_) => expected.push((i, FaultKind::Reset, 1)),
            _ => {}
        }
    }
    let got: Vec<_> = sc
        .faults
        .locations
        .iter()
        .map(|l| (l.op_index, l.kind, l.outcomes.len()))
        .collect();
    assert_eq!(got, expected);
}

/// Every single circuit fault (any location, any Pauli) is corrected.
#[test]
fn every_single_circuit_fault_is_corrected() {
    for d in [3, 5] {
        let sc = SurfaceCode::new(d, d);
        for loc in &sc.faults.locations {
            for (j, sig) in loc.outcomes.iter().enumerate() {
                let pred = sc.decoder.decode(&sig.detectors);
                assert_eq!(
                    pred, sig.flips_logical,
                    "d={d}: fault at op {} ({:?}) outcome {j} -> {:?} not corrected",
                    loc.op_index, loc.kind, sig
                );
            }
        }
    }
}

#[test]
fn circuit_derived_graph_has_full_distance() {
    for d in [3, 5, 7] {
        let sc = SurfaceCode::new(d, d);
        let r = &sc.graph_report;
        assert_eq!(r.undetectable_logical, 0, "d={d}: {r:?}");
        assert_eq!(
            r.hyperedges_decomposable, r.hyperedge_signatures,
            "d={d}: {r:?}"
        );
        assert_eq!(
            sc.decoder.graph.min_logical_weight(),
            Some(d),
            "d={d}: graph-like circuit distance"
        );
    }
}

/// Two simultaneous X errors on data qubits (inserted between rounds) are
/// always corrected at d = 5.
#[test]
fn weight_two_data_errors_are_corrected_at_d5() {
    let d = 5;
    let sc = SurfaceCode::new(d, d);
    let circuit = sc.build_circuit();
    let dets = sc.detector_records();
    let obs = sc.observable_records();
    // Insert after the last measurement of round 1, i.e. after a reset-free point.
    let per_round = sc.z_stabilizers.len() + sc.x_stabilizers.len();
    let mut seen = 0;
    let mut op_idx = 0;
    for (i, op) in circuit.ops.iter().enumerate() {
        if matches!(op, Op::Measure(_)) {
            seen += 1;
            if seen == 2 * per_round {
                op_idx = i;
                break;
            }
        }
    }
    // An X after a measurement on a data qubit is equivalent to the 1q-gate
    // fault model; use the forward frame with a Gate-kind insertion at a
    // following H gate on an ancilla (which acts trivially on data).
    let h_idx = (op_idx + 1..circuit.ops.len())
        .find(|&i| matches!(circuit.ops[i], Op::Gate(g) if g.qubits().len() == 1))
        .unwrap();
    for q1 in 0..d * d {
        for q2 in q1 + 1..d * d {
            let sig = propagate_forward(
                &circuit,
                h_idx,
                FaultKind::Gate1q,
                &[(q1, Pauli::X), (q2, Pauli::X)],
                &dets,
                &obs,
            )
            .unwrap();
            assert_eq!(
                sc.decoder.decode(&sig.detectors),
                sig.flips_logical,
                "d=5: X on data ({q1}, {q2}) not corrected"
            );
        }
    }
}

/// Quick (non-ignored) statistical check that DEM sampling reproduces the
/// tableau. The full-power version is `tests/qec_dem_audit.rs`.
#[test]
fn dem_sampling_matches_tableau_quick() {
    let sc = SurfaceCode::new(3, 3);
    let noise = NoiseModel::circuit_level(0.01, 0.01);
    let shots = 3000;
    let mut rng = StdRng::seed_from_u64(101);
    let t = sc.detector_rates(&noise, shots, SamplingMethod::Tableau, &mut rng);
    let f = sc.detector_rates(&noise, shots, SamplingMethod::DetectorErrorModel, &mut rng);
    for (k, (a, b)) in t.iter().zip(&f).enumerate() {
        let p = (a + b) / 2.0;
        let se = (p * (1.0 - p) * 2.0 / shots as f64).sqrt().max(1e-9);
        assert!(
            ((a - b) / se).abs() < 5.0,
            "detector {k}: tableau {a} vs dem {b}"
        );
    }
    let rt = sc.run_experiment(&noise, shots, SamplingMethod::Tableau, &mut rng);
    let rf = sc.run_experiment(&noise, shots, SamplingMethod::DetectorErrorModel, &mut rng);
    let (a, b) = (rt.logical_error_rate, rf.logical_error_rate);
    let p = (a + b) / 2.0;
    let se = (p * (1.0 - p) * 2.0 / shots as f64).sqrt().max(1e-9);
    assert!(((a - b) / se).abs() < 5.0, "decoded: tableau {a} vs dem {b}");
    // and noise is not ignored
    assert!(rt.logical_errors > 0 && rf.logical_errors > 0);
}

#[test]
fn larger_distance_suppresses_errors_below_threshold() {
    let mut rng = StdRng::seed_from_u64(5);
    let noise = NoiseModel::circuit_level(0.002, 0.002);
    let r3 = SurfaceCode::new(3, 3).run_experiment_dem(&noise, 20_000, &mut rng);
    let r5 = SurfaceCode::new(5, 5).run_experiment_dem(&noise, 20_000, &mut rng);
    assert!(
        r5.logical_errors < r3.logical_errors,
        "d=5 ({}) should beat d=3 ({}) at p=0.2%",
        r5.logical_errors,
        r3.logical_errors
    );
}
