//! Repetition code memory experiment tests.

use qsim_lab::gate::Gate;
use qsim_lab::noise::NoiseModel;
use qsim_lab::qec::RepetitionCode;
use qsim_lab::Tableau;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn repetition_noiseless_has_zero_logical_errors() {
    let mut rng = StdRng::seed_from_u64(1);
    for d in [3, 5, 7] {
        let code = RepetitionCode::new(d, d);
        let res = code.run_experiment(&NoiseModel::none(), 500, &mut rng);
        assert_eq!(
            res.logical_errors, 0,
            "noiseless repetition code d={d} had errors"
        );
    }
}

#[test]
fn single_injected_data_errors_always_corrected() {
    let mut rng = StdRng::seed_from_u64(2);
    for d in [3, 5, 7] {
        let code = RepetitionCode::new(d, d);
        let base_circuit = code.build_circuit();
        let n = RepetitionCode::num_qubits(d);

        // Inject an X error on every possible data qubit
        for data_q in 0..d {
            let mut c = base_circuit.clone();
            // Prepend or insert X on data_q before measurements
            // In base_circuit, data qubits start in |0>.
            // Let's insert X(data_q) as the very first operation:
            c.ops
                .insert(0, qsim_lab::circuit::Op::Gate(Gate::X(data_q)));

            let mut tab = Tableau::new(n);
            let raw_bits = c.run(&mut tab, &mut rng).unwrap();
            let (defects, raw_logical) = code.extract_defects(&raw_bits);
            let pred_flip = code.decoder.decode(&defects);
            let corrected = raw_logical ^ pred_flip;

            assert!(
                !corrected,
                "d={d}: single injected error on data qubit {data_q} caused logical error"
            );
        }
    }
}

#[test]
fn single_injected_measurement_errors_always_corrected() {
    let mut rng = StdRng::seed_from_u64(3);
    for d in [3, 5, 7] {
        let rounds = d;
        let code = RepetitionCode::new(d, rounds);
        let base_circuit = code.build_circuit();
        let n = RepetitionCode::num_qubits(d);

        // Run noiseless circuit to get clean baseline
        let mut tab = Tableau::new(n);
        let clean_bits = base_circuit.run(&mut tab, &mut rng).unwrap();

        // Invert any single ancilla measurement in the output
        let num_anc_meas = rounds * (d - 1);
        for meas_idx in 0..num_anc_meas {
            let mut corrupted_bits = clean_bits.clone();
            corrupted_bits[meas_idx] = !corrupted_bits[meas_idx];

            let (defects, raw_logical) = code.extract_defects(&corrupted_bits);
            let pred_flip = code.decoder.decode(&defects);
            let corrected = raw_logical ^ pred_flip;

            assert!(
                !corrected,
                "d={d}: single measurement error at index {meas_idx} caused logical error"
            );
        }
    }
}

#[test]
fn repetition_subthreshold_error_suppression() {
    let mut rng = StdRng::seed_from_u64(42);
    // At low physical error rate, larger distance must have strictly lower logical error rate
    let p = 0.02; // 2% circuit-level depolarizing
    let noise = NoiseModel::circuit_level(p, p);
    let shots = 3000;

    let r3 = RepetitionCode::new(3, 3).run_experiment(&noise, shots, &mut rng);
    let r5 = RepetitionCode::new(5, 5).run_experiment(&noise, shots, &mut rng);
    let r7 = RepetitionCode::new(7, 7).run_experiment(&noise, shots, &mut rng);

    // Verify error suppression: P_L(d=7) < P_L(d=5) < P_L(d=3)
    assert!(
        r5.logical_error_rate < r3.logical_error_rate,
        "Expected P_L(d=5) < P_L(d=3) at p={p}: got {} vs {}",
        r5.logical_error_rate,
        r3.logical_error_rate
    );
    assert!(
        r7.logical_error_rate < r5.logical_error_rate,
        "Expected P_L(d=7) < P_L(d=5) at p={p}: got {} vs {}",
        r7.logical_error_rate,
        r5.logical_error_rate
    );
}
