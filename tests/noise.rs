//! Tests for stochastic noise channels, noise models, and teleportation with classical feedback.

use num_complex::Complex64;
use qsim_lab::circuit::Circuit;
use qsim_lab::gate::Gate;
use qsim_lab::noise::NoiseModel;
use qsim_lab::{Mps, StateVectorF64, Tableau};
use rand::rngs::StdRng;
use rand::SeedableRng;

// ----- Teleportation ---------------------------------------------------------

fn teleport_circuit(prep_gates: &[Gate], meas_basis_h: bool) -> Circuit {
    let mut c = Circuit::new(3);
    // Prepare payload on qubit 0
    for &g in prep_gates {
        c.gate(g);
    }
    // Entangle ancillas 1 and 2 in Bell state |00> + |11>
    c.h(1).cnot(1, 2);
    // Bell-basis measurement on qubits 0 and 1
    c.cnot(0, 1).h(0);
    c.measure(0); // classical bit 0
    c.measure(1); // classical bit 1
                  // Classical correction on Bob's qubit 2:
                  // If bit 1 == 1, apply X
    c.c_if(1, Gate::X(2));
    // If bit 0 == 1, apply Z
    c.c_if(0, Gate::Z(2));

    // Optional change of basis before measuring Bob's qubit 2
    if meas_basis_h {
        c.h(2);
    }
    c.measure(2); // classical bit 2
    c
}

#[test]
fn teleportation_on_tableau_all_six_stabilizer_states() {
    let mut rng = StdRng::seed_from_u64(42);
    // 6 eigenstates of X, Y, Z:
    // |0>: expect Z=+1 (meas 0)
    // |1>: expect Z=-1 (meas 1)
    // |+>: expect X=+1 (meas 0 after H)
    // |->: expect X=-1 (meas 1 after H)
    // |+i>: S |+>, expect Y=+1 (after Sdg then H: meas 0)
    // |-i>: S |->, expect Y=-1 (after Sdg then H: meas 1)
    let cases: Vec<(Vec<Gate>, bool, bool, &'static str)> = vec![
        (vec![], false, false, "|0>"),
        (vec![Gate::X(0)], false, true, "|1>"),
        (vec![Gate::H(0)], true, false, "|+>"),
        (vec![Gate::X(0), Gate::H(0)], true, true, "|->"),
        (vec![Gate::H(0), Gate::S(0)], false, false, "|+i>"),
        (
            vec![Gate::X(0), Gate::H(0), Gate::S(0)],
            false,
            true,
            "|-i>",
        ),
    ];

    for (prep, meas_h, expected_outcome, label) in cases {
        if label.contains('i') {
            // For Y eigenstates, measure by applying Sdg then H
            let mut c = Circuit::new(3);
            for &g in &prep {
                c.gate(g);
            }
            c.h(1).cnot(1, 2);
            c.cnot(0, 1).h(0);
            c.measure(0).measure(1);
            c.c_if(1, Gate::X(2));
            c.c_if(0, Gate::Z(2));
            c.sdg(2).h(2).measure(2);

            for _ in 0..30 {
                let mut tab = Tableau::new(3);
                let bits = c.run(&mut tab, &mut rng).unwrap();
                assert_eq!(
                    bits[2], expected_outcome,
                    "teleportation failed on tableau for {label}"
                );
            }
        } else {
            let c = teleport_circuit(&prep, meas_h);
            for _ in 0..30 {
                let mut tab = Tableau::new(3);
                let bits = c.run(&mut tab, &mut rng).unwrap();
                assert_eq!(
                    bits[2], expected_outcome,
                    "teleportation failed on tableau for {label}"
                );
            }
        }
    }
}

#[test]
fn teleportation_on_statevector_arbitrary_state() {
    let mut rng = StdRng::seed_from_u64(100);
    // Arbitrary single-qubit state: Ry(0.7) Rz(1.2) |0>
    let c = teleport_circuit(&[Gate::Rz(0, 1.2), Gate::Ry(0, 0.7)], false);

    // Let's check expectation value of Bob's qubit <Z_2> over many shots
    // Target state: cos(0.7/2)|0> + e^{i 1.2} sin(0.7/2)|1>
    // P(1) = sin^2(0.35) ≈ 0.11767
    let target_p1 = (0.35f64).sin().powi(2);

    let shots = 5000;
    let mut ones = 0;
    for _ in 0..shots {
        let mut sv = StateVectorF64::new(3);
        let bits = c.run(&mut sv, &mut rng).unwrap();
        if bits[2] {
            ones += 1;
        }
    }
    let p_obs = ones as f64 / shots as f64;
    assert!(
        (p_obs - target_p1).abs() < 0.02,
        "observed P(1) = {p_obs}, expected {target_p1}"
    );
}

#[test]
fn reset_restores_computational_zero() {
    let mut rng = StdRng::seed_from_u64(77);
    for n in [1, 3] {
        // Start in |111>
        let mut c = Circuit::new(n);
        for q in 0..n {
            c.x(q);
        }
        for q in 0..n {
            c.reset(q);
            c.measure(q);
        }

        // Tableau
        let mut tab = Tableau::new(n);
        let bits = c.run(&mut tab, &mut rng).unwrap();
        assert!(bits.iter().all(|&b| !b), "Tableau reset failed");

        // StateVector
        let mut sv = StateVectorF64::new(n);
        let bits = c.run(&mut sv, &mut rng).unwrap();
        assert!(bits.iter().all(|&b| !b), "StateVector reset failed");

        // MPS
        let mut mps = Mps::new(n, 16);
        let bits = c.run(&mut mps, &mut rng).unwrap();
        assert!(bits.iter().all(|&b| !b), "MPS reset failed");
    }
}

// ----- Density Matrix Reference & Noisy Trajectories -------------------------

/// Exact 2x2 density matrix for verifying single-qubit noise channels.
#[derive(Clone, Copy, Debug)]
struct DensityMatrix2x2 {
    rho: [[Complex64; 2]; 2],
}

impl DensityMatrix2x2 {
    fn zero() -> Self {
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        Self {
            rho: [[one, zero], [zero, zero]],
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn apply_gate(&mut self, g: &Gate) {
        let m = g.matrix_1q().expect("1q gate");
        let mut out = [[Complex64::default(); 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        out[i][j] += m[i][k] * self.rho[k][l] * m[j][l].conj();
                    }
                }
            }
        }
        self.rho = out;
    }

    #[allow(clippy::needless_range_loop)]
    fn depolarize(&mut self, p: f64) {
        let id_weight = 1.0 - p;
        let mut out = [[Complex64::default(); 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                out[i][j] += id_weight * self.rho[i][j];
            }
        }
        for g in [Gate::X(0), Gate::Y(0), Gate::Z(0)] {
            let m = g.matrix_1q().unwrap();
            let w = p / 3.0;
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            out[i][j] += w * m[i][k] * self.rho[k][l] * m[j][l].conj();
                        }
                    }
                }
            }
        }
        self.rho = out;
    }

    fn prob_one_with_readout(&self, p_meas: f64) -> f64 {
        let p1 = self.rho[1][1].re;
        let p0 = self.rho[0][0].re;
        (1.0 - p_meas) * p1 + p_meas * p0
    }
}

/// Test single-qubit depolarizing noise starting from |0> (Z-basis).
///
/// This is discriminating: starting from |0>, the depolarizing channel
/// produces P(|1>) = p/3 (from X and Y errors), and readout flip
/// further perturbs it. The noiseless case must give exactly zero errors.
#[test]
fn single_qubit_depolarizing_z_basis_matches_density_matrix() {
    let mut rng = StdRng::seed_from_u64(2026);

    // --- Z-basis test: start from |0>, measure in Z ---
    let p_depol = 0.18;
    let p_meas = 0.04;

    let mut c_z = Circuit::new(1);
    c_z.measure(0); // measure |0> directly (no H)

    let noise = NoiseModel::none().with_p1(p_depol).with_meas(p_meas);

    // Exact density-matrix expectation for |0> with depolarizing + readout:
    // After depolarizing: P(|1>) = p/3 (X and Y each flip |0> to |1>, Z doesn't)
    // Wait: depolarizing on |0> before measurement — but there's no gate to
    // attach the noise to. We need a gate.
    // Let's use: apply an identity-like gate (Z) that doesn't change |0>, then noise after.
    let mut c_z = Circuit::new(1);
    c_z.gate(Gate::Z(0)); // Z|0> = |0>, but triggers 1q depolarizing noise
    c_z.measure(0);

    let mut dm = DensityMatrix2x2::zero();
    dm.apply_gate(&Gate::Z(0));
    dm.depolarize(p_depol);
    let expected_p1_z = dm.prob_one_with_readout(p_meas);

    let shots = 20000;

    // Test on Tableau
    let mut ones_tab = 0;
    for _ in 0..shots {
        let mut tab = Tableau::new(1);
        let bits = c_z.run_noisy(&mut tab, &noise, &mut rng).unwrap();
        if bits[0] {
            ones_tab += 1;
        }
    }
    let p_tab_z = ones_tab as f64 / shots as f64;
    assert!(
        (p_tab_z - expected_p1_z).abs() < 0.012,
        "Tableau Z-basis: observed {p_tab_z}, expected {expected_p1_z}"
    );

    // Test on StateVector
    let mut ones_sv = 0;
    for _ in 0..shots {
        let mut sv = StateVectorF64::new(1);
        let bits = c_z.run_noisy(&mut sv, &noise, &mut rng).unwrap();
        if bits[0] {
            ones_sv += 1;
        }
    }
    let p_sv_z = ones_sv as f64 / shots as f64;
    assert!(
        (p_sv_z - expected_p1_z).abs() < 0.012,
        "StateVector Z-basis: observed {p_sv_z}, expected {expected_p1_z}"
    );

    // --- Noiseless sanity check: p=0 must give exactly zero errors ---
    let noise_none = NoiseModel::none();
    for _ in 0..500 {
        let mut tab = Tableau::new(1);
        let bits = c_z.run_noisy(&mut tab, &noise_none, &mut rng).unwrap();
        assert!(!bits[0], "noiseless |0> measured as 1");
    }
}

/// Test single-qubit depolarizing noise in X-basis (H|0> = |+>, then H, measure).
///
/// This measures in the X basis: H followed by measurement effectively measures X.
/// Depolarizing noise after the first H gives a known P(X=-1) = p/3 (from Z and Y).
#[test]
fn single_qubit_depolarizing_x_basis_matches_density_matrix() {
    let mut rng = StdRng::seed_from_u64(3333);
    let p_depol = 0.15;
    let p_meas = 0.03;

    // H|0> = |+>, then H again before measurement: measures X eigenvalue.
    // Depolarizing after first H: Y and Z cause X flip, X doesn't.
    let mut c_x = Circuit::new(1);
    c_x.h(0).h(0).measure(0);
    // After first H: |+>. Depolarize. Then H: maps X eigenstates to Z eigenstates.

    let noise = NoiseModel::none().with_p1(p_depol).with_meas(p_meas);

    let mut dm = DensityMatrix2x2::zero();
    dm.apply_gate(&Gate::H(0));
    dm.depolarize(p_depol);
    dm.apply_gate(&Gate::H(0)); // noise only after first H (no noise after second H
    // because noise model applies after the second H too)

    // Actually, with noise model, depolarizing is applied after BOTH H gates.
    // First H: |0> → |+>, then depolarize(p)
    // Second H: transform, then depolarize(p) again
    // So density matrix should have two rounds of depolarizing:
    let mut dm2 = DensityMatrix2x2::zero();
    dm2.apply_gate(&Gate::H(0));
    dm2.depolarize(p_depol);
    dm2.apply_gate(&Gate::H(0));
    dm2.depolarize(p_depol);
    let expected_p1_x = dm2.prob_one_with_readout(p_meas);

    let shots = 20000;

    let mut ones = 0;
    for _ in 0..shots {
        let mut tab = Tableau::new(1);
        let bits = c_x.run_noisy(&mut tab, &noise, &mut rng).unwrap();
        if bits[0] {
            ones += 1;
        }
    }
    let p_obs = ones as f64 / shots as f64;
    assert!(
        (p_obs - expected_p1_x).abs() < 0.012,
        "Tableau X-basis: observed {p_obs}, expected {expected_p1_x}"
    );

    // Noiseless: H H = I, so |0> → |0>, must get exactly 0
    let noise_none = NoiseModel::none();
    for _ in 0..500 {
        let mut tab = Tableau::new(1);
        let bits = c_x.run_noisy(&mut tab, &noise_none, &mut rng).unwrap();
        assert!(!bits[0], "noiseless H·H|0> measured as 1");
    }
}

#[test]
fn two_qubit_depolarizing_noise_after_cnot() {
    let mut rng = StdRng::seed_from_u64(999);
    let p_2q = 0.15;

    // Prepare Bell state: H(0), CNOT(0, 1)
    // Then measure Z_0 and Z_1: outcomes should agree unless error occurs
    let mut c = Circuit::new(2);
    c.h(0).cnot(0, 1).measure(0).measure(1);

    let noise = NoiseModel::none().with_p2(p_2q);

    // Analytical expectation for <Z_0 Z_1>:
    // In ideal Bell state, Z_0 Z_1 = +1.
    // Of the 15 Pauli pairs, 7 commute with Z_0 Z_1 and 8 anticommute.
    // So <Z_0 Z_1> = (1 - p) + (7/15)p - (8/15)p = 1 - (16/15)p.
    // P(outcomes differ) = (1 - <Z_0 Z_1>) / 2 = (8/15) * p.
    let expected_diff_p = (8.0 / 15.0) * p_2q;

    let shots = 15000;

    // Test on Tableau
    let mut diff_count_tab = 0;
    for _ in 0..shots {
        let mut tab = Tableau::new(2);
        let bits = c.run_noisy(&mut tab, &noise, &mut rng).unwrap();
        if bits[0] != bits[1] {
            diff_count_tab += 1;
        }
    }
    let p_obs_tab = diff_count_tab as f64 / shots as f64;
    assert!(
        (p_obs_tab - expected_diff_p).abs() < 0.015,
        "Tableau: observed difference prob {p_obs_tab}, expected {expected_diff_p}"
    );

    // Test on StateVector
    let mut diff_count_sv = 0;
    for _ in 0..shots {
        let mut sv = StateVectorF64::new(2);
        let bits = c.run_noisy(&mut sv, &noise, &mut rng).unwrap();
        if bits[0] != bits[1] {
            diff_count_sv += 1;
        }
    }
    let p_obs_sv = diff_count_sv as f64 / shots as f64;
    assert!(
        (p_obs_sv - expected_diff_p).abs() < 0.015,
        "StateVector: observed difference prob {p_obs_sv}, expected {expected_diff_p}"
    );
}

#[test]
fn explicit_stochastic_pauli_channels_in_circuit() {
    let mut rng = StdRng::seed_from_u64(12345);
    let p = 0.25;

    // Start in |0>, apply X flip with probability p, then measure
    let mut c = Circuit::new(1);
    c.x_flip(0, p).measure(0);

    let shots = 10000;
    let mut ones = 0;
    for _ in 0..shots {
        let mut tab = Tableau::new(1);
        let bits = c.run(&mut tab, &mut rng).unwrap();
        if bits[0] {
            ones += 1;
        }
    }
    let p_obs = ones as f64 / shots as f64;
    assert!(
        (p_obs - p).abs() < 0.015,
        "x_flip: observed {p_obs}, expected {p}"
    );
}
