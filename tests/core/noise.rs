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

/// Fraction of shots (over `shots` fresh simulators from `make`) in which
/// `pred(bits)` holds.
fn frequency<S: qsim_lab::circuit::Simulator>(
    c: &Circuit,
    noise: &NoiseModel,
    shots: usize,
    rng: &mut StdRng,
    mut make: impl FnMut() -> S,
    pred: impl Fn(&[bool]) -> bool,
) -> f64 {
    let mut hits = 0;
    for _ in 0..shots {
        let mut sim = make();
        let bits = c.run_noisy(&mut sim, noise, rng).unwrap();
        hits += pred(&bits) as usize;
    }
    hits as f64 / shots as f64
}

/// Asserts `observed` is within 5 binomial standard errors of `expected`,
/// and that every value in `alternatives` (what a broken implementation
/// would produce) is more than 8 standard errors away from `expected`, so
/// the check is discriminating.
fn assert_binomial(
    label: &str,
    observed: f64,
    expected: f64,
    shots: usize,
    alternatives: &[(&str, f64)],
) {
    let se = (expected * (1.0 - expected) / shots as f64).sqrt();
    assert!(
        (observed - expected).abs() < 5.0 * se,
        "{label}: observed {observed:.5}, expected {expected:.5} (5σ = {:.5})",
        5.0 * se
    );
    for (what, alt) in alternatives {
        assert!(
            (alt - expected).abs() > 8.0 * se,
            "{label}: test cannot tell the correct model from '{what}' ({alt:.5} vs {expected:.5})"
        );
    }
}

/// Single-qubit depolarizing + readout noise, starting from |0>, checked in
/// the Z basis AND the X basis on the tableau, the state vector and the MPS
/// against an exact density-matrix calculation.
///
/// * Z basis: `Z(0)` (acts trivially on |0>, but carries 1q noise) then
///   measure. Only the X and Y components of the depolarizing channel and
///   the readout flip can produce a 1.
/// * X basis: `H, H`, measure (noise after both H). The Z and Y components
///   after the first H and the X and Y components after the second H flip
///   the outcome.
///
/// Each check also asserts that ignoring the gate noise, ignoring the
/// readout noise, or ignoring all noise would give a value more than 8σ
/// away, so a backend that skipped any of them would fail.
#[test]
fn single_qubit_depolarizing_and_readout_from_zero_in_z_and_x_basis() {
    let mut rng = StdRng::seed_from_u64(2026);
    let p = 0.18;
    let pm = 0.06;
    let noise = NoiseModel::none().with_p1(p).with_meas(pm);
    let shots = 20_000;

    let mut cz = Circuit::new(1);
    cz.gate(Gate::Z(0)).measure(0);
    let mut cx = Circuit::new(1);
    cx.h(0).h(0).measure(0);

    let exact = |gates: &[Gate], p: f64, pm: f64| {
        let mut dm = DensityMatrix2x2::zero();
        for g in gates {
            dm.apply_gate(g);
            dm.depolarize(p);
        }
        dm.prob_one_with_readout(pm)
    };
    for (label, c, gates) in [
        ("Z basis", &cz, vec![Gate::Z(0)]),
        ("X basis", &cx, vec![Gate::H(0), Gate::H(0)]),
    ] {
        let expected = exact(&gates, p, pm);
        let alternatives = [
            ("no noise", exact(&gates, 0.0, 0.0)),
            ("no gate noise", exact(&gates, 0.0, pm)),
            ("no readout noise", exact(&gates, p, 0.0)),
        ];
        let one = |b: &[bool]| b[0];
        let f_tab = frequency(c, &noise, shots, &mut rng, || Tableau::new(1), one);
        assert_binomial(
            &format!("tableau {label}"),
            f_tab,
            expected,
            shots,
            &alternatives,
        );
        let f_sv = frequency(c, &noise, shots, &mut rng, || StateVectorF64::new(1), one);
        assert_binomial(
            &format!("statevector {label}"),
            f_sv,
            expected,
            shots,
            &alternatives,
        );
        let f_mps = frequency(c, &noise, shots, &mut rng, || Mps::new(1, 4), one);
        assert_binomial(
            &format!("mps {label}"),
            f_mps,
            expected,
            shots,
            &alternatives,
        );
    }
}

/// Two-qubit depolarizing noise after a CNOT that prepares a Bell state,
/// checked through both stabilizers ZZ and XX on the tableau and the state
/// vector. Of the 15 non-identity Paulis, 8 anticommute with ZZ and 8 with
/// XX (different subsets), so P(parity flips) = 8p/15 in both bases; a
/// channel that only inserted X-type (or only Z-type) errors would leave
/// one of the two parities untouched and fail.
#[test]
fn two_qubit_depolarizing_after_cnot_zz_and_xx_parity() {
    let mut rng = StdRng::seed_from_u64(999);
    let p2 = 0.15;
    // p_1q = 0: the H gates are noiseless, so only the CNOT's channel acts.
    let noise = NoiseModel::none().with_p2(p2);
    let shots = 15_000;
    let expected = 8.0 / 15.0 * p2;

    let mut czz = Circuit::new(2);
    czz.h(0).cnot(0, 1).measure(0).measure(1);
    let mut cxx = Circuit::new(2);
    cxx.h(0).cnot(0, 1).h(0).h(1).measure(0).measure(1);

    let differ = |b: &[bool]| b[0] != b[1];
    for (label, c) in [("ZZ", &czz), ("XX", &cxx)] {
        let alternatives = [("no noise", 0.0)];
        let f_tab = frequency(c, &noise, shots, &mut rng, || Tableau::new(2), differ);
        assert_binomial(
            &format!("tableau {label}"),
            f_tab,
            expected,
            shots,
            &alternatives,
        );
        let f_sv = frequency(
            c,
            &noise,
            shots,
            &mut rng,
            || StateVectorF64::new(2),
            differ,
        );
        assert_binomial(
            &format!("statevector {label}"),
            f_sv,
            expected,
            shots,
            &alternatives,
        );
    }
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
