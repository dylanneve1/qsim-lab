//! Exactness of phase folding (`compile::phase_fold`): the output circuit
//! equals the input up to the tracked global phase (full unitary for small
//! n, random input states up to n = 10), the non-Clifford count never grows,
//! and the pass composes with the DAG peephole and the plan options.

#[path = "../common/mod.rs"]
mod common;

use common::*;
use num_complex::Complex64;
use proptest::prelude::*;
use qsim_lab::compile::plan::PlanOptions;
use qsim_lab::compile::{phase_fold, phase_fold_with_stats};
use qsim_lab::{Circuit, Gate, Op, StateVectorF64};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn non_clifford(c: &Circuit) -> usize {
    c.gates().filter(|g| !g.is_clifford()).count()
}

/// `|a - e^{iφ} b|_max` over amplitudes.
fn diff_with_phase(a: &StateVectorF64, b: &StateVectorF64, phase: f64) -> f64 {
    let ph = Complex64::from_polar(1.0, phase);
    a.amplitudes()
        .iter()
        .zip(b.amplitudes())
        .map(|(x, y)| (x - ph * y).norm())
        .fold(0.0, f64::max)
}

/// Random Clifford+T / Clifford+Rz circuit (optionally with non-affine
/// gates mixed in), biased to repeat parities.
fn random_ct<R: Rng>(n: usize, len: usize, rz: bool, extras: bool, rng: &mut R) -> Circuit {
    let mut c = Circuit::new(n);
    for _ in 0..len {
        let q = rng.random_range(0..n);
        let other = |rng: &mut R, not: &[usize]| loop {
            let x = rng.random_range(0..n);
            if !not.contains(&x) {
                break x;
            }
        };
        let k = rng.random_range(
            0..if n >= 3 && extras {
                20
            } else if n >= 2 {
                16
            } else {
                10
            },
        );
        let ang = if rng.random_bool(0.5) {
            rng.random_range(-8..8) as f64 * std::f64::consts::FRAC_PI_4
        } else {
            rng.random_range(-3.2..3.2)
        };
        let g = match k {
            0 => Gate::H(q),
            1 => Gate::S(q),
            2 | 3 => Gate::T(q),
            4 | 5 => Gate::Tdg(q),
            6 => Gate::X(q),
            7 => Gate::Z(q),
            8 => Gate::Y(q),
            9 => {
                if rz {
                    Gate::Rz(q, ang)
                } else {
                    Gate::Sdg(q)
                }
            }
            10..=12 => Gate::Cnot(q, other(rng, &[q])),
            13 => Gate::Cz(q, other(rng, &[q])),
            14 => Gate::Swap(q, other(rng, &[q])),
            15 => Gate::Cnot(q, other(rng, &[q])),
            16 => Gate::Ccx(q, other(rng, &[q]), {
                let b = other(rng, &[q]);
                if b == q {
                    unreachable!()
                } else {
                    b
                }
            }),
            17 => Gate::CPhase(q, other(rng, &[q]), ang),
            18 => Gate::Ry(q, ang),
            _ => Gate::Sx(q),
        };
        // Ccx needs three distinct qubits
        let g = if let Gate::Ccx(a, b, _) = g {
            let t = loop {
                let x = rng.random_range(0..n);
                if x != a && x != b {
                    break x;
                }
            };
            Gate::Ccx(a, b, t)
        } else {
            g
        };
        c.gate(g);
    }
    c
}

/// Prepares a random state with a universal prefix, then runs `body`.
fn run_on_random_state(prep: &Circuit, body: &Circuit) -> StateVectorF64 {
    let mut s = StateVectorF64::new(prep.num_qubits);
    s.apply_circuit(prep).unwrap();
    s.apply_circuit(body).unwrap();
    s
}

fn check_state(c: &Circuit, seed: u64) {
    let n = c.num_qubits;
    let o = phase_fold(c);
    let mut rng = StdRng::seed_from_u64(seed);
    let prep = random_universal(n, 4 * n + 6, &mut rng);
    let a = run_on_random_state(&prep, c);
    let b = run_on_random_state(&prep, &o.circuit);
    let d = diff_with_phase(&a, &b, o.global_phase);
    assert!(d < 1e-10, "diff {d}\n{:?}\n{:?}", c.ops, o.circuit.ops);
    assert!(non_clifford(&o.circuit) <= non_clifford(c));
}

fn check_unitary(c: &Circuit) {
    let n = c.num_qubits;
    let o = phase_fold(c);
    for col in 0..(1usize << n) {
        let mut prep = Circuit::new(n);
        for q in 0..n {
            if col >> q & 1 == 1 {
                prep.x(q);
            }
        }
        let a = run_on_random_state(&prep, c);
        let b = run_on_random_state(&prep, &o.circuit);
        let d = diff_with_phase(&a, &b, o.global_phase);
        assert!(
            d < 1e-10,
            "col {col} diff {d}\n{:?}\n{:?}",
            c.ops,
            o.circuit.ops
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn clifford_t_unitary_small(seed in any::<u64>(), n in 1usize..=4, len in 1usize..60) {
        let mut rng = StdRng::seed_from_u64(seed);
        check_unitary(&random_ct(n, len, false, false, &mut rng));
    }

    #[test]
    fn clifford_rz_unitary_small(seed in any::<u64>(), n in 1usize..=4, len in 1usize..60) {
        let mut rng = StdRng::seed_from_u64(seed);
        check_unitary(&random_ct(n, len, true, true, &mut rng));
    }

    #[test]
    fn clifford_t_state_to_10(seed in any::<u64>(), n in 1usize..=10, len in 1usize..200) {
        let mut rng = StdRng::seed_from_u64(seed);
        check_state(&random_ct(n, len, false, false, &mut rng), seed ^ 1);
    }

    #[test]
    fn clifford_rz_extras_state_to_10(seed in any::<u64>(), n in 1usize..=10, len in 1usize..200) {
        let mut rng = StdRng::seed_from_u64(seed);
        check_state(&random_ct(n, len, true, true, &mut rng), seed ^ 2);
    }

    #[test]
    fn idempotent_and_stats(seed in any::<u64>(), n in 2usize..=8, len in 1usize..150) {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = random_ct(n, len, false, false, &mut rng);
        let (o, st) = phase_fold_with_stats(&c);
        let (o2, _) = phase_fold_with_stats(&o.circuit);
        prop_assert_eq!(non_clifford(&o2.circuit), non_clifford(&o.circuit));
        prop_assert!(st.rotations_out <= st.rotations_in);
        prop_assert!(non_clifford(&o.circuit) <= non_clifford(&c));
    }

    /// peephole -> fold -> peephole (as the plan pass does) stays exact and
    /// does not increase the T count.
    #[test]
    fn composes_with_peephole(seed in any::<u64>(), n in 1usize..=8, len in 1usize..150) {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = random_ct(n, len, false, false, &mut rng);
        let p1 = qsim_lab::compile::optimize(&c);
        let f = phase_fold(&p1.circuit);
        let p2 = qsim_lab::compile::optimize(&f.circuit);
        let ph = p1.global_phase + f.global_phase + p2.global_phase;
        let prep = random_universal(n, 4 * n + 6, &mut rng);
        let a = run_on_random_state(&prep, &c);
        let b = run_on_random_state(&prep, &p2.circuit);
        prop_assert!(diff_with_phase(&a, &b, ph) < 1e-10);
        prop_assert!(non_clifford(&p2.circuit) <= non_clifford(&p1.circuit));
        prop_assert!(non_clifford(&p2.circuit) <= non_clifford(&c));
    }
}

#[test]
fn textbook_cases() {
    // T T = S; T Tdg = I; T across CNOT on a shared parity.
    let mut c = Circuit::new(2);
    c.t(0).cnot(0, 1).cnot(0, 1).t(0);
    let o = phase_fold(&c);
    assert_eq!(o.circuit.t_count(), 0);
    check_unitary(&c);

    // Classic: T(a) CNOT(a,b) Tdg(b) CNOT(a,b)... same parity merges.
    let mut c = Circuit::new(3);
    c.t(2).cnot(0, 1).cnot(1, 2).cnot(0, 1).t(2).cnot(1, 2).t(1);
    check_unitary(&c);
    assert!(phase_fold(&c).circuit.t_count() <= c.t_count());

    // across an H on a different wire it still merges
    let mut c = Circuit::new(2);
    c.t(1).h(0).t(1);
    let o = phase_fold(&c);
    assert_eq!(o.circuit.t_count(), 0);
    check_unitary(&c);

    // but not across an H on the same wire
    let mut c = Circuit::new(1);
    c.t(0).h(0).t(0);
    assert_eq!(phase_fold(&c).circuit.t_count(), 2);

    // Toffoli (7 T) decomposed twice with X-conjugation merges to identity
    let mut c = Circuit::new(1);
    c.t(0).x(0).t(0).x(0);
    check_unitary(&c);
}

#[test]
fn measurement_blocks_merging_on_its_wire() {
    let mut c = Circuit::new(2);
    c.t(0).h(1);
    c.ops.push(Op::Measure(0));
    c.t(0);
    let o = phase_fold(&c);
    assert_eq!(o.circuit.t_count(), 2);
}

#[test]
fn plan_option_keeps_simulate_identical() {
    use qsim_lab::compile::compile_unitary;
    let mut rng = StdRng::seed_from_u64(7);
    for n in [3, 6, 9] {
        let c = random_ct(n, 120, false, false, &mut rng);
        let base = compile_unitary(&c, PlanOptions::default()).unwrap();
        let folded = compile_unitary(
            &c,
            PlanOptions {
                phase_fold: true,
                ..PlanOptions::default()
            },
        )
        .unwrap();
        let a = base.factored::<f64>().unwrap();
        let b = folded.factored::<f64>().unwrap();
        for x in 0..(1u128 << n).min(64) {
            let d = (a.amplitude(x) - b.amplitude(x)).norm();
            assert!(d < 1e-10, "n={n} x={x} d={d}");
        }
    }
}
