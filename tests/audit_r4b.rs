//! Audit §16 (round 4, second pass): regression tests for findings.
use qsim_lab::simulability::support_bound;
use qsim_lab::{Gate, StateVectorF64};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn true_support_log2(n: usize, gates: &[Gate]) -> f64 {
    let mut sv = StateVectorF64::new(n);
    for g in gates {
        sv.apply_gate(g).unwrap();
    }
    let nnz = sv
        .amplitudes()
        .iter()
        .filter(|z| z.norm_sqr() > 1e-20)
        .count();
    (nnz as f64).log2()
}

#[test]
fn support_bound_constant_control_counterexample() {
    use Gate::*;
    // q0 = |0>: the Toffoli is the identity, q1 ends |0>, q2 = H-variable,
    // then H on q1: support 4. The old rule returned 1 (2^1 = 2 < 4).
    let g = [H(1), Cnot(1, 2), Ccx(0, 1, 2), Cnot(2, 1), H(1)];
    assert_eq!(true_support_log2(3, &g), 2.0);
    assert!(support_bound(3, &g) >= 2);
}

#[test]
fn support_bound_is_an_upper_bound_fuzz() {
    let mut rng = StdRng::seed_from_u64(16);
    for _ in 0..4000 {
        let n = rng.random_range(2..=6usize);
        let len = rng.random_range(1..=20usize);
        let mut g = Vec::new();
        for _ in 0..len {
            let q = |r: &mut StdRng| r.random_range(0..n);
            let a = q(&mut rng);
            let mut b = q(&mut rng);
            while b == a {
                b = q(&mut rng);
            }
            let mut c = q(&mut rng);
            while c == a || c == b {
                if n < 3 {
                    break;
                }
                c = q(&mut rng);
            }
            g.push(match rng.random_range(0..8) {
                0 => Gate::H(a),
                1 => Gate::X(a),
                2 => Gate::Cnot(a, b),
                3 if n >= 3 => Gate::Ccx(a, b, c),
                4 => Gate::Swap(a, b),
                5 => Gate::T(a),
                6 => Gate::Rx(a, 0.7),
                _ => Gate::Cz(a, b),
            });
        }
        let t = true_support_log2(n, &g);
        let b = support_bound(n, &g) as f64;
        assert!(b + 1e-9 >= t, "bound {b} < true {t} for {g:?}");
    }
}
