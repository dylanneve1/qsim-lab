//! Differential tests for the simulability study (src/simulability.rs):
//! every engine answers `<Z^{⊗n}>` like the reference state vector, and the
//! cheap features are valid bounds.

use qsim_lab::adaptive;
use qsim_lab::simulability::{build, features, run_engine, support_bound, Spec, ENGINES};
use qsim_lab::{Circuit, Gate, Mps, SparseState, StateVectorF64};

fn reference_parity(c: &Circuit) -> f64 {
    let mut sv = StateVectorF64::new(c.num_qubits);
    sv.apply_circuit(c).unwrap();
    sv.amplitudes()
        .iter()
        .enumerate()
        .map(|(x, a)| {
            if x.count_ones() % 2 == 1 {
                -a.norm_sqr()
            } else {
                a.norm_sqr()
            }
        })
        .sum()
}

const SPECS: &[&str] = &[
    "ct:n=9,L=3,t=0,nn=1",
    "ct:n=9,L=3,t=5,nn=1",
    "ct:n=10,L=4,t=12,nn=0",
    "brick:n=8,D=3,nn=1",
    "brick:n=9,D=2,nn=0",
    "arith:bits=3,h=1,reps=2",
    "arith:bits=4,h=0,reps=1",
    "arith:bits=3,h=3,reps=3",
    "qaoa:n=8,p=2,deg=3,nn=0",
    "qaoa:n=9,p=1,deg=2,nn=1",
];

#[test]
fn every_engine_matches_reference() {
    for spec in SPECS {
        for seed in 1..=3 {
            let c = build(&Spec::parse(spec).unwrap(), seed).unwrap();
            let want = reference_parity(&c);
            let clifford = c.gates().all(|g| g.is_clifford());
            for e in ENGINES {
                if e == "tableau" && !clifford {
                    assert!(run_engine(e, &c, 1 << 30).is_err());
                    continue;
                }
                let got = run_engine(e, &c, 1 << 30)
                    .unwrap_or_else(|err| panic!("{e} on {spec} s{seed}: {err:?}"))
                    .value;
                assert!(
                    (got - want).abs() < 1e-9,
                    "{e} on {spec} seed {seed}: {got} vs {want}"
                );
            }
        }
    }
}

#[test]
fn mps_z_product_matches_statevector() {
    let c = build(&Spec::parse("brick:n=7,D=4,nn=0").unwrap(), 5).unwrap();
    let mut sv = StateVectorF64::new(7);
    sv.apply_circuit(&c).unwrap();
    let mut m = Mps::new(7, 1 << 10);
    for g in c.gates() {
        m.apply_gate(g).unwrap();
    }
    for qs in [
        vec![0],
        vec![3],
        vec![1, 4],
        vec![0, 2, 6],
        vec![0, 1, 2, 3, 4, 5, 6],
    ] {
        let want: f64 = sv
            .amplitudes()
            .iter()
            .enumerate()
            .map(|(x, a)| {
                let par = qs.iter().filter(|&&q| x >> q & 1 == 1).count() % 2;
                if par == 1 {
                    -a.norm_sqr()
                } else {
                    a.norm_sqr()
                }
            })
            .sum();
        assert!(
            (m.expectation_z_product(&qs) - want).abs() < 1e-10,
            "{qs:?}"
        );
    }
}

#[test]
fn profile_ends_at_active_dimension() {
    for spec in SPECS {
        let c = build(&Spec::parse(spec).unwrap(), 2).unwrap();
        let prof = adaptive::active_dimension_profile(&c).unwrap();
        assert_eq!(
            prof.last().copied().unwrap_or(0),
            adaptive::active_dimension(&c).unwrap()
        );
        assert!(prof.windows(2).all(|w| w[0] <= w[1] && w[1] <= w[0] + 1));
    }
}

#[test]
fn bounds_hold() {
    // support bound >= log2 nnz; bond bound >= log2 max MPS bond.
    for spec in SPECS {
        for seed in 1..=3 {
            let c = build(&Spec::parse(spec).unwrap(), seed).unwrap();
            let f = features(&c, true).unwrap();
            let mut s = SparseState::new(c.num_qubits);
            let mut m = Mps::new(c.num_qubits, 1 << 12);
            for g in c.gates() {
                s.apply_gate(g).unwrap();
                m.apply_gate(g).unwrap();
            }
            let nnz = s.iter().filter(|(_, a)| a.norm_sqr() > 1e-20).count();
            assert!(
                nnz <= 1usize << f.sup,
                "{spec} s{seed}: nnz {nnz} > 2^{}",
                f.sup
            );
            assert!(
                m.max_bond_dim() <= 1usize << f.chi_bits,
                "{spec} s{seed}: bond {} > 2^{}",
                m.max_bond_dim(),
                f.chi_bits
            );
            assert!(f.d <= f.rotations && f.d <= c.num_qubits);
        }
    }
}

#[test]
fn support_bound_basic_cases() {
    use Gate::*;
    // H on 3 qubits then a CNOT ladder: support 2^3.
    let g = [H(0), H(1), H(2), Cnot(0, 3), Cnot(1, 4), Ccx(0, 1, 5)];
    assert_eq!(support_bound(6, &g), 3);
    // Permutations alone: support 1.
    assert_eq!(
        support_bound(4, &[X(0), Cnot(0, 1), Ccx(0, 1, 2), Swap(2, 3)]),
        0
    );
    // Diagonal gates do not branch.
    assert_eq!(support_bound(2, &[T(0), Cz(0, 1), Rz(1, 0.3)]), 0);
}

#[test]
fn mps_svd_nonconvergence_regression() {
    // faer's thin SVD failed to converge on this circuit (the MPS engine
    // panicked); exact <Z^n> is 0 (state vector, frame and dense agree).
    let c = build(&Spec::parse("ct:n=24,L=32,t=16,nn=1").unwrap(), 1).unwrap();
    let r = run_engine("mps", &c, 1 << 30).expect("mps runs");
    assert!(r.value.abs() < 1e-9, "{}", r.value);
}

#[test]
fn z_product_vanishing_certificate_is_sound() {
    // Whenever the O(gates·n) certificate says <Z^n> = 0, it is 0.
    let mut certified = 0;
    for spec in SPECS {
        for seed in 1..=4 {
            let c = build(&Spec::parse(spec).unwrap(), seed).unwrap();
            let all: Vec<usize> = (0..c.num_qubits).collect();
            if adaptive::z_product_vanishes(&c, &all).unwrap() {
                certified += 1;
                assert!(reference_parity(&c).abs() < 1e-12, "{spec} s{seed}");
            }
        }
    }
    assert!(certified > 0);
}
