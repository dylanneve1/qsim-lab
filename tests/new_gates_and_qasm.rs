//! Cross-checks for the gates added with OpenQASM support (Sx, Sxdg, U, ISwap,
//! ISwapdg): decompositions against the matrices, MPS and tableau against the
//! state vector, a QASM round trip compared by state overlap (not just gate
//! counts), and the parameter-expression grammar.
use qsim_lab::circuit::Circuit;
use qsim_lab::gate::Gate;
use qsim_lab::{Mps, StateVectorF64, Tableau};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn sv_of(c: &Circuit) -> StateVectorF64 {
    let mut sv = StateVectorF64::new(c.num_qubits);
    let mut rng = StdRng::seed_from_u64(0);
    c.run(&mut sv, &mut rng).unwrap();
    sv
}

fn overlap(a: &StateVectorF64, b: &StateVectorF64) -> f64 {
    let mut s = num_complex::Complex64::new(0.0, 0.0);
    for i in 0..a.amplitudes().len() {
        s += a.amplitude(i).conj() * b.amplitude(i);
    }
    s.norm()
}

fn new_gate(rng: &mut StdRng, n: usize) -> Gate {
    let q = rng.random_range(0..n);
    let mut r = rng.random_range(0..n);
    while r == q {
        r = rng.random_range(0..n);
    }
    let t = |rng: &mut StdRng| rng.random_range(-3.2..3.2);
    match rng.random_range(0..10) {
        0 => Gate::Sx(q),
        1 => Gate::Sxdg(q),
        2 => Gate::U(q, t(rng), t(rng), t(rng)),
        3 => Gate::ISwap(q, r),
        4 => Gate::ISwapdg(q, r),
        5 => Gate::H(q),
        6 => Gate::Cnot(q, r),
        7 => Gate::T(q),
        8 => Gate::Ry(q, t(rng)),
        _ => Gate::Cz(q, r),
    }
}

#[test]
fn decompositions_match_matrices() {
    let mut rng = StdRng::seed_from_u64(1);
    for _ in 0..200 {
        let n = 3;
        let mut prep = Circuit::new(n);
        for q in 0..n {
            prep.ry(q, rng.random_range(0.0..3.0))
                .rz(q, rng.random_range(0.0..3.0));
        }
        prep.cnot(0, 1).cnot(1, 2);
        let g = new_gate(&mut rng, n);
        let mut a = prep.clone();
        a.ops.push(qsim_lab::circuit::Op::Gate(g));
        let mut b = prep.clone();
        for d in g.decompose_to_clifford_rz() {
            b.ops.push(qsim_lab::circuit::Op::Gate(d));
        }
        let o = overlap(&sv_of(&a), &sv_of(&b));
        assert!((o - 1.0).abs() < 1e-10, "{g:?}: overlap {o}");
        // inverse
        let mut c = a.clone();
        c.ops.push(qsim_lab::circuit::Op::Gate(g.inverse()));
        let o2 = overlap(&sv_of(&c), &sv_of(&prep));
        assert!((o2 - 1.0).abs() < 1e-10, "{g:?} inverse: overlap {o2}");
    }
}

#[test]
fn mps_matches_sv_with_new_gates() {
    let mut rng = StdRng::seed_from_u64(2);
    for _ in 0..50 {
        let n = 5;
        let mut c = Circuit::new(n);
        for _ in 0..40 {
            c.ops
                .push(qsim_lab::circuit::Op::Gate(new_gate(&mut rng, n)));
        }
        let sv = sv_of(&c);
        let mut m = Mps::new(n, 64);
        let mut r2 = StdRng::seed_from_u64(0);
        c.run(&mut m, &mut r2).unwrap();
        let mut s = num_complex::Complex64::new(0.0, 0.0);
        for i in 0..(1usize << n) {
            s += sv.amplitude(i).conj() * m.amplitude(i as u128);
        }
        assert!((s.norm() - 1.0).abs() < 1e-9, "overlap {}", s.norm());
    }
}

#[test]
fn tableau_matches_sv_with_new_clifford_gates() {
    let mut rng = StdRng::seed_from_u64(3);
    for _ in 0..200 {
        let n = 4;
        let mut c = Circuit::new(n);
        for _ in 0..30 {
            let q = rng.random_range(0..n);
            let mut r = rng.random_range(0..n);
            while r == q {
                r = rng.random_range(0..n);
            }
            let g = match rng.random_range(0..8) {
                0 => Gate::Sx(q),
                1 => Gate::Sxdg(q),
                2 => Gate::ISwap(q, r),
                3 => Gate::ISwapdg(q, r),
                4 => Gate::H(q),
                5 => Gate::S(q),
                6 => Gate::Cnot(q, r),
                _ => Gate::Cz(q, r),
            };
            c.ops.push(qsim_lab::circuit::Op::Gate(g));
        }
        let probs = sv_of(&c).probabilities();
        // Measure all on tableau many times; every outcome must have nonzero SV probability,
        // and the support size must be a power of two matching uniform weights.
        let mut cm = c.clone();
        cm.measure_all();
        let mut seen = std::collections::HashSet::new();
        for s in 0..64 {
            let mut t = Tableau::new(n);
            let mut r = StdRng::seed_from_u64(s);
            let bits = cm.run(&mut t, &mut r).unwrap();
            let idx: usize = bits
                .iter()
                .enumerate()
                .map(|(i, &b)| (b as usize) << i)
                .sum();
            assert!(
                probs[idx] > 1e-9,
                "tableau outcome {idx} has SV prob {}",
                probs[idx]
            );
            seen.insert(idx);
        }
        let support = probs.iter().filter(|&&p| p > 1e-9).count();
        for (i, &p) in probs.iter().enumerate() {
            if p > 1e-9 {
                assert!(
                    (p - 1.0 / support as f64).abs() < 1e-9,
                    "non-uniform at {i}"
                );
            }
        }
    }
}

#[test]
fn qasm_roundtrip_preserves_state() {
    let mut rng = StdRng::seed_from_u64(4);
    for _ in 0..50 {
        let n = 4;
        let mut c = Circuit::new(n);
        for _ in 0..40 {
            c.ops
                .push(qsim_lab::circuit::Op::Gate(new_gate(&mut rng, n)));
        }
        let back = Circuit::from_qasm(&c.to_qasm().unwrap()).unwrap();
        let o = overlap(&sv_of(&c), &sv_of(&back));
        assert!(
            (o - 1.0).abs() < 1e-9,
            "roundtrip overlap {o}\n{}",
            c.to_qasm().unwrap()
        );
    }
}

fn rz_angle(src: &str) -> f64 {
    let c = Circuit::from_qasm(&format!(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\nrz({src}) q[0];\n"
    ))
    .unwrap();
    match c.ops[0] {
        qsim_lab::circuit::Op::Gate(Gate::Rz(_, t)) => t,
        ref o => panic!("unexpected {o:?}"),
    }
}

#[test]
fn qasm_parameter_expressions() {
    use std::f64::consts::PI;
    let cases = [
        ("pi/2", PI / 2.0),
        ("-pi/4", -PI / 4.0),
        ("3*pi/4", 3.0 * PI / 4.0),
        ("1/2*pi", PI / 2.0),
        ("pi/2/2", PI / 4.0),
        ("0.1", 0.1),
        ("1e-3", 1e-3),
        ("2.5E+2", 250.0),
        ("pi-0.1", PI - 0.1),
        ("pi + pi/2", 1.5 * PI),
        ("-(pi/2)", -PI / 2.0),
        ("((pi))/2", PI / 2.0),
        ("2*-pi", -2.0 * PI),
        ("2^3", 8.0),
        ("2^3^2", 512.0),
        ("-2^2", -4.0),
        ("cos(0)", 1.0),
        ("sqrt(4)*pi/8", PI / 4.0),
        ("ln(exp(1.5))", 1.5),
        ("sin(pi/2) + tan(0)", 1.0),
    ];
    let mut bad = vec![];
    for (s, want) in cases {
        let got = rz_angle(s);
        if (got - want).abs() > 1e-12 {
            bad.push(format!("{s}: got {got}, want {want}"));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn qasm_nested_parentheses_and_multi_param_gates() {
    use std::f64::consts::PI;
    let c = Circuit::from_qasm(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\n\
         rz((pi)/2) q[0];\nu3(pi/2, (0.1+0.2)*2, -(pi)) q[1];\n",
    )
    .unwrap();
    match c.ops[0] {
        qsim_lab::circuit::Op::Gate(Gate::Rz(0, t)) => assert!((t - PI / 2.0).abs() < 1e-12),
        ref o => panic!("unexpected {o:?}"),
    }
    match c.ops[1] {
        qsim_lab::circuit::Op::Gate(Gate::U(1, th, ph, la)) => {
            assert!((th - PI / 2.0).abs() < 1e-12);
            assert!((ph - 0.6).abs() < 1e-12);
            assert!((la + PI).abs() < 1e-12);
        }
        ref o => panic!("unexpected {o:?}"),
    }
}

#[test]
fn qasm_rejects_malformed_expressions() {
    for bad in ["pi/", "2**3", "foo(1)", "(pi", "pi)", "1/0", "sin 1"] {
        let src = format!("OPENQASM 2.0;\nqreg q[1];\nrz({bad}) q[0];\n");
        assert!(Circuit::from_qasm(&src).is_err(), "accepted '{bad}'");
    }
}

#[test]
fn qasm_repeated_measurements_get_their_own_bits() {
    let mut c = Circuit::new(2);
    c.h(0)
        .measure(0)
        .measure(0)
        .cnot(0, 1)
        .measure(1)
        .measure(0);
    let q = c.to_qasm().unwrap();
    assert!(q.contains("creg c[4];"), "{q}");
    for k in 0..4 {
        assert!(q.contains(&format!("-> c[{k}];")), "{q}");
    }
    let back = Circuit::from_qasm(&q).unwrap();
    assert_eq!(back.ops, c.ops);
}

#[test]
fn qasm_rejects_ops_it_cannot_express() {
    let mut c = Circuit::new(2);
    c.h(0).measure(0).c_if(0, Gate::X(1));
    assert!(
        c.to_qasm().is_err(),
        "classically conditioned gate was silently dropped"
    );
    let mut d = Circuit::new(1);
    d.x_flip(0, 0.1);
    assert!(d.to_qasm().is_err(), "noise channel was silently dropped");
}

#[test]
fn qasm_angles_round_trip_exactly() {
    let mut c = Circuit::new(2);
    c.rz(0, 1e-20)
        .rx(1, std::f64::consts::PI / 3.0)
        .u(0, 0.1, -2.5e-9, 3.0)
        .cphase(0, 1, 0.123_456_789_012_345_68);
    let back = Circuit::from_qasm(&c.to_qasm().unwrap()).unwrap();
    assert_eq!(back.ops, c.ops);
}

#[test]
fn blocked_executor_handles_new_gates() {
    use qsim_lab::blocked::BlockConfig;
    let mut rng = StdRng::seed_from_u64(5);
    for _ in 0..50 {
        let n = 6;
        let mut c = Circuit::new(n);
        for _ in 0..60 {
            c.ops
                .push(qsim_lab::circuit::Op::Gate(new_gate(&mut rng, n)));
        }
        let want = sv_of(&c);
        let mut got = StateVectorF64::new(n);
        got.apply_circuit_blocked(&c, &BlockConfig::default())
            .unwrap();
        let o = overlap(&want, &got);
        assert!((o - 1.0).abs() < 1e-10, "blocked overlap {o}");
        for i in 0..(1usize << n) {
            assert!((want.amplitude(i) - got.amplitude(i)).norm() < 1e-10);
        }
    }
}

#[test]
fn qasm_rejects_wrong_arity_and_bad_registers() {
    let header = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nqreg r[1];\n";
    for bad in [
        "rz() q[0];",
        "cx q[0];",
        "u3(1) q[0];",
        "h q[0],q[1];",
        "rx(0.1,0.2) q[0];",
        "h q[2];",
        "h r[1];",
        "h s[0];",
    ] {
        let src = format!("{header}{bad}\n");
        assert!(Circuit::from_qasm(&src).is_err(), "accepted '{bad}'");
    }
    // Registers are laid out in declaration order: r[0] is qubit 2.
    let c = Circuit::from_qasm(&format!("{header}x r[0];\ncx q[1],r[0];\n")).unwrap();
    assert_eq!(c.num_qubits, 3);
    assert_eq!(
        c.ops,
        vec![
            qsim_lab::circuit::Op::Gate(Gate::X(2)),
            qsim_lab::circuit::Op::Gate(Gate::Cnot(1, 2)),
        ]
    );
}

/// `reset_all` must leave the tableau indistinguishable from a fresh one,
/// including the inverse-row signs used by sign tracking.
#[test]
fn tableau_reset_all_matches_fresh_tableau() {
    let mut rng = StdRng::seed_from_u64(9);
    for _ in 0..30 {
        let n = 7;
        let mut c = Circuit::new(n);
        for _ in 0..40 {
            let q = rng.random_range(0..n);
            let mut r = rng.random_range(0..n);
            while r == q {
                r = rng.random_range(0..n);
            }
            match rng.random_range(0..6) {
                0 => c.h(q),
                1 => c.s(q),
                2 => c.cnot(q, r),
                3 => c.cz(q, r),
                4 => c.measure(q),
                _ => c.x(q),
            };
        }
        c.measure_all();
        let seed = rng.random::<u64>();
        let mut fresh = Tableau::new(n);
        let want = c.run(&mut fresh, &mut StdRng::seed_from_u64(seed)).unwrap();
        let mut used = Tableau::new(n);
        c.run(&mut used, &mut StdRng::seed_from_u64(seed ^ 1))
            .unwrap();
        used.reset_all();
        let got = c.run(&mut used, &mut StdRng::seed_from_u64(seed)).unwrap();
        assert_eq!(want, got);
    }
}
