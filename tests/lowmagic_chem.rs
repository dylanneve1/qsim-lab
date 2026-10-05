//! Tests of `qsim_lab::chem` (research/simulability/lowmagic-chem.md): the Jordan–Wigner
//! Hamiltonian against Slater–Condon and against PySCF/OpenFermion numbers, the filtered
//! compressed-state energy against the state vector, d = GF(2) span dimension, Rotosolve.
use num_complex::Complex64 as C64;
use qsim_lab::adaptive::CompressedState;
use qsim_lab::chem::{self, Fcidump, POp, Program, Span};
use qsim_lab::{Gate, StateVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const H2: &str = include_str!("data/lowmagic-chem/h2.fcidump");
const H4: &str = include_str!("data/lowmagic-chem/h4.fcidump");
const H4_SEL4: &str = include_str!("data/lowmagic-chem/h4.sel4.jw.prog");

/// Random real integrals with the 8-fold symmetry, `(pq|rs) = Σ_k L^k_pq L^k_rs`.
fn random_fcidump(norb: usize, nelec: usize, seed: u64) -> Fcidump {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut h1 = vec![0.0; norb * norb];
    for p in 0..norb {
        for q in 0..=p {
            let v = rng.random_range(-1.0..1.0);
            h1[p * norb + q] = v;
            h1[q * norb + p] = v;
        }
    }
    let nk = 3;
    let mut l = vec![0.0; nk * norb * norb];
    for k in 0..nk {
        for p in 0..norb {
            for q in 0..=p {
                let v = rng.random_range(-0.5..0.5);
                l[(k * norb + p) * norb + q] = v;
                l[(k * norb + q) * norb + p] = v;
            }
        }
    }
    let n2 = norb * norb;
    let mut eri = vec![0.0; n2 * n2];
    for pq in 0..n2 {
        for rs in 0..n2 {
            eri[pq * n2 + rs] = (0..nk).map(|k| l[k * n2 + pq] * l[k * n2 + rs]).sum();
        }
    }
    Fcidump {
        norb,
        nelec,
        ms2: 0,
        ecore: rng.random_range(-1.0..1.0),
        h1,
        eri,
    }
}

fn random_program(n: usize, nel: usize, rots: usize, clifford_h: bool, seed: u64) -> Program {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut p = Program {
        n,
        ..Default::default()
    };
    for q in 0..nel {
        p.ops.push(POp::Clifford(Gate::X(q)));
    }
    if clifford_h {
        p.ops.push(POp::Clifford(Gate::H(0)));
        p.ops.push(POp::Clifford(Gate::Cnot(0, n - 1)));
    }
    for _ in 0..rots {
        let mut pauli = Vec::new();
        for q in 0..n {
            match rng.random_range(0..5) {
                0 => pauli.push((q, b'X')),
                1 => pauli.push((q, b'Y')),
                2 => pauli.push((q, b'Z')),
                _ => {}
            }
        }
        p.ops.push(POp::Rot {
            pauli,
            angle: rng.random_range(-1.0..1.0),
            param: None,
            mult: 0.0,
        });
    }
    p
}

fn sv_of(p: &Program) -> StateVector<f64> {
    let c = p.circuit(None);
    let mut sv = StateVector::<f64>::new(p.n);
    for g in c.gates() {
        sv.apply_gate(g).unwrap();
    }
    sv
}

#[test]
fn diagonal_terms_match_slater_condon() {
    for seed in 0..3 {
        let fd = random_fcidump(3, 2, seed);
        let (h, _) = chem::jw_hamiltonian(&fd, None, 0.0);
        let n = fd.qubits();
        for b in 0..(1usize << n) {
            let mut amps = vec![C64::new(0.0, 0.0); 1 << n];
            amps[b] = C64::new(1.0, 0.0);
            let occ: Vec<usize> = (0..n).filter(|q| b >> q & 1 == 1).collect();
            let e = chem::sv_expectation(&amps, &h);
            assert!((e - fd.determinant_energy(&occ)).abs() < 1e-10, "b={b:b}");
        }
    }
}

#[test]
fn h2_and_h4_match_pyscf_and_openfermion() {
    // PySCF RHF / FCI and an OpenFermion (expm of the fermionic generators,
    // OpenFermion's own JW Hamiltonian) evaluation of the same UCC state.
    let h2 = Fcidump::parse(H2).unwrap();
    assert!((h2.determinant_energy(&[0, 1]) - -1.116684387085341).abs() < 1e-10);
    let h4 = Fcidump::parse(H4).unwrap();
    assert!((h4.determinant_energy(&[0, 1, 2, 3]) - -2.0985459369979056).abs() < 1e-10);
    let prog = Program::parse(H4_SEL4).unwrap();
    let span = Span::from_program(&prog);
    let (h, st) = chem::jw_hamiltonian(&h4, Some(&span), 0.0);
    assert!(st.monomials_kept < st.monomials_total);
    let (e, cs) = chem::energy(&prog, None, &h, 20).unwrap();
    assert_eq!(cs.active_qubits(), 4);
    assert!((e - -2.149348823640304).abs() < 1e-9, "{e}");
}

#[test]
fn rotosolve_reaches_h2_fci() {
    let fd = Fcidump::parse(H2).unwrap();
    // one double excitation (spin orbitals 0,1 -> 2,3), JW, as written by chem.py
    let text = "n 4\nparam 0 0.05\nx 0\nx 1\n\
        prot 0 0.25 X0 X1 Y2 X3\nprot 0 -0.25 Y0 Y1 Y2 X3\nprot 0 0.25 Y0 X1 Y2 Y3\nprot 0 0.25 X0 Y1 Y2 Y3\n\
        prot 0 -0.25 Y0 X1 X2 X3\nprot 0 -0.25 X0 Y1 X2 X3\nprot 0 0.25 X0 X1 X2 Y3\nprot 0 -0.25 Y0 Y1 X2 Y3\n";
    let prog = Program::parse(text).unwrap();
    let (h, _) = chem::jw_hamiltonian(&fd, Some(&Span::from_program(&prog)), 0.0);
    let (th, hist, _) = chem::rotosolve(&prog, &h, 3, 1e-12, 10, usize::MAX).unwrap();
    let e = *hist.last().unwrap();
    assert!((e - -1.137270174660904).abs() < 1e-9, "{e} {th:?}");
    // the register {HF, doubly excited} holds the exact ground state
    let (_, st) = chem::energy(&prog, None, &h, 10).unwrap();
    let (hist, _) = chem::register_ground(&st, &h, 10, 1e-12);
    assert!((hist.last().unwrap() - -1.137270174660904).abs() < 1e-9);
}

#[test]
fn register_ground_bounds_the_circuit() {
    let h4 = Fcidump::parse(H4).unwrap();
    let prog = Program::parse(H4_SEL4).unwrap();
    let (h, _) = chem::jw_hamiltonian(&h4, Some(&Span::from_program(&prog)), 0.0);
    let (e, st) = chem::energy(&prog, None, &h, 20).unwrap();
    let (hist, _) = chem::register_ground(&st, &h, 40, 1e-12);
    let er = *hist.last().unwrap();
    // FCI (PySCF) <= register optimum <= circuit energy
    assert!(er <= e + 1e-12 && er >= -2.1663874486347625 - 1e-9, "{er} {e}");
}

#[test]
fn filtered_compressed_energy_equals_state_vector() {
    for seed in 0..6u64 {
        let norb = 3 + (seed as usize % 2);
        let fd = random_fcidump(norb, 2, 100 + seed);
        let n = fd.qubits();
        let prog = random_program(n, 2, 4 + seed as usize, false, seed);
        assert!(prog.x_preserving());
        let (hfull, _) = chem::jw_hamiltonian(&fd, None, 0.0);
        let span = Span::from_program(&prog);
        let (hf, _) = chem::jw_hamiltonian(&fd, Some(&span), 0.0);
        let sv = sv_of(&prog);
        let e_sv = chem::sv_expectation(sv.amplitudes(), &hfull);
        let (e_cs, st) = chem::energy(&prog, None, &hf, 20).unwrap();
        assert!((e_sv - e_cs).abs() < 1e-9, "seed {seed}: {e_sv} vs {e_cs}");
        // the compressed register is exactly the span of the rotation x vectors
        assert_eq!(st.active_qubits(), span.dim(), "seed {seed}");
    }
}

#[test]
fn non_x_preserving_program_with_full_hamiltonian() {
    let fd = random_fcidump(3, 2, 7);
    let prog = random_program(fd.qubits(), 2, 5, true, 9);
    assert!(!prog.x_preserving());
    let (h, _) = chem::jw_hamiltonian(&fd, None, 0.0);
    let sv = sv_of(&prog);
    let e_sv = chem::sv_expectation(sv.amplitudes(), &h);
    let cs = CompressedState::new(&prog.circuit(None), 20).unwrap();
    assert!((e_sv - cs.expectation(&h)).abs() < 1e-9);
}

#[test]
fn trig2_min_finds_the_minimum() {
    let mut rng = StdRng::seed_from_u64(5);
    for _ in 0..20 {
        let c: Vec<f64> = (0..5).map(|_| rng.random_range(-1.0..1.0)).collect();
        let f = |t: f64| {
            c[0] + c[1] * t.cos() + c[2] * t.sin() + c[3] * (2.0 * t).cos() + c[4] * (2.0 * t).sin()
        };
        let t0 = rng.random_range(-3.0..3.0);
        let step = 2.0 * std::f64::consts::PI / 5.0;
        let vals: [f64; 5] = std::array::from_fn(|k| f(t0 + step * k as f64));
        let (t, v) = chem::trig2_min(t0, &vals);
        assert!((f(t) - v).abs() < 1e-9);
        let grid = (0..20000)
            .map(|i| f(i as f64 * 2.0 * std::f64::consts::PI / 20000.0))
            .fold(f64::INFINITY, f64::min);
        assert!(v <= grid + 1e-9, "{v} > {grid}");
    }
}

#[test]
fn program_parse_errors() {
    assert!(Program::parse("n 2\nrot 0.1 X5\nfoo\n").is_err());
    assert!(Program::parse("n 2\nrot abc X0\n").is_err());
    assert!(Program::parse("n 2\nprot 0 1.0 Q0\n").is_err());
    let p = Program::parse("n 2\n# k v\nx 0\nh 1\ncx 1 0\nrot 0.3 X0 Y1\nprot 1 2.0 Z0\n").unwrap();
    assert_eq!(p.params.len(), 2);
    assert_eq!(p.rotations(), 2);
    assert!(!p.x_preserving());
    assert_eq!(p.meta[0].0, "k");
}
