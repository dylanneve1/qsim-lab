//! The tensor-network engine against the independent reference state vector
//! of `tests/audit_common` (research/simulability/tn.md §5): amplitudes,
//! batches over open qubits and Pauli expectation values, f64 to 1e-12 and
//! f32 to 1e-5, with every simplification pass, both tree methods, and
//! slicing on and off.

#[path = "../audit_common/mod.rs"]
mod audit_common;

use audit_common::{random_circuit, RefSv};
use num_complex::Complex64;
use qsim_lab::circuit::{Circuit, SimError};
use qsim_lab::engines::tn::{
    self, amplitude, amplitudes, bits_of, expectation, parse_pauli, Network, PairStrategy,
    PathOptions, Pauli, Precision, SimplifyOptions, TnOptions,
};
use qsim_lab::gate::Gate;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

const TOL: f64 = 1e-12;
const TOL32: f64 = 1e-5;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// 2x2 matrices of the gates `RefSv` does not know, from their textbook
/// definitions (not from the library).
fn extra_1q(g: &Gate) -> Option<[[Complex64; 2]; 2]> {
    Some(match *g {
        Gate::I(_) => [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]],
        Gate::Sx(_) => [[c(0.5, 0.5), c(0.5, -0.5)], [c(0.5, -0.5), c(0.5, 0.5)]],
        Gate::Sxdg(_) => [[c(0.5, -0.5), c(0.5, 0.5)], [c(0.5, 0.5), c(0.5, -0.5)]],
        Gate::U(_, th, ph, la) => {
            let (s, co) = ((th / 2.0).sin(), (th / 2.0).cos());
            let e = |t: f64| c(t.cos(), t.sin());
            [[c(co, 0.0), -e(la) * s], [e(ph) * s, e(ph + la) * co]]
        }
        _ => return None,
    })
}

/// The reference state of `c`, with the extra gates applied directly.
fn reference(circ: &Circuit) -> RefSv {
    let mut s = RefSv::new(circ.num_qubits);
    for g in circ.gates() {
        if let Some(m) = extra_1q(g) {
            let q = g.qubits()[0];
            let old = s.a.clone();
            for (i, out) in s.a.iter_mut().enumerate() {
                let r = (i >> q) & 1;
                *out = m[r][0] * old[i & !(1 << q)] + m[r][1] * old[i | (1 << q)];
            }
        } else if let Gate::ISwap(a, b) | Gate::ISwapdg(a, b) = *g {
            let ph = if matches!(g, Gate::ISwap(..)) {
                c(0.0, 1.0)
            } else {
                c(0.0, -1.0)
            };
            let old = s.a.clone();
            for (i, out) in s.a.iter_mut().enumerate() {
                let (ba, bb) = ((i >> a) & 1, (i >> b) & 1);
                if ba != bb {
                    *out = ph * old[i ^ (1 << a) ^ (1 << b)];
                } else {
                    *out = old[i];
                }
            }
        } else {
            s.apply(g);
        }
    }
    s
}

fn opts() -> TnOptions {
    TnOptions {
        path: PathOptions {
            trials: 8,
            max_secs: 5.0,
            ..PathOptions::default()
        },
        threads: 2,
        ..TnOptions::default()
    }
}

/// Every gate type, edge-biased angles included (RefSv's generator plus
/// the gates it does not cover).
fn random_any(rng: &mut StdRng, n: usize, depth: usize) -> Circuit {
    let mut circ = random_circuit(rng, n, depth, false, false);
    let extra = depth / 3;
    for _ in 0..extra {
        let q = rng.random_range(0..n);
        let g = match rng.random_range(0..6) {
            0 => Gate::Sx(q),
            1 => Gate::Sxdg(q),
            2 => Gate::U(q, rng.random_range(-PI..PI), rng.random_range(-PI..PI), rng.random_range(-PI..PI)),
            3 => Gate::I(q),
            4 if n >= 2 => {
                let b = (q + 1 + rng.random_range(0..n - 1)) % n;
                Gate::ISwap(q, b)
            }
            5 if n >= 2 => {
                let b = (q + 1 + rng.random_range(0..n - 1)) % n;
                Gate::ISwapdg(q, b)
            }
            _ => Gate::H(q),
        };
        let pos = rng.random_range(0..=circ.ops.len());
        circ.ops.insert(pos, qsim_lab::circuit::Op::Gate(g));
    }
    circ
}

fn brickwork(rng: &mut StdRng, n: usize, layers: usize, cz: bool) -> Circuit {
    let mut circ = Circuit::new(n);
    for l in 0..layers {
        for q in 0..n {
            circ.u(q, rng.random_range(0.0..PI), rng.random_range(-PI..PI), rng.random_range(-PI..PI));
        }
        let mut q = l % 2;
        while q + 1 < n {
            if cz {
                circ.cz(q, q + 1);
            } else {
                circ.gate(Gate::ISwap(q, q + 1));
                circ.cphase(q, q + 1, rng.random_range(-PI..PI));
            }
            q += 2;
        }
    }
    circ
}

fn qft_circuit(n: usize, input: u64) -> Circuit {
    let mut circ = Circuit::new(n);
    for q in 0..n {
        if (input >> q) & 1 == 1 {
            circ.x(q);
        }
    }
    circ.append(&qsim_lab::algorithms::qft(n));
    circ
}

fn check_all_amplitudes(circ: &Circuit, o: &TnOptions, tol: f64) -> f64 {
    let r = reference(circ);
    let n = circ.num_qubits;
    // every amplitude at once: all qubits open
    let open: Vec<usize> = (0..n).collect();
    let (v, _) = amplitudes(circ, &vec![false; n], &open, o).expect("tn amplitudes");
    let err = v
        .iter()
        .zip(&r.a)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max);
    assert!(err <= tol, "n={n}: all-open batch differs by {err}");
    err
}

#[test]
fn random_circuits_every_amplitude() {
    let mut rng = StdRng::seed_from_u64(0x7e57_0001);
    let o = opts();
    let mut worst: f64 = 0.0;
    for n in 1..=10 {
        for rep in 0..3 {
            let depth = [4, 3 * n, 8 * n][rep];
            let circ = random_any(&mut rng, n, depth);
            worst = worst.max(check_all_amplitudes(&circ, &o, TOL));
        }
    }
    eprintln!("worst all-open error {worst:e}");
}

#[test]
fn single_amplitudes_and_batches() {
    let mut rng = StdRng::seed_from_u64(0x7e57_0002);
    let o = opts();
    for n in [3usize, 6, 9, 12] {
        let circ = random_any(&mut rng, n, 6 * n);
        let r = reference(&circ);
        for _ in 0..4 {
            let x: u64 = rng.random_range(0..1u64 << n);
            let (a, rep) = amplitude(&circ, &bits_of(x as u128, n), &o).unwrap();
            assert!((a - r.a[x as usize]).norm() <= TOL, "n={n} x={x}: {a} vs {}", r.a[x as usize]);
            assert_eq!(rep.qubits, n);
        }
        // batches over random open sets, in random order
        for k in 1..=n.min(5) {
            let mut open: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = rng.random_range(0..=i);
                open.swap(i, j);
            }
            open.truncate(k);
            let x: u64 = rng.random_range(0..1u64 << n);
            let bits = bits_of(x as u128, n);
            let (v, _) = amplitudes(&circ, &bits, &open, &o).unwrap();
            assert_eq!(v.len(), 1 << k);
            for (j, a) in v.iter().enumerate() {
                let mut idx = x as usize;
                for (t, &q) in open.iter().enumerate() {
                    idx = (idx & !(1 << q)) | (((j >> t) & 1) << q);
                }
                assert!((a - r.a[idx]).norm() <= TOL, "n={n} open={open:?} j={j}");
            }
        }
    }
}

#[test]
fn brickwork_qft_and_wide_circuits() {
    let mut rng = StdRng::seed_from_u64(0x7e57_0003);
    let o = opts();
    for (n, layers, cz) in [(12, 8, true), (14, 10, false), (16, 6, true), (20, 5, true)] {
        let circ = brickwork(&mut rng, n, layers, cz);
        let r = reference(&circ);
        for _ in 0..3 {
            let x: u64 = rng.random_range(0..1u64 << n);
            let (a, _) = amplitude(&circ, &bits_of(x as u128, n), &o).unwrap();
            assert!((a - r.a[x as usize]).norm() <= TOL, "brick n={n}");
        }
        // a 4-qubit batch
        let open = [0usize, n / 3, n / 2, n - 1];
        let (v, _) = amplitudes(&circ, &vec![false; n], &open, &o).unwrap();
        for (j, a) in v.iter().enumerate() {
            let idx: usize = open.iter().enumerate().map(|(t, &q)| ((j >> t) & 1) << q).sum();
            assert!((a - r.a[idx]).norm() <= TOL);
        }
    }
    for n in [5usize, 9, 14] {
        let circ = qft_circuit(n, 0b1011);
        let r = reference(&circ);
        let (v, _) = amplitudes(&circ, &vec![false; n], &(0..n.min(8)).collect::<Vec<_>>(), &o).unwrap();
        for (j, a) in v.iter().enumerate() {
            assert!((a - r.a[j]).norm() <= TOL, "qft n={n} j={j}");
        }
    }
}

#[test]
fn disconnected_components_idle_qubits_and_empty_circuits() {
    let mut rng = StdRng::seed_from_u64(0x7e57_0004);
    let o = opts();
    // two blocks on interleaved qubits plus idle qubits
    let n = 11;
    let mut circ = Circuit::new(n);
    let blk_a = [0usize, 3, 5, 8];
    let blk_b = [1usize, 2, 7];
    for _ in 0..12 {
        let (x, y) = (blk_a[rng.random_range(0..4)], blk_a[rng.random_range(0..4)]);
        circ.u(x, rng.random(), rng.random(), rng.random());
        if x != y {
            circ.cnot(x, y);
        }
        let (x, y) = (blk_b[rng.random_range(0..3)], blk_b[rng.random_range(0..3)]);
        circ.ry(x, rng.random());
        if x != y {
            circ.cphase(x, y, rng.random());
        }
    }
    check_all_amplitudes(&circ, &o, TOL);
    // empty circuit: |0^n>
    let empty = Circuit::new(5);
    let (a, _) = amplitude(&empty, &[false; 5], &o).unwrap();
    assert!((a - c(1.0, 0.0)).norm() <= TOL);
    let (a, _) = amplitude(&empty, &[false, true, false, false, false], &o).unwrap();
    assert!(a.norm() <= TOL);
    check_all_amplitudes(&empty, &o, TOL);
    // zero qubits
    let (v, _) = amplitudes(&Circuit::new(0), &[], &[], &o).unwrap();
    assert_eq!(v, vec![c(1.0, 0.0)]);
}

#[test]
fn every_single_gate_alone() {
    let o = opts();
    let gates = [
        Gate::I(0),
        Gate::H(0),
        Gate::X(1),
        Gate::Y(0),
        Gate::Z(0),
        Gate::S(1),
        Gate::Sdg(0),
        Gate::T(0),
        Gate::Tdg(1),
        Gate::Sx(0),
        Gate::Sxdg(1),
        Gate::Rx(0, 0.3),
        Gate::Ry(1, -1.1),
        Gate::Rz(0, 2.0),
        Gate::Phase(1, 0.7),
        Gate::U(0, 1.2, -0.5, 0.8),
        Gate::Cnot(0, 1),
        Gate::Cnot(1, 0),
        Gate::Cz(0, 1),
        Gate::Swap(0, 2),
        Gate::ISwap(1, 2),
        Gate::ISwapdg(2, 0),
        Gate::CPhase(0, 2, 0.9),
        Gate::Ccx(0, 1, 2),
        Gate::Ccx(2, 0, 1),
    ];
    for g in gates {
        // prepare a generic state first so every matrix entry matters
        let mut circ = Circuit::new(3);
        circ.u(0, 0.7, 0.2, -0.4).u(1, 1.9, -1.0, 0.3).u(2, 0.4, 0.8, 1.1);
        circ.gate(g);
        check_all_amplitudes(&circ, &o, TOL);
        // and the bare gate on |000>
        let mut bare = Circuit::new(3);
        bare.gate(g);
        check_all_amplitudes(&bare, &o, TOL);
    }
}

#[test]
fn pauli_expectations_with_light_cones() {
    let mut rng = StdRng::seed_from_u64(0x7e57_0005);
    let o = opts();
    let mut worst: f64 = 0.0;
    for n in [1usize, 2, 4, 7, 10, 13] {
        for _ in 0..4 {
            let circ = random_any(&mut rng, n, 5 * n);
            let r = reference(&circ);
            let w = rng.random_range(1..=n.min(4));
            let mut s: Vec<char> = vec!['I'; n];
            for _ in 0..w {
                s[rng.random_range(0..n)] = ['X', 'Y', 'Z'][rng.random_range(0..3)];
            }
            let ps: String = s.iter().collect();
            let p = parse_pauli(&ps).unwrap();
            let (v, _) = expectation(&circ, &p, &o).unwrap();
            let e = r.pauli_expectation(&ps);
            worst = worst.max((v - e).abs());
            assert!((v - e).abs() <= TOL, "n={n} P={ps}: {v} vs {e}");
        }
    }
    // a wide brickwork whose local observable has a small light cone
    let circ = brickwork(&mut rng, 20, 3, true);
    let r = reference(&circ);
    for ps in ["IIIIIIIIIZZIIIIIIIII", "XIIIIIIIIIIIIIIIIIIY", "IIIIIIIIIIYIIIIIIIII"] {
        let (v, rep) = expectation(&circ, &parse_pauli(ps).unwrap(), &o).unwrap();
        assert!((v - r.pauli_expectation(ps)).abs() <= TOL, "{ps}");
        assert!(rep.qubits < 20, "light cone not used for {ps}");
    }
    // identity observable and a Pauli on an idle qubit
    let mut circ = Circuit::new(3);
    circ.h(0).cnot(0, 1);
    let (v, _) = expectation(&circ, &[], &o).unwrap();
    assert!((v - 1.0).abs() <= TOL);
    let (v, _) = expectation(&circ, &[(2, Pauli::Z)], &o).unwrap();
    assert!((v - 1.0).abs() <= TOL);
    let (v, _) = expectation(&circ, &[(2, Pauli::X)], &o).unwrap();
    assert!(v.abs() <= TOL);
    let (v, _) = expectation(&circ, &[(0, Pauli::X), (1, Pauli::X)], &o).unwrap();
    assert!((v - 1.0).abs() <= TOL);
    eprintln!("worst expectation error {worst:e}");
}

#[test]
fn slicing_on_and_off_agree() {
    let mut rng = StdRng::seed_from_u64(0x7e57_0006);
    for (n, layers) in [(10usize, 8usize), (14, 8), (16, 10)] {
        let cz = rng.random_bool(0.5);
        let circ = brickwork(&mut rng, n, layers, cz);
        let x: u64 = rng.random_range(0..1u64 << n);
        let bits = bits_of(x as u128, n);
        let open = [1usize, n - 2];
        let unsliced = TnOptions {
            path: PathOptions {
                target_log2_size: Some(60.0),
                ..opts().path
            },
            ..opts()
        };
        let (v0, r0) = amplitudes(&circ, &bits, &open, &unsliced).unwrap();
        assert_eq!(r0.exec.slices, 1);
        let r = reference(&circ);
        // targets below the unsliced width (the root holds the 2 open qubits)
        let w = r0.path.log2_max_size;
        assert!(w >= 4.0, "n={n}: unsliced width {w} too small for this test");
        for target in [w - 1.0, w - 2.0, (w - 3.0).max(2.0)] {
            let sliced = TnOptions {
                path: PathOptions {
                    target_log2_size: Some(target),
                    ..opts().path
                },
                ..opts()
            };
            let (v1, r1) = amplitudes(&circ, &bits, &open, &sliced).unwrap();
            assert!(r1.exec.slices > 1, "n={n} target={target}: nothing sliced");
            assert!(r1.path.log2_sliced_max_size <= target + 1e-9);
            for (j, (a, b)) in v0.iter().zip(&v1).enumerate() {
                assert!((a - b).norm() <= TOL, "n={n} target={target} j={j}: {a} vs {b}");
                let mut idx = x as usize;
                for (t, &q) in open.iter().enumerate() {
                    idx = (idx & !(1 << q)) | (((j >> t) & 1) << q);
                }
                assert!((b - r.a[idx]).norm() <= TOL);
            }
        }
    }
    // sliced expectation values, sliced across workers
    let circ = brickwork(&mut rng, 12, 6, true);
    let ps = "IIXZIIIIYIII";
    let e = reference(&circ).pauli_expectation(ps);
    for threads in [1, 4] {
        let o = TnOptions {
            path: PathOptions {
                target_log2_size: Some(4.0),
                ..opts().path
            },
            threads,
            ..opts()
        };
        let (v, rep) = expectation(&circ, &parse_pauli(ps).unwrap(), &o).unwrap();
        assert!(rep.exec.slices > 1);
        assert!((v - e).abs() <= TOL, "threads={threads}");
    }
}

#[test]
fn every_configuration_knob_is_exact() {
    let mut rng = StdRng::seed_from_u64(0x7e57_0007);
    let circs: Vec<Circuit> = vec![
        random_any(&mut rng, 8, 40),
        brickwork(&mut rng, 9, 6, true),
        qft_circuit(7, 0b101),
    ];
    let simps = [
        SimplifyOptions::default(),
        SimplifyOptions::none(),
        SimplifyOptions {
            diagonal: false,
            ..SimplifyOptions::default()
        },
        SimplifyOptions {
            column: false,
            ..SimplifyOptions::default()
        },
        SimplifyOptions {
            split: false,
            ..SimplifyOptions::default()
        },
    ];
    for circ in &circs {
        for simp in simps {
            for (greedy, bisect, reconf) in [(true, false, false), (false, true, true), (true, true, true)] {
                let o = TnOptions {
                    simplify: simp,
                    path: PathOptions {
                        greedy,
                        bisect,
                        reconf,
                        trials: 6,
                        ..PathOptions::default()
                    },
                    threads: 2,
                    ..TnOptions::default()
                };
                check_all_amplitudes(circ, &o, TOL);
            }
        }
        // both pairwise strategies, sliced and not
        for strategy in [PairStrategy::Permute, PairStrategy::Loops] {
            for target in [None, Some(3.0)] {
                let o = TnOptions {
                    strategy,
                    path: PathOptions {
                        target_log2_size: target,
                        ..opts().path
                    },
                    ..opts()
                };
                check_all_amplitudes(circ, &o, TOL);
            }
        }
        // f32
        let o = TnOptions {
            precision: Precision::F32,
            ..opts()
        };
        check_all_amplitudes(circ, &o, TOL32);
    }
}

#[test]
fn simplification_preserves_the_value() {
    let mut rng = StdRng::seed_from_u64(0x7e57_0008);
    for _ in 0..10 {
        let n = rng.random_range(2..7);
        let circ = random_any(&mut rng, n, 20);
        let open: Vec<usize> = (0..n).filter(|_| rng.random_bool(0.4)).collect();
        let bits: Vec<bool> = (0..n).map(|_| rng.random_bool(0.5)).collect();
        let raw = Network::amplitude(&circ, &bits, &open).unwrap();
        let want = raw.contract_naive();
        let mut simp = raw.clone();
        let st = simp.simplify(&SimplifyOptions::default());
        assert!(st.tensors_after <= st.tensors_before);
        let got = simp.contract_naive();
        for (a, b) in want.iter().zip(&got) {
            assert!((a - b).norm() <= TOL);
        }
    }
}

#[test]
fn errors_are_reported() {
    let o = opts();
    let mut circ = Circuit::new(2);
    circ.h(0).measure(0);
    assert!(matches!(
        amplitude(&circ, &[false, false], &o),
        Err(SimError::MeasurementNotSupported { backend: "tn", .. })
    ));
    assert!(matches!(
        expectation(&circ, &[(0, Pauli::Z)], &o),
        Err(SimError::MeasurementNotSupported { .. })
    ));
    let mut ok = Circuit::new(2);
    ok.h(0).cnot(0, 1);
    assert!(matches!(
        amplitudes(&ok, &[false, false], &[2], &o),
        Err(SimError::QubitOutOfRange { .. })
    ));
    assert!(amplitude(&ok, &[false], &o).is_err());
    assert!(matches!(
        expectation(&ok, &[(5, Pauli::Z)], &o),
        Err(SimError::QubitOutOfRange { .. })
    ));
    // a budget too small for even one slice
    let mut rng = StdRng::seed_from_u64(9);
    let wide = brickwork(&mut rng, 10, 6, true);
    let tiny = TnOptions {
        max_bytes: 64,
        path: PathOptions {
            target_log2_size: Some(40.0),
            ..opts().path
        },
        ..opts()
    };
    assert!(matches!(
        amplitudes(&wide, &[false; 10], &(0..10).collect::<Vec<_>>(), &tiny),
        Err(SimError::TooLarge { .. })
    ));
}

#[test]
fn tree_search_reports_consistent_costs() {
    let mut rng = StdRng::seed_from_u64(0x7e57_0009);
    let circ = brickwork(&mut rng, 16, 8, false);
    let mut nw = Network::amplitude(&circ, &[false; 16], &[]).unwrap();
    nw.simplify(&SimplifyOptions::default());
    let hg = tn::Hypergraph::from_network(&nw);
    let p = tn::search(
        &hg,
        &PathOptions {
            trials: 16,
            target_log2_size: Some(5.0),
            ..PathOptions::default()
        },
    );
    assert!(p.tree.is_valid());
    let tc = tn::tree_cost(&hg, &p.tree, &p.sliced);
    assert!(tc.log2_max <= 5.0 + 1e-9);
    assert!((tc.log10_total() - p.stats.log10_sliced_flops).abs() < 1e-9);
    assert!(p.stats.log10_sliced_flops + 1e-9 >= p.stats.log10_flops);
    assert!(p.stats.overhead >= 1.0 - 1e-9);
}

#[test]
fn big_tensor_kernels_match_the_reference() {
    // all-open outputs of 11-13 qubits make intermediates of 2^11+ entries,
    // so the single-pass big-times-rank-2 kernels run (Auto), next to the
    // permute plan and the loop plan
    let mut rng = StdRng::seed_from_u64(0x7e57_000a);
    for n in [11usize, 12, 13] {
        let circ = random_any(&mut rng, n, 6 * n);
        let brick = brickwork(&mut rng, n, 6, n % 2 == 0);
        for c in [&circ, &brick] {
            for strategy in [PairStrategy::Auto, PairStrategy::Permute, PairStrategy::Loops] {
                let o = TnOptions {
                    strategy,
                    ..opts()
                };
                check_all_amplitudes(c, &o, TOL);
            }
            let o = TnOptions {
                precision: Precision::F32,
                ..opts()
            };
            check_all_amplitudes(c, &o, TOL32);
        }
    }
}

#[test]
fn light_cone_drops_diagonal_gates_that_commute_with_z() {
    let o = opts();
    let mut c = Circuit::new(6);
    for q in 0..6 {
        c.h(q);
    }
    c.cnot(0, 1).cnot(1, 2).ry(2, 0.7).cnot(2, 3).rx(3, 0.4);
    // trailing diagonal gates around qubit 3
    c.cz(3, 4).cphase(2, 3, 0.9).rz(3, 1.3).t(5).cz(4, 5);
    let r = reference(&c);
    // Z3: every trailing diagonal gate commutes; the cone is the causal cone
    // of the non-diagonal part (qubits 0..=3)
    let (cone_c, cone) = tn::pauli_light_cone(&c, &[(3, Pauli::Z)]).unwrap();
    assert_eq!(cone, vec![0, 1, 2, 3]);
    assert!(cone_c.gates().all(|g| !matches!(g, Gate::Cz(..) | Gate::CPhase(..) | Gate::Rz(..))));
    let (v, rep) = expectation(&c, &[(3, Pauli::Z)], &o).unwrap();
    assert!((v - r.pauli_expectation("IIIZII")).abs() <= TOL);
    assert_eq!(rep.qubits, 4);
    // X3 does not commute with them: they stay and the cone grows
    let (v, rep) = expectation(&c, &[(3, Pauli::X)], &o).unwrap();
    assert!((v - r.pauli_expectation("IIIXII")).abs() <= TOL);
    assert!(rep.qubits >= 5);
    // Z3 Z4: still diagonal, the CZ(3,4) commutes
    let (v, _) = expectation(&c, &[(3, Pauli::Z), (4, Pauli::Z)], &o).unwrap();
    assert!((v - r.pauli_expectation("IIIZZI")).abs() <= TOL);
}
