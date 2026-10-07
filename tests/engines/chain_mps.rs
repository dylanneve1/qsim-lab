//! Boundary-MPS chain sweep (`engines::chain_mps`): with a bond cap at
//! least the register's Schmidt ranks it must reproduce the exact sweep, the
//! dense register state after any prefix, and the state vector; with a
//! small cap it must report a truncation estimate below one.

use num_complex::Complex64;
use qsim_lab::circuit::Circuit;
use qsim_lab::engines::blocked::BlockConfig;
use qsim_lab::engines::chain_mps::{amplitude_mps, run_prefix};
use qsim_lab::engines::chain_sweep::{self, compile, truncate, ChainCircuit};
use qsim_lab::engines::statevector::StateVector;
use qsim_lab::gate::Gate;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

const QASM: &str = include_str!("../../research/chain-sweep/nq70_depth70_checks27_doped.qasm");

fn brickwork(n: usize, d: usize, rng: &mut StdRng) -> Circuit {
    let mut c = Circuit::new(n);
    for layer in 0..d {
        for q in 0..n {
            for _ in 0..rng.random_range(0..3) {
                c.gate(match rng.random_range(0..5) {
                    0 => Gate::H(q),
                    1 => Gate::S(q),
                    2 => Gate::T(q),
                    3 => Gate::Rz(q, rng.random_range(-3.0..3.0)),
                    _ => Gate::Ry(q, rng.random_range(-3.0..3.0)),
                });
            }
        }
        let mut a = layer % 2;
        while a + 1 < n {
            if rng.random_bool(0.85) {
                c.gate(Gate::Cz(a, a + 1));
            }
            a += 2;
        }
    }
    c
}

#[test]
fn exact_cap_matches_state_vector() {
    let mut rng = StdRng::seed_from_u64(11);
    for (n, d) in [(3, 5), (5, 8), (6, 11), (8, 10)] {
        let c = brickwork(n, d, &mut rng);
        let mut s = StateVector::<f64>::new(n);
        s.apply_circuit(&c).unwrap();
        let cc = ChainCircuit::from_circuit(&c).unwrap();
        let scale = (1.0 / (1u64 << n) as f64).sqrt();
        for x in 0..1u128 << n {
            let plan = compile(&cc, x, &HashMap::new());
            let r = amplitude_mps(&plan, 1 << 12, 1e-14);
            let e = s.amplitude(x as usize);
            assert!(
                (r.amp - e).norm() <= 1e-10 * scale,
                "n={n} d={d} x={x}: {} vs {e}",
                r.amp
            );
            assert!(r.fid_est > 1.0 - 1e-9, "fid_est {}", r.fid_est);
        }
    }
}

#[test]
fn exact_cap_matches_dense_register_after_every_qubit() {
    let c = truncate(&Circuit::from_qasm(QASM).unwrap(), 12, 16);
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let plan = compile(&cc, 0b1011_0110_0101, &HashMap::new());
    for upto in [1, 3, 6, 9, 12] {
        let mps = run_prefix(&plan, upto, 1 << 10, 1e-14);
        let mut sv = StateVector::<f64>::try_new(plan.width).unwrap();
        sv.apply_kops_blocked(&plan.ops[..plan.qubit_ops[upto]], &BlockConfig::default());
        let dense = mps.to_dense();
        let norm: f64 = dense.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        for (i, z) in dense.iter().enumerate() {
            let e = sv.amplitude(i);
            assert!(
                (z - e).norm() <= 1e-10 * norm.max(1e-300),
                "upto={upto} i={i}: {z} vs {e}"
            );
        }
    }
}

#[test]
fn doped_circuit_exact_and_truncated() {
    let c = truncate(&Circuit::from_qasm(QASM).unwrap(), 20, 24);
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let mut rng = StdRng::seed_from_u64(3);
    let mut low = 0;
    for _ in 0..4 {
        let x: u128 = rng.random_range(0..1u128 << 20);
        let plan = compile(&cc, x, &HashMap::new());
        let ex = chain_sweep::amplitude(&cc, x).unwrap();
        let r = amplitude_mps(&plan, 1 << 7, 1e-14);
        let scale = (1.0f64 / (1u64 << 20) as f64).sqrt();
        assert!(
            (r.amp - ex).norm() <= 1e-9 * scale,
            "x={x}: {} vs {ex}",
            r.amp
        );
        // the register reaches width 12 and Schmidt rank 2^6: a cap of 8
        // must truncate
        let t = amplitude_mps(&plan, 8, 1e-14);
        assert!(t.fid_est < 0.99 && t.counters.truncations > 0);
        if (t.amp - ex).norm() > 1e-6 * scale {
            low += 1;
        }
        let _: Complex64 = t.amp;
    }
    assert!(low > 0);
}
