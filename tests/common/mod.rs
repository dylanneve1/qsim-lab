#![allow(dead_code)]

use qsim_lab::circuit::Circuit;
use qsim_lab::gate::Gate;
use qsim_lab::{Mps, StateVectorF64};
use rand::Rng;

/// A random circuit using every gate type.
pub fn random_universal<R: Rng>(n: usize, len: usize, rng: &mut R) -> Circuit {
    let mut c = Circuit::new(n);
    for _ in 0..len {
        let q = rng.random_range(0..n);
        let th = rng.random_range(-3.2..3.2);
        let pick = if n >= 3 {
            17
        } else if n == 2 {
            16
        } else {
            12
        };
        let other = |rng: &mut R, not: &[usize]| loop {
            let x = rng.random_range(0..n);
            if !not.contains(&x) {
                break x;
            }
        };
        let g = match rng.random_range(0..pick) {
            0 => Gate::H(q),
            1 => Gate::X(q),
            2 => Gate::Y(q),
            3 => Gate::Z(q),
            4 => Gate::S(q),
            5 => Gate::Sdg(q),
            6 => Gate::T(q),
            7 => Gate::Tdg(q),
            8 => Gate::Rx(q, th),
            9 => Gate::Ry(q, th),
            10 => Gate::Rz(q, th),
            11 => Gate::Phase(q, th),
            12 => Gate::Cnot(q, other(rng, &[q])),
            13 => Gate::Cz(q, other(rng, &[q])),
            14 => Gate::Swap(q, other(rng, &[q])),
            15 => Gate::CPhase(q, other(rng, &[q]), th),
            _ => {
                let b = other(rng, &[q]);
                let t = other(rng, &[q, b]);
                Gate::Ccx(q, b, t)
            }
        };
        c.gate(g);
    }
    c
}

pub fn sv_of(c: &Circuit) -> StateVectorF64 {
    let mut s = StateVectorF64::new(c.num_qubits);
    s.apply_circuit(c).unwrap();
    s
}

pub fn mps_probabilities(m: &Mps) -> Vec<f64> {
    (0..1u128 << m.num_qubits())
        .map(|i| m.amplitude(i).norm_sqr())
        .collect()
}

pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}
