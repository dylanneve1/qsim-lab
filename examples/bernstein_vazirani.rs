//! Bernstein–Vazirani: recover a hidden bit string with one oracle query.
//! The circuit is Clifford, so the stabilizer backend runs it at sizes far
//! beyond what a state vector could hold.

use qsim_lab::algorithms;
use qsim_lab::{StateVectorF64, Tableau};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn main() {
    let mut rng = StdRng::seed_from_u64(7);
    let secret: u64 = 0b1101_0010_1110;
    let c = algorithms::bernstein_vazirani(12, secret);
    let mut sv = StateVectorF64::new(13);
    let bits = c.run(&mut sv, &mut rng).unwrap();
    let got = bits
        .iter()
        .rev()
        .map(|&b| if b { '1' } else { '0' })
        .collect::<String>();
    println!("state vector, 13 qubits: secret {secret:012b}, measured {got}");

    // 2,000-bit secret on the tableau
    let n = 2000;
    let secret_bits: Vec<bool> = (0..n).map(|_| rng.random_bool(0.5)).collect();
    let mut t = Tableau::new(n + 1);
    t.x(n);
    t.h(n);
    for q in 0..n {
        t.h(q);
    }
    for (q, &b) in secret_bits.iter().enumerate() {
        if b {
            t.cnot(q, n);
        }
    }
    for q in 0..n {
        t.h(q);
    }
    let measured: Vec<bool> = (0..n).map(|q| t.measure_qubit(q, &mut rng)).collect();
    println!(
        "stabilizer, {} qubits: recovered all {n} secret bits: {}",
        n + 1,
        measured == secret_bits
    );
}
