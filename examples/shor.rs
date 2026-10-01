//! Shor's algorithm on the state vector: factor 15 (12 qubits) and
//! 21 (15 qubits). Modular multiplication is a permutation oracle; the
//! inverse QFT is built from gates; the order is recovered with continued
//! fractions.

use qsim_lab::algorithms;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let mut rng = StdRng::seed_from_u64(15);
    for n in [15, 21] {
        let (f, runs) = algorithms::shor_factor(n, &mut rng);
        for r in &runs {
            println!(
                "N={n}: a={:2} on {} qubits, measured {:4}, order {:?}, factor {:?}",
                r.a, r.qubits, r.measured, r.order, r.factor
            );
        }
        match f {
            Some((p, q)) => println!("N={n}: {n} = {p} x {q}\n"),
            None => println!("N={n}: no factor found\n"),
        }
    }
}
