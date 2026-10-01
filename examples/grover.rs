//! Grover search over 2^n items for one marked index.

use qsim_lab::algorithms;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let mut rng = StdRng::seed_from_u64(3);
    for n in [4, 8, 12, 16] {
        let marked = 0xBEEF & ((1 << n) - 1);
        let iters = (std::f64::consts::FRAC_PI_4 * ((1u64 << n) as f64).sqrt()).floor();
        let (found, p) = algorithms::grover::<f64, _>(n, marked, &mut rng);
        println!(
            "n={n:2}: {iters:4} iterations, marked {marked:5}, measured {found:5}, P(success) = {p:.5}"
        );
    }
}
