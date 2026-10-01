//! The quantum Fourier transform of a basis state has uniform magnitudes
//! and linearly increasing phases; QFT followed by its inverse is the
//! identity.

use qsim_lab::algorithms;
use qsim_lab::StateVectorF64;
use std::f64::consts::PI;

fn main() {
    let n = 4;
    let x = 3;
    let mut s = StateVectorF64::basis_state(n, x);
    s.apply_circuit(&algorithms::qft(n)).unwrap();
    println!("QFT|{x}> on {n} qubits:");
    for k in 0..1 << n {
        let a = s.amplitude(k);
        println!(
            "  |{k:2}>  |a| = {:.4}   phase = {:6.3} π   (expected {:6.3} π)",
            a.norm(),
            a.arg() / PI,
            {
                let e = 2.0 * (x * k) as f64 / (1 << n) as f64;
                let e = e.rem_euclid(2.0);
                if e > 1.0 {
                    e - 2.0
                } else {
                    e
                }
            }
        );
    }
    let mut c = algorithms::qft(n);
    c.append(&algorithms::qft(n).inverse());
    let mut s = StateVectorF64::basis_state(n, x);
    s.apply_circuit(&c).unwrap();
    println!(
        "after QFT then QFT†: |<{x}|ψ>|² = {:.6}",
        s.amplitude(x).norm_sqr()
    );
}
