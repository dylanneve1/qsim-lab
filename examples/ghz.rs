//! GHZ states across the backends, at sizes each one can handle:
//! 20 qubits as a state vector, 10,000 on the tableau, 1,000 as an MPS.

use qsim_lab::algorithms;
use qsim_lab::{Gate, Mps, StateVectorF32, Tableau};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

fn main() {
    let mut rng = StdRng::seed_from_u64(1);

    let n = 20;
    let t = Instant::now();
    let mut sv = StateVectorF32::new(n);
    sv.apply_circuit(&algorithms::ghz(n)).unwrap();
    let shots = sv.sample(5, &mut rng);
    println!(
        "state vector, {n} qubits ({} KiB): samples {:?} in {:.3}s",
        sv.bytes() / 1024,
        shots,
        t.elapsed().as_secs_f64()
    );

    let n = 10_000;
    let t = Instant::now();
    let mut tab = Tableau::new(n);
    tab.h(0);
    for q in 1..n {
        tab.cnot(q - 1, q);
    }
    let bits = tab.measure_all(&mut rng);
    let ones = bits.iter().filter(|&&b| b).count();
    println!(
        "stabilizer, {n} qubits ({} MiB): {ones} ones out of {n} in {:.3}s",
        tab.bytes() >> 20,
        t.elapsed().as_secs_f64()
    );

    let n = 1000;
    let t = Instant::now();
    let mut m = Mps::new(n, 16);
    m.apply_gate(&Gate::H(0)).unwrap();
    for q in 1..n {
        m.apply_gate(&Gate::Cnot(q - 1, q)).unwrap();
    }
    println!(
        "mps, {n} qubits ({} KiB, max bond {}): <0..0|ψ> = {:.4} in {:.3}s",
        m.bytes() / 1024,
        m.max_bond_dim(),
        m.amplitude(0).re,
        t.elapsed().as_secs_f64()
    );
}
