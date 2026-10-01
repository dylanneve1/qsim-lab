//! Prepare a Bell pair on three backends and sample it.

use qsim_lab::algorithms;
use qsim_lab::circuit::Simulator;
use qsim_lab::{Mps, StateVectorF64, Tableau};
use rand::rngs::StdRng;
use rand::SeedableRng;

type MakeBackend = fn() -> Box<dyn Simulator>;

fn main() {
    let mut rng = StdRng::seed_from_u64(1);
    let mut c = algorithms::bell();
    c.measure_all();
    let shots = 1000;
    let backends: [(&str, MakeBackend); 3] = [
        ("state vector", || Box::new(StateVectorF64::new(2))),
        ("stabilizer", || Box::new(Tableau::new(2))),
        ("mps", || Box::new(Mps::new(2, 2))),
    ];
    for (name, make) in backends {
        let mut counts = [0usize; 4];
        for _ in 0..shots {
            let mut sim = make();
            let b = c.run(sim.as_mut(), &mut rng).unwrap();
            counts[usize::from(b[0]) | usize::from(b[1]) << 1] += 1;
        }
        println!(
            "{name:>12}: |00> {}  |01> {}  |10> {}  |11> {}",
            counts[0], counts[2], counts[1], counts[3]
        );
    }
    let mut t = Tableau::new(2);
    for g in algorithms::bell().gates() {
        t.apply_gate(g).unwrap();
    }
    println!("stabilizers of the Bell state: {:?}", t.stabilizers());
}
