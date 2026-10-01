//! Bit-flip repetition code memory experiment: logical error rate vs physical error rate.
//!
//! Runs repeated syndrome rounds on the stabilizer tableau with circuit-level depolarizing noise
//! and decodes using the Union-Find decoder.

use qsim_lab::noise::NoiseModel;
use qsim_lab::qec::RepetitionCode;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    println!("=== Bit-Flip Repetition Code Memory Experiment ===");
    println!("Backend: Stabilizer Tableau (CHP) | Decoder: Union-Find");
    println!("Noise: Circuit-level depolarizing (gates + measurement)\n");

    let distances = [3, 5, 7, 9];
    let ps = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10];
    let shots = 1000;

    print!("{:<8}", "p");
    for &d in &distances {
        print!("  d={:<6}", d);
    }
    println!();
    println!("{}", "-".repeat(40));

    let mut rng = StdRng::seed_from_u64(42);

    for &p in &ps {
        print!("{:<8.4}", p);
        for &d in &distances {
            let code = RepetitionCode::new(d, d);
            let noise = NoiseModel::circuit_level(p, p);
            let res = code.run_experiment(&noise, shots, &mut rng);
            print!("  {:<8.4}", res.logical_error_rate);
        }
        println!();
    }
}
