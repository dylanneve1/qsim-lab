//! Ad-hoc ablation probe for the compiler (development aid).
use qsim_lab::algorithms;
use qsim_lab::compile::plan::{compile_sampling, prepare_statevector, PlanOptions};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

fn t<F: FnMut()>(mut f: F) -> f64 {
    (0..5)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let n = 22;
    let mut c = algorithms::qft(n);
    let mut rng = StdRng::seed_from_u64(1);
    println!(
        "raw apply: {:.4}",
        t(|| {
            let mut s = qsim_lab::StateVectorF32::new(n);
            s.apply_circuit(&c).unwrap();
        })
    );
    println!(
        "prepare no prefix: {:.4}",
        t(|| {
            prepare_statevector::<f32>(&c, false).unwrap();
        })
    );
    println!(
        "prepare prefix: {:.4}",
        t(|| {
            prepare_statevector::<f32>(&c, true).unwrap();
        })
    );
    c.measure_all();
    for (name, o) in [
        ("none", PlanOptions::none()),
        ("all", PlanOptions::default()),
        (
            "no prefix",
            PlanOptions {
                clifford_prefix: false,
                ..PlanOptions::default()
            },
        ),
        (
            "no suffix",
            PlanOptions {
                suffix: false,
                ..PlanOptions::default()
            },
        ),
    ] {
        let p = compile_sampling(&c, o).unwrap();
        println!(
            "{name}: {:.4}",
            t(|| {
                p.sample::<f32, _>(1000, &mut rng).unwrap();
            })
        );
    }
}
