//! Single-thread kernel microbenchmark: ns per amplitude for a dense k-qubit
//! op (k = number of targets) at the given buffer bit positions.
//! usage: sv_micro <f32|f64> t0[,t1[,t2[,t3]]] ...   (buffer is 2^15 amplitudes)
use qsim_lab::blocked::micro_kernel;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let prec = a[1].as_str();
    println!("| prec | targets | portable ns/amp | avx2+fma ns/amp |");
    println!("|---|---|---|---|");
    for spec in &a[2..] {
        let t: Vec<usize> = spec.split(',').map(|x| x.parse().unwrap()).collect();
        let l = 15;
        let reps = 300;
        let time = |avx: bool| {
            (0..7)
                .map(|_| {
                    if prec == "f32" {
                        micro_kernel::<f32>(l, &t, reps, avx)
                    } else {
                        micro_kernel::<f64>(l, &t, reps, avx)
                    }
                })
                .fold(f64::INFINITY, f64::min)
                * 1e9
                / (1u64 << l) as f64
        };
        println!("| {prec} | {spec} | {:.3} | {:.3} |", time(false), time(true));
    }
}
