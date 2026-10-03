//! Statistical validation of the noisy trajectory sampler
//! (`qsim_lab::shor::noisy`) against the stock noisy `Circuit::run`
//! (independent sampling code in `circuit.rs` / `noise.rs`) on a dense
//! state vector (ripple oracle, N = 15, 16 qubits) and on the sparse state
//! (windowed oracle, N = 15 and 21; too many qubits for a dense vector).
//!
//! For each case: two-sample chi-square homogeneity test on the recorded
//! integer (bins with expected count < 5 pooled; p-value from the
//! Wilson–Hilferty approximation) and a two-proportion z-test on the
//! success probability (peak criterion: y within 2^t/(2r^2)·... of s·2^t/r).
//!
//! `cargo run --release --example shor_noise_validate -- M_engine M_ref seed`

use qsim_lab::shor::noisy::{self, NoiseKind, NoisyCircuit};
use qsim_lab::shor::{Instance, Oracle};
use qsim_lab::{SparseState, StateVectorF64};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

fn chi2_two_sample(a: &[u64], b: &[u64]) -> (f64, usize) {
    let (na, nb) = (a.iter().sum::<u64>() as f64, b.iter().sum::<u64>() as f64);
    let mut bins: Vec<(f64, f64)> = Vec::new();
    let (mut pa, mut pb) = (0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let tot = (x + y) as f64;
        if tot * na.min(nb) / (na + nb) >= 5.0 {
            bins.push((x as f64, y as f64));
        } else {
            pa += x as f64;
            pb += y as f64;
        }
    }
    if pa + pb > 0.0 {
        bins.push((pa, pb));
    }
    let mut chi = 0.0;
    for &(x, y) in &bins {
        let tot = x + y;
        let ea = tot * na / (na + nb);
        let eb = tot * nb / (na + nb);
        chi += (x - ea).powi(2) / ea + (y - eb).powi(2) / eb;
    }
    (chi, bins.len() - 1)
}

fn erfc(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let e = y * (-x * x).exp();
    if x >= 0.0 {
        e
    } else {
        2.0 - e
    }
}

fn chi2_pvalue(chi: f64, df: usize) -> f64 {
    let k = df as f64;
    let z = ((chi / k).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / (2.0 / (9.0 * k)).sqrt();
    0.5 * erfc(z / std::f64::consts::SQRT_2)
}

fn peak_ok(y: u128, r: u64, t: usize) -> bool {
    // |y/2^t - s/r| < 1/(2 r^2) for the nearest s
    let two_t = 1u128 << t;
    let yr = y * u128::from(r);
    let s = (yr + two_t / 2) / two_t;
    let d = yr.abs_diff(s * two_t);
    d * 2 * u128::from(r) < two_t
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let me: usize = args.first().map_or(20000, |s| s.parse().unwrap());
    let mr: usize = args.get(1).map_or(4000, |s| s.parse().unwrap());
    let seed: u64 = args.get(2).map_or(1, |s| s.parse().unwrap());
    // (N, a, oracle, dense reference, ancilla reset after every round)
    let cases: Vec<(u64, u64, Oracle, bool, bool)> = vec![
        (15, 7, Oracle::Ripple, true, false),
        (15, 7, Oracle::Windowed(1), false, false),
        (21, 2, Oracle::Windowed(2), false, false),
        (21, 2, Oracle::Windowed(2), false, true),
    ];
    println!("case,kind,p,L,mean_faults,M_engine,M_ref,chi2,df,chi2_pvalue,succ_engine,succ_ref,z,z_pvalue,secs_engine,secs_ref");
    for (n, a, oracle, dense, reset) in cases {
        let inst = Instance::new(n, a, oracle);
        let r = noisy::order_of(a, n);
        for kind in [NoiseKind::Depolarizing, NoiseKind::BitFlip, NoiseKind::PhaseFlip] {
            let nc = NoisyCircuit::new(&inst, kind);
            let l = nc.num_locations();
            let p = 1.5 / l as f64; // ~1.5 faults per run
            let t = inst.t;
            let t0 = std::time::Instant::now();
            let ys: Vec<u128> = (0..me)
                .into_par_iter()
                .map(|j| {
                    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(1_000_003) + j as u64);
                    let fs = nc.sample_p(p, &mut rng);
                    noisy::run_trajectory_opts::<f64, _>(&nc, &fs, usize::MAX, reset, &mut rng)
                        .measured
                        .unwrap()
                })
                .collect();
            let se = t0.elapsed().as_secs_f64();
            let circ = noisy::reference_circuit_opts(&nc, p, reset);
            let t1 = std::time::Instant::now();
            let yr: Vec<u128> = (0..mr)
                .into_par_iter()
                .map(|j| {
                    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(7_000_003) + 0x5555 + j as u64);
                    let bits = if dense {
                        let mut s = StateVectorF64::new(nc.nq);
                        circ.run(&mut s, &mut rng).unwrap()
                    } else {
                        let mut s = SparseState::new(nc.nq);
                        circ.run(&mut s, &mut rng).unwrap()
                    };
                    bits.iter()
                        .enumerate()
                        .fold(0u128, |acc, (i, &b)| acc | (u128::from(b) << i))
                })
                .collect();
            let sr = t1.elapsed().as_secs_f64();
            let mut ha = vec![0u64; 1 << t];
            let mut hb = vec![0u64; 1 << t];
            for &y in &ys {
                ha[y as usize] += 1;
            }
            for &y in &yr {
                hb[y as usize] += 1;
            }
            let (chi, df) = chi2_two_sample(&ha, &hb);
            let pv = chi2_pvalue(chi, df);
            let s1 = ys.iter().filter(|&&y| peak_ok(y, r, t)).count() as f64 / me as f64;
            let s2 = yr.iter().filter(|&&y| peak_ok(y, r, t)).count() as f64 / mr as f64;
            let pool = (s1 * me as f64 + s2 * mr as f64) / (me + mr) as f64;
            let se_ = (pool * (1.0 - pool) * (1.0 / me as f64 + 1.0 / mr as f64)).sqrt();
            let z = if se_ > 0.0 { (s1 - s2) / se_ } else { 0.0 };
            let zp = erfc(z.abs() / std::f64::consts::SQRT_2);
            println!(
                "N={n} {:?}{}{},{},{p:.4e},{l},{:.3},{me},{mr},{chi:.2},{df},{pv:.4},{s1:.4},{s2:.4},{z:.3},{zp:.4},{se:.1},{sr:.1}",
                oracle,
                if dense { " dense" } else { " sparse" },
                if reset { " +reset" } else { "" },
                kind.name(),
                p * l as f64
            );
        }
    }
}
