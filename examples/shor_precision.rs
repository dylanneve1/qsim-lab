//! f32 vs f64 amplitudes in the exact sliced Shor state: total-variation
//! distance between the exact outcome distributions and the change in the
//! probability that one run recovers a factor.
//!
//! cargo run --release --example shor_precision
use qsim_lab::algorithms::gcd;
use qsim_lab::shor::{self, sliced::SlicedState, Instance, Oracle};

fn main() {
    println!("N\ta\tr\tn\tTV(f32,f64)\tmax|dp|\tP_factor_f64\tP_factor_f32\t|dP_factor|");
    for n_mod in [
        15u64, 21, 33, 35, 39, 51, 55, 57, 65, 69, 77, 85, 87, 91, 95, 115, 119, 133, 143, 187,
        209, 221, 247,
    ] {
        let bases: Vec<u64> = (2..n_mod - 1)
            .filter(|&a| gcd(a, n_mod) == 1)
            .step_by(7)
            .take(3)
            .collect();
        for a in bases {
            let inst = Instance::new(n_mod, a, Oracle::Windowed(4));
            let d64 = shor::semiclassical_distribution(&inst, SlicedState::<f64>::new(&inst), 0.0);
            let d32 = shor::semiclassical_distribution(&inst, SlicedState::<f32>::new(&inst), 0.0);
            let tv: f64 = d64
                .iter()
                .zip(&d32)
                .map(|(x, y)| (x - y).abs())
                .sum::<f64>()
                / 2.0;
            let mx = d64
                .iter()
                .zip(&d32)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max);
            let (mut pf64, mut pf32) = (0.0, 0.0);
            let mut r = None;
            for (y, (p64, p32)) in d64.iter().zip(&d32).enumerate() {
                if *p64 == 0.0 && *p32 == 0.0 {
                    continue;
                }
                let (ord, f) = shor::postprocess(n_mod, a, y as u128, inst.t as u32);
                if ord.is_some() && r.is_none() {
                    r = ord;
                }
                if f.is_some() {
                    pf64 += p64;
                    pf32 += p32;
                }
            }
            println!(
                "{n_mod}\t{a}\t{}\t{}\t{tv:.3e}\t{mx:.3e}\t{pf64:.6}\t{pf32:.6}\t{:.3e}",
                r.map_or("-".into(), |r| r.to_string()),
                inst.m,
                (pf64 - pf32).abs()
            );
        }
    }
}
