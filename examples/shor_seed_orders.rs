//! For `qsim run shor --modulus N --seed S --tries 1` (random base), print the
//! base each seed draws, its order r (from the factorisation of λ(N)) and
//! the support-law prediction of the peak support max(r_odd, r/2) and of
//! the peak RSS (≈ 33 B per element in f32, research/shor/shor.md). Usage:
//! `cargo run --release --example shor_seed_orders -- N lambda_prime... -- seeds`
//! e.g. `... 3631204201 2 7 7 17 37 29453` (λ = 1 815 541 826).
use qsim_lab::algorithms::{gcd, pow_mod};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn main() {
    let args: Vec<u64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().unwrap())
        .collect();
    let n = args[0];
    let primes = &args[1..];
    let lambda: u64 = primes.iter().product();
    for seed in 1..=40u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let a = rng.random_range(2..n - 1);
        if gcd(a, n) > 1 {
            println!("seed={seed} a={a} shares a factor");
            continue;
        }
        assert_eq!(pow_mod(a, lambda, n), 1);
        let mut r = lambda;
        for &p in primes {
            if r % p == 0 && pow_mod(a, r / p, n) == 1 {
                r /= p;
            }
        }
        let nu = r.trailing_zeros();
        let r_odd = r >> nu;
        let peak = if nu == 0 { r } else { r_odd.max(r / 2) };
        let ok = r % 2 == 0 && pow_mod(a, r / 2, n) != n - 1;
        println!(
            "seed={seed:2} a={a} r={r} nu2={nu} peak_support={peak} est_rss_f32={:.2} GB classical_ok={ok}",
            33.0 * peak as f64 / 1e9
        );
    }
}
