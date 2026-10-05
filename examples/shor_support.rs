//! Reachable support of the exact gate-level semiclassical Shor circuit.
//!
//! Prediction: before round i (i bits measured, exponents 2^(t-1)..2^(t-i)
//! applied), the work register is supported on {a^(m·2^(t-i)) : 0 <= m < 2^i},
//! whose size is B_i = min(2^i, r / gcd(r, 2^(t-i))). The real support can
//! only be smaller (exact amplitude cancellation). The simulation work is
//! W = Σ_i 2·|S_i|·G_i gate applications (control 0 and 1 branches).
//!
//! For random semiprimes N (seeded) and random bases, runs one seeded
//! trajectory with the windowed oracle and compares the measured
//! support trace with B_i, and the measured gate·branch counter with the
//! prediction from B_i.
//!
//! cargo run --release --example shor_support -- <bits_lo> <bits_hi> <per_size> <seeds>
use qsim_lab::algorithms::gcd;
use qsim_lab::shor::{self, sliced::SlicedState, Instance, Oracle, OrderFindingState};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    let mut d = 2;
    while d * d <= n {
        if n.is_multiple_of(d) {
            return false;
        }
        d += 1;
    }
    true
}

fn order(a: u64, n: u64) -> u64 {
    let mut r = 1;
    let mut x = a % n;
    while x != 1 {
        x = (u128::from(x) * u128::from(a) % u128::from(n)) as u64;
        r += 1;
    }
    r
}

/// B_i = min(2^i, r / gcd(r, 2^(t-i))).
fn bound(r: u64, t: usize, i: usize) -> u64 {
    // gcd(r, 2^k) = 2^min(nu2(r), k)
    let g = 1u64 << (r.trailing_zeros() as usize).min(t - i);
    if i >= 63 {
        r / g
    } else {
        (1u64 << i).min(r / g)
    }
}

fn main() {
    let args: Vec<u64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().unwrap())
        .collect();
    let (lo, hi, per, seeds) = (args[0], args[1], args[2], args[3]);
    let mut rng = StdRng::seed_from_u64(2026);
    println!("N\tbits\ta\tr\tnu2(r)\tseed\trounds\trounds_eq\trounds_lt\tmax_deficit\tfinal_S\tW_measured\tW_predicted\tratio");
    let (mut tot_rounds, mut tot_eq) = (0usize, 0usize);
    for bits in lo..=hi {
        let mut done = 0;
        while done < per {
            let h = bits / 2;
            let p = rng.random_range(1u64 << (h - 1)..1u64 << h) | 1;
            let q = rng.random_range(1u64 << (bits - h - 1)..1u64 << (bits - h)) | 1;
            let n = p * q;
            if p == q || !is_prime(p) || !is_prime(q) || 64 - n.leading_zeros() as u64 != bits {
                continue;
            }
            let a = rng.random_range(2..n - 1);
            if gcd(a, n) != 1 {
                continue;
            }
            done += 1;
            let r = order(a, n);
            let inst = Instance::new(n, a, Oracle::Windowed(4));
            let t = inst.t;
            for seed in 0..seeds {
                let mut srng = StdRng::seed_from_u64(seed);
                let mut s = SlicedState::<f64>::new(&inst);
                s.keep_final = true;
                let mut y = 0u128;
                let mut w_pred: u128 = 0;
                for i in 0..t {
                    // gates of this round's block
                    let (c, _) = shor::sliced::oracle_block(&inst, inst.mults[t - 1 - i]);
                    let g = c.ops.len() as u128;
                    let b_i = bound(r, t, i);
                    w_pred += 2 * u128::from(b_i) * g;
                    s.round(&inst, i, y);
                    let p1 = s.prob_one(0);
                    let bit = srng.random::<f64>() < p1;
                    s.collapse(0, bit);
                    if bit {
                        y |= 1 << i;
                    }
                }
                let mut eq = 0;
                let mut lt = 0;
                let mut maxdef = 0f64;
                for (i, &si) in s.support_trace.iter().enumerate() {
                    let b_i = bound(r, t, i) as usize;
                    assert!(
                        si <= b_i,
                        "support {si} exceeds bound {b_i} (N={n} a={a} i={i})"
                    );
                    if si == b_i {
                        eq += 1;
                    } else {
                        lt += 1;
                        maxdef = maxdef.max(1.0 - si as f64 / b_i as f64);
                    }
                }
                tot_rounds += t;
                tot_eq += eq;
                let wm = s.gate_branch_ops;
                println!(
                    "{n}\t{bits}\t{a}\t{r}\t{}\t{seed}\t{t}\t{eq}\t{lt}\t{maxdef:.3}\t{}\t{wm}\t{w_pred}\t{:.6}",
                    r.trailing_zeros(),
                    s.nnz(),
                    wm as f64 / w_pred as f64
                );
            }
        }
    }
    eprintln!("rounds with |S_i| = B_i: {tot_eq} / {tot_rounds}");
}
