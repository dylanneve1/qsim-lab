//! The Poisson-hit `FastSampler` draws the same distribution as
//! `SymPhaseSampler` (which `tests/symphase.rs` checks exactly against the
//! tableau). Here:
//! * the hit algebra, exactly (series over the hit count, every pattern);
//! * the Poisson and uniform-integer primitives, statistically;
//! * whole-circuit outcome distributions of small random circuits against
//!   the exact distribution (chi-square), with a perturbed-circuit control
//!   that must be rejected;
//! * surface-code detector marginals and pair rates against the old path.

use qsim_lab::stabilizer::fast_sampler::{
    hit_identity_prob, hit_rate, poisson, uniform_below, FastSampler, WyRand,
};
use qsim_lab::stabilizer::symphase::{SymPhaseSampler, VarDist};
use qsim_lab::{Circuit, Gate, NoiseModel, SurfaceCode};
use rand::rngs::{SmallRng, StdRng};
use rand::{Rng, RngCore, SeedableRng};
use std::collections::HashMap;
use std::f64::consts::FRAC_PI_2;

type Dist = HashMap<Vec<bool>, f64>;

/// Distribution of the net pattern after Poisson(lambda) hits, each a
/// uniform non-zero element of Z_2^b, by summing the series over the hit
/// count (independent of the closed form).
fn hit_series(lambda: f64, b: u32) -> Vec<f64> {
    let n = 1usize << b;
    let m = (n - 1) as f64;
    let mut cur = vec![0.0; n];
    cur[0] = 1.0;
    let mut acc = vec![0.0; n];
    let mut pk = (-lambda).exp();
    for k in 0..200 {
        for x in 0..n {
            acc[x] += pk * cur[x];
        }
        let mut nxt = vec![0.0; n];
        for x in 0..n {
            for s in 1..n {
                nxt[x ^ s] += cur[x] / m;
            }
        }
        cur = nxt;
        pk *= lambda / (k + 1) as f64;
    }
    acc
}

#[test]
fn hit_model_reproduces_every_group_distribution_exactly() {
    for (dist, b) in [
        (VarDist::Flip as fn(f64) -> VarDist, 1u32),
        (VarDist::Depol1, 2),
        (VarDist::Depol2, 4),
    ] {
        let m = (1u64 << b) - 1;
        for p in [1e-6, 1e-4, 1e-3, 3e-3, 0.01, 0.1, 0.25] {
            let lam = hit_rate(p, m);
            assert!((hit_identity_prob(lam, m) - (1.0 - p)).abs() < 1e-15);
            let series = hit_series(lam, b);
            // pattern bit i = variable i of the group (VarDist::outcomes)
            for (pat, want) in dist(p).outcomes() {
                let got = series[pat as usize];
                assert!(
                    (got - want).abs() < 1e-14,
                    "b={b} p={p} pattern {pat}: series {got} vs {want}"
                );
            }
        }
    }
}

#[test]
fn poisson_moments_and_pmf() {
    let mut rng = SmallRng::seed_from_u64(1);
    for lam in [0.3, 2.0, 9.99, 10.0, 37.5, 512.0, 20_000.0] {
        let n = 400_000;
        let xs: Vec<f64> = (0..n).map(|_| poisson(&mut rng, lam) as f64).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let zm = (mean - lam) / (lam / n as f64).sqrt();
        // Var(sample var) ~ (mu4 - sigma^4)/n with mu4 = lam(1+3 lam)
        let zv = (var - lam) / ((lam * (1.0 + 3.0 * lam) - lam * lam) / n as f64).sqrt();
        assert!(
            zm.abs() < 5.0 && zv.abs() < 5.0,
            "lam={lam} zm={zm} zv={zv}"
        );
        if lam < 50.0 {
            // chi-square over cells with expectation >= 20
            let mut counts = HashMap::<u64, f64>::new();
            for &x in &xs {
                *counts.entry(x as u64).or_default() += 1.0;
            }
            let (mut chi, mut df, mut lpk) = (0.0, 0usize, -lam);
            for k in 0..400u64 {
                if k > 0 {
                    lpk += lam.ln() - (k as f64).ln();
                }
                let e = n as f64 * lpk.exp();
                if e >= 20.0 {
                    let o = counts.get(&k).copied().unwrap_or(0.0);
                    chi += (o - e) * (o - e) / e;
                    df += 1;
                }
            }
            let df = df as f64;
            assert!(
                chi < df + 6.0 * (2.0 * df).sqrt(),
                "lam={lam} chi={chi} df={df}"
            );
        }
    }
}

#[test]
fn uniform_below_is_uniform() {
    let mut rng = WyRand(7);
    for n in [1u64, 3, 15, 1000, (1 << 40) + 3] {
        let cells = n.min(1000);
        let draws = 300_000u64;
        let mut c = vec![0f64; cells as usize];
        for _ in 0..draws {
            let x = uniform_below(&mut rng, n);
            assert!(x < n);
            c[(x * cells / n) as usize] += 1.0;
        }
        // cells of (near-)equal size; for n = 2^40 + 3 the size difference is negligible
        let e = draws as f64 / cells as f64;
        let chi: f64 = c.iter().map(|o| (o - e) * (o - e) / e).sum();
        let df = (cells - 1).max(1) as f64;
        assert!(chi < df + 6.0 * (2.0 * df).sqrt() + 10.0, "n={n} chi={chi}");
    }
}

/// Exact distribution of the sampler's affine map (as in tests/symphase.rs).
fn sampler_dist(s: &SymPhaseSampler) -> Dist {
    let mut assignments: Vec<(Vec<u64>, f64)> = vec![(vec![0; s.num_vars()], 1.0)];
    for g in s.groups() {
        let mut nxt = vec![];
        for (vals, p) in &assignments {
            for (pat, pp) in g.dist.outcomes() {
                if pp == 0.0 {
                    continue;
                }
                let mut v = vals.clone();
                for k in 0..g.dist.len() {
                    v[g.first as usize + k] = ((pat >> k) & 1) as u64;
                }
                nxt.push((v, p * pp));
            }
        }
        assignments = nxt;
    }
    let mut acc = Dist::new();
    let mut out = vec![0u64; s.num_measurements()];
    for (vals, p) in assignments {
        s.eval(&vals, &mut out);
        let key: Vec<bool> = out.iter().map(|w| w & 1 == 1).collect();
        *acc.entry(key).or_insert(0.0) += p;
    }
    acc
}

fn fast_counts<R: RngCore>(f: &FastSampler, shots: usize, words: usize, rng: &mut R) -> Dist {
    let rows = f.rows();
    let st = f.stride();
    let mut out = vec![0u64; st * words];
    let mut acc = Dist::new();
    let mut done = 0;
    while done < shots {
        f.sample_batch(rng, &mut out);
        for s in 0..64 * words {
            let (w, b) = (s / 64, s % 64);
            let key: Vec<bool> = (0..rows).map(|r| (out[w * st + r] >> b) & 1 == 1).collect();
            *acc.entry(key).or_insert(0.0) += 1.0;
        }
        done += 64 * words;
    }
    acc
}

/// Chi-square of observed counts against an exact distribution (cells with
/// expectation < 5 pooled); returns (statistic, df).
fn chi_square(obs: &Dist, exact: &Dist) -> (f64, f64) {
    let n: f64 = obs.values().sum();
    let (mut chi, mut df) = (0.0, 0.0);
    let (mut po, mut pe) = (0.0, 0.0);
    for (k, &p) in exact {
        let e = n * p;
        let o = obs.get(k).copied().unwrap_or(0.0);
        if e >= 5.0 {
            chi += (o - e) * (o - e) / e;
            df += 1.0;
        } else {
            po += o;
            pe += e;
        }
    }
    for (k, &o) in obs {
        assert!(
            exact.get(k).copied().unwrap_or(0.0) > 0.0,
            "sampled an impossible outcome {k:?}"
        );
        let _ = o;
    }
    if pe > 0.0 {
        chi += (po - pe) * (po - pe) / pe.max(1e-300);
        df += 1.0;
    }
    (chi, (df - 1.0f64).max(1.0))
}

/// Standard-normal equivalent of a chi-square statistic (Wilson-Hilferty).
fn wilson_hilferty(chi: f64, df: f64) -> f64 {
    let v = 2.0 / (9.0 * df);
    ((chi / df).cbrt() - (1.0 - v)) / v.sqrt()
}

fn random_noisy_circuit(n: usize, len: usize, rng: &mut StdRng) -> Circuit {
    let mut c = Circuit::new(n);
    for _ in 0..len {
        let a = rng.random_range(0..n);
        let b = (a + rng.random_range(1..n.max(2))) % n;
        let p = [0.01, 0.1, 0.25, 0.4][rng.random_range(0..4)];
        match rng.random_range(0..16) {
            0..=2 => {
                c.gate(Gate::H(a));
            }
            3 => {
                c.gate(Gate::S(a));
            }
            4 => {
                c.gate(Gate::Phase(a, FRAC_PI_2 * rng.random_range(1..4) as f64));
            }
            5..=6 if n > 1 => {
                c.gate(Gate::Cnot(a, b));
            }
            7 if n > 1 => {
                c.gate(Gate::Cz(a, b));
            }
            8 => {
                c.measure(a);
            }
            9 => {
                c.reset(a);
            }
            10 => {
                c.x_flip(a, p);
            }
            11 => {
                c.y_flip(a, p);
            }
            12 => {
                c.z_flip(a, p);
            }
            13 => {
                c.depolarize_1q(a, p);
            }
            14 if n > 1 => {
                c.depolarize_2q(a, b, p);
            }
            _ => {
                c.gate(Gate::X(a));
            }
        }
    }
    c.measure_all();
    c
}

fn enum_size(s: &SymPhaseSampler) -> f64 {
    s.groups()
        .iter()
        .map(|g| g.dist.outcomes().len() as f64)
        .product()
}

/// Small random noisy circuits (rare and dense groups, coins, gate noise):
/// the fast sampler's outcome frequencies pass a chi-square test against the
/// exact distribution; the same samples are rejected against the exact
/// distribution of a perturbed circuit (power check).
#[test]
fn fast_sampler_matches_exact_distribution_on_random_circuits() {
    let mut rng = StdRng::seed_from_u64(11);
    let mut checked = 0;
    let mut tries = 0;
    let mut worst: f64 = 0.0;
    while checked < 60 {
        tries += 1;
        assert!(tries < 5000);
        let n = rng.random_range(1..4);
        let len = rng.random_range(4..14);
        let c = random_noisy_circuit(n, len, &mut rng);
        let noise = match rng.random_range(0..3) {
            0 => NoiseModel::none().with_meas(0.05),
            1 => NoiseModel::gate_depolarizing(0.02, 0.03).with_reset(0.3),
            _ => NoiseModel::none(),
        };
        let s = SymPhaseSampler::new(&c, &noise).unwrap();
        if enum_size(&s) > 2e4 || s.num_measurements() > 8 {
            continue;
        }
        let exact = sampler_dist(&s);
        // cycle through the code paths: blocked u16 table (default), u32
        // table, unblocked table, column-by-column; two RNGs
        let mut f = FastSampler::new(&s);
        match checked % 4 {
            1 => f.set_narrow(false),
            2 => f.set_blocked(false),
            3 => f.force_column_path(),
            _ => {}
        }
        let words = [1, 2, 4, 16][(checked / 4) % 4];
        let obs = if checked % 8 < 4 {
            fast_counts(
                &f,
                1 << 18,
                words,
                &mut SmallRng::seed_from_u64(100 + checked as u64),
            )
        } else {
            fast_counts(
                &f,
                1 << 18,
                words,
                &mut WyRand(0x9e37_79b9 * (1 + checked as u64)),
            )
        };
        let (chi, df) = chi_square(&obs, &exact);
        let z = wilson_hilferty(chi, df);
        worst = worst.max(z);
        // one-sided, Bonferroni over 60 circuits: per-test 3.4e-6, family 2e-4
        assert!(
            z < 4.5,
            "circuit {c:?} noise {noise:?}: chi={chi} df={df} z={z}"
        );
        checked += 1;
    }
    eprintln!("worst standardized chi-square over {checked} circuits: {worst:.2}");
}

#[test]
fn chi_square_test_rejects_a_perturbed_circuit() {
    // one rare channel at 0.02 vs 0.025 (+25%), 2^20 shots
    let build = |p: f64| {
        let mut c = Circuit::new(2);
        c.h(0).cnot(0, 1);
        c.depolarize_2q(0, 1, p);
        c.x_flip(1, 0.01);
        c.measure(0).measure(1);
        c
    };
    let s = SymPhaseSampler::new(&build(0.02), &NoiseModel::none()).unwrap();
    let s_bad = SymPhaseSampler::new(&build(0.025), &NoiseModel::none()).unwrap();
    let f = FastSampler::new(&s_bad);
    let obs = fast_counts(&f, 1 << 20, 4, &mut SmallRng::seed_from_u64(3));
    let (chi, df) = chi_square(&obs, &sampler_dist(&s));
    assert!(
        wilson_hilferty(chi, df) > 10.0,
        "no power: chi={chi} df={df}"
    );
    let (chi, df) = chi_square(&obs, &sampler_dist(&s_bad));
    assert!(wilson_hilferty(chi, df) < 4.5, "chi={chi} df={df}");
}

/// Surface-code detectors (+ observable), d = 5, p = 0.5%: per-row marginals
/// and all pair rates of the fast sampler agree with the original sampler.
#[test]
fn surface_code_detector_statistics_match_old_sampler() {
    let d = 5;
    let sc = SurfaceCode::new(d, d);
    let sets: Vec<Vec<usize>> = sc
        .detector_records()
        .into_iter()
        .chain(std::iter::once(sc.observable_records()))
        .collect();
    let s = SymPhaseSampler::new(
        &sc.build_circuit(),
        &NoiseModel::circuit_level(0.005, 0.005),
    )
    .unwrap()
    .with_parities(&sets);
    let rows = s.num_measurements();
    let f = FastSampler::new(&s);
    assert_eq!(f.layout().1, 0, "detectors: no coins, no dense groups");
    let batches = 1 << 12; // x 256 shots = 2^20
    let shots = (batches * 256) as f64;
    let count = |words: Vec<Vec<u64>>| {
        let mut c1 = vec![0f64; rows];
        let mut c2 = vec![0f64; rows * rows];
        for w in &words {
            for i in 0..rows {
                c1[i] += w[i].count_ones() as f64;
                for j in i + 1..rows {
                    c2[i * rows + j] += (w[i] & w[j]).count_ones() as f64;
                }
            }
        }
        (c1, c2)
    };
    let mut rng = SmallRng::seed_from_u64(21);
    let mut out = vec![0u64; f.stride() * 4];
    let mut fw = Vec::new();
    for _ in 0..batches {
        f.sample_batch(&mut rng, &mut out);
        fw.extend(out.chunks(f.stride()).map(|c| c[..rows].to_vec()));
    }
    let mut rng = StdRng::seed_from_u64(22);
    let mut vals = vec![0u64; s.num_vars()];
    let mut o = vec![0u64; rows];
    let mut ow = Vec::new();
    for _ in 0..batches * 4 {
        s.sample_batch(&mut rng, &mut vals, &mut o);
        ow.push(o.clone());
    }
    // the other code paths on the same circuit
    let mut variants = Vec::new();
    for v in 0..3 {
        let mut fc = f.clone();
        match v {
            0 => fc.force_column_path(),
            1 => fc.set_blocked(false),
            _ => fc.set_narrow(false),
        }
        let mut fcw = Vec::new();
        let mut wy = WyRand(77 + v);
        for _ in 0..batches {
            fc.sample_batch(&mut wy, &mut out);
            fcw.extend(out.chunks(f.stride()).map(|c| c[..rows].to_vec()));
        }
        variants.push(count(fcw));
    }
    let (a1, a2) = count(fw);
    let (b1, b2) = count(ow);
    let z = |a: f64, b: f64| {
        let pp = (a + b) / (2.0 * shots);
        if a + b == 0.0 {
            0.0
        } else {
            (a - b) / shots / (pp * (1.0 - pp) * 2.0 / shots).sqrt()
        }
    };
    let mut zs: Vec<f64> = Vec::new();
    let mut all = vec![(&a1, &a2)];
    all.extend(variants.iter().map(|(x, y)| (x, y)));
    for (x1, x2) in all {
        zs.extend((0..rows).map(|i| z(x1[i], b1[i])));
        for i in 0..rows {
            for j in i + 1..rows {
                if x2[i * rows + j] + b2[i * rows + j] >= 50.0 {
                    zs.push(z(x2[i * rows + j], b2[i * rows + j]));
                }
            }
        }
    }
    let ntests = zs.len() as f64;
    // Bonferroni at 1% family-wise (Gaussian tail bound, slightly conservative)
    let zcrit = (2.0 * (2.0 * ntests / 0.01).ln()).sqrt();
    let max = zs.iter().fold(0f64, |m, z| m.max(z.abs()));
    eprintln!("{} tests, max |z| = {max:.2}, crit {zcrit:.2}", zs.len());
    assert!(max < zcrit);
}
