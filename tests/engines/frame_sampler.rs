//! The Pauli-frame detector sampler (`FrameSampler`, used by `sample-x` for
//! short runs) against the exact distribution of the detector matrix:
//! * random small `.stim` programs (every instruction of the subset, nested
//!   REPEAT blocks, high noise so dense groups and coins appear): the full
//!   joint histogram passes a chi-square test against the exact
//!   distribution, enumerated from `compile_stim`'s columns (itself checked
//!   `==` against the SymPhase route in `detector_compiler.rs`);
//! * a perturbed program is rejected (power);
//! * the AVX-512 build of the word loops is bit-identical;
//! * on a d = 5 surface code, per-detector and pair rates agree with the
//!   FastSampler (tables) at 2^20 shots.

use qsim_lab::engines::stabilizer::detector_compiler::{compile_stim, Columns};
use qsim_lab::engines::stabilizer::fast_sampler::{batch_rng, FastSampler};
use qsim_lab::engines::stabilizer::frame_sampler::{FrameSampler, FrameState};
use qsim_lab::io::stim::parse_stim_circuit;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

type Dist = HashMap<Vec<bool>, f64>;

/// Exact distribution of the rows of `c` (relative to the reference): every
/// joint outcome of the variable groups, XOR of the toggled rows.
fn exact(c: &Columns) -> Dist {
    let mut acc: Dist = HashMap::new();
    let mut states: Vec<(Vec<bool>, f64)> = vec![(vec![false; c.rows], 1.0)];
    for g in &c.groups {
        let mut nxt = Vec::new();
        for (rows, p) in &states {
            for (pat, pp) in g.dist.outcomes() {
                if pp == 0.0 {
                    continue;
                }
                let mut r = rows.clone();
                for k in 0..g.dist.len() {
                    if pat >> k & 1 == 1 {
                        for &row in c.col(g.first as usize + k) {
                            r[row as usize] ^= true;
                        }
                    }
                }
                nxt.push((r, p * pp));
            }
        }
        // merge equal row states to keep the enumeration small
        let mut m: HashMap<Vec<bool>, f64> = HashMap::new();
        for (r, p) in nxt {
            *m.entry(r).or_insert(0.0) += p;
        }
        states = m.into_iter().collect();
    }
    for (r, p) in states {
        *acc.entry(r).or_insert(0.0) += p;
    }
    acc
}

fn frame_counts(f: &FrameSampler, words: usize, batches: usize, seed: u64) -> Dist {
    let rows = f.rows();
    let mut st = FrameState::default();
    let mut out = vec![0u64; words * rows];
    let mut acc = Dist::new();
    for b in 0..batches {
        let mut rng = batch_rng(seed, b as u64);
        f.sample_batch(words, &mut rng, &mut st, &mut out);
        for s in 0..64 * words {
            let (w, bit) = (s / 64, s % 64);
            let key: Vec<bool> = (0..rows)
                .map(|r| (out[w * rows + r] >> bit) & 1 == 1)
                .collect();
            *acc.entry(key).or_insert(0.0) += 1.0;
        }
    }
    acc
}

/// Chi-square (cells with expectation < 5 pooled) -> Wilson-Hilferty z.
fn chi_z(obs: &Dist, exact: &Dist) -> f64 {
    let n: f64 = obs.values().sum();
    let (mut chi, mut df, mut po, mut pe) = (0.0, 0.0, 0.0, 0.0);
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
    for k in obs.keys() {
        assert!(
            exact.get(k).copied().unwrap_or(0.0) > 0.0,
            "impossible outcome {k:?}"
        );
    }
    if pe > 0.0 {
        chi += (po - pe) * (po - pe) / pe;
        df += 1.0;
    }
    let df = (df - 1.0f64).max(1.0);
    let v = 2.0 / (9.0 * df);
    ((chi / df).cbrt() - (1.0 - v)) / v.sqrt()
}

/// A random program of the subset (small enough to enumerate exactly).
fn random_program(rng: &mut StdRng) -> String {
    let n = rng.random_range(1..4);
    let mut t = String::new();
    let mut meas = 0usize;
    let pm = [0.0, 0.05][rng.random_range(0..2)];
    let mpre = |p: f64| {
        if p > 0.0 {
            format!("({p})")
        } else {
            String::new()
        }
    };
    let lines = rng.random_range(3..12);
    let body = |t: &mut String, meas: &mut usize, rng: &mut StdRng| {
        let q = rng.random_range(0..n);
        let r = (q + rng.random_range(1..n.max(2))) % n;
        let p = [0.02, 0.1, 0.3][rng.random_range(0..3)];
        match rng.random_range(0..14) {
            0 | 1 => t.push_str(&format!("H {q}\n")),
            2 => t.push_str(&format!(
                "{} {q}\n",
                ["S", "S_DAG", "X", "Y", "Z"][rng.random_range(0..5)]
            )),
            3 | 4 if n > 1 => t.push_str(&format!(
                "{} {q} {r}\n",
                ["CX", "CZ", "SWAP"][rng.random_range(0..3)]
            )),
            5 => t.push_str(&format!("{} {q}\n", ["R", "RX"][rng.random_range(0..2)])),
            6 | 7 => {
                let name = ["M", "MX", "MR", "MRX"][rng.random_range(0..4)];
                t.push_str(&format!("{name}{} {q}\n", mpre(pm)));
                *meas += 1;
            }
            8 => t.push_str(&format!(
                "{}({p}) {q}\n",
                ["X_ERROR", "Y_ERROR", "Z_ERROR"][rng.random_range(0..3)]
            )),
            9 => t.push_str(&format!("DEPOLARIZE1({p}) {q}\n")),
            10 if n > 1 => t.push_str(&format!("DEPOLARIZE2({p}) {q} {r}\n")),
            11 if *meas > 0 => {
                let k = rng.random_range(1..=(*meas).min(3));
                t.push_str(&format!("DETECTOR rec[-{k}]\n"));
            }
            _ => t.push_str(&format!("H {q}\n")),
        }
    };
    for _ in 0..lines {
        body(&mut t, &mut meas, rng);
    }
    if rng.random_bool(0.5) {
        t.push_str("REPEAT 2 {\n");
        let before = meas;
        for _ in 0..rng.random_range(1..4) {
            body(&mut t, &mut meas, rng);
        }
        meas = before + 2 * (meas - before);
        t.push_str("}\n");
    }
    t.push_str(&format!("M{}", mpre(pm)));
    for q in 0..n {
        t.push_str(&format!(" {q}"));
    }
    t.push('\n');
    meas += n;
    for _ in 0..rng.random_range(1..4) {
        let k = rng.random_range(1..=meas.min(4));
        t.push_str(&format!("DETECTOR rec[-{k}] rec[-1]\n"));
    }
    t.push_str("OBSERVABLE_INCLUDE(0) rec[-1]\n");
    t
}

#[test]
fn frame_sampler_matches_exact_distribution_on_random_programs() {
    let mut rng = StdRng::seed_from_u64(31);
    let (mut checked, mut worst) = (0, 0f64);
    while checked < 60 {
        let text = random_program(&mut rng);
        let prog = parse_stim_circuit(&text).unwrap();
        let cols = compile_stim(&prog);
        let size: f64 = cols
            .groups
            .iter()
            .map(|g| g.dist.outcomes().len() as f64)
            .product();
        if size > 3e4 || cols.rows > 10 {
            continue;
        }
        let ex = exact(&cols);
        let mut f = FrameSampler::new(&prog);
        if checked % 2 == 1 {
            f.set_simd(true);
        }
        let words = [1, 2, 4, 16][checked % 4];
        let obs = frame_counts(&f, words, (1 << 18) / (64 * words), 1000 + checked as u64);
        let z = chi_z(&obs, &ex);
        worst = worst.max(z);
        // one-sided, Bonferroni over 60 programs
        assert!(z < 4.5, "program:\n{text}\nz = {z}");
        checked += 1;
    }
    eprintln!("worst standardized chi-square over {checked} programs: {worst:.2}");
}

#[test]
fn frame_sampler_rejects_a_perturbed_program() {
    let text = |p: f64| {
        format!(
            "R 0 1\nH 0\nCX 0 1\nDEPOLARIZE2({p}) 0 1\nX_ERROR(0.01) 1\nM 0 1\nDETECTOR rec[-1] rec[-2]\nDETECTOR rec[-1]\n"
        )
    };
    let good = parse_stim_circuit(&text(0.02)).unwrap();
    let bad = parse_stim_circuit(&text(0.025)).unwrap();
    let ex = exact(&compile_stim(&good));
    let f = FrameSampler::new(&bad);
    let obs = frame_counts(&f, 16, (1 << 20) / 1024, 3);
    assert!(chi_z(&obs, &ex) > 10.0, "no power");
    let ex_bad = exact(&compile_stim(&bad));
    assert!(chi_z(&obs, &ex_bad) < 4.5);
}

/// AVX-512 build of the word loops: same stream, bit-identical output; and
/// the per-detector / pair rates of a d = 5 surface code agree with the
/// FastSampler at 2^20 shots.
#[test]
fn frame_sampler_surface_code_statistics() {
    let path = format!(
        "{}/tests/data/stim_rotated_memory_x_d5_p0.002.stim",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).unwrap();
    let prog = parse_stim_circuit(&text).unwrap();
    let f = FrameSampler::new(&prog);
    let mut fs = FrameSampler::new(&prog);
    if fs.set_simd(true) {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        f.write_ptb64(10_000, 7, 8, &mut a).unwrap();
        fs.write_ptb64(10_000, 7, 8, &mut b).unwrap();
        assert!(a == b, "AVX-512 frame sampler differs");
    }
    let rows = f.rows();
    let shots = 1usize << 20;
    let (mut fr, mut fa) = (Vec::new(), Vec::new());
    f.write_ptb64(shots, 11, 16, &mut fr).unwrap();
    FastSampler::from_columns(compile_stim(&prog), true)
        .write_ptb64(shots, 12, 1, &mut fa)
        .unwrap();
    let words = |b: &[u8]| -> Vec<u64> {
        b.chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let count = |w: &[u64]| {
        let mut c1 = vec![0f64; rows];
        let mut c2 = vec![0f64; rows * rows];
        for blk in w.chunks_exact(rows) {
            for i in 0..rows {
                c1[i] += blk[i].count_ones() as f64;
                for j in i + 1..rows {
                    c2[i * rows + j] += (blk[i] & blk[j]).count_ones() as f64;
                }
            }
        }
        (c1, c2)
    };
    let (a1, a2) = count(&words(&fr));
    let (b1, b2) = count(&words(&fa));
    let n = shots as f64;
    let z = |a: f64, b: f64| {
        let pp = (a + b) / (2.0 * n);
        if a + b == 0.0 {
            0.0
        } else {
            (a - b) / n / (pp * (1.0 - pp) * 2.0 / n).sqrt()
        }
    };
    let mut zs: Vec<f64> = (0..rows).map(|i| z(a1[i], b1[i])).collect();
    for i in 0..rows {
        for j in i + 1..rows {
            if a2[i * rows + j] + b2[i * rows + j] >= 50.0 {
                zs.push(z(a2[i * rows + j], b2[i * rows + j]));
            }
        }
    }
    let zcrit = (2.0 * (2.0 * zs.len() as f64 / 0.01).ln()).sqrt();
    let max = zs.iter().fold(0f64, |m, z| m.max(z.abs()));
    eprintln!("{} tests, max |z| = {max:.2}, crit {zcrit:.2}", zs.len());
    assert!(max < zcrit);
}
