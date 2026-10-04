//! Branching-rank profiles and demos (research/theory-rank.md).
//!
//! ```text
//! theory_rank profile <spec> <seed> <cap>     # one JSON line: r_k profile + atlas d
//! theory_rank verify  <spec> <seed>           # rank engine vs state vector (n <= 20)
//! theory_rank grover  <n> <it> <seed> <samples>   # exact Grover, Toffoli-ladder oracle
//! ```
use num_complex::Complex64 as C64;
use qsim_lab::magic_atlas::{self, families, AtlasOptions};
use qsim_lab::stab_rank::{bits, RankState};
use qsim_lab::StateVector;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(|s| s.as_str()) {
        Some("profile") => profile(&a[2], a[3].parse().unwrap(), a[4].parse().unwrap()),
        Some("verify") => verify(&a[2], a[3].parse().unwrap()),
        Some("grover") => grover(
            a[2].parse().unwrap(),
            a[3].parse().unwrap(),
            a[4].parse().unwrap(),
            a[5].parse().unwrap(),
        ),
        _ => eprintln!("usage: see source"),
    }
}

fn profile(spec: &str, seed: u64, cap: usize) {
    let c = families::build(spec, seed).unwrap();
    let opts = AtlasOptions {
        checkpoints: 0,
        entanglement: false,
        cut: None,
        support: false,
    };
    let ap = magic_atlas::profile(&c, &opts).unwrap();
    let mut rs = RankState::new(c.num_qubits);
    rs.max_terms = cap;
    if let Ok(v) = std::env::var("RANK_PM_MAXR") {
        rs.pair_merge_max_r = v.parse().unwrap();
    }
    if let Ok(v) = std::env::var("RANK_PM_S") {
        rs.pair_merge_s = v.parse().unwrap();
    }
    let t0 = Instant::now();
    let ok = rs.run(&c);
    let secs = t0.elapsed().as_secs_f64();
    // ground-truth nullity at 20 checkpoints (n <= 12)
    let (mut nu_max, mut nu_end, mut nu_at) = (-1.0f64, -1.0f64, Vec::new());
    if c.num_qubits <= 12 {
        let gl: Vec<_> = c.gates().cloned().collect();
        let mut sv = StateVector::<f64>::new(c.num_qubits);
        let g = gl.len();
        let marks: Vec<usize> = (1..=20).map(|i| (i * g / 20).max(1)).collect();
        for (k, gate) in gl.iter().enumerate() {
            sv.apply_gate(gate).unwrap();
            if marks.contains(&(k + 1)) {
                let amps: Vec<C64> = sv
                    .amplitudes()
                    .iter()
                    .map(|a| C64::new(a.re, a.im))
                    .collect();
                let nu = magic_atlas::state_magic(&amps).nullity;
                nu_max = nu_max.max(nu);
                nu_end = nu;
                nu_at.push((nu * 100.0).round() / 100.0);
            }
        }
    }
    let g = c.num_gates();
    let r = &rs.stats.r;
    let prof: Vec<usize> = (1..=20)
        .map(|i| {
            r.get(
                (i * g / 20)
                    .saturating_sub(1)
                    .min(r.len().saturating_sub(1)),
            )
            .copied()
            .unwrap_or(0)
        })
        .collect();
    println!(
        "{{\"spec\":\"{spec}\",\"n\":{},\"gates\":{g},\"toffolis\":{},\"rot\":{},\"t\":{},\"d\":{},\"f\":{},\"ok\":{ok},\"stopped_at\":{},\"max_r\":{},\"r_end\":{},\"branch\":{},\"cliff\":{},\"diag\":{},\"merges\":{},\"pair_merges\":{},\"cancel\":{},\"secs\":{secs:.4},\"prof\":{:?},\"nu_max\":{nu_max:.2},\"nu_end\":{nu_end:.2},\"nu_prof\":{nu_at:?}}}",
        c.num_qubits,
        ap.toffolis,
        ap.rotations,
        ap.t_count,
        ap.d,
        ap.f,
        r.len(),
        rs.stats.max_r,
        rs.rank(),
        rs.stats.branch_events,
        rs.stats.clifford_events,
        rs.stats.diag_events,
        rs.stats.merges,
        rs.stats.pair_merges,
        rs.stats.cancellations,
        prof
    );
}

fn verify(spec: &str, seed: u64) {
    let c = families::build(spec, seed).unwrap();
    assert!(c.num_qubits <= 20);
    let mut sv = StateVector::<f64>::new(c.num_qubits);
    sv.apply_circuit(&c).unwrap();
    let mut rs = RankState::new(c.num_qubits);
    rs.run(&c);
    let v = rs.to_statevector();
    let err = sv
        .amplitudes()
        .iter()
        .zip(&v)
        .map(|(a, b)| (C64::new(a.re, a.im) - b).norm())
        .fold(0.0, f64::max);
    println!(
        "{{\"spec\":\"{spec}\",\"n\":{},\"max_r\":{},\"r_end\":{},\"max_amp_err\":{err:.3e}}}",
        c.num_qubits,
        rs.stats.max_r,
        rs.rank()
    );
}

/// Grover on `n` search qubits with the atlas's Toffoli-ladder oracle
/// (`families::build("grover:n=..,it=..")`, `n − 2` ancillas, so `2n − 2`
/// qubits). Checks the exact amplitudes against the closed form:
/// each iteration is `−(2|s⟩⟨s| − I)(I − 2|w⟩⟨w|)`, so after k iterations
/// `⟨w|Ψ⟩ = (−1)^k sin((2k+1)θ)`, `⟨x|Ψ⟩ = (−1)^k cos((2k+1)θ)/√(N−1)` for
/// `x ≠ w` (ancillas 0), and every amplitude with a dirty ancilla is 0.
fn grover(n: usize, it: usize, seed: u64, samples: usize) {
    let c = families::build(&format!("grover:n={n},it={it}"), seed).unwrap();
    let nq = c.num_qubits;
    let mut rng = StdRng::seed_from_u64(seed);
    let marked: Vec<bool> = (0..n).map(|_| rng.random()).collect();
    let mut rs = RankState::new(nq);
    let t0 = Instant::now();
    assert!(rs.run(&c));
    let secs = t0.elapsed().as_secs_f64();
    // closed form, in long double-ish via f64 (θ = asin 2^{-n/2})
    let theta = (2f64.powf(-(n as f64) / 2.0)).asin();
    let sgn = if it % 2 == 0 { 1.0 } else { -1.0 };
    let aw = sgn * ((2 * it + 1) as f64 * theta).sin();
    let nn = 2f64.powi(n as i32);
    let ao = sgn * ((2 * it + 1) as f64 * theta).cos() / (nn - 1.0).sqrt();
    let wbits: Vec<usize> = (0..n).filter(|&i| marked[i]).collect();
    let amp_w = rs.amplitude(&bits(nq, &wbits));
    let mut err_rel: f64 = ((amp_w - C64::new(aw, 0.0)).norm()) / aw.abs();
    let mut err_dirty: f64 = 0.0;
    let mut rng2 = StdRng::seed_from_u64(seed ^ 0xabc);
    for k in 0..samples {
        let mut on: Vec<usize> = (0..n).filter(|_| rng2.random::<bool>()).collect();
        if on == wbits {
            on.retain(|&q| q != 0);
            if !marked[0] {
                on.insert(0, 0);
            }
        }
        if k % 2 == 1 && n > 2 {
            // dirty ancilla
            on.push(n + rng2.random_range(0..n - 2));
            let a = rs.amplitude(&bits(nq, &on));
            err_dirty = err_dirty.max(a.norm() / ao.abs());
        } else {
            let a = rs.amplitude(&bits(nq, &on));
            err_rel = err_rel.max((a - C64::new(ao, 0.0)).norm() / ao.abs());
        }
    }
    eprintln!(
        "t_merge {:.3} t_pair {:.3} pair_tests {}",
        rs.stats.t_merge, rs.stats.t_pair, rs.stats.pair_tests
    );
    let r = &rs.stats.r;
    // ranks at the iteration boundaries
    // the circuit is n H gates followed by `it` identical iterations
    let per_it = (c.num_gates() - n) / it.max(1);
    let bnd: Vec<usize> = (1..=it).map(|k| r[n + k * per_it - 1]).collect();
    let bnd_max = bnd.iter().copied().max().unwrap_or(0);
    let bnd: Vec<usize> = if bnd.len() > 16 {
        bnd[..16].to_vec()
    } else {
        bnd
    };
    println!(
        "{{\"n_search\":{n},\"qubits\":{nq},\"it\":{it},\"gates\":{},\"toffolis\":{},\"max_r\":{},\"r_end\":{},\"r_at_iteration_ends\":{bnd:?},\"max_r_at_iteration_ends\":{bnd_max},\"branch\":{},\"cliff\":{},\"merges\":{},\"pair_merges\":{},\"cancel\":{},\"secs\":{secs:.3},\"amp_w\":{:.6e},\"amp_w_exact\":{aw:.6e},\"amp_other_exact\":{ao:.6e},\"max_rel_err\":{err_rel:.2e},\"max_dirty_rel\":{err_dirty:.2e}}}",
        c.num_gates(),
        c.gates().filter(|g| matches!(g, qsim_lab::Gate::Ccx(..))).count(),
        rs.stats.max_r,
        rs.rank(),
        rs.stats.branch_events,
        rs.stats.clifford_events,
        rs.stats.merges,
        rs.stats.pair_merges,
        rs.stats.cancellations,
        amp_w.re
    );
}
