//! Statistical validation of the generic noisy trajectory sampler
//! (`shor::noisy_gen`) against independent samplers:
//!
//! * `windowed-opt` (reversible): the stock noisy `Circuit::run`
//!   (`noisy::reference_circuit`, `noise.rs` channels) on the sparse state;
//! * `mbu-lookup`, `mbu` (X-basis measurements, outcome-dependent fix-ups):
//!   a lazy gate-by-gate interpreter of the oracle's *logical* ops on the
//!   sparse state — real `H` + projective measurements with their true
//!   probabilities (no importance weights), fix-ups generated from the
//!   measured outcomes as they happen, noise sampled per location with its
//!   own RNG use (no location indexing, no geometric skipping).
//!
//! Per case: two-sample chi-square homogeneity test on the recorded integer
//! (engine histogram weighted by the importance weights; bins with expected
//! count < 5 pooled; Wilson–Hilferty p-value) and a two-proportion z-test
//! on the success probability (peak criterion).
//!
//! `cargo run --release --example noise_oracles_validate -- M_engine M_ref seed`

use qsim_lab::shor::noisy::{self, NoiseKind, NoisyCircuit};
use qsim_lab::shor::noisy_gen::{self, GenCircuit};
use qsim_lab::shor::{Instance, Oracle};
use qsim_lab::shor_mbu::{self, LOp, MbuLayout, MbuOpts};
use qsim_lab::{Gate, SparseState};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

fn chi2_two_sample(a: &[f64], b: &[f64]) -> (f64, usize) {
    let (na, nb) = (a.iter().sum::<f64>(), b.iter().sum::<f64>());
    let mut bins: Vec<(f64, f64)> = Vec::new();
    let (mut pa, mut pb) = (0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let tot = x + y;
        if tot * na.min(nb) / (na + nb) >= 5.0 {
            bins.push((x, y));
        } else {
            pa += x;
            pb += y;
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
    let y = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
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

/// Independent lazy reference for the MBU oracles.
struct LazyRef {
    s: SparseState,
    rng: StdRng,
    p: f64,
    kind: NoiseKind,
}

impl LazyRef {
    fn noise(&mut self, q: usize) {
        if self.rng.random::<f64>() >= self.p {
            return;
        }
        let g = match self.kind {
            NoiseKind::Depolarizing => match self.rng.random_range(0..3) {
                0 => Gate::X(q),
                1 => Gate::Y(q),
                _ => Gate::Z(q),
            },
            NoiseKind::BitFlip => Gate::X(q),
            NoiseKind::PhaseFlip => Gate::Z(q),
        };
        self.s.apply_gate(&g).unwrap();
    }
    fn flip(&mut self) -> bool {
        !matches!(self.kind, NoiseKind::PhaseFlip) && self.rng.random::<f64>() < self.p
    }
    fn gate(&mut self, g: Gate) {
        self.s.apply_gate(&g).unwrap();
        for q in g.qubits() {
            self.noise(q);
        }
    }
    /// X-basis measurement + reset; returns the recorded outcome.
    fn measx(&mut self, q: usize) -> bool {
        self.s.apply_gate(&Gate::H(q)).unwrap();
        let p1 = self.s.prob_one(q);
        let m = self.rng.random::<f64>() < p1;
        self.s.collapse(q, m);
        if m {
            self.s.apply_gate(&Gate::X(q)).unwrap();
        }
        let rec = m ^ self.flip();
        if self.flip() {
            self.s.apply_gate(&Gate::X(q)).unwrap();
        }
        rec
    }
    fn exec(&mut self, ops: &[LOp]) {
        for op in ops {
            match op {
                LOp::G(g) => self.gate(*g),
                LOp::GlobalNeg => {}
                LOp::And(a, b, t) => self.gate(Gate::Ccx(*a, *b, *t)),
                LOp::UnAnd(a, b, t) => {
                    if self.measx(*t) {
                        self.gate(Gate::Cz(*a, *b));
                    }
                }
                LOp::Lookup(s) => self.exec(&shor_mbu::lookup_ops(s)),
                LOp::FlagCompute(f) => self.exec(&f.compute),
                LOp::FlagUncompute(f) => {
                    if self.measx(f.t) {
                        self.exec(&f.fix);
                    }
                }
                LOp::Unlookup(s) => {
                    if !s.meas_unlookup {
                        self.exec(&shor_mbu::lookup_ops(s));
                        continue;
                    }
                    let mut mask = 0u64;
                    for (j, &q) in s.out.iter().enumerate() {
                        if self.measx(q) {
                            mask |= 1 << j;
                        }
                    }
                    let g: Vec<bool> = s
                        .table
                        .iter()
                        .map(|&t| (t & mask).count_ones() & 1 == 1)
                        .collect();
                    if g.iter().any(|&b| b) {
                        let scratch: Vec<usize> = s.and.iter().chain(&s.out).copied().collect();
                        self.exec(&shor_mbu::phase_table(s.ctrl, &s.addr, &g, &scratch));
                    }
                }
            }
        }
    }
}

fn lazy_run(inst: &Instance, o: &MbuOpts, w: usize, kind: NoiseKind, p: f64, seed: u64) -> u128 {
    let lay = MbuLayout::new(inst.m, w, o);
    let mut r = LazyRef {
        s: SparseState::new(lay.num_qubits()),
        rng: StdRng::seed_from_u64(seed),
        p,
        kind,
    };
    r.s.apply_gate(&Gate::X(1)).unwrap();
    let mut y = 0u128;
    for i in 0..inst.t {
        let mult = inst.mults[inst.t - 1 - i];
        let ops = shor_mbu::controlled_ua_ops(&lay, mult, inst.n_mod, o);
        if r.flip() {
            r.s.apply_gate(&Gate::X(0)).unwrap(); // Prep
        }
        r.s.apply_gate(&Gate::H(0)).unwrap();
        r.noise(0);
        r.exec(&ops);
        let y_low = y & ((1u128 << i) - 1);
        if y_low != 0 {
            r.s.apply_gate(&Gate::Phase(0, Instance::correction(i, y_low)))
                .unwrap();
        }
        r.noise(0);
        r.s.apply_gate(&Gate::H(0)).unwrap();
        r.noise(0);
        let p1 = r.s.prob_one(0);
        let bit = r.rng.random::<f64>() < p1;
        r.s.collapse(0, bit);
        if bit {
            r.s.apply_gate(&Gate::X(0)).unwrap();
        }
        if bit ^ r.flip() {
            y |= 1 << i;
        }
    }
    y
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let me: usize = args.first().map_or(20000, |s| s.parse().unwrap());
    let mr: usize = args.get(1).map_or(3000, |s| s.parse().unwrap());
    let seed: u64 = args.get(2).map_or(1, |s| s.parse().unwrap());
    let cases: Vec<(u64, u64, Oracle)> = vec![
        (15, 7, Oracle::WindowedOpt(2)),
        (21, 2, Oracle::WindowedOpt(2)),
        (15, 7, Oracle::WindowedMbuLookup(2)),
        (21, 2, Oracle::WindowedMbuLookup(2)),
        (15, 7, Oracle::WindowedMbu(2)),
        (21, 2, Oracle::WindowedMbu(2)),
    ];
    println!("case,kind,p,Lbar,M_engine,M_ref,w_ne1,mean_w,chi2,df,chi2_pvalue,succ_engine,succ_ref,z,z_pvalue,secs_engine,secs_ref");
    for (n, a, oracle) in cases {
        let inst = Instance::new(n, a, oracle);
        let r = noisy::order_of(a, n);
        let t = inst.t;
        for kind in [
            NoiseKind::Depolarizing,
            NoiseKind::BitFlip,
            NoiseKind::PhaseFlip,
        ] {
            let gc = GenCircuit::new(&inst, kind);
            let mut lr = StdRng::seed_from_u64(seed ^ 0xABCD);
            let lbar = (0..20)
                .map(|_| gc.resolve_rng(&mut lr).num_locations() as f64)
                .sum::<f64>()
                / 20.0;
            let p = 1.5 / lbar;
            let t0 = std::time::Instant::now();
            let ys: Vec<(u128, f64)> = (0..me)
                .into_par_iter()
                .map(|j| {
                    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(1_000_003) + j as u64);
                    let res = gc.resolve_rng(&mut rng);
                    let fs = res.sample_p(p, &mut rng);
                    let tr = noisy_gen::run_trajectory::<u128, f64, _>(
                        &gc,
                        &res,
                        &fs,
                        usize::MAX,
                        false,
                        &mut rng,
                    );
                    (tr.measured.unwrap(), tr.weight)
                })
                .collect();
            let se = t0.elapsed().as_secs_f64();
            let t1 = std::time::Instant::now();
            let yr: Vec<u128> = match oracle {
                Oracle::WindowedOpt(_) => {
                    let nc = NoisyCircuit::new(&inst, kind);
                    let circ = noisy::reference_circuit(&nc, p);
                    (0..mr)
                        .into_par_iter()
                        .map(|j| {
                            let mut rng = StdRng::seed_from_u64(
                                seed.wrapping_mul(7_000_003) + 0x5555 + j as u64,
                            );
                            let mut s = SparseState::new(nc.nq);
                            let bits = circ.run(&mut s, &mut rng).unwrap();
                            bits.iter()
                                .enumerate()
                                .fold(0u128, |acc, (i, &b)| acc | (u128::from(b) << i))
                        })
                        .collect()
                }
                Oracle::WindowedMbu(w) | Oracle::WindowedMbuLookup(w) => {
                    let o = if matches!(oracle, Oracle::WindowedMbu(_)) {
                        MbuOpts::ALL
                    } else {
                        MbuOpts::LOOKUPS
                    };
                    (0..mr)
                        .into_par_iter()
                        .map(|j| {
                            lazy_run(
                                &inst,
                                &o,
                                w,
                                kind,
                                p,
                                seed.wrapping_mul(7_000_003) + 0x7777 + j as u64,
                            )
                        })
                        .collect()
                }
                _ => unreachable!(),
            };
            let sr = t1.elapsed().as_secs_f64();
            let mut ha = vec![0f64; 1 << t];
            let mut hb = vec![0f64; 1 << t];
            for &(y, w) in &ys {
                ha[y as usize] += w;
            }
            for &y in &yr {
                hb[y as usize] += 1.0;
            }
            let (chi, df) = chi2_two_sample(&ha, &hb);
            let pv = chi2_pvalue(chi, df);
            let wsum: f64 = ys.iter().map(|e| e.1).sum();
            let s1 = ys
                .iter()
                .filter(|e| noisy_gen::peak_ok(e.0, t, r))
                .map(|e| e.1)
                .sum::<f64>()
                / wsum;
            let s2 = yr.iter().filter(|&&y| noisy_gen::peak_ok(y, t, r)).count() as f64 / mr as f64;
            let pool = (s1 * me as f64 + s2 * mr as f64) / (me + mr) as f64;
            let se_ = (pool * (1.0 - pool) * (1.0 / me as f64 + 1.0 / mr as f64)).sqrt();
            let z = if se_ > 0.0 { (s1 - s2) / se_ } else { 0.0 };
            let zp = erfc(z.abs() / std::f64::consts::SQRT_2);
            let wne1 = ys.iter().filter(|e| (e.1 - 1.0).abs() > 1e-9).count();
            println!(
                "N={n} {oracle:?},{},{p:.4e},{lbar:.0},{me},{mr},{wne1},{:.5},{chi:.2},{df},{pv:.4},{s1:.4},{s2:.4},{z:.3},{zp:.4},{se:.1},{sr:.1}",
                kind.name(),
                wsum / me as f64
            );
        }
    }
}
