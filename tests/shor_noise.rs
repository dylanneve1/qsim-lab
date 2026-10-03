//! Differential tests of the noisy bit-sliced Shor engine
//! (`qsim_lab::shor::noisy`) against independent gate-by-gate references.
//!
//! * Fixed fault patterns (every site kind, every Pauli): the exact
//!   distribution of the recorded integer (whole measurement tree) equals
//!   the dense state vector (ripple, N = 15) and the sparse state
//!   (windowed, N = 15, 21) applying the same Paulis as real gates
//!   (`Y` as the actual `Y` matrix, the preparation error as an `X` before
//!   the first H) to < 1e-12; for 6–8-bit N the per-round `P(1)` along a
//!   sampled measurement path agrees to < 1e-12.
//! * Sampling: the trajectory sampler at rate `p` vs the stock noisy
//!   `Circuit::run` (independent RNG use and `noise.rs` channels) on the
//!   same circuit; chi-square homogeneity test of the recorded integers.

use qsim_lab::shor::noisy::{self, Fault, NoiseKind, NoisyCircuit, Pauli, Site};
use qsim_lab::shor::{Instance, Oracle};
use qsim_lab::{Gate, SparseState, StateVectorF64};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

trait RefSim: Clone {
    fn g(&mut self, g: &Gate);
    fn p1(&self) -> f64;
    fn coll(&mut self, b: bool);
}
impl RefSim for SparseState {
    fn g(&mut self, g: &Gate) {
        self.apply_gate(g).unwrap();
    }
    fn p1(&self) -> f64 {
        self.prob_one(0)
    }
    fn coll(&mut self, b: bool) {
        self.collapse(0, b);
    }
}
impl RefSim for StateVectorF64 {
    fn g(&mut self, g: &Gate) {
        self.apply_gate(g).unwrap();
    }
    fn p1(&self) -> f64 {
        self.prob_one(0)
    }
    fn coll(&mut self, b: bool) {
        self.collapse(0, b);
    }
}

fn pauli_gate(p: Pauli, q: usize) -> Gate {
    match p {
        Pauli::X => Gate::X(q),
        Pauli::Y => Gate::Y(q),
        Pauli::Z => Gate::Z(q),
    }
}

/// One round on the reference simulator, gate by gate. Returns `(P(1),
/// readout flipped)`.
fn ref_round<S: RefSim>(s: &mut S, nc: &NoisyCircuit, i: usize, y_low: u128, fs: &[Fault]) -> (f64, bool) {
    let at = |site: Site| fs.iter().find(|f| f.site == site).map(|f| f.pauli);
    if at(Site::Prep).is_some() {
        s.g(&Gate::X(0));
    }
    s.g(&Gate::H(0));
    if let Some(p) = at(Site::H1) {
        s.g(&pauli_gate(p, 0));
    }
    for (gi, g) in nc.rounds[i].gates.iter().enumerate() {
        s.g(g);
        for (slot, q) in g.qubits().into_iter().enumerate() {
            if let Some(p) = at(Site::Gate {
                gate: gi as u32,
                slot: slot as u8,
            }) {
                s.g(&pauli_gate(p, q));
            }
        }
    }
    if y_low != 0 {
        s.g(&Gate::Phase(0, Instance::correction(i, y_low)));
    }
    if let Some(p) = at(Site::Phase) {
        s.g(&pauli_gate(p, 0));
    }
    s.g(&Gate::H(0));
    if let Some(p) = at(Site::H2) {
        s.g(&pauli_gate(p, 0));
    }
    (s.p1(), at(Site::Meas).is_some())
}

fn ref_distribution<S: RefSim>(s: S, nc: &NoisyCircuit, faults: &[Fault]) -> Vec<f64> {
    let t = nc.inst.t;
    let mut out = vec![0.0; 1 << t];
    fn walk<S: RefSim>(s: S, nc: &NoisyCircuit, faults: &[Fault], i: usize, y: u128, p: f64, out: &mut [f64]) {
        if i == nc.inst.t {
            out[y as usize] += p;
            return;
        }
        let fs: Vec<Fault> = faults.iter().filter(|f| f.round as usize == i).copied().collect();
        let mut s = s;
        let (p1, flip) = ref_round(&mut s, nc, i, y, &fs);
        for bit in [false, true] {
            let pb = if bit { p1 } else { 1.0 - p1 };
            if pb <= 1e-300 {
                continue;
            }
            let mut c = s.clone();
            c.coll(bit);
            if bit {
                c.g(&Gate::X(0)); // reset
            }
            walk(c, nc, faults, i + 1, y | (u128::from(bit ^ flip) << i), p * pb, out);
        }
    }
    walk(s, nc, faults, 0, 0, 1.0, &mut out);
    out
}

/// Fault patterns covering every site kind and Pauli: 1–4 faults.
fn patterns(nc: &NoisyCircuit, count: usize, seed: u64) -> Vec<Vec<Fault>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::new();
    let t = nc.inst.t as u32;
    // every control site in an early and a late round, every Pauli
    for round in [0, t / 2, t - 1] {
        for site in [Site::Prep, Site::H1, Site::Phase, Site::H2, Site::Meas] {
            for pauli in [Pauli::X, Pauli::Y, Pauli::Z] {
                if matches!(site, Site::Prep | Site::Meas) && pauli != Pauli::X {
                    continue;
                }
                out.push(vec![Fault { round, site, pauli }]);
            }
        }
    }
    for k in 0..count {
        out.push(nc.sample_k(1 + k % 4, &mut rng));
    }
    out
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

#[test]
fn fixed_faults_match_dense_ripple() {
    let inst = Instance::new(15, 7, Oracle::Ripple);
    let nc = NoisyCircuit::new(&inst, NoiseKind::Depolarizing);
    for (j, fs) in patterns(&nc, 40, 1).iter().enumerate() {
        let d = noisy::trajectory_distribution(&nc, fs);
        let r = ref_distribution(StateVectorF64::basis_state(nc.nq, 2), &nc, fs);
        let m = max_diff(&d, &r);
        assert!(m < 1e-12, "pattern {j} {fs:?}: {m:e}");
        assert!((d.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}

#[test]
fn fixed_faults_match_sparse_windowed() {
    for (n, a, w, seed) in [(15u64, 2u64, 2usize, 2u64), (21, 5, 1, 3), (21, 2, 4, 4)] {
        let inst = Instance::new(n, a, Oracle::Windowed(w));
        for kind in [NoiseKind::Depolarizing, NoiseKind::BitFlip, NoiseKind::PhaseFlip] {
            let nc = NoisyCircuit::new(&inst, kind);
            for (j, fs) in patterns(&nc, 30, seed).iter().enumerate() {
                let d = noisy::trajectory_distribution(&nc, fs);
                let r = ref_distribution(SparseState::basis_state(nc.nq, 2), &nc, fs);
                let m = max_diff(&d, &r);
                assert!(m < 1e-12, "N={n} w={w} {kind:?} pattern {j} {fs:?}: {m:e}");
            }
        }
    }
}

/// Larger N (6–8 bits): per-round P(1) along one sampled path.
#[test]
fn fixed_faults_match_sparse_on_a_path() {
    for (n, a, seed) in [(35u64, 2u64, 5u64), (143, 5, 6), (247, 7, 7)] {
        let inst = Instance::new(n, a, Oracle::Windowed(4));
        let nc = NoisyCircuit::new(&inst, NoiseKind::Depolarizing);
        let mut rng = StdRng::seed_from_u64(seed);
        for k in [1usize, 2, 3, 6] {
            for _ in 0..4 {
                let fs = nc.sample_k(k, &mut rng);
                let mut s = noisy::NoisyState::<f64>::new();
                let mut r = SparseState::basis_state(nc.nq, 2);
                let mut y = 0u128;
                for i in 0..inst.t {
                    let fi: Vec<Fault> = fs.iter().filter(|f| f.round as usize == i).copied().collect();
                    let (p, flip) = s.round(&nc, i, y, &fi, usize::MAX).unwrap();
                    let (pr, flip_r) = ref_round(&mut r, &nc, i, y, &fi);
                    assert_eq!(flip, flip_r);
                    assert!((p - pr).abs() < 1e-12, "N={n} k={k} round {i}: {p} vs {pr}");
                    let bit = if p < 1e-9 { false } else if p > 1.0 - 1e-9 { true } else { rng.random::<f64>() < p };
                    s.collapse(bit);
                    r.collapse(0, bit);
                    if bit {
                        r.apply_gate(&Gate::X(0)).unwrap();
                    }
                    y |= u128::from(bit ^ flip) << i;
                }
            }
        }
    }
}

/// Chi-square homogeneity statistic and degrees of freedom of two samples
/// (bins with expected count < 5 in either sample are pooled).
pub fn chi2_two_sample(a: &[u64], b: &[u64]) -> (f64, usize) {
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

/// Wilson–Hilferty upper-tail p-value of a chi-square statistic.
pub fn chi2_pvalue(chi: f64, df: usize) -> f64 {
    let k = df as f64;
    let z = ((chi / k).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / (2.0 / (9.0 * k)).sqrt();
    0.5 * erfc(z / std::f64::consts::SQRT_2)
}

fn erfc(x: f64) -> f64 {
    // Abramowitz–Stegun 7.1.26 (|err| < 1.5e-7)
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let e = y * (-x * x).exp();
    if x >= 0.0 {
        e
    } else {
        2.0 - e
    }
}

#[test]
fn sampler_matches_stock_noisy_circuit_run() {
    // windowed N = 15 (w = 1, 21 qubits): sparse reference; ~2 faults per run
    let inst = Instance::new(15, 7, Oracle::Windowed(1));
    for kind in [NoiseKind::Depolarizing, NoiseKind::BitFlip] {
        let nc = NoisyCircuit::new(&inst, kind);
        let p = 2.0 / nc.num_locations() as f64;
        let circ = noisy::reference_circuit(&nc, p);
        let m = 3000;
        let t = inst.t;
        let mut ha = vec![0u64; 1 << t];
        let mut hb = vec![0u64; 1 << t];
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..m {
            let fs = nc.sample_p(p, &mut rng);
            let tr = noisy::run_trajectory::<f64, _>(&nc, &fs, usize::MAX, &mut rng);
            ha[tr.measured.unwrap() as usize] += 1;
        }
        let mut rng = StdRng::seed_from_u64(12);
        for _ in 0..m {
            let mut s = SparseState::basis_state(nc.nq, 0);
            let bits = circ.run(&mut s, &mut rng).unwrap();
            let y = bits.iter().enumerate().fold(0usize, |acc, (i, &b)| acc | (usize::from(b) << i));
            hb[y] += 1;
        }
        let (chi, df) = chi2_two_sample(&ha, &hb);
        let pv = chi2_pvalue(chi, df);
        eprintln!("{kind:?}: p = {p:.3e}, chi2 = {chi:.1}, df = {df}, p-value = {pv:.3}");
        assert!(pv > 1e-3, "{kind:?}: chi2 {chi} df {df} p-value {pv}");
        let _ = PI;
    }
}

#[test]
fn ancilla_reset_sampler_matches_stock_circuit_with_resets() {
    let inst = Instance::new(21, 2, Oracle::Windowed(2));
    let nc = NoisyCircuit::new(&inst, NoiseKind::Depolarizing);
    // noiseless: the reset is a no-op
    let mut rng = StdRng::seed_from_u64(5);
    let a = noisy::run_trajectory_opts::<f64, _>(&nc, &[], usize::MAX, true, &mut rng);
    let mut rng = StdRng::seed_from_u64(5);
    let b = noisy::run_trajectory::<f64, _>(&nc, &[], usize::MAX, &mut rng);
    assert_eq!(a.measured, b.measured);
    let p = 1.5 / nc.num_locations() as f64;
    let circ = noisy::reference_circuit_opts(&nc, p, true);
    let m = 2000;
    let mut ha = vec![0u64; 1 << inst.t];
    let mut hb = vec![0u64; 1 << inst.t];
    let mut rng = StdRng::seed_from_u64(21);
    for _ in 0..m {
        let fs = nc.sample_p(p, &mut rng);
        let tr = noisy::run_trajectory_opts::<f64, _>(&nc, &fs, usize::MAX, true, &mut rng);
        ha[tr.measured.unwrap() as usize] += 1;
    }
    let mut rng = StdRng::seed_from_u64(22);
    for _ in 0..m {
        let mut s = SparseState::basis_state(nc.nq, 0);
        let bits = circ.run(&mut s, &mut rng).unwrap();
        let y = bits.iter().enumerate().fold(0usize, |acc, (i, &b)| acc | (usize::from(b) << i));
        hb[y] += 1;
    }
    let (chi, df) = chi2_two_sample(&ha, &hb);
    let pv = chi2_pvalue(chi, df);
    eprintln!("reset: chi2 = {chi:.1}, df = {df}, p-value = {pv:.3}");
    assert!(pv > 1e-3, "chi2 {chi} df {df} p-value {pv}");
}
