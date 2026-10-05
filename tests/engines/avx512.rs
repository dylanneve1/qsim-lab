//! The AVX-512 kernel tier (`BlockConfig::avx512`) must agree with the
//! AVX2+FMA and portable kernels, with gate-by-gate application and with the
//! independent reference state vector of `audit_common`: max |Δamplitude|
//! <= 1e-12 (f64) and <= 1e-5 (f32).
//!
//! Coverage: every target and control position, in particular the lowest
//! bits that sit inside one 512-bit vector (bits 0..3 for f32, 0..2 for f64);
//! every explicit kernel (1-qubit X / real / complex, in-vector and
//! above-vector controls, pairs with and without their CNOT for every
//! in-vector/vertical combination, dense k = 2 and 3 for every pattern of
//! in-vector targets); the generic kernels compiled for AVX-512 (diagonal
//! blocks, swaps, L1 tiling); every `BlockConfig` knob; and edge sizes
//! (registers smaller than one vector, equal to a block, just above it).
//! On a CPU without AVX-512 (or with `QSIM_NO_AVX512` set) the knob falls
//! back to AVX2+FMA and these tests compare the fallback with itself.

#[path = "../audit_common/mod.rs"]
mod audit_common;
#[path = "../common/mod.rs"]
mod common;

use common::random_universal;
use num_complex::Complex64;
use proptest::prelude::*;
use qsim_lab::circuit::{Circuit, Op};
use qsim_lab::engines::blocked::{
    avx512_available, fusion_stats, lower_gates, simd_available, BlockConfig, KOp,
};
use qsim_lab::engines::statevector::{Real, StateVector};
use qsim_lab::gate::Mat2;
use qsim_lab::Gate;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn max_diff<T: Real>(a: &StateVector<T>, b: &StateVector<T>) -> f64 {
    a.amplitudes()
        .iter()
        .zip(b.amplitudes())
        .map(|(x, y)| {
            let d = *x - *y;
            (d.re.to_f64().powi(2) + d.im.to_f64().powi(2)).sqrt()
        })
        .fold(0.0, f64::max)
}

/// A random product state, so every amplitude matters.
fn start<T: Real>(n: usize, seed: u64) -> StateVector<T> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut s = StateVector::<T>::new(n);
    for q in 0..n {
        s.apply_gate(&Gate::Ry(q, rng.random::<f64>() * 3.0))
            .unwrap();
        s.apply_gate(&Gate::Rz(q, rng.random::<f64>() * 3.0))
            .unwrap();
    }
    s
}

/// The kernel sets compared: AVX-512 (falls back to AVX2+FMA without it),
/// AVX2+FMA, portable.
const ISAS: [(bool, bool, &str); 3] = [
    (true, true, "avx512"),
    (true, false, "avx2"),
    (false, false, "portable"),
];

fn tol<T: Real>() -> f64 {
    if std::mem::size_of::<T>() == 4 {
        1e-5
    } else {
        1e-12
    }
}

/// One block geometry: (block bytes, slots, L1 tile bytes, small_n).
type Geom = (usize, usize, usize, usize);

/// Whole register in one block, tiny blocks (gathers, many stages, blocks
/// with fewer index bits than a vector has lanes), blocks of one or two
/// vectors' worth of index bits above the lanes, nested L1 tiles.
const GEOMS: [Geom; 7] = [
    (1 << 20, 6, 0, 64),
    (64, 2, 0, 0),
    (256, 3, 0, 0),
    (512, 1, 0, 0),
    (2048, 4, 256, 0),
    (8192, 6, 1024, 0),
    (64 << 10, 6, 0, 4),
];

fn cfg(g: Geom, simd: bool, avx512: bool) -> BlockConfig {
    BlockConfig {
        block_bytes: g.0,
        slots: g.1,
        l1_tile_bytes: g.2,
        small_n: g.3,
        simd,
        avx512,
        ..BlockConfig::default()
    }
}

#[test]
fn reports_detection() {
    // Documents which paths the other tests exercised.
    eprintln!(
        "avx512 available: {}, avx2+fma available: {}",
        avx512_available(),
        simd_available()
    );
}

/// `QSIM_NO_AVX512` (any value) turns the AVX-512 tier off for the whole
/// process: checked in a child process of this test binary.
#[test]
fn env_var_disables_avx512() {
    if std::env::var_os("QSIM_NO_AVX512").is_some() {
        assert!(!avx512_available());
        return;
    }
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "env_var_disables_avx512", "--test-threads", "1"])
        .env("QSIM_NO_AVX512", "1")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "child failed: {}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("1 passed"));
}

fn mats() -> Vec<(Mat2, &'static str)> {
    let c = |re: f64, im: f64| Complex64::new(re, im);
    let x: Mat2 = [[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]];
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let h: Mat2 = [[c(s, 0.0), c(s, 0.0)], [c(s, 0.0), c(-s, 0.0)]];
    let ry = Gate::Ry(0, 0.7).matrix_1q().unwrap();
    let u = Gate::U(0, 1.1, 0.3, -0.8).matrix_1q().unwrap();
    vec![(x, "x"), (h, "h"), (ry, "ry"), (u, "u")]
}

/// Controlled 1-qubit gates on every target with every control set of size
/// <= 2 (exhaustive below 7 qubits), against `apply_multi_controlled_1q`
/// (the plain, unblocked kernel) for every kernel set and geometry.
fn controlled_u1_case<T: Real>(n: usize) {
    let mut sets: Vec<Vec<usize>> = vec![vec![]];
    for a in 0..n {
        sets.push(vec![a]);
        for b in a + 1..n {
            if n <= 7 || b == a + 1 || a == 0 {
                sets.push(vec![a, b]);
            }
        }
    }
    let init = start::<T>(n, n as u64);
    for (m, name) in mats() {
        for t in 0..n {
            for ctrl in sets.iter().filter(|s| !s.contains(&t)) {
                let mut want = init.clone();
                want.apply_multi_controlled_1q(ctrl, t, &m);
                let mask = ctrl.iter().map(|&c| 1usize << c).sum();
                let ops = [KOp::U1 {
                    q: t,
                    m,
                    ctrl: mask,
                }];
                for g in GEOMS {
                    for (simd, avx512, isa) in ISAS {
                        let mut got = init.clone();
                        got.apply_kops_blocked(&ops, &cfg(g, simd, avx512));
                        let d = max_diff(&got, &want);
                        assert!(
                            d <= tol::<T>(),
                            "{name} t={t} ctrl={ctrl:?} n={n} {g:?} {isa}: {d:e}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn controlled_u1_every_position() {
    for n in [1, 2, 3, 4, 5, 6, 7, 9] {
        controlled_u1_case::<f32>(n);
        controlled_u1_case::<f64>(n);
    }
}

/// Two 1-qubit gates on bits `t1 != t2` followed by nothing or a CNOT in
/// either direction: the shape the executor turns into one pair sweep.
fn pair_circuit(n: usize, t1: usize, t2: usize, cx: u8, real: bool, rng: &mut StdRng) -> Circuit {
    let mut c = Circuit::new(n);
    for q in [t1, t2] {
        if real {
            c.ry(q, rng.random::<f64>() * 3.0);
        } else {
            c.gate(Gate::U(
                q,
                rng.random::<f64>() * 3.0,
                rng.random::<f64>() * 3.0,
                rng.random::<f64>() * 3.0,
            ));
        }
    }
    match cx {
        1 => {
            c.cnot(t1, t2);
        }
        2 => {
            c.cnot(t2, t1);
        }
        _ => {}
    }
    c
}

fn pairs_case<T: Real>(n: usize) {
    let mut rng = StdRng::seed_from_u64(77 + n as u64);
    let init = start::<T>(n, 5);
    for t1 in 0..n {
        for t2 in 0..n {
            if t1 == t2 {
                continue;
            }
            for cx in 0..3u8 {
                for real in [true, false] {
                    let c = pair_circuit(n, t1, t2, cx, real, &mut rng);
                    let mut want = init.clone();
                    want.apply_circuit(&c).unwrap();
                    for g in GEOMS {
                        for (simd, avx512, isa) in ISAS {
                            let mut got = init.clone();
                            got.apply_circuit_blocked(&c, &cfg(g, simd, avx512))
                                .unwrap();
                            let d = max_diff(&got, &want);
                            assert!(
                                d <= tol::<T>(),
                                "pair t1={t1} t2={t2} cx={cx} real={real} n={n} {g:?} {isa}: {d:e}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn pairs_every_position() {
    for n in [2, 3, 5, 8] {
        pairs_case::<f32>(n);
        pairs_case::<f64>(n);
    }
}

/// Random 1-qubit gates and CNOTs on the qubit set `qs` only, enough for
/// dense fusion (forced) to build `2^k x 2^k` blocks on exactly `qs`.
fn group_circuit(n: usize, qs: &[usize], rng: &mut StdRng) -> Circuit {
    let mut c = Circuit::new(n);
    for _ in 0..3 {
        for &q in qs {
            c.gate(Gate::U(
                q,
                rng.random::<f64>() * 3.0,
                rng.random::<f64>() * 3.0,
                rng.random::<f64>() * 3.0,
            ));
        }
        for w in qs.windows(2) {
            c.cnot(w[0], w[1]);
        }
    }
    c
}

fn gates_of(c: &Circuit) -> Vec<Gate> {
    c.ops
        .iter()
        .map(|o| match o {
            Op::Gate(g) => *g,
            _ => unreachable!(),
        })
        .collect()
}

fn dense_case<T: Real>(n: usize, k: usize) {
    let mut rng = StdRng::seed_from_u64(1000 + n as u64 * 10 + k as u64);
    let init = start::<T>(n, 9);
    // every k-subset of the lowest 6 qubits (all in-vector patterns), plus
    // a few sets reaching the top qubit
    let mut sets: Vec<Vec<usize>> = Vec::new();
    let lim = n.min(6);
    for m in 0usize..1 << lim {
        if m.count_ones() as usize == k {
            sets.push((0..lim).filter(|&q| m >> q & 1 == 1).collect());
        }
    }
    if n > 6 {
        sets.push((n - k..n).collect());
        sets.push([0].into_iter().chain(n - k + 1..n).collect());
    }
    for qs in sets {
        // shuffle the order so the group's first gate is not always on the
        // lowest qubit
        let mut qs2 = qs.clone();
        if rng.random_bool(0.5) {
            qs2.reverse();
        }
        let c = group_circuit(n, &qs2, &mut rng);
        let mut want = init.clone();
        want.apply_circuit(&c).unwrap();
        let ops = lower_gates(&gates_of(&c));
        for g in GEOMS {
            for (simd, avx512, isa) in ISAS {
                let mut cf = cfg(g, simd, avx512);
                cf.dense_fusion = k;
                cf.dense_min_ops = 1;
                if g == GEOMS[0] {
                    let st = fusion_stats::<T>(&ops, n, &cf);
                    assert!(st.dense2 + st.dense3 > 0, "no dense op for {qs:?}: {st:?}");
                }
                let mut got = init.clone();
                got.apply_circuit_blocked(&c, &cf).unwrap();
                let d = max_diff(&got, &want);
                assert!(
                    d <= tol::<T>(),
                    "dense k={k} qs={qs2:?} n={n} {g:?} {isa}: {d:e}"
                );
            }
        }
    }
}

#[test]
fn dense_every_target_pattern() {
    for n in [3, 4, 6, 7, 9] {
        for k in [2, 3] {
            if k <= n {
                dense_case::<f32>(n, k);
                dense_case::<f64>(n, k);
            }
        }
    }
}

/// Diagonal-heavy circuits (the single-pass diagonal kernel of the AVX-512
/// tier): layers of CZ / controlled phases on random and neighbouring pairs
/// with single-qubit phases and an Rx mixer (QAOA-like), and the QFT. Tiny
/// blocks put many terms on outer qubits (per-chunk terms), large ones put
/// terms across the low-table/row split.
#[test]
fn diagonal_blocks() {
    let mut rng = StdRng::seed_from_u64(4242);
    for n in [2usize, 3, 4, 5, 8, 9, 10, 12, 14] {
        for layers in [1usize, 3] {
            let mut c = Circuit::new(n);
            for q in 0..n {
                c.h(q);
            }
            for _ in 0..layers {
                for _ in 0..2 * n {
                    let a = rng.random_range(0..n);
                    let b = rng.random_range(0..n);
                    if a != b {
                        c.cphase(a, b, rng.random::<f64>() * 6.0 - 3.0);
                        c.gate(Gate::Phase(a, rng.random::<f64>()));
                    }
                }
                for q in (0..n.saturating_sub(1)).step_by(2) {
                    c.cz(q, q + 1);
                }
                for q in 0..n {
                    c.rx(q, rng.random::<f64>() * 3.0);
                }
            }
            for circ in [c, qsim_lab::algorithms::qft(n)] {
                for prec in [4, 8] {
                    macro_rules! go {
                        ($t:ty) => {{
                            let init = start::<$t>(n, 3);
                            let mut want = init.clone();
                            want.apply_circuit(&circ).unwrap();
                            for g in GEOMS {
                                for (simd, avx512, isa) in ISAS {
                                    let mut got = init.clone();
                                    got.apply_circuit_blocked(&circ, &cfg(g, simd, avx512))
                                        .unwrap();
                                    let d = max_diff(&got, &want);
                                    assert!(d <= tol::<$t>(), "diag n={n} {g:?} {isa}: {d:e}");
                                }
                            }
                        }};
                    }
                    if prec == 4 {
                        go!(f32)
                    } else {
                        go!(f64)
                    }
                }
            }
        }
    }
}

/// Every knob combination used below, for one kernel set.
fn knob_configs(simd: bool, avx512: bool) -> Vec<BlockConfig> {
    let mut v = Vec::new();
    for g in GEOMS {
        for (fuse, sched, dense, dmin) in [
            (true, true, 0, 0),
            (false, false, 0, 0),
            (true, true, 2, 1),
            (true, false, 3, 1),
            (true, true, 2, 0),
        ] {
            let mut c = cfg(g, simd, avx512);
            c.fuse_1q = fuse;
            c.schedule_diag = sched;
            c.dense_fusion = dense;
            c.dense_min_ops = dmin;
            v.push(c);
        }
    }
    v
}

/// Edge-biased random circuits (every gate kind, qubits 0 and n-1 favoured,
/// special angles) against the independent reference state vector, for
/// every kernel set and knob combination.
#[test]
fn matches_audit_reference_all_knobs() {
    let mut rng = StdRng::seed_from_u64(audit_common::base_seed() ^ 0xa512);
    for &n in &[1usize, 2, 3, 4, 5, 6, 8, 9, 11, 13] {
        for _ in 0..audit_common::iters().max(3) {
            let depth = rng.random_range(1..150);
            let c = audit_common::random_circuit(&mut rng, n, depth, false, false);
            let r = audit_common::RefSv::run(&c);
            for (simd, avx512, isa) in ISAS {
                for cf in knob_configs(simd, avx512) {
                    let mut a = StateVector::<f64>::new(n);
                    a.apply_circuit_blocked(&c, &cf).unwrap();
                    let d = audit_common::max_amp_diff(&r.a, a.amplitudes().iter().copied());
                    assert!(d <= 1e-12, "f64 n={n} {isa} {cf:?}: {d:e}");
                    let mut b = StateVector::<f32>::new(n);
                    b.apply_circuit_blocked(&c, &cf).unwrap();
                    let d = audit_common::max_amp_diff(
                        &r.a,
                        b.amplitudes()
                            .iter()
                            .map(|z| Complex64::new(z.re as f64, z.im as f64)),
                    );
                    assert!(d <= 1e-5, "f32 n={n} {isa} {cf:?}: {d:e}");
                }
            }
        }
    }
}

/// The default configuration from 1 to 20 qubits: below, at and above the
/// default block size, AVX-512 on vs off vs gate-by-gate.
#[test]
fn default_config_edge_sizes() {
    let mut rng = StdRng::seed_from_u64(31);
    for n in 1..=20usize {
        let c = random_universal(n, 30 + 6 * n, &mut rng);
        for prec in [4, 8] {
            macro_rules! go {
                ($t:ty) => {{
                    let init = start::<$t>(n, n as u64);
                    let mut want = init.clone();
                    want.apply_circuit(&c).unwrap();
                    for (simd, avx512, isa) in ISAS {
                        let cf = BlockConfig {
                            simd,
                            avx512,
                            ..BlockConfig::default()
                        };
                        let mut got = init.clone();
                        got.apply_circuit_blocked(&c, &cf).unwrap();
                        let d = max_diff(&got, &want);
                        assert!(d <= tol::<$t>(), "n={n} {isa}: {d:e}");
                    }
                }};
            }
            if prec == 4 {
                go!(f32)
            } else {
                go!(f64)
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// AVX-512 vs AVX2+FMA vs portable vs gate-by-gate on random universal
    /// circuits, random geometry and dense-fusion width.
    #[test]
    fn random_circuits_agree(
        n in 1usize..13,
        len in 1usize..160,
        seed in 0u64..100_000,
        gi in 0usize..GEOMS.len(),
        dense in 0usize..4,
    ) {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = random_universal(n, len, &mut rng);
        for prec in [4, 8] {
            macro_rules! go {
                ($t:ty) => {{
                    let init = start::<$t>(n, seed);
                    let mut want = init.clone();
                    want.apply_circuit(&c).unwrap();
                    for (simd, avx512, isa) in ISAS {
                        let mut cf = cfg(GEOMS[gi], simd, avx512);
                        cf.dense_fusion = dense;
                        cf.dense_min_ops = 1;
                        let mut got = init.clone();
                        got.apply_circuit_blocked(&c, &cf).unwrap();
                        let d = max_diff(&got, &want);
                        prop_assert!(d <= tol::<$t>(), "n={} {} {:?}: {:e}", n, isa, cf, d);
                    }
                }};
            }
            if prec == 4 {
                go!(f32)
            } else {
                go!(f64)
            }
        }
    }
}
