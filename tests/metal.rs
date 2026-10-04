//! Differential tests of the Metal (Apple GPU, f32) backend against the
//! independent audit reference state vector (f64), tolerance 1e-5.
#![cfg(all(feature = "metal", target_os = "macos"))]

mod audit_common;

use audit_common::{random_circuit, RefSv};
use num_complex::Complex64;
use qsim_lab::blocked::lower_gates;
use qsim_lab::metal_sv::{circuit_gates, MetalConfig, MetalSim};
use qsim_lab::{algorithms, Circuit, Gate};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TOL: f64 = 1e-5;

fn sim() -> MetalSim {
    MetalSim::new().expect("Metal device")
}

fn diff(gpu: &[num_complex::Complex32], r: &[Complex64]) -> f64 {
    gpu.iter()
        .zip(r)
        .map(|(a, b)| (Complex64::new(a.re as f64, a.im as f64) - b).norm())
        .fold(0.0, f64::max)
}

/// Configurations that force many stages, gathers and outer controls even
/// on small registers: the default (register kernel), the shared-memory
/// kernel with every batch width, and register-kernel shapes covering
/// every split of buffer bits into lane / threadgroup-memory / local bits.
fn configs() -> Vec<MetalConfig> {
    let mut v = vec![MetalConfig::default()];
    // shared-memory kernel
    for (tg, slots, threads, batch) in [
        (3, 2, 4, 3),
        (3, 2, 1, 2),
        (4, 2, 8, 4),
        (5, 3, 32, 1),
        (6, 2, 64, 4),
        (8, 4, 1024, 2),
        (11, 6, 256, 4),
        (12, 8, 128, 1),
    ] {
        v.push(MetalConfig {
            tg_bits: tg,
            slots,
            threads,
            batch,
            regs: false,
            ..MetalConfig::default()
        });
    }
    // register kernel
    for (tg, slots, threads) in [
        (3, 2, 2),
        (5, 2, 32),
        (5, 2, 4),
        (6, 3, 8),
        (8, 2, 64),
        (8, 3, 32),
        (10, 5, 32),
        (11, 5, 64),
        (12, 5, 128),
        (12, 4, 256),
        (12, 6, 1024),
    ] {
        v.push(MetalConfig {
            tg_bits: tg,
            slots,
            threads,
            regs: true,
            ..MetalConfig::default()
        });
    }
    v.push(MetalConfig {
        tg_bits: 7,
        slots: 2,
        threads: 64,
        regs: true,
        fuse_1q: false,
        schedule_diag: false,
        ..MetalConfig::default()
    });
    v.push(MetalConfig {
        tg_bits: 5,
        slots: 2,
        threads: 8,
        regs: false,
        fuse_1q: false,
        schedule_diag: false,
        ..MetalConfig::default()
    });
    v
}

/// Every gate type, including the ones `random_circuit` never draws.
fn extra_gates(rng: &mut StdRng, n: usize, c: &mut Circuit) {
    for _ in 0..3 * n {
        let q = rng.random_range(0..n);
        let a = rng.random_range(-7.0..7.0);
        let g = match rng.random_range(0..6) {
            0 => Gate::Sx(q),
            1 => Gate::Sxdg(q),
            2 => Gate::U(q, a, 0.3 * a, -1.7 * a),
            3 => Gate::I(q),
            _ if n >= 2 => {
                let mut b = rng.random_range(0..n - 1);
                if b >= q {
                    b += 1;
                }
                if rng.random_bool(0.5) {
                    Gate::ISwap(q, b)
                } else {
                    Gate::ISwapdg(q, b)
                }
            }
            _ => Gate::H(q),
        };
        c.gate(g);
    }
}

/// Reference for the gates `RefSv` does not cover, from textbook matrices
/// (independent of `qsim_lab::gate`).
fn ref_apply(r: &mut RefSv, g: &Gate) {
    let c = |re: f64, im: f64| Complex64::new(re, im);
    let one_q = |r: &mut RefSv, q: usize, m: [[Complex64; 2]; 2]| {
        for i in 0..r.a.len() {
            if i >> q & 1 == 0 {
                let (x, y) = (r.a[i], r.a[i | 1 << q]);
                r.a[i] = m[0][0] * x + m[0][1] * y;
                r.a[i | 1 << q] = m[1][0] * x + m[1][1] * y;
            }
        }
    };
    match *g {
        Gate::I(_) => {}
        Gate::Sx(q) => one_q(
            r,
            q,
            [[c(0.5, 0.5), c(0.5, -0.5)], [c(0.5, -0.5), c(0.5, 0.5)]],
        ),
        Gate::Sxdg(q) => one_q(
            r,
            q,
            [[c(0.5, -0.5), c(0.5, 0.5)], [c(0.5, 0.5), c(0.5, -0.5)]],
        ),
        Gate::U(q, th, ph, lam) => {
            // OpenQASM 3 U(θ, φ, λ)
            let (s, co) = (th / 2.0).sin_cos();
            let e = |t: f64| c(t.cos(), t.sin());
            one_q(
                r,
                q,
                [[c(co, 0.0), -e(lam) * s], [e(ph) * s, e(ph + lam) * co]],
            )
        }
        Gate::ISwap(a, b) | Gate::ISwapdg(a, b) => {
            // |01> <-> ±i|10>, |00>, |11> fixed
            let ph = if matches!(g, Gate::ISwap(..)) {
                c(0.0, 1.0)
            } else {
                c(0.0, -1.0)
            };
            for i in 0..r.a.len() {
                if i >> a & 1 == 1 && i >> b & 1 == 0 {
                    let j = i ^ (1 << a) ^ (1 << b);
                    let (x, y) = (r.a[i], r.a[j]);
                    r.a[i] = ph * y;
                    r.a[j] = ph * x;
                }
            }
        }
        _ => r.apply(g),
    }
}

fn ref_run(c: &Circuit) -> RefSv {
    let mut r = RefSv::new(c.num_qubits);
    for g in c.gates() {
        ref_apply(&mut r, g);
    }
    r
}

#[test]
fn reference_extras_sane() {
    // ISwap = SWAP · CZ · (S ⊗ S) (exact decomposition): check the
    // test's own reference against RefSv on a random state.
    let mut rng = StdRng::seed_from_u64(1);
    let mut c = random_circuit(&mut rng, 3, 30, false, false);
    let mut d = c.clone();
    c.gate(Gate::ISwap(0, 2));
    for g in [Gate::Swap(0, 2), Gate::Cz(0, 2), Gate::S(0), Gate::S(2)] {
        d.gate(g);
    }
    let (x, y) = (ref_run(&c), RefSv::run(&d));
    let e =
        x.a.iter()
            .zip(&y.a)
            .map(|(p, q)| (p - q).norm())
            .fold(0.0, f64::max);
    assert!(e < 1e-12, "{e}");
}

#[test]
fn random_circuits_match_reference() {
    let s = sim();
    let cfgs = configs();
    let mut rng = StdRng::seed_from_u64(0x6D74_6C00);
    let mut worst = 0.0f64;
    for n in 1..=16usize {
        let reps = if n <= 10 { 4 } else { 2 };
        for _ in 0..reps {
            let depth = rng.random_range(1..=12 * n + 20);
            let mut c = random_circuit(&mut rng, n, depth, false, false);
            extra_gates(&mut rng, n, &mut c);
            // interleave more random gates after the extras
            let tail = random_circuit(&mut rng, n, depth / 2 + 1, false, false);
            c.append(&tail);
            let r = ref_run(&c);
            let gates = circuit_gates(&c).unwrap();
            for cfg in &cfgs {
                let mut st = s.alloc(n).unwrap();
                s.apply_gates(&mut st, &gates, cfg).unwrap();
                let d = diff(st.amplitudes(), &r.a);
                worst = worst.max(d);
                assert!(d < TOL, "n={n} depth={depth} cfg={cfg:?}: max |dAmp| {d:e}");
            }
            // unfused one-dispatch-per-gate baseline
            let mut st = s.alloc(n).unwrap();
            s.apply_kops_naive(&mut st, &lower_gates(&gates)).unwrap();
            let d = diff(st.amplitudes(), &r.a);
            assert!(d < TOL, "naive n={n}: {d:e}");
        }
    }
    eprintln!("worst max |dAmp| = {worst:e}");
}

#[test]
fn qft_and_brickwork_match_reference() {
    let s = sim();
    for n in [5usize, 9, 13, 16] {
        let mut rng = StdRng::seed_from_u64(42);
        for c in [
            algorithms::qft(n),
            algorithms::random_brickwork(n, 8, &mut rng),
        ] {
            let r = RefSv::run(&c);
            for cfg in configs() {
                let mut st = s.alloc(n).unwrap();
                s.apply_circuit(&mut st, &c, &cfg).unwrap();
                let d = diff(st.amplitudes(), &r.a);
                assert!(d < TOL, "n={n} cfg={cfg:?}: {d:e}");
            }
        }
    }
}

#[test]
fn qft_of_basis_states_matches_reference() {
    // QFT|0> is a uniform real state; from |x> every amplitude has a
    // different phase, which exercises the diagonal tables properly.
    let s = sim();
    let mut rng = StdRng::seed_from_u64(99);
    for n in [6usize, 11, 14, 16] {
        let c = algorithms::qft(n);
        let x = rng.random_range(0..1usize << n);
        let mut r = RefSv::new(n);
        r.a[0] = Complex64::new(0.0, 0.0);
        r.a[x] = Complex64::new(1.0, 0.0);
        for g in c.gates() {
            r.apply(g);
        }
        for cfg in configs() {
            let mut st = s.alloc(n).unwrap();
            s.set_basis(&st, x);
            s.apply_circuit(&mut st, &c, &cfg).unwrap();
            let d = diff(st.amplitudes(), &r.a);
            assert!(d < TOL, "n={n} x={x} cfg={cfg:?}: {d:e}");
        }
    }
}

#[test]
fn nonzero_basis_state_and_reuse() {
    let s = sim();
    let n = 11;
    let mut rng = StdRng::seed_from_u64(7);
    let c1 = random_circuit(&mut rng, n, 150, false, false);
    let c2 = random_circuit(&mut rng, n, 150, false, false);
    let idx = 0b101_1001_0110;
    let mut r = RefSv::new(n);
    r.a[0] = Complex64::new(0.0, 0.0);
    r.a[idx] = Complex64::new(1.0, 0.0);
    for g in c1.gates().chain(c2.gates()) {
        r.apply(g);
    }
    let cfg = MetalConfig {
        tg_bits: 6,
        slots: 3,
        ..MetalConfig::default()
    };
    let mut st = s.alloc(n).unwrap();
    s.set_basis(&st, idx);
    // two plans run back to back on the same buffer
    let p1 = s
        .compile(n, &lower_gates(&circuit_gates(&c1).unwrap()), &cfg)
        .unwrap();
    let p2 = s
        .compile(n, &lower_gates(&circuit_gates(&c2).unwrap()), &cfg)
        .unwrap();
    s.run(&mut st, &p1).unwrap();
    s.run(&mut st, &p2).unwrap();
    let d = diff(st.amplitudes(), &r.a);
    assert!(d < TOL, "{d:e}");
}

#[test]
fn rejects_bad_input() {
    let s = sim();
    let mut st = s.alloc(3).unwrap();
    assert!(s
        .apply_gates(&mut st, &[Gate::H(3)], &MetalConfig::default())
        .is_err());
    assert!(s
        .apply_gates(&mut st, &[Gate::Cnot(1, 1)], &MetalConfig::default())
        .is_err());
    let mut c = Circuit::new(3);
    c.h(0).measure(0);
    assert!(s
        .apply_circuit(&mut st, &c, &MetalConfig::default())
        .is_err());
    assert!(s.alloc(31).is_err());
    for bad in [
        MetalConfig {
            tg_bits: 13,
            ..MetalConfig::default()
        },
        MetalConfig {
            slots: 1,
            ..MetalConfig::default()
        },
        MetalConfig {
            threads: 100,
            ..MetalConfig::default()
        },
        MetalConfig {
            batch: 5,
            ..MetalConfig::default()
        },
    ] {
        assert!(s.apply_gates(&mut st, &[Gate::H(0)], &bad).is_err());
    }
}
