//! autoimprove correctness gate (copied into `tests/ai_gate.rs` of the tree
//! under test by `tools/autoimprove/autoimprove.py`; not part of the crate).
//!
//! Differential fuzz of the blocked executor against the independent
//! reference state vector of `tests/audit_common`: every `Gate` variant
//! (the audit generator plus I/Sx/Sxdg/U/ISwap/ISwapdg, whose reference
//! matrices are written out below from their textbook definitions), random
//! sizes 1..=14 plus a few 16..=18-qubit registers on the default config,
//! adversarial block configurations that force multi-stage plans, gathers,
//! tiling and both kernel builds, f64 and f32, through both
//! `apply_circuit_blocked` and `compile_kops`/`run_compiled`.
//!
//! env: AI_GATE_CASES (default 4000), AI_GATE_SEED (extra seed mixed into
//! half of the cases; the other half always use the fixed seeds).

mod audit_common;

use audit_common::{cx, edge_angle, edge_pair, random_gate, RefSv};
use num_complex::Complex64 as C;
use qsim_lab::blocked::{lower_gates, BlockConfig};
use qsim_lab::circuit::Circuit;
use qsim_lab::statevector::{Real, StateVector};
use qsim_lab::Gate;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn env(k: &str, d: u64) -> u64 {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

/// Gates the audit generator never emits.
fn extra_gate(rng: &mut StdRng, n: usize) -> Gate {
    let q = rng.random_range(0..n);
    let k = if n >= 2 {
        rng.random_range(0..6)
    } else {
        rng.random_range(0..4)
    };
    match k {
        0 => Gate::I(q),
        1 => Gate::Sx(q),
        2 => Gate::Sxdg(q),
        3 => Gate::U(q, edge_angle(rng), edge_angle(rng), edge_angle(rng)),
        4 => {
            let (a, b) = edge_pair(rng, n);
            Gate::ISwap(a, b)
        }
        _ => {
            let (a, b) = edge_pair(rng, n);
            Gate::ISwapdg(a, b)
        }
    }
}

/// Reference for the extra gates, from the textbook matrices.
fn ref_apply(s: &mut RefSv, g: &Gate) {
    let h = 0.5;
    let m1 = |s: &mut RefSv, q: usize, m: [[C; 2]; 2]| {
        let old = s.a.clone();
        for (i, out) in s.a.iter_mut().enumerate() {
            let r = (i >> q) & 1;
            *out = m[r][0] * old[i & !(1 << q)] + m[r][1] * old[i | (1 << q)];
        }
    };
    match *g {
        Gate::I(_) => {}
        Gate::Sx(q) => m1(s, q, [[cx(h, h), cx(h, -h)], [cx(h, -h), cx(h, h)]]),
        Gate::Sxdg(q) => m1(s, q, [[cx(h, -h), cx(h, h)], [cx(h, h), cx(h, -h)]]),
        Gate::U(q, th, ph, la) => {
            let (c, sn) = ((th / 2.0).cos(), (th / 2.0).sin());
            let e = |t: f64| cx(t.cos(), t.sin());
            m1(
                s,
                q,
                [[cx(c, 0.0), -e(la) * sn], [e(ph) * sn, e(ph + la) * c]],
            )
        }
        Gate::ISwap(a, b) | Gate::ISwapdg(a, b) => {
            // |01> -> ±i|10>, |10> -> ±i|01>
            let ph = if matches!(g, Gate::ISwap(..)) {
                cx(0.0, 1.0)
            } else {
                cx(0.0, -1.0)
            };
            let old = s.a.clone();
            for i in 0..old.len() {
                let (ba, bb) = ((i >> a) & 1, (i >> b) & 1);
                if ba != bb {
                    let j = i ^ (1 << a) ^ (1 << b);
                    s.a[j] = ph * old[i];
                }
            }
        }
        _ => s.apply(g),
    }
}

fn reference(c: &Circuit) -> RefSv {
    let mut s = RefSv::new(c.num_qubits);
    for g in c.gates() {
        ref_apply(&mut s, g);
    }
    s
}

#[allow(clippy::too_many_arguments)]
fn cfg(
    kib_x16: usize,
    slots: usize,
    fuse: bool,
    sched: bool,
    simd: bool,
    tile_b: usize,
    small_n: usize,
) -> BlockConfig {
    BlockConfig {
        block_bytes: kib_x16,
        slots,
        fuse_1q: fuse,
        schedule_diag: sched,
        simd,
        l1_tile_bytes: tile_b,
        small_n,
        ..BlockConfig::default()
    }
}

/// Adversarial configurations: tiny blocks (multi-stage plans and gathers
/// even at a few qubits), 0..=6 slots, tiles, fusion and scheduling on and
/// off, both kernel builds.
fn configs() -> Vec<BlockConfig> {
    let mut v = vec![
        cfg(64, 2, true, true, true, 0, 0),
        cfg(64, 0, true, true, false, 0, 0),
        cfg(128, 1, false, true, true, 0, 0),
        cfg(256, 2, true, false, true, 64, 0),
        cfg(256, 3, true, true, false, 128, 0),
        cfg(512, 4, true, true, true, 128, 1),
        cfg(1024, 3, false, false, true, 256, 0),
        cfg(1024, 6, true, true, true, 0, 2),
        cfg(2048, 5, true, true, false, 512, 0),
        cfg(4096, 6, true, true, true, 1024, 0),
        cfg(16384, 2, true, true, true, 0, 0),
        cfg(65536, 6, true, true, true, 4096, 0),
        BlockConfig::default(),
    ];
    // dense fusion widths / cost rules, when the tree has them
    let mut d = Vec::new(); // @dense
    for (k, m) in [(2, 1), (3, 1), (2, 0), (3, 2)] {
        // @dense
        let mut c = v[d.len() * 2 + 1].clone(); // @dense
        c.dense_fusion = k; // @dense
        c.dense_min_ops = m; // @dense
        d.push(c); // @dense
    } // @dense
    v.extend(d); // @dense
    v
}

fn run<T: Real>(c: &Circuit, cfg: &BlockConfig, compiled: bool) -> Vec<C> {
    let mut s = StateVector::<T>::new(c.num_qubits);
    if compiled {
        let ops = lower_gates(c.gates());
        let plan = StateVector::<T>::compile_kops(c.num_qubits, &ops, cfg);
        s.run_compiled(&plan);
    } else {
        s.apply_circuit_blocked(c, cfg).unwrap();
    }
    s.amplitudes()
        .iter()
        .map(|a| cx(a.re.to_f64(), a.im.to_f64()))
        .collect()
}

fn diff(a: &[C], b: &[C]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

fn case(rng: &mut StdRng, n: usize, depth: usize) -> Circuit {
    let mut c = Circuit::new(n);
    // generic start state: every amplitude matters
    for q in 0..n {
        c.gate(Gate::Ry(q, rng.random_range(0.1..3.0)));
        c.gate(Gate::Rz(q, rng.random_range(-3.0..3.0)));
    }
    for _ in 0..depth {
        let g = if rng.random_range(0..7) == 0 {
            extra_gate(rng, n)
        } else {
            random_gate(rng, n, false, false)
        };
        c.gate(g);
    }
    c
}

#[test]
fn blocked_matches_reference() {
    let cases = env("AI_GATE_CASES", 4000) as usize;
    let extra = env("AI_GATE_SEED", 0);
    let cfgs = configs();
    let (mut w64, mut w32) = (0.0f64, 0.0f64);
    let mut counts = [0usize; 2];
    for i in 0..cases {
        let seed = if i % 2 == 0 {
            0xA1_0000 + i as u64
        } else {
            (0xA1_0000 + i as u64) ^ extra.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        };
        let mut rng = StdRng::seed_from_u64(seed);
        let big = i % 200 == 199;
        let n = if big {
            rng.random_range(16..=18)
        } else {
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14][rng.random_range(0..14)]
        };
        let depth = if big { 60 } else { rng.random_range(1..=120) };
        let c = case(&mut rng, n, depth);
        let cf = if big {
            BlockConfig::default()
        } else {
            cfgs[i % cfgs.len()].clone()
        };
        let compiled = i % 5 == 3;
        let r = reference(&c);
        let f64p = i % 3 != 2;
        let d = if f64p {
            run::<f64>(&c, &cf, compiled)
        } else {
            run::<f32>(&c, &cf, compiled)
        };
        let e = diff(&r.a, &d);
        let tol = if f64p { 1e-10 } else { 5e-5 };
        assert!(
            e <= tol,
            "case {i} seed {seed:#x} n={n} depth={depth} f64={f64p} compiled={compiled} \
             err={e:e} cfg={cf:?}\ncircuit: {:?}",
            c.gates().collect::<Vec<_>>()
        );
        if f64p {
            w64 = w64.max(e);
            counts[0] += 1;
        } else {
            w32 = w32.max(e);
            counts[1] += 1;
        }
    }
    println!(
        "AI_GATE cases={cases} f64={} f32={} worst_f64={w64:.3e} worst_f32={w32:.3e}",
        counts[0], counts[1]
    );
}
