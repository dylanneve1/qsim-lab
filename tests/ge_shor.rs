//! Exact checks of the Gidney–Ekerå techniques (`src/shor_ge.rs`):
//! exponent-windowed phase estimation against the textbook distribution,
//! the permutation oracle and a gate-by-gate quantum reference; the
//! support law for windowed rounds; Ekerå–Håstad against its exact
//! textbook distribution; coset arithmetic as an approximation whose
//! effect is measured exactly.
use num_complex::Complex64;
use qsim_lab::algorithms::{gcd, pow_mod};
use qsim_lab::shor::{self, Instance, Oracle};
use qsim_lab::shor_ge::{self, ExpReg, GeOpts, GeState, WindowProg};
use qsim_lab::shor_mbu::{MbuOpts, Outcomes};

fn bases(n: u64, count: usize) -> Vec<u64> {
    (2..n).filter(|&a| gcd(a, n) == 1).take(count).collect()
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn opts(we: usize, wm: usize, mbu: MbuOpts) -> GeOpts {
    GeOpts {
        we,
        wm,
        mbu,
        coset: 0,
    }
}

/// The exponent-windowed circuit has exactly the textbook distribution
/// (3n-qubit circuit with a full inverse QFT), for every window size.
#[test]
fn windowed_distribution_matches_textbook() {
    for (n, count) in [(15u64, 3), (21, 2), (35, 1)] {
        for a in bases(n, count) {
            let full = shor::full_qft_distribution(n, a);
            let perm = Instance::new(n, a, Oracle::Permutation);
            let d_perm =
                shor::semiclassical_distribution(&perm, shor::sparse_initial(&perm), 1e-15);
            assert!(max_diff(&full, &d_perm) < 1e-12);
            for we in [1usize, 2, 3, 4] {
                for (wm, mbu) in [(2, MbuOpts::ALL), (3, MbuOpts::LOOKUPS), (1, MbuOpts::NONE)] {
                    let o = opts(we, wm, mbu);
                    let d = shor_ge::distribution(n, &shor_ge::shor_regs(n, a), &o, 1e-15);
                    let e = max_diff(&full, &d);
                    assert!(e < 1e-12, "N={n} a={a} {o:?}: {e:e}");
                    let s: f64 = d.iter().sum();
                    assert!((s - 1.0).abs() < 1e-12);
                }
            }
        }
    }
}

/// Independent quantum reference: every gate on a sparse state vector, the
/// exponent qubits as real qubits (H, block, Phase, H, projective
/// measurement), every MBU X-measurement as H + projection with P = 1/2.
#[test]
fn windowed_gate_by_gate_matches() {
    for (n, a) in [(15u64, 7u64), (21, 2)] {
        let full = shor::full_qft_distribution(n, a);
        for (we, wm, mbu) in [
            (2, 2, MbuOpts::ALL),
            (3, 1, MbuOpts::LOOKUPS),
            (2, 3, MbuOpts::NONE),
        ] {
            let o = opts(we, wm, mbu);
            let lay = shor_ge::GeLayout::new(shor::work_bits(n), &o);
            if lay.nq > 64 {
                continue;
            }
            let d = shor_ge::distribution_sparse(n, &shor_ge::shor_regs(n, a), &o, 1e-15);
            let e = max_diff(&full, &d);
            assert!(e < 1e-10, "N={n} a={a} {o:?}: {e:e}");
        }
    }
}

/// Theorem 1(b) of research/theory-shor.md (exact support of the
/// semiclassical state after `i` measured bits).
fn support_closed(n_mod: u64, a: u64, t: usize, i: usize, y: u128) -> usize {
    let gi = shor_ge::pow2k(a, t - i, n_mod);
    let mut r = 1u128;
    let mut x = gi;
    while x != 1 {
        x = shor::mul_mod(x, gi, n_mod);
        r += 1;
    }
    let p = 1u128 << i;
    if p <= r {
        return p as usize;
    }
    let s = p % r;
    let zeta_ne_1 = (r * y) % p != 0;
    let b1 = zeta_ne_1 && (s * y) % p == 0;
    let b2 = zeta_ne_1 && ((r - s) * y) % p == 0;
    (r - if b1 { r - s } else { 0 } - if b2 { s } else { 0 }) as usize
}

/// Support law for windowed rounds: inside a window of `w` exponent bits,
/// after `j` of them are measured the state is `Σ_{e''} |e''⟩ V^{e''} ψ_{i0+j}`
/// — `2^{w−j}` arrays, each of size exactly `|S_{i0+j}(y)|` (Thm 1b), and all
/// with the same multiset of amplitudes. Checked on whole measurement trees,
/// plus the work identity `W = Σ_windows 2^w |S_{i0}| G`.
#[test]
fn windowed_support_law_on_trees() {
    for (n_mod, a, we) in [
        (15u64, 7u64, 2usize),
        (21, 2, 3),
        (143, 2, 2),
        (65, 2, 4),
        (91, 3, 3),
    ] {
        let n = shor::work_bits(n_mod);
        let t = 2 * n;
        let o = opts(we, 3, MbuOpts::LOOKUPS);
        let lay = shor_ge::GeLayout::new(n, &o);
        let regs = shor_ge::shor_regs(n_mod, a);
        let wins = shor_ge::windows(t, we);
        let progs: Vec<(usize, usize, WindowProg)> = wins
            .iter()
            .enumerate()
            .map(|(k, &(i0, w))| {
                let g = shor_ge::pow2k(regs[0].base, t - i0 - w, n_mod);
                let mut oc = Outcomes::new(k as u64 + 1, 0);
                let ops = shor_ge::window_block(&lay, g, n_mod, &o, &mut oc);
                (i0, w, WindowProg::new(&lay, &ops))
            })
            .collect();
        let mut nodes = 0usize;
        // depth-first over all outcomes with p > 1e-13, random subset at depth
        #[allow(clippy::too_many_arguments)]
        fn walk(
            n_mod: u64,
            a: u64,
            t: usize,
            progs: &[(usize, usize, WindowProg)],
            k: usize,
            st: GeState<f64>,
            y: u128,
            nodes: &mut usize,
            budget: &mut u64,
        ) {
            if k == progs.len() {
                return;
            }
            let (i0, w, ref wp) = progs[k];
            let mut st = st;
            let s0 = st.psi.len();
            assert_eq!(
                s0,
                support_closed(n_mod, a, t, i0, y),
                "N={n_mod} i={i0} y={y}"
            );
            let before = st.gate_branch_ops;
            let wa = st.window(wp, w);
            assert_eq!(
                st.gate_branch_ops - before,
                ((1u128 << w) * s0 as u128) * wp.prog.gates as u128
            );
            #[allow(clippy::type_complexity)]
            fn inner(
                ctx: (u64, u64, usize, &[(usize, usize, WindowProg)], usize),
                st: &GeState<f64>,
                wa: shor_ge::WindowArrays<f64>,
                j: usize,
                y: u128,
                nodes: &mut usize,
                budget: &mut u64,
            ) {
                let (n_mod, a, t, progs, k) = ctx;
                let (i0, w, _) = progs[k];
                let i = i0 + j;
                *nodes += 1;
                let want = support_closed(n_mod, a, t, i, y);
                let cs = wa.materialize();
                assert_eq!(cs.len(), 1 << (w - j));
                assert_eq!(wa.bits_left(), w - j);
                let mut ref_amps: Option<Vec<(i64, i64)>> = None;
                for c in &cs {
                    assert_eq!(c.len(), want, "N={n_mod} i={i} j={j} y={y}");
                    let mut am: Vec<(i64, i64)> = c
                        .iter()
                        .map(|v| ((v.1.re * 1e9).round() as i64, (v.1.im * 1e9).round() as i64))
                        .collect();
                    am.sort_unstable();
                    match &ref_amps {
                        None => ref_amps = Some(am),
                        Some(r) => assert_eq!(r, &am, "amplitude multisets differ"),
                    }
                }
                if j == w {
                    let mut s2 = st.clone();
                    s2.finish(wa);
                    walk(n_mod, a, t, progs, k + 1, s2, y, nodes, budget);
                    return;
                }
                let phi = shor_ge::correction(i, y);
                let (p0, p1) = wa.probs(phi);
                for (bit, pb) in [(false, p0), (true, p1)] {
                    if pb <= 1e-13 {
                        continue;
                    }
                    // xorshift budget: explore both children near the root,
                    // then only a pseudo-random one
                    *budget ^= *budget << 13;
                    *budget ^= *budget >> 7;
                    *budget ^= *budget << 17;
                    if i > 8 && (*budget & 1 == 1) != bit && p0 > 1e-13 && p1 > 1e-13 {
                        continue;
                    }
                    let mut w2 = wa.clone();
                    w2.collapse(phi, bit, pb);
                    inner(
                        ctx,
                        st,
                        w2,
                        j + 1,
                        y | (u128::from(bit) << i),
                        nodes,
                        budget,
                    );
                }
            }
            inner((n_mod, a, t, progs, k), &st, wa, 0, y, nodes, budget);
        }
        let st = GeState::<f64>::basis(1);
        let mut budget = 0x9E37_79B9_7F4A_7C15u64 ^ n_mod;
        walk(n_mod, a, t, &progs, 0, st, 0, &mut nodes, &mut budget);
        assert!(nodes >= 12, "N={n_mod}: {nodes} nodes");
        eprintln!("support law N={n_mod} a={a} we={we}: {nodes} tree nodes checked");
    }
}

/// The textbook Ekerå–Håstad distribution of `(j, k)`:
/// `P(j,k) = Σ_z |2^{−3m} Σ_{a,b: g^a y^{−b} = z} e^{−2πi(aj/2^{2m} + bk/2^m)}|²`.
fn eh_textbook(n_mod: u64, g: u64) -> Vec<f64> {
    let m = shor_ge::eh_m(n_mod);
    let y = shor_ge::eh_target(n_mod, g);
    let yi = shor::mod_inverse(y, n_mod);
    let (na, nb) = (1usize << (2 * m), 1usize << m);
    let mut z_of = vec![0u64; na * nb];
    for a in 0..na {
        for b in 0..nb {
            z_of[a + na * b] = shor::mul_mod(
                pow_mod(g, a as u64, n_mod),
                pow_mod(yi, b as u64, n_mod),
                n_mod,
            );
        }
    }
    let mut zs: Vec<u64> = z_of.clone();
    zs.sort_unstable();
    zs.dedup();
    let mut out = vec![0.0; na * nb];
    let norm = 1.0 / (na * nb) as f64;
    for j in 0..na {
        for k in 0..nb {
            let mut acc: std::collections::HashMap<u64, Complex64> = Default::default();
            for a in 0..na {
                for b in 0..nb {
                    let ph = -2.0
                        * std::f64::consts::PI
                        * ((a * j) as f64 / na as f64 + (b * k) as f64 / nb as f64);
                    *acc.entry(z_of[a + na * b]).or_default() += Complex64::from_polar(norm, ph);
                }
            }
            out[j + na * k] = acc.values().map(|v| v.norm_sqr()).sum();
        }
    }
    out
}

/// Ekerå–Håstad with exponent windows: the gate-level distribution of
/// `(j, k)` equals the textbook one, and the classical post-processing
/// recovers `p, q` with the exact success probability reported.
#[test]
fn eh_distribution_matches_textbook() {
    for (n_mod, gs) in [(35u64, vec![2u64, 3]), (77, vec![2]), (143, vec![2])] {
        for g in gs {
            let m = shor_ge::eh_m(n_mod);
            let regs = shor_ge::eh_regs(n_mod, g);
            assert_eq!(regs[0].len + regs[1].len, 3 * m);
            let book = eh_textbook(n_mod, g);
            let na = 1u128 << (2 * m);
            let nb = 1usize << m;
            for o in [opts(1, 2, MbuOpts::LOOKUPS), opts(2, 2, MbuOpts::ALL)] {
                // the engine runs register b (k, m bits) first: index k + 2^m j
                let d0 = shor_ge::distribution(n_mod, &regs, &o, 1e-15);
                let mut d = vec![0.0; d0.len()];
                for (idx, &p) in d0.iter().enumerate() {
                    let (k, j) = (idx % nb, idx / nb);
                    d[j + (na as usize) * k] = p;
                }
                let e = max_diff(&book, &d);
                assert!(e < 1e-12, "N={n_mod} g={g} {o:?}: {e:e}");
            }
            let mut ok = 0.0;
            for (idx, &p) in book.iter().enumerate() {
                if p < 1e-15 {
                    continue;
                }
                let (j, k) = (idx as u128 % na, idx as u128 / na);
                if shor_ge::eh_postprocess(n_mod, g, j, k, 4096).0.is_some() {
                    ok += p;
                }
            }
            eprintln!("EH N={n_mod} g={g}: exact P(success, one run) = {ok:.4}");
            assert!(ok > 0.3, "N={n_mod} g={g}: P(success) = {ok}");
        }
    }
}

/// Coset arithmetic: a valid probability distribution whose deviation from
/// the exact one shrinks with the padding; the deviation is measured
/// exactly (total variation distance), not assumed.
#[test]
fn coset_deviation_shrinks_with_padding() {
    let (n_mod, a) = (21u64, 2u64);
    let full = shor::full_qft_distribution(n_mod, a);
    let mut tv = Vec::new();
    for c in [1usize, 2, 4] {
        let o = GeOpts {
            we: 2,
            wm: 2,
            mbu: MbuOpts::ALL,
            coset: c,
        };
        let d = shor_ge::distribution(n_mod, &shor_ge::shor_regs(n_mod, a), &o, 1e-15);
        let s: f64 = d.iter().sum();
        assert!((s - 1.0).abs() < 1e-9, "c={c}: Σ = {s}");
        let t: f64 = 0.5 * full.iter().zip(&d).map(|(x, y)| (x - y).abs()).sum::<f64>();
        eprintln!("coset N={n_mod} c={c}: TV = {t:.5}");
        tv.push(t);
    }
    // measured (Mac, exact): TV = 0.2454 / 0.1673 / 0.0712 for c = 1 / 2 / 4
    assert!(tv[0] > 0.1 && tv[2] < tv[1] && tv[1] < tv[0], "{tv:?}");
}

/// Same measured integer as the reversible-oracle engine with the same RNG
/// stream (the windowed circuit has the same conditional probabilities).
#[test]
fn windowed_run_reproduces_semiclassical_bits() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    for (n_mod, seed) in [(143u64, 1u64), (1003, 2), (1_005_973, 1)] {
        let mut rng = StdRng::seed_from_u64(seed);
        let a = rng.random_range(2..n_mod - 1);
        if gcd(a, n_mod) > 1 {
            continue;
        }
        let mut rng2 = rng.clone();
        let inst = Instance::new(n_mod, a, Oracle::Permutation);
        let base = shor::run_semiclassical(&inst, shor::sparse_initial(&inst), &mut rng);
        for we in [1usize, 2, 3] {
            let mut r3 = rng2.clone();
            let o = opts(we, 3, MbuOpts::LOOKUPS);
            let (run, _, _) = shor_ge::shor_run::<f64>(n_mod, a, &o, &mut || r3.random::<f64>());
            assert_eq!(run.y[0], base.measured, "N={n_mod} we={we}");
        }
        let _ = &mut rng2;
    }
}

#[test]
fn eh_regs_compute_short_dlog() {
    // y = g^d with d = (p + q − 2)/2 < 2^m
    for (n_mod, p, q) in [(35u64, 5u64, 7u64), (143, 11, 13), (1_005_973, 997, 1009)] {
        let d = (p + q - 2) / 2;
        assert!(d < 1 << shor_ge::eh_m(n_mod));
        for g in bases(n_mod, 5) {
            assert_eq!(pow_mod(g, d, n_mod), shor_ge::eh_target(n_mod, g));
        }
        let _: Vec<ExpReg> = shor_ge::eh_regs(n_mod, 2);
    }
}
