//! Magic atlas: the profile invariants and the factored engine against the
//! reference state vector, the algorithm circuits against their classical
//! semantics, and the state-magic estimator against known values.

use num_complex::Complex64;
use qsim_lab::adaptive::{self, CompressedState};
use qsim_lab::circuit::Circuit;
use qsim_lab::gate::Gate;
use qsim_lab::magic_atlas::{families, profile, state_magic, AtlasOptions, FactoredState};
use qsim_lab::StateVectorF64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TOL: f64 = 1e-10;

fn sv_of(c: &Circuit) -> StateVectorF64 {
    let mut sv = StateVectorF64::new(c.num_qubits);
    for g in c.gates() {
        sv.apply_gate(g).unwrap();
    }
    sv
}

fn sv_pauli(sv: &StateVectorF64, p: &[u8]) -> f64 {
    let mut phi = sv.clone();
    for (q, &ch) in p.iter().enumerate() {
        match ch {
            b'X' => phi.apply_gate(&Gate::X(q)).unwrap(),
            b'Y' => phi.apply_gate(&Gate::Y(q)).unwrap(),
            b'Z' => phi.apply_gate(&Gate::Z(q)).unwrap(),
            _ => {}
        }
    }
    sv.inner(&phi).re
}

const SMALL: &[&str] = &[
    "qft:n=10,in=basis",
    "qft:n=10,in=graph",
    "qft:n=10,in=plus",
    "qft:n=10,in=graph,cut=3",
    "cuccaro:bits=4,in=plusa",
    "gidney:bits=4,in=plusab",
    "draper:bits=5,in=plusab",
    "draper:bits=5,in=basis,cut=2",
    "shorwin:nbits=2,w=1,in=half",
    "grover:n=6,it=2",
    "ising:n=10,steps=3,dt=0.2",
    "heis:n=10,steps=2",
    "qaoa:n=10,p=2,graph=reg3",
    "hea:n=10,layers=2",
    "qpe:t=5,s=6,kind=stab",
    "qpe:t=3,s=6,kind=trotter",
    "walk:m=4,steps=3",
    "hhl:t=4,m=3",
    "rct:n=12,L=6,t=14",
];

#[test]
fn profile_matches_adaptive_and_engines_match_statevector() {
    let mut rng = StdRng::seed_from_u64(7);
    for (i, spec) in SMALL.iter().enumerate() {
        let c = families::build(spec, i as u64 + 1).unwrap();
        let n = c.num_qubits;
        let p = profile(&c, &AtlasOptions::default()).unwrap();
        let want = adaptive::active_dimension_profile(&c).unwrap();
        let got: Vec<usize> = p.d_prof.iter().map(|&x| x as usize).collect();
        assert_eq!(got, want, "{spec}: d profile");
        assert!(p.f <= p.d, "{spec}");
        let sv = sv_of(&c);
        // compressed state: full state up to a global phase
        let cs = CompressedState::new(&c, 24).unwrap();
        assert_eq!(cs.active_qubits(), p.d, "{spec}");
        let fid = sv.inner(&cs.to_statevector()).norm();
        assert!(
            (1.0 - fid).abs() < TOL,
            "{spec}: cstate infidelity {}",
            1.0 - fid
        );
        // factored state: largest factor = profile's f, Paulis exact
        let fs = FactoredState::new(&c, 24).unwrap();
        assert_eq!(fs.stats.f, p.f, "{spec}: factored f");
        let da = fs.dense_active();
        let ov: Complex64 = da
            .iter()
            .zip(cs.active_amplitudes())
            .map(|(a, b)| a.conj() * b)
            .sum();
        assert!((1.0 - ov.norm()).abs() < TOL, "{spec}: factored vs cstate");
        let fr = FactoredState::with_recycling(&c, 24, 16).unwrap();
        assert!(
            fr.stats.f <= p.f,
            "{spec}: recycling made the register larger"
        );
        let mut nonzero = 0;
        for k in 0..60 {
            let r = if k % 3 == 0 { 4 } else { 14 };
            let pst: Vec<u8> = (0..n)
                .map(|_| [b'X', b'Y', b'Z', b'I'][rng.random_range(0..r).min(3)])
                .collect();
            let x: Vec<bool> = pst.iter().map(|&c| c == b'X' || c == b'Y').collect();
            let z: Vec<bool> = pst.iter().map(|&c| c == b'Z' || c == b'Y').collect();
            let a = fs.pauli_expectation(&x, &z);
            let b = sv_pauli(&sv, &pst);
            let ar = fr.pauli_expectation(&x, &z);
            assert!(
                (ar - b).abs() < TOL,
                "{spec}: recycled <{}> {ar} vs {b}",
                String::from_utf8_lossy(&pst)
            );
            if b.abs() > 1e-6 {
                nonzero += 1;
            }
            assert!(
                (a - b).abs() < TOL,
                "{spec}: <{}> {a} vs {b}",
                String::from_utf8_lossy(&pst)
            );
        }
        assert!(nonzero > 0, "{spec}: only trivial Paulis tested");
    }
}

#[test]
fn qft_of_basis_state_factorises_completely() {
    for n in [6, 12, 40, 200] {
        let c = families::build(&format!("qft:n={n},in=basis"), 3).unwrap();
        let p = profile(&c, &AtlasOptions::default()).unwrap();
        assert_eq!(p.f, 1, "n={n}");
        assert!(p.d >= n - 1);
    }
}

#[test]
fn factored_qft_large_n_matches_product_formula() {
    // QFT|x> (with bit reversal) = ⊗_k (|0> + e^{2πi x / 2^{k+1}} |1>)/√2 on
    // output qubit n-1-k ... checked through <X_q> and <Y_q> at n = 300.
    let n = 300;
    let mut c = Circuit::new(n);
    let mut rng = StdRng::seed_from_u64(5);
    let xbits: Vec<bool> = (0..n).map(|_| rng.random()).collect();
    for (q, &b) in xbits.iter().enumerate() {
        if b {
            c.x(q);
        }
    }
    families::qft(&mut c, &(0..n).collect::<Vec<_>>(), 0, false, true);
    let fs = FactoredState::new(&c, 4).unwrap();
    assert_eq!(fs.stats.f, 1);
    for out in [0usize, 1, 2, 7, 150, 298, 299] {
        // after the swaps, output qubit `out` holds the factor of qubit
        // j = n-1-out before the swaps, phase 2π·0.x_j x_{j-1} … x_0 (binary
        // fraction of the low j+1 bits of x).
        let j = n - 1 - out;
        let mut frac = 0.0f64;
        for i in 0..=j.min(60) {
            // bit (j - i) contributes 2^{-(i+1)}
            if xbits[j - i] {
                frac += 0.5f64.powi(i as i32 + 1);
            }
        }
        let ang = 2.0 * std::f64::consts::PI * frac;
        let mut xs = vec![false; n];
        let zs = vec![false; n];
        xs[out] = true;
        let ex = fs.pauli_expectation(&xs, &zs);
        let mut zy = vec![false; n];
        zy[out] = true;
        let ey = fs.pauli_expectation(&xs, &zy);
        assert!(
            (ex - ang.cos()).abs() < 1e-9,
            "X_{out}: {ex} vs {}",
            ang.cos()
        );
        assert!(
            (ey - ang.sin()).abs() < 1e-9,
            "Y_{out}: {ey} vs {}",
            ang.sin()
        );
    }
}

fn basis_value(sv: &StateVectorF64) -> usize {
    let a = sv.amplitudes();
    let (i, m) = a
        .iter()
        .enumerate()
        .map(|(i, v)| (i, v.norm()))
        .fold((0, 0.0), |acc, x| if x.1 > acc.1 { x } else { acc });
    assert!((m - 1.0).abs() < 1e-9, "not a basis state");
    i
}

#[test]
fn adders_add() {
    let mut rng = StdRng::seed_from_u64(11);
    for bits in 1..=4usize {
        for _ in 0..6 {
            let a: usize = rng.random_range(0..1 << bits);
            let b: usize = rng.random_range(0..1 << bits);
            // gidney (Clifford+T AND gadget): a in 0..bits, b in bits..2bits, anc after
            let n = 2 * bits + bits.saturating_sub(1);
            let mut c = Circuit::new(n);
            for i in 0..bits {
                if a >> i & 1 == 1 {
                    c.x(i);
                }
                if b >> i & 1 == 1 {
                    c.x(bits + i);
                }
            }
            let av: Vec<usize> = (0..bits).collect();
            let bv: Vec<usize> = (bits..2 * bits).collect();
            let anc: Vec<usize> = (2 * bits..n).collect();
            let mut g = c.clone();
            families::gidney_add(&mut g, &av, &bv, &anc, false);
            let out = basis_value(&sv_of(&g));
            assert_eq!(out & ((1 << bits) - 1), a, "gidney a");
            assert_eq!(out >> bits, (a + b) % (1 << bits), "gidney sum (anc clean)");
            let mut d = c.clone();
            families::draper_add(&mut d, &av, &bv, 0);
            let out = basis_value(&sv_of(&d));
            assert_eq!(out, a | ((a + b) % (1 << bits)) << bits, "draper");
        }
    }
}

#[test]
fn and_gadget_is_toffoli_on_clean_target() {
    for a in 0..2 {
        for b in 0..2 {
            let mut c = Circuit::new(3);
            if a == 1 {
                c.x(0);
            }
            if b == 1 {
                c.x(1);
            }
            families::and_compute(&mut c, 0, 1, 2);
            assert_eq!(basis_value(&sv_of(&c)), a | b << 1 | (a & b) << 2);
            families::and_uncompute(&mut c, 0, 1, 2);
            assert_eq!(basis_value(&sv_of(&c)), a | b << 1);
        }
    }
    // also on a superposition (phase-exact up to global phase)
    let mut c = Circuit::new(3);
    c.h(0).h(1);
    let mut t = c.clone();
    families::and_compute(&mut c, 0, 1, 2);
    t.ccx(0, 1, 2);
    assert!((1.0 - sv_of(&c).inner(&sv_of(&t)).norm()).abs() < TOL);
}

#[test]
fn hhl_inverts_eigenvalues() {
    for (t, m) in [(3usize, 2usize), (4, 2), (4, 3), (5, 3)] {
        let c = families::build(&format!("hhl:t={t},m={m}"), 1).unwrap();
        let sv = sv_of(&c);
        let amps = sv.amplitudes();
        // clock (low t bits) back to 0; P(anc = 1) = 2^-m Σ_x 1/(x+1)^2
        let anc = t + m;
        let mut p1 = 0.0;
        let mut clock_dirty = 0.0;
        for (i, a) in amps.iter().enumerate() {
            let pr = a.norm_sqr();
            if i & ((1 << t) - 1) != 0 {
                clock_dirty += pr;
            }
            if i >> anc & 1 == 1 {
                p1 += pr;
            }
        }
        let want: f64 = (0..1usize << m)
            .map(|x| 1.0 / ((x + 1) as f64).powi(2))
            .sum::<f64>()
            / (1 << m) as f64;
        assert!(clock_dirty < 1e-12, "t={t} m={m}: clock {clock_dirty}");
        assert!((p1 - want).abs() < 1e-10, "t={t} m={m}: {p1} vs {want}");
    }
}

#[test]
fn qpe_stab_reads_the_eigenphase() {
    // eigenvalue e^{-iΦ}; the counting register (qubit 0 = LSB) should peak
    // at y ≈ -Φ/(2π)·2^t mod 2^t, distribution |Σ_x e^{2πi x δ}|²/4^t.
    let (t, s) = (5usize, 4usize);
    let c = families::build(&format!("qpe:t={t},s={s},kind=stab"), 9).unwrap();
    let sv = sv_of(&c);
    let mut py = vec![0.0; 1 << t];
    for (i, a) in sv.amplitudes().iter().enumerate() {
        py[i & ((1 << t) - 1)] += a.norm_sqr();
    }
    // recover Φ from the circuit's angles the same way the builder drew them
    let mut rng = StdRng::seed_from_u64(9);
    let theta: f64 = rng.random_range(0.1..3.0);
    let phis: f64 = (0..s - 1).map(|_| rng.random_range(0.1..3.0)).sum();
    let phi = -(theta + phis) / (2.0 * std::f64::consts::PI);
    let tt = (1 << t) as f64;
    for (y, &p) in py.iter().enumerate() {
        let delta = phi - y as f64 / tt;
        let amp: Complex64 = (0..1 << t)
            .map(|x| Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI * x as f64 * delta))
            .sum::<Complex64>()
            / tt;
        assert!(
            (p - amp.norm_sqr()).abs() < 1e-10,
            "y={y}: {p} vs {}",
            amp.norm_sqr()
        );
    }
}

#[test]
fn state_magic_known_values_and_nullity_bounded_by_d() {
    // stabilizer state
    let mut c = Circuit::new(4);
    c.h(0).cnot(0, 1).s(1).h(2).cz(2, 3);
    let m = state_magic(sv_of(&c).amplitudes());
    assert!(m.nullity.abs() < 1e-12 && m.m2.abs() < 1e-12);
    // T|+>: nullity 1, M2 = -log2(3/4)
    let mut c = Circuit::new(1);
    c.h(0).t(0);
    let m = state_magic(sv_of(&c).amplitudes());
    assert!((m.nullity - 1.0).abs() < 1e-12);
    assert!((m.m2 - (-(0.75f64).log2())).abs() < 1e-12);
    // ν ≤ d on random Clifford+T and algorithm circuits
    for (i, spec) in [
        "rct:n=8,L=6,t=6",
        "qft:n=8,in=basis",
        "cuccaro:bits=3,in=plusa",
        "grover:n=5,it=1",
    ]
    .iter()
    .enumerate()
    {
        let c = families::build(spec, i as u64).unwrap();
        let p = profile(&c, &AtlasOptions::default()).unwrap();
        let m = state_magic(sv_of(&c).amplitudes());
        assert!(
            m.nullity <= p.d as f64 + 1e-9,
            "{spec}: ν={} d={}",
            m.nullity,
            p.d
        );
    }
}

/// Rényi-2 entanglement entropy of `|ψ>` across `{0..cut} | rest`.
fn renyi2(sv: &StateVectorF64, cut: usize) -> f64 {
    let a = sv.amplitudes();
    let la = 1usize << cut;
    let lb = a.len() / la;
    // ρ_A[i][j] = Σ_k ψ(i + k·la) conj(ψ(j + k·la))
    let mut rho = vec![Complex64::new(0.0, 0.0); la * la];
    for k in 0..lb {
        for i in 0..la {
            let vi = a[i + k * la];
            if vi.norm_sqr() == 0.0 {
                continue;
            }
            for j in 0..la {
                rho[i * la + j] += vi * a[j + k * la].conj();
            }
        }
    }
    -rho.iter().map(|v| v.norm_sqr()).sum::<f64>().log2()
}

#[test]
fn skeleton_entanglement_is_exact_for_cliffords_and_bounds_the_state() {
    let mut rng = StdRng::seed_from_u64(3);
    for trial in 0..20 {
        let n = 8;
        let c = Circuit::random_clifford(n, 6 + trial % 5, &mut rng);
        let p = profile(&c, &AtlasOptions::default()).unwrap();
        assert_eq!(p.d, 0);
        let e = p.checkpoints.last().unwrap().e_stab as f64;
        let s2 = renyi2(&sv_of(&c), n / 2);
        assert!((e - s2).abs() < 1e-9, "Clifford: E_stab {e} vs S2 {s2}");
    }
    for (i, spec) in SMALL.iter().enumerate() {
        let c = families::build(spec, i as u64 + 1).unwrap();
        if c.num_qubits > 14 {
            continue;
        }
        let p = profile(&c, &AtlasOptions::default()).unwrap();
        let ck = p.checkpoints.last().unwrap();
        let bound = (ck.e_stab + ck.d).min(p.cut.min(c.num_qubits - p.cut)) as f64;
        let s2 = renyi2(&sv_of(&c), p.cut);
        assert!(s2 <= bound + 1e-9, "{spec}: S2 {s2} > bound {bound}");
    }
}

#[test]
fn recycled_shor_oracle_is_exact_far_beyond_state_vector() {
    // windowed controlled-U_a (8-bit modulus, 38 qubits, T-count 5,880,
    // d = n): control |+>, x = 1  ->  (|0>|1> + |1>|a mod N>)/√2 ⊗ |0…>.
    let c = families::build("shorwin:nbits=8,w=2,in=one", 1).unwrap();
    let n = c.num_qubits;
    let p = profile(&c, &AtlasOptions::default()).unwrap();
    assert_eq!(p.d, n);
    let fr = FactoredState::with_recycling(&c, 8, 8).unwrap();
    assert!(fr.stats.f <= 2, "register {}", fr.stats.f);
    assert!(
        fr.stats.live.iter().all(|&(l, _)| l == 0),
        "not stabilizer at a gate boundary"
    );
    let (n_mod, a) = families::shor_modulus(8);
    let u: u64 = 1;
    let v: u64 = a % n_mod;
    let bit = |val: u64, q: usize| q >= 1 && q <= 8 && (val >> (q - 1)) & 1 == 1;
    for q in 0..n {
        let mut z = vec![false; n];
        z[q] = true;
        let e = fr.pauli_expectation(&vec![false; n], &z);
        let want = if q == 0 {
            0.0
        } else {
            match (bit(u, q), bit(v, q)) {
                (false, false) => 1.0,
                (true, true) => -1.0,
                _ => 0.0,
            }
        };
        assert!((e - want).abs() < TOL, "<Z_{q}> = {e}, want {want}");
    }
    let mut x = vec![false; n];
    x[0] = true;
    for q in 1..=8 {
        x[q] = bit(u, q) != bit(v, q);
    }
    assert!((fr.pauli_expectation(&x, &vec![false; n]) - 1.0).abs() < TOL);
    let mut zy = vec![false; n];
    zy[0] = true; // Y on the control
    assert!(fr.pauli_expectation(&x, &zy).abs() < TOL);
}

#[test]
fn stabilizer_synth_recognises_exactly_the_stabilizer_states() {
    let mut rng = StdRng::seed_from_u64(17);
    for _ in 0..200 {
        let k = rng.random_range(1..=6usize);
        let c = Circuit::random_clifford(k, rng.random_range(1..8), &mut rng);
        let sv = sv_of(&c);
        let g =
            qsim_lab::magic_atlas::stabilizer_synth(sv.amplitudes(), k).expect("stabilizer state");
        let mut kc = Circuit::new(k);
        for gt in g {
            kc.gate(gt);
        }
        assert!((1.0 - sv.inner(&sv_of(&kc)).norm()).abs() < 1e-10);
        // one T on a qubit in superposition usually breaks it; check that a
        // non-stabilizer state is never accepted
        let mut c2 = c.clone();
        c2.t(rng.random_range(0..k));
        let sv2 = sv_of(&c2);
        if qsim_lab::magic_atlas::state_magic(sv2.amplitudes()).nullity > 0.5 {
            assert!(qsim_lab::magic_atlas::stabilizer_synth(sv2.amplitudes(), k).is_none());
        }
    }
}
