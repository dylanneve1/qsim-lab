//! Exactness and theorem checks for `stab_rank` (research/theory/theory-rank.md).
#![allow(clippy::needless_range_loop)]
mod audit_common;
use audit_common::{max_amp_diff, random_circuit, RefSv};
use qsim_lab::engines::stab_rank::RankState;
use qsim_lab::Circuit;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn check(c: &Circuit, tol: f64) -> (usize, f64) {
    let reference = RefSv::run(c);
    let mut rs = RankState::new(c.num_qubits);
    assert!(rs.run(c));
    let sv = rs.to_statevector();
    let d = max_amp_diff(&reference.a, sv.into_iter());
    assert!(d < tol, "amplitude error {d} (rank {})", rs.rank());
    (rs.stats.max_r, d)
}

#[test]
fn ch_form_random_clifford_exact() {
    let mut rng = StdRng::seed_from_u64(1);
    for it in 0..300 {
        let n = 1 + it % 7;
        let c = random_circuit(&mut rng, n, 60, true, false);
        let (r, _) = check(&c, 1e-10);
        assert_eq!(r, 1);
    }
}

#[test]
fn random_clifford_t_exact() {
    let mut rng = StdRng::seed_from_u64(2);
    for it in 0..200 {
        let n = 2 + it % 6;
        let c = random_circuit(&mut rng, n, 40, false, true);
        check(&c, 1e-9);
    }
}

#[test]
fn random_universal_exact() {
    let mut rng = StdRng::seed_from_u64(3);
    for it in 0..200 {
        let n = 3 + it % 5;
        let c = random_circuit(&mut rng, n, 30, false, false);
        check(&c, 1e-9);
    }
}

// ---------------------------------------------------------------------------
// dense helpers

use num_complex::Complex64 as C;
use qsim_lab::engines::stab_rank::Pauli;
use qsim_lab::magic_atlas::{families, state_magic};
use qsim_lab::Gate;
use rand::Rng;

/// Dense `P|ψ⟩` for `P = i^e X^x Z^z` (x, z as bit masks over n ≤ 16).
fn pauli_apply(psi: &[C], x: usize, z: usize, e: u8) -> Vec<C> {
    let ph = [
        C::new(1.0, 0.0),
        C::new(0.0, 1.0),
        C::new(-1.0, 0.0),
        C::new(0.0, -1.0),
    ][e as usize % 4];
    let mut out = vec![C::new(0.0, 0.0); psi.len()];
    for (y, a) in psi.iter().enumerate() {
        // Z^z first, then X^x
        let s = if (y & z).count_ones() % 2 == 1 {
            -1.0
        } else {
            1.0
        };
        out[y ^ x] += ph * a * s;
    }
    out
}

fn to_mask(b: &[u64]) -> usize {
    b[0] as usize
}

fn random_clifford(rng: &mut StdRng, n: usize, depth: usize) -> Circuit {
    random_circuit(rng, n, depth, true, false)
}

fn state_of(c: &Circuit) -> RankState {
    let mut rs = RankState::new(c.num_qubits);
    assert!(rs.run(c));
    rs
}

fn ray_equal(a: &[C], b: &[C]) -> bool {
    let ov: C = a.iter().zip(b).map(|(x, y)| x.conj() * y).sum();
    let na: f64 = a.iter().map(|x| x.norm_sqr()).sum();
    let nb: f64 = b.iter().map(|x| x.norm_sqr()).sum();
    (ov.norm() - (na * nb).sqrt()).abs() < 1e-9
}

#[test]
fn canonical_key_identifies_rays() {
    let mut rng = StdRng::seed_from_u64(11);
    let mut same = 0;
    for it in 0..400 {
        let n = 1 + it % 4;
        let c1 = random_clifford(&mut rng, n, 25);
        // c2: c1 followed by a random Clifford w and w^{-1}, plus a Pauli
        // stabilizer of the state half the time (same ray) or a random
        // Clifford (other ray)
        let mut c2 = c1.clone();
        let w = random_clifford(&mut rng, n, 10);
        for g in w.gates() {
            c2.gate(*g);
        }
        let gl: Vec<Gate> = w.gates().cloned().collect();
        for g in gl.iter().rev() {
            c2.gate(g.inverse());
        }
        if it % 2 == 1 {
            let extra = random_clifford(&mut rng, n, 3);
            for g in extra.gates() {
                c2.gate(*g);
            }
        }
        let (s1, s2) = (state_of(&c1), state_of(&c2));
        let (a, b) = (s1.to_statevector(), s2.to_statevector());
        let k_eq = s1.terms[0].st.canonical_key() == s2.terms[0].st.canonical_key();
        assert_eq!(k_eq, ray_equal(&a, &b), "key/ray mismatch at it {it}");
        same += k_eq as usize;
    }
    assert!(same > 200 && same < 400);
}

/// Theorem R3: a projector gate `I + (λ−1)Π` leaves a stabilizer state a
/// stabilizer state iff the engine classifies it as Diagonal/Clifford, and
/// the engine's update equals the dense one.
#[test]
fn thm_r3_branching_criterion_is_exact() {
    let mut rng = StdRng::seed_from_u64(12);
    let lambdas = [
        C::new(-1.0, 0.0),
        C::new(0.0, 1.0),
        C::new(0.0, -1.0),
        C::from_polar(1.0, std::f64::consts::FRAC_PI_4),
        C::from_polar(1.0, 0.3),
        C::new(0.5, 0.0),
    ];
    let (mut nb, mut nc, mut nd) = (0, 0, 0);
    for it in 0..1500 {
        let n = 2 + it % 4;
        let c = random_clifford(&mut rng, n, 30);
        let rs = state_of(&c);
        let phi = rs.to_statevector();
        // commuting factors: W Z_k W† for a random Clifford W, random signs,
        // m = 1..3 factors (possibly products of Z's)
        let wc = random_clifford(&mut rng, n, 20);
        let m = 1 + rng.random_range(0..3usize);
        let mut factors = Vec::new();
        for _ in 0..m {
            let mut zmask = 0usize;
            while zmask == 0 {
                zmask = rng.random_range(1..1usize << n);
            }
            let mut p = Pauli::identity(n);
            for q in 0..n {
                if (zmask >> q) & 1 == 1 {
                    p.mul_assign(&Pauli::z(n, q, false));
                }
            }
            if rng.random::<bool>() {
                p.e = (p.e + 2) % 4;
            }
            for g in wc.gates() {
                for h in g.decompose_to_clifford_rz() {
                    qsim_lab::engines::stab_rank::conj_gate(&mut p, &h);
                }
            }
            factors.push(p);
        }
        let lambda = lambdas[it % lambdas.len()];
        // dense U φ
        let mut pi_phi = phi.clone();
        for p in &factors {
            let pp = pauli_apply(&pi_phi, to_mask(&p.x), to_mask(&p.z), p.e);
            pi_phi = pi_phi.iter().zip(&pp).map(|(a, b)| (a + b) * 0.5).collect();
        }
        let uphi: Vec<C> = phi
            .iter()
            .zip(&pi_phi)
            .map(|(a, b)| a + (lambda - 1.0) * b)
            .collect();
        let nrm: f64 = uphi.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        let unit: Vec<C> = uphi.iter().map(|x| x / nrm).collect();
        let is_stab = state_magic(&unit).nullity < 0.5;
        let (act, _, _, _) = RankState::classify(&rs.terms[0].st, lambda, &factors);
        use qsim_lab::engines::stab_rank::Action;
        match act {
            Action::Branch => nb += 1,
            Action::Clifford => nc += 1,
            Action::Diagonal => nd += 1,
        }
        if (lambda.norm() - 1.0).abs() < 1e-12 {
            assert_eq!(
                is_stab,
                act != Action::Branch,
                "it {it}: λ={lambda} act={act:?}"
            );
        } else if act != Action::Branch {
            // non-unitary λ: Clifford/Diagonal still must give a stabilizer ray
            assert!(is_stab || act == Action::Diagonal);
        }
        // engine update equals dense
        let mut rs2 = rs.clone();
        rs2.projector_gate(lambda, &factors, C::new(1.0, 0.0));
        let got = rs2.to_statevector();
        let err = max_amp_diff(&uphi, got.into_iter());
        assert!(err < 1e-10, "it {it}: err {err} act {act:?}");
    }
    eprintln!("R3 cases: branch {nb} clifford {nc} diagonal {nd}");
    assert!(nb > 100 && nc > 100 && nd > 100, "{nb} {nc} {nd}");
}

/// Count of Paulis (with sign) stabilizing a dense state: 2^{n-ν}.
fn stab_set(psi: &[C], n: usize) -> Vec<(usize, usize, u8)> {
    let mut out = Vec::new();
    for x in 0..1usize << n {
        for z in 0..1usize << n {
            let e = ((x & z).count_ones() % 2) as u8; // Hermitian phase i^{x·z}
            for sgn in [0u8, 2] {
                let pp = pauli_apply(psi, x, z, (e + sgn) % 4);
                let d: f64 = pp.iter().zip(psi).map(|(a, b)| (a - b).norm_sqr()).sum();
                if d < 1e-18 {
                    out.push((x, z, (e + sgn) % 4));
                }
            }
        }
    }
    out
}

/// Theorem R2: for two non-proportional stabilizer states,
/// ν(αφ1 + βφ2) ≤ s := n − log2|Stab φ1 ∩ Stab φ2| for every ratio, with
/// equality for generic ratios; and |⟨φ1|φ2⟩|² = 2^{−s} when nonzero.
#[test]
fn thm_r2_rank_two_nullity() {
    let mut rng = StdRng::seed_from_u64(13);
    let mut drops = 0;
    for it in 0..300 {
        let n = 1 + it % 4;
        let a = state_of(&random_clifford(&mut rng, n, 20)).to_statevector();
        let b = state_of(&random_clifford(&mut rng, n, 20)).to_statevector();
        if ray_equal(&a, &b) {
            continue;
        }
        let sa = stab_set(&a, n);
        let sb = stab_set(&b, n);
        let common = sa.iter().filter(|p| sb.contains(p)).count();
        let s = n as f64 - (common as f64).log2();
        let ov: C = a.iter().zip(&b).map(|(x, y)| x.conj() * y).sum();
        if ov.norm() > 1e-9 {
            assert!((ov.norm_sqr() - 2f64.powf(-s)).abs() < 1e-9, "overlap law");
        } else {
            assert!(s >= 1.0);
        }
        let ratios = [
            C::new(0.37, 0.81),
            C::new(1.0, 0.0),
            C::new(-1.0, 0.0),
            C::new(0.0, 1.0),
            C::new(0.0, -1.0),
            C::new(2.0, 0.0),
            C::new(-0.5, 0.0),
            C::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
            C::new(-std::f64::consts::SQRT_2, 0.0),
            C::from_polar(1.0, std::f64::consts::FRAC_PI_4),
        ];
        for (k, r) in ratios.iter().enumerate() {
            let v: Vec<C> = a.iter().zip(&b).map(|(x, y)| x + r * y).collect();
            let nrm = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            if nrm < 1e-9 {
                continue;
            }
            let u: Vec<C> = v.iter().map(|x| x / nrm).collect();
            let nu = state_magic(&u).nullity;
            assert!(nu <= s + 1e-9, "ν {nu} > s {s}");
            if k == 0 {
                assert!((nu - s).abs() < 1e-9, "generic ratio: ν {nu} ≠ s {s}");
            } else if nu < s - 0.5 {
                drops += 1;
            }
        }
    }
    assert!(drops > 0, "special ratios never lowered ν (expected some)");
}

/// Families at n ≤ 14: the rank engine equals the state vector.
#[test]
fn families_exact_vs_statevector() {
    let specs = [
        "qft:n=8,in=basis",
        "qft:n=8,in=plus",
        "qft:n=5,in=graph",
        "cuccaro:bits=4,in=basis",
        "cuccaro:bits=4,in=plusa",
        "cuccaro:bits=3,in=plusab",
        "gidney:bits=2,in=plusa",
        "draper:bits=4,in=basis",
        "draper:bits=3,in=plusab",
        "grover:n=5,it=3",
        "grover:n=7,it=2",
        "ising:n=6,steps=2",
        "heis:n=6,steps=1",
        "qaoa:n=6,p=1",
        "hea:n=6,layers=1",
        "qpe:t=4,s=4,kind=stab",
        "qpe:t=2,s=4,kind=trotter",
        "walk:m=3,steps=2",
        "hhl:t=3,m=2",
        "rct:n=8,L=6,t=8",
    ];
    for spec in specs {
        let c = families::build(spec, 1).unwrap();
        assert!(c.num_qubits <= 16, "{spec}");
        let (r, d) = check(&c, 1e-9);
        eprintln!("{spec}: n={} max_r={r} err={d:.1e}", c.num_qubits);
    }
}

/// Grover with the Toffoli-ladder oracle: r = 2 at every iteration end,
/// r ≤ n_search during the oracle, and exact amplitudes.
#[test]
fn grover_rank_two_at_iteration_boundaries() {
    for n in [4usize, 6, 9, 12, 16, 20] {
        let it = 3;
        let c = families::build(&format!("grover:n={n},it={it}"), 7).unwrap();
        let mut rs = RankState::new(c.num_qubits);
        assert!(rs.run(&c));
        let per = (c.num_gates() - n) / it;
        for k in 1..=it {
            assert_eq!(rs.stats.r[n + k * per - 1], 2, "n={n} iteration {k}");
        }
        assert!(rs.stats.max_r <= n, "n={n} max_r={}", rs.stats.max_r);
        if c.num_qubits <= 14 {
            check(&c, 1e-10);
        }
    }
}

/// Theorem R4: reversible (X/CNOT/Toffoli) circuits on an input with at most
/// two basis branches never branch (r = 1 at every gate).
#[test]
fn thm_r4_two_branch_permutation_circuits_rank_one() {
    for spec in [
        "shorwin:nbits=4,w=2,in=one",
        "shorwin:nbits=8,w=3,in=one",
        "cuccaro:bits=16,in=basis",
    ] {
        let c = families::build(spec, 3).unwrap();
        let mut rs = RankState::new(c.num_qubits);
        assert!(rs.run(&c));
        assert_eq!(rs.stats.max_r, 1, "{spec}");
        assert_eq!(rs.stats.branch_events, 0, "{spec}");
    }
    // random reversible circuits on (|u> + i^k |v>)/√2 inputs
    let mut rng = StdRng::seed_from_u64(14);
    for it in 0..100 {
        let n = 4 + it % 5;
        let mut c = Circuit::new(n);
        // two-branch input: H on q0, CNOTs from q0, X's
        c.h(0);
        for q in 1..n {
            if rng.random::<bool>() {
                c.cnot(0, q);
            }
            if rng.random::<bool>() {
                c.x(q);
            }
        }
        if it % 2 == 1 {
            c.s(0);
        }
        for _ in 0..40 {
            let a = rng.random_range(0..n);
            let mut b = rng.random_range(0..n);
            while b == a {
                b = rng.random_range(0..n);
            }
            let mut t = rng.random_range(0..n);
            while t == a || t == b {
                t = rng.random_range(0..n);
            }
            match rng.random_range(0..3) {
                0 => c.x(a),
                1 => c.cnot(a, b),
                _ => c.ccx(a, b, t),
            };
        }
        let (r, _) = check(&c, 1e-10);
        assert_eq!(r, 1);
    }
}
