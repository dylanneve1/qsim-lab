//! Differential tests for the sparse-Pauli-dynamics engine (`qsim_lab::spd`):
//! lattice construction against IBM's published coupling map, exact (δ = 0)
//! SPD against the dense state vector and against the exact Clifford+Rz
//! Pauli-path engine at 127 qubits, the rigorous l1 truncation bound, and the
//! noise-aware mode against an exact density matrix.

use num_complex::Complex64 as C;
use qsim_lab::pauli_path::{self, PauliSum};
use qsim_lab::spd::{simulate, KickedIsing, Lattice, PauliObs, SpdOptions};
use qsim_lab::StateVector;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

/// The 144 Eagle edges exactly as published with Kim et al. (Fig3a.ipynb of
/// github.com/youngseok-kim1/Evidence-for-the-utility-of-quantum-computing-before-fault-tolerance),
/// in its three edge-colour layers of 48.
const EAGLE_EDGES: [(usize, usize); 144] = [(2,1),(33,39),(59,60),(66,67),(72,81),(118,119),(21,20),(26,25),(13,12),(31,32),(70,74),(122,123),(97,96),(57,56),(63,64),(107,108),(103,104),(46,45),(28,35),(7,6),(79,78),(5,4),(109,114),(62,61),(58,71),(37,52),(76,77),(0,14),(36,51),(106,105),(73,85),(88,87),(68,55),(116,115),(94,95),(100,110),(17,30),(92,102),(50,49),(83,84),(48,47),(98,99),(8,9),(121,120),(23,24),(44,43),(22,15),(53,41),(53,60),(123,124),(21,22),(11,12),(67,68),(2,3),(66,65),(122,121),(110,118),(6,5),(94,90),(28,29),(14,18),(62,63),(111,104),(100,99),(45,44),(4,15),(20,19),(57,58),(77,71),(76,75),(26,27),(16,8),(35,47),(31,30),(48,49),(69,70),(125,126),(89,74),(80,79),(116,117),(114,113),(10,9),(106,93),(101,102),(92,83),(98,91),(82,81),(54,64),(96,109),(85,84),(87,86),(108,112),(34,24),(42,43),(40,41),(39,38),(10,11),(54,45),(111,122),(64,65),(60,61),(103,102),(72,62),(4,3),(33,20),(58,59),(26,16),(28,27),(8,7),(104,105),(66,73),(87,93),(85,86),(55,49),(68,69),(89,88),(80,81),(117,118),(101,100),(114,115),(96,95),(29,30),(106,107),(83,82),(91,79),(0,1),(56,52),(90,75),(126,112),(36,32),(46,47),(77,78),(97,98),(17,12),(119,120),(22,23),(24,25),(43,34),(42,41),(40,39),(37,38),(125,124),(50,51),(18,19)];

#[test]
fn eagle_matches_published_coupling_map() {
    let l = Lattice::eagle127();
    let p = Lattice::from_edges(127, &EAGLE_EDGES);
    assert_eq!(l.n, 127);
    assert_eq!(l.edges.len(), 144);
    assert_eq!(l, p);
    // The three published layers are perfect matchings of disjoint edges.
    for layer in EAGLE_EDGES.chunks(48) {
        let mut seen = [false; 127];
        for &(a, b) in layer {
            assert!(!seen[a] && !seen[b]);
            seen[a] = true;
            seen[b] = true;
        }
    }
}

#[test]
fn larger_heavy_hex_lattices() {
    for (l, n, e) in [
        (Lattice::osprey433(), 433, 504),
        (Lattice::condor1121(), 1121, 1320),
    ] {
        assert_eq!(l.n, n);
        assert_eq!(l.edges.len(), e, "edges of {n}");
        assert!(l.adj.iter().all(|a| (1..=3).contains(&a.len())));
        assert!(l.distances_from(&[0]).iter().all(|&d| d != usize::MAX));
        // heavy hex: every edge joins a degree-2 bridge/row qubit and a degree <=3 one,
        // and no two degree-3 qubits are adjacent.
        for &(a, b) in &l.edges {
            assert!(!(l.adj[a].len() == 3 && l.adj[b].len() == 3));
        }
    }
}

fn pauli_masks(n: usize, s: &[(usize, char)]) -> (usize, usize) {
    let (mut x, mut z) = (0usize, 0usize);
    for &(q, p) in s {
        assert!(q < n);
        match p {
            'X' => x |= 1 << q,
            'Y' => {
                x |= 1 << q;
                z |= 1 << q
            }
            'Z' => z |= 1 << q,
            _ => unreachable!(),
        }
    }
    (x, z)
}

/// <psi| P |psi> with P = i^{|x&z|} X^x Z^z.
fn sv_pauli(a: &[C], x: usize, z: usize) -> f64 {
    let ph = [C::new(1.0, 0.0), C::new(0.0, 1.0), C::new(-1.0, 0.0), C::new(0.0, -1.0)]
        [((x & z).count_ones() % 4) as usize];
    let mut acc = C::new(0.0, 0.0);
    for (b, amp) in a.iter().enumerate() {
        let s = if (z & b).count_ones() % 2 == 1 { -1.0 } else { 1.0 };
        acc += a[b ^ x].conj() * amp * s;
    }
    (acc * ph).re
}

fn sv_expect(model: &KickedIsing, obs: &PauliObs) -> f64 {
    let mut sv = StateVector::new(model.lattice.n);
    sv.apply_circuit(&model.to_circuit()).unwrap();
    obs.terms
        .iter()
        .map(|(s, c)| {
            let (x, z) = pauli_masks(model.lattice.n, s);
            c * sv_pauli(sv.amplitudes(), x, z)
        })
        .sum()
}

fn random_pauli(rng: &mut StdRng, n: usize, w: usize) -> Vec<(usize, char)> {
    let mut qs: Vec<usize> = (0..n).collect();
    for i in 0..w {
        let j = rng.random_range(i..n);
        qs.swap(i, j);
    }
    qs[..w]
        .iter()
        .map(|&q| (q, ['X', 'Y', 'Z'][rng.random_range(0..3)]))
        .collect()
}

#[test]
fn exact_spd_matches_statevector_on_heavy_hex_patches() {
    let eagle = Lattice::eagle127();
    let mut rng = StdRng::seed_from_u64(0x5bd1);
    let mut nonzero = 0;
    let mut total = 0;
    for (root, k) in [(62usize, 12usize), (0, 14), (37, 16), (75, 13)] {
        let patch = eagle.bfs_patch(root, k);
        for steps in 1..=4 {
            for final_rx in [false, true] {
                let theta = rng.random_range(0.0..PI);
                let mut m = KickedIsing::new(patch.clone(), steps, theta);
                m.final_rx = final_rx;
                let mut obs = vec![PauliObs::magnetisation(&(0..k).collect::<Vec<_>>())];
                for w in [1, 2, 3, 5] {
                    obs.push(PauliObs::single(random_pauli(&mut rng, k, w)));
                }
                // a multi-term observable with repeated strings
                let mut multi = PauliObs::single(random_pauli(&mut rng, k, 2));
                multi.terms.push((random_pauli(&mut rng, k, 3), -0.7));
                multi.terms.push((multi.terms[0].0.clone(), 0.25));
                obs.push(multi);
                for o in &obs {
                    let want = sv_expect(&m, o);
                    for light_cone in [true, false] {
                        let r = simulate(
                            &m,
                            o,
                            &SpdOptions {
                                light_cone,
                                ..SpdOptions::default()
                            },
                        );
                        assert!(
                            (r.value - want).abs() < 1e-10,
                            "root {root} k {k} steps {steps} rx {final_rx} θ {theta}: spd {} sv {want}",
                            r.value
                        );
                        assert_eq!(r.discarded_l1, 0.0);
                    }
                    total += 1;
                    if want.abs() > 1e-3 {
                        nonzero += 1;
                    }
                }
            }
        }
    }
    // Guard against a degenerate test (all-zero expectation values).
    assert!(nonzero * 2 > total, "only {nonzero}/{total} non-zero values");
}

#[test]
fn exact_spd_matches_statevector_at_special_angles() {
    let patch = Lattice::eagle127().bfs_patch(62, 12);
    for theta in [0.0, PI / 4.0, PI / 2.0, PI, -PI / 2.0, 3.0 * PI / 2.0] {
        for steps in 1..=3 {
            let m = KickedIsing::new(patch.clone(), steps, theta);
            for o in [PauliObs::z(0), PauliObs::parse("X0 Y1 Z2"), PauliObs::parse("Y3 Y5")] {
                let want = sv_expect(&m, &o);
                let got = simulate(&m, &o, &SpdOptions::default()).value;
                assert!((got - want).abs() < 1e-10, "θ {theta} steps {steps}: {got} vs {want}");
            }
        }
    }
}

#[test]
fn truncation_error_is_within_the_l1_bound_and_converges() {
    let patch = Lattice::eagle127().bfs_patch(62, 16);
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..6 {
        let theta = rng.random_range(0.2..1.4);
        let m = KickedIsing::new(patch.clone(), 4, theta);
        let o = PauliObs::single(random_pauli(&mut rng, 16, 2));
        let exact = sv_expect(&m, &o);
        let mut last_err = f64::INFINITY;
        for delta in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6] {
            let r = simulate(&m, &o, &SpdOptions { delta, ..SpdOptions::default() });
            let err = (r.value - exact).abs();
            assert!(err <= r.discarded_l1 + 1e-12, "δ {delta}: err {err} > bound {}", r.discarded_l1);
            assert!(r.norm2 <= 1.0 + 1e-12);
            last_err = err;
        }
        assert!(last_err < 1e-4, "δ=1e-6 error {last_err}");
        for w in [2usize, 4, 8] {
            let r = simulate(&m, &o, &SpdOptions { max_weight: w, ..SpdOptions::default() });
            assert!((r.value - exact).abs() <= r.discarded_l1 + 1e-12);
        }
    }
}

/// Exact 127-qubit cross-check against the independent Clifford+Rz
/// Pauli-path engine (frame compiler, exact pruning) at shallow depth.
#[test]
fn exact_spd_matches_pauli_path_engine_at_127_qubits() {
    let eagle = Lattice::eagle127();
    let obs = [
        "Z62",
        "X13 X29 X31 Y9 Y30 Z8 Z12 Z17 Z28 Z32",
        "X37 X41 X52 X56 X57 X58 X62 X79 Y75 Z38 Z40 Z42 Z63 Z72 Z80 Z90 Z91",
        "Y0 Z1",
    ];
    for (steps, theta) in [(1usize, 0.3), (2, 0.7), (2, 1.1), (3, 0.2)] {
        for final_rx in [false, true] {
            let mut m = KickedIsing::new(eagle.clone(), steps, theta);
            m.final_rx = final_rx;
            let circ = m.to_circuit();
            for s in obs {
                let o = PauliObs::parse(s);
                let mut chars = vec!['I'; 127];
                for &(q, p) in &o.terms[0].0 {
                    chars[q] = p;
                }
                let ps = PauliSum::from_str_single(&chars.iter().collect::<String>());
                let (want, _) = pauli_path::expectation(&circ, &ps, 1 << 24).unwrap();
                let got = simulate(&m, &o, &SpdOptions::default());
                assert!(
                    (got.value - want).abs() < 1e-10,
                    "{s} steps {steps} θ {theta} rx {final_rx}: spd {} path {want}",
                    got.value
                );
            }
        }
    }
}

/// At θ_h = π/2 the circuit is Clifford: the two weight-17 / weight-10
/// observables of Kim et al. are stabilizers (value ±1), and SPD keeps a
/// single term for any depth.
#[test]
fn clifford_point_matches_pauli_path_at_depth_20() {
    let eagle = Lattice::eagle127();
    for (steps, final_rx, s) in [
        (5, false, "X13 X29 X31 Y9 Y30 Z8 Z12 Z17 Z28 Z32"),
        (5, false, "X37 X41 X52 X56 X57 X58 X62 X79 Y75 Z38 Z40 Z42 Z63 Z72 Z80 Z90 Z91"),
        (5, true, "X37 X41 X52 X56 X57 X58 X62 X79 Y38 Y40 Y42 Y63 Y72 Y80 Y90 Y91 Z75"),
        (20, false, "Z62"),
        (20, false, "X0 Z1 Y126"),
    ] {
        let mut m = KickedIsing::new(eagle.clone(), steps, PI / 2.0);
        m.final_rx = final_rx;
        let o = PauliObs::parse(s);
        let mut chars = vec!['I'; 127];
        for &(q, p) in &o.terms[0].0 {
            chars[q] = p;
        }
        let ps = PauliSum::from_str_single(&chars.iter().collect::<String>());
        let (want, _) = pauli_path::expectation(&m.to_circuit(), &ps, 1 << 20).unwrap();
        let got = simulate(&m, &o, &SpdOptions::default());
        assert!((got.value - want).abs() < 1e-12, "{s}: {} vs {want}", got.value);
        assert_eq!(got.peak_terms, 1);
        if steps == 5 {
            assert!((want.abs() - 1.0).abs() < 1e-12, "{s} should be a stabilizer, got {want}");
        }
    }
}

// --- noise: exact density matrix on a small patch ---------------------------

type Mat = Vec<Vec<C>>;

fn apply_1q(rho: &mut Mat, n: usize, q: usize, u: [[C; 2]; 2]) {
    let d = 1 << n;
    // rho <- U rho U†, U acting on qubit q
    for side in 0..2 {
        for r in 0..d {
            for c in 0..d {
                let (i, j) = if side == 0 { (r, c) } else { (c, r) };
                if (i >> q) & 1 == 1 {
                    continue;
                }
                let i1 = i | (1 << q);
                let (a0, a1) = if side == 0 { (rho[i][j], rho[i1][j]) } else { (rho[j][i], rho[j][i1]) };
                let (u00, u01, u10, u11) = if side == 0 {
                    (u[0][0], u[0][1], u[1][0], u[1][1])
                } else {
                    (u[0][0].conj(), u[0][1].conj(), u[1][0].conj(), u[1][1].conj())
                };
                let b0 = u00 * a0 + u01 * a1;
                let b1 = u10 * a0 + u11 * a1;
                if side == 0 {
                    rho[i][j] = b0;
                    rho[i1][j] = b1;
                } else {
                    rho[j][i] = b0;
                    rho[j][i1] = b1;
                }
            }
        }
    }
}

fn apply_diag(rho: &mut Mat, f: impl Fn(usize) -> C) {
    let d = rho.len();
    for i in 0..d {
        for j in 0..d {
            rho[i][j] = f(i) * rho[i][j] * f(j).conj();
        }
    }
}

fn depolarize(rho: &mut Mat, n: usize, q: usize, p: f64) {
    let o = C::new(1.0, 0.0);
    let z = C::new(0.0, 0.0);
    let i = C::new(0.0, 1.0);
    let paulis = [[[z, o], [o, z]], [[z, -i], [i, z]], [[o, z], [z, -o]]];
    let mut out: Mat = rho.iter().map(|r| r.iter().map(|v| v * (1.0 - p)).collect()).collect();
    for pm in paulis {
        let mut t = rho.clone();
        apply_1q(&mut t, n, q, pm);
        for a in 0..rho.len() {
            for b in 0..rho.len() {
                out[a][b] += t[a][b] * (p / 3.0);
            }
        }
    }
    *rho = out;
}

#[test]
fn noisy_spd_matches_density_matrix() {
    let patch = Lattice::eagle127().bfs_patch(62, 6);
    let n = 6;
    let p = 0.03;
    let theta = 0.6;
    for steps in 1..=3 {
        let m = KickedIsing::new(patch.clone(), steps, theta);
        let d = 1 << n;
        let mut rho: Mat = vec![vec![C::new(0.0, 0.0); d]; d];
        rho[0][0] = C::new(1.0, 0.0);
        let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
        let rx = [[C::new(c, 0.0), C::new(0.0, -s)], [C::new(0.0, -s), C::new(c, 0.0)]];
        for _ in 0..steps {
            for q in 0..n {
                apply_1q(&mut rho, n, q, rx);
            }
            let edges = patch.edges.clone();
            apply_diag(&mut rho, |b| {
                // exp(+i π/4 Σ Z_a Z_b)
                let e: f64 = edges
                    .iter()
                    .map(|&(a, bb)| if ((b >> a) ^ (b >> bb)) & 1 == 0 { 1.0 } else { -1.0 })
                    .sum();
                C::new(0.0, PI / 4.0 * e).exp()
            });
            for q in 0..n {
                depolarize(&mut rho, n, q, p);
            }
        }
        for o in [PauliObs::z(0), PauliObs::parse("X0 Y1 Z3"), PauliObs::parse("Y2 Z4")] {
            let (x, z) = pauli_masks(n, &o.terms[0].0);
            // Tr(P rho) = Σ_b <b|P rho|b>; P|b'> = i^{|x&z|}(-1)^{z·b'}|b'^x>
            let ph = [C::new(1.0, 0.0), C::new(0.0, 1.0), C::new(-1.0, 0.0), C::new(0.0, -1.0)]
                [((x & z).count_ones() % 4) as usize];
            let mut tr = C::new(0.0, 0.0);
            for b in 0..d {
                // <b| P = (P† |b>)† = (P|b>)† since P Hermitian; P|b> = ph (-1)^{z·b} |b^x>
                let sgn = if (z & b).count_ones() % 2 == 1 { -1.0 } else { 1.0 };
                tr += (ph * sgn).conj() * rho[b ^ x][b];
            }
            let r = simulate(&m, &o, &SpdOptions { depol: p, ..SpdOptions::default() });
            assert!((r.value - tr.re).abs() < 1e-10, "steps {steps}: spd {} dm {}", r.value, tr.re);
            let clean = simulate(&m, &o, &SpdOptions::default());
            if clean.value.abs() > 1e-3 {
                assert!((r.value - clean.value).abs() > 1e-6, "noise had no effect");
            }
        }
    }
}
