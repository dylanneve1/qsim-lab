//! The free-fermion (Gaussian) detector and engine (docs/ENGINE_GAUSSIAN.md):
//! random matchgate circuits against the state-vector engine to 1e-10
//! (`⟨Z_i⟩`, `⟨Z_i Z_j⟩`, three-point Z products, every basis-state
//! probability, a sampling check), in the qubit order, in a permuted layout,
//! and behind a SWAP network; non-Gaussian circuits are rejected; and the
//! SU(2) hadron-dynamics tracker circuits reproduce the free-fermion values
//! of research/data/su2-hadron.

use qsim_lab::circuit::{Circuit, SimError};
use qsim_lab::engines::gaussian::{
    self, DetectOptions, GaussianOptions, GaussianState, InteractionPolicy, Ordering,
};
use qsim_lab::engines::statevector::StateVectorF64;
use qsim_lab::gate::Gate;
use qsim_lab::planner::{self, Engine, PlanRequest, PlannerConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

const TOL: f64 = 1e-10;

/// Appends `g` with every qubit mapped through `perm`.
fn push(c: &mut Circuit, perm: &[usize], g: Gate) {
    use Gate::*;
    let p = |q: usize| perm[q];
    let h = match g {
        H(a) => H(p(a)),
        X(a) => X(p(a)),
        Y(a) => Y(p(a)),
        Z(a) => Z(p(a)),
        S(a) => S(p(a)),
        Sdg(a) => Sdg(p(a)),
        T(a) => T(p(a)),
        Rz(a, t) => Rz(p(a), t),
        Phase(a, t) => Phase(p(a), t),
        Rx(a, t) => Rx(p(a), t),
        Cnot(a, b) => Cnot(p(a), p(b)),
        Cz(a, b) => Cz(p(a), p(b)),
        Swap(a, b) => Swap(p(a), p(b)),
        ISwap(a, b) => ISwap(p(a), p(b)),
        ISwapdg(a, b) => ISwapdg(p(a), p(b)),
        CPhase(a, b, t) => CPhase(p(a), p(b), t),
        Ccx(a, b, d) => Ccx(p(a), p(b), p(d)),
        g => panic!("push: {g:?}"),
    };
    c.ops.push(qsim_lab::Op::Gate(h));
}

/// Basis change taking Z to `axis` (0 = X, 1 = Y), in time order, and back.
fn to_axis(axis: u8, q: usize) -> (Vec<Gate>, Vec<Gate>) {
    match axis {
        0 => (vec![Gate::H(q)], vec![Gate::H(q)]),
        _ => (vec![Gate::Sdg(q), Gate::H(q)], vec![Gate::H(q), Gate::S(q)]),
    }
}

/// `exp(-i θ/2 P_a P_b)` with `P ∈ {X, Y}`: a quadratic Majorana term on
/// adjacent modes, written with CNOTs in a random orientation.
fn pauli_pair(a: usize, b: usize, pa: u8, pb: u8, t: f64, flip: bool) -> Vec<Gate> {
    let (ia, oa) = to_axis(pa, a);
    let (ib, ob) = to_axis(pb, b);
    let mut v: Vec<Gate> = ia.into_iter().chain(ib).collect();
    let (c, tg) = if flip { (b, a) } else { (a, b) };
    v.extend([Gate::Cnot(c, tg), Gate::Rz(tg, t), Gate::Cnot(c, tg)]);
    v.extend(oa.into_iter().chain(ob));
    v
}

/// A random matchgate circuit on the line `0..n` (logical qubits): one-site
/// phases, X/Y flips (parity-odd), the four quadratic pair terms, iSWAPs and
/// fermionic SWAPs written as `SWAP; CZ`.
fn random_matchgate(n: usize, depth: usize, rng: &mut StdRng) -> Vec<Gate> {
    let mut g = Vec::new();
    for q in 0..n {
        if rng.random_bool(0.5) {
            g.push(Gate::X(q));
        }
    }
    for _ in 0..depth {
        let r = rng.random_range(0..10);
        let q = rng.random_range(0..n);
        let t = rng.random_range(-PI..PI);
        match r {
            0 => g.push(Gate::Rz(q, t)),
            1 => g.push(if rng.random_bool(0.5) {
                Gate::T(q)
            } else {
                Gate::Phase(q, t)
            }),
            2 => g.push(if rng.random_bool(0.5) {
                Gate::X(q)
            } else {
                Gate::Y(q)
            }),
            _ if n >= 2 => {
                let a = rng.random_range(0..n - 1);
                let b = a + 1;
                let (a, b) = if rng.random_bool(0.5) { (a, b) } else { (b, a) };
                match r {
                    3..=6 => {
                        let (pa, pb) = (rng.random_range(0..2), rng.random_range(0..2));
                        g.extend(pauli_pair(a, b, pa, pb, t, rng.random_bool(0.5)));
                    }
                    7 => g.push(if rng.random_bool(0.5) {
                        Gate::ISwap(a, b)
                    } else {
                        Gate::ISwapdg(a, b)
                    }),
                    8 => g.extend([Gate::Swap(a, b), Gate::Cz(a, b)]),
                    _ => {
                        // a hopping term: XX + YY with the same angle
                        g.extend(pauli_pair(a, b, 0, 0, t, false));
                        g.extend(pauli_pair(a, b, 1, 1, t, true));
                    }
                }
            }
            _ => g.push(Gate::Rz(q, t)),
        }
    }
    g
}

fn reference(c: &Circuit) -> Vec<f64> {
    let mut sv = StateVectorF64::new(c.num_qubits);
    for g in c.gates() {
        sv.apply_gate(g).unwrap();
    }
    sv.amplitudes().iter().map(|a| a.norm_sqr()).collect()
}

fn z_product(p: &[f64], qs: &[usize]) -> f64 {
    let mask = qs.iter().fold(0usize, |m, &q| m ^ (1 << q));
    p.iter()
        .enumerate()
        .map(|(x, v)| {
            if (x & mask).count_ones() % 2 == 1 {
                -v
            } else {
                *v
            }
        })
        .sum()
}

/// Every read-out of the engine against the state vector.
fn check_against_sv(c: &Circuit, rng: &mut StdRng, what: &str) -> gaussian::GaussianReport {
    let n = c.num_qubits;
    let (st, rep) = GaussianState::from_circuit(c, &GaussianOptions::default())
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    assert!(rep.exact && rep.max_residual < TOL, "{what}: {rep:?}");
    let p = reference(c);
    for i in 0..n {
        let want = z_product(&p, &[i]);
        let got = st.expectation_z(i).unwrap();
        assert!((got - want).abs() < TOL, "{what}: <Z_{i}> {got} vs {want}");
        for j in i + 1..n {
            let want = z_product(&p, &[i, j]);
            let got = st.z_correlation(i, j).unwrap();
            assert!(
                (got - want).abs() < TOL,
                "{what}: <Z_{i} Z_{j}> {got} vs {want}"
            );
        }
    }
    for _ in 0..8 {
        let k = rng.random_range(1..=n.min(5));
        let qs: Vec<usize> = (0..k).map(|_| rng.random_range(0..n)).collect();
        let want = z_product(&p, &qs);
        let got = st.expectation_z_product(&qs).unwrap();
        assert!(
            (got - want).abs() < TOL,
            "{what}: <Z{qs:?}> {got} vs {want}"
        );
    }
    let xs: Vec<usize> = if n <= 8 {
        (0..1 << n).collect()
    } else {
        (0..64).map(|_| rng.random_range(0..1usize << n)).collect()
    };
    for x in xs {
        let got = st.probability(x as u128).unwrap();
        assert!(
            (got - p[x]).abs() < TOL,
            "{what}: p({x:b}) {got} vs {}",
            p[x]
        );
    }
    rep
}

#[test]
fn random_matchgate_circuits_match_the_state_vector() {
    let mut rng = StdRng::seed_from_u64(0x6a55);
    for trial in 0..24 {
        let n = rng.random_range(1..=12);
        let depth = rng.random_range(5..60);
        let gates = random_matchgate(n, depth, &mut rng);
        let ident: Vec<usize> = (0..n).collect();
        let mut c = Circuit::new(n);
        for &g in &gates {
            push(&mut c, &ident, g);
        }
        check_against_sv(&c, &mut rng, &format!("trial {trial} (n={n})"));
    }
}

#[test]
fn a_permuted_layout_is_reordered_along_the_chain() {
    let mut rng = StdRng::seed_from_u64(0x6a56);
    for trial in 0..10 {
        let n = rng.random_range(4..=10);
        let gates = random_matchgate(n, 80, &mut rng);
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            perm.swap(i, rng.random_range(0..=i));
        }
        let mut c = Circuit::new(n);
        for &g in &gates {
            push(&mut c, &perm, g);
        }
        let rep = check_against_sv(&c, &mut rng, &format!("permuted {trial}"));
        let id_ok = (0..n.saturating_sub(1)).all(|k| perm[k].abs_diff(perm[k + 1]) == 1);
        if !id_ok {
            assert_eq!(rep.ordering, Some(Ordering::Paths), "permuted {trial}");
        }
    }
}

#[test]
fn a_swap_network_is_undone_by_relabelling() {
    // the logical chain moves through the register by SWAPs; the gates act
    // on whatever qubit holds the logical site at that moment
    let mut rng = StdRng::seed_from_u64(0x6a57);
    for trial in 0..10 {
        let n = rng.random_range(3..=9);
        let gates = random_matchgate(n, 60, &mut rng);
        let mut at: Vec<usize> = (0..n).collect(); // logical -> physical
        let mut c = Circuit::new(n);
        for &g in &gates {
            if rng.random_bool(0.3) {
                let (p, q) = (rng.random_range(0..n), rng.random_range(0..n));
                if p != q {
                    c.swap(p, q);
                    let (lp, lq) = (
                        at.iter().position(|&x| x == p).unwrap(),
                        at.iter().position(|&x| x == q).unwrap(),
                    );
                    at.swap(lp, lq);
                }
            }
            push(&mut c, &at, g);
        }
        let rep = check_against_sv(&c, &mut rng, &format!("swap network {trial}"));
        assert!(rep.swaps_relabelled > 0 || n < 2);
    }
}

#[test]
fn samples_follow_the_born_rule() {
    // 20000 shots: binomial standard deviation <= 0.0036 per outcome, the
    // tolerance 0.025 is about 7 sigma.
    let mut rng = StdRng::seed_from_u64(0x6a58);
    let n = 5;
    let gates = random_matchgate(n, 50, &mut rng);
    let ident: Vec<usize> = (0..n).collect();
    let mut c = Circuit::new(n);
    for &g in &gates {
        push(&mut c, &ident, g);
    }
    let (st, _) = GaussianState::from_circuit(&c, &GaussianOptions::default()).unwrap();
    let shots = 20000;
    let s = st.sample(shots, &mut rng).unwrap();
    let p = reference(&c);
    let mut freq = vec![0.0; 1 << n];
    for x in s {
        freq[x as usize] += 1.0 / shots as f64;
    }
    for x in 0..1 << n {
        assert!(
            (freq[x] - p[x]).abs() < 0.025,
            "{x:b}: {} vs {}",
            freq[x],
            p[x]
        );
    }
}

fn line_hops(n: usize) -> Circuit {
    let mut c = Circuit::new(n);
    for q in (0..n).step_by(2) {
        c.x(q);
    }
    for _ in 0..3 {
        for a in 0..n - 1 {
            for g in pauli_pair(a, a + 1, 0, 0, 0.4, false)
                .into_iter()
                .chain(pauli_pair(a, a + 1, 1, 1, 0.4, false))
            {
                c.ops.push(qsim_lab::Op::Gate(g));
            }
        }
    }
    c
}

#[test]
fn non_gaussian_circuits_are_rejected() {
    let opts = GaussianOptions::default();
    let reject = |c: &Circuit, what: &str| {
        let r = gaussian::detect(c, &DetectOptions::default());
        assert!(!r.exact, "{what}: {r:?}");
        assert!(r.gaussian_fraction < 1.0, "{what}");
        match GaussianState::from_circuit(c, &opts) {
            Err(SimError::NotSupported { .. }) => {}
            other => panic!("{what}: {:?}", other.map(|x| x.1)),
        }
        r
    };
    // a lone Hadamard (superposition of parity sectors)
    let mut c = line_hops(4);
    c.h(2);
    let r = reject(&c, "hadamard");
    assert!(r.max_residual > 0.1 && r.non_gaussian >= 1);
    // a lone CNOT
    let mut c = line_hops(4);
    c.cnot(1, 2);
    reject(&c, "cnot");
    // Rx on the first mode (a linear Majorana term)
    let mut c = line_hops(3);
    c.rx(0, 0.3);
    reject(&c, "rx");
    // a Toffoli
    let mut c = line_hops(4);
    c.ccx(0, 1, 2);
    reject(&c, "toffoli");
    // hopping on a ring: no Jordan-Wigner order makes every edge adjacent
    let mut c = line_hops(5);
    for g in pauli_pair(4, 0, 0, 0, 0.3, false) {
        c.ops.push(qsim_lab::Op::Gate(g));
    }
    let r = reject(&c, "ring");
    assert!(r.nonadjacent >= 1, "{r:?}");
    // a star (degree 3)
    let mut c = line_hops(4);
    for g in pauli_pair(1, 3, 1, 1, 0.3, false) {
        c.ops.push(qsim_lab::Op::Gate(g));
    }
    let r = reject(&c, "star");
    assert!(r.nonadjacent >= 1, "{r:?}");
    // a measurement in the middle
    let mut c = line_hops(3);
    c.measure(1).x(1);
    assert!(matches!(
        GaussianState::from_circuit(&c, &opts),
        Err(SimError::MeasurementNotSupported { .. })
    ));
}

#[test]
fn interaction_phases_are_reported_and_only_dropped_on_request() {
    let mut c = line_hops(4);
    c.cphase(1, 2, 0.25);
    c.cphase(0, 3, -0.5); // non-adjacent: a diagonal block needs no adjacency
    for g in pauli_pair(0, 1, 0, 1, 0.3, false) {
        c.ops.push(qsim_lab::Op::Gate(g));
    }
    let r = gaussian::detect(&c, &DetectOptions::default());
    assert!(r.free && !r.exact, "{r:?}");
    assert_eq!(r.interactions.len(), 2);
    assert!((r.interaction_total - 0.75).abs() < 1e-12);
    assert!((r.interaction_max - 0.5).abs() < 1e-12);
    assert!(matches!(
        GaussianState::from_circuit(&c, &GaussianOptions::default()),
        Err(SimError::NotSupported { .. })
    ));
    // CPhase(θ) is exp(iθ n_a n_b) with no one-site phase, so dropping the
    // interaction phases is the same as deleting the two gates
    let drop = GaussianOptions {
        interactions: InteractionPolicy::Drop,
        ..Default::default()
    };
    let (st, _) = GaussianState::from_circuit(&c, &drop).unwrap();
    let mut free = line_hops(4);
    for g in pauli_pair(0, 1, 0, 1, 0.3, false) {
        free.ops.push(qsim_lab::Op::Gate(g));
    }
    let p = reference(&free);
    for q in 0..4 {
        assert!((st.expectation_z(q).unwrap() - z_product(&p, &[q])).abs() < TOL);
    }
    // the hook sees every interaction phase, in time order
    let prog = gaussian::compile(&c, &DetectOptions::default());
    let mut seen = Vec::new();
    GaussianState::evolve_with(&prog, 1 << 20, |_, ip| seen.push(ip.g)).unwrap();
    assert_eq!(seen, vec![0.25, -0.5]);
}

#[test]
fn cz_is_a_fermionic_swap_plus_a_relabelling() {
    // CZ = SWAP · fSWAP: an interaction phase of π is Gaussian up to a
    // renaming of the wires
    // (afterwards qubit 1 holds mode 2 and vice versa, so only the pair
    // (1, 2) may interact again without leaving the chain)
    let mut c = line_hops(4);
    c.cz(1, 2);
    c.rz(1, 0.7).rz(2, -0.2);
    for g in pauli_pair(1, 2, 0, 0, 0.5, false) {
        c.ops.push(qsim_lab::Op::Gate(g));
    }
    let mut rng = StdRng::seed_from_u64(1);
    let r = check_against_sv(&c, &mut rng, "cz");
    assert!(r.swaps_relabelled >= 1);
}

#[test]
fn the_planner_chooses_the_gaussian_engine() {
    let mut rng = StdRng::seed_from_u64(0x6a59);
    // too large for a state vector
    let n = 48;
    let gates = random_matchgate(n, 600, &mut rng);
    let ident: Vec<usize> = (0..n).collect();
    let mut c = Circuit::new(n);
    for &g in &gates {
        push(&mut c, &ident, g);
    }
    let cfg = PlannerConfig {
        cache: false,
        ..Default::default()
    };
    let obs = vec![3, 17];
    let p = planner::plan(&c, &PlanRequest::Expectation(obs.clone()), &cfg).unwrap();
    assert_eq!(p.engine, Engine::Gaussian, "{:?}", p.ranked);
    let ex = planner::execute_expectation(&p, &c, &obs, &cfg).unwrap();
    let want = gaussian::expectation_z_product(&c, &obs, &GaussianOptions::default()).unwrap();
    assert_eq!(ex.engine, Engine::Gaussian);
    assert!((ex.value - want).abs() < 1e-12);
    let p = planner::plan(&c, &PlanRequest::Samples(10), &cfg).unwrap();
    assert_eq!(p.engine, Engine::Gaussian);
    let s = planner::execute_samples(&p, &c, 10, &mut rng, &cfg).unwrap();
    assert_eq!(s.samples.len(), 10);
    // amplitudes need the global phase: never the Gaussian engine
    let p = planner::plan(&c, &PlanRequest::Amplitudes(1), &cfg).unwrap();
    assert_ne!(p.engine, Engine::Gaussian);

    // small: the debug reference checks the planned value
    let n = 10;
    let gates = random_matchgate(n, 400, &mut rng);
    let ident: Vec<usize> = (0..n).collect();
    let mut c = Circuit::new(n);
    for &g in &gates {
        push(&mut c, &ident, g);
    }
    let dbg = PlannerConfig {
        cache: false,
        debug_reference: true,
        debug_tol: 1e-10,
        ..Default::default()
    };
    let ex = planner::expectation(&c, &[2, 5, 7], &dbg).unwrap();
    assert!(ex.reference.is_some());

    // a non-Gaussian circuit of the same size is planned elsewhere
    let mut c = line_hops(40);
    c.h(7).t(7).cnot(7, 8);
    let p = planner::plan(&c, &PlanRequest::Expectation(vec![3]), &cfg).unwrap();
    assert_ne!(p.engine, Engine::Gaussian);
}

// ---------------------------------------------------------------------------
// SU(2) hadron dynamics (quantum advantage tracker, research/data/su2-hadron)

fn su2(name: &str) -> Circuit {
    let path = format!(
        "{}/tests/data/su2-hadron/x_100_{name}.qasm",
        env!("CARGO_MANIFEST_DIR")
    );
    let src = std::fs::read_to_string(&path).unwrap();
    qsim_lab::io::qasm::from_qasm(&src).unwrap()
}

/// Staggered occupation `Σ_r (-1)^r (n_{r,0} + n_{r,1})` along the two
/// chains the detector found, and the total occupation.
fn stag(c: &Circuit) -> (f64, f64, gaussian::GaussianReport) {
    let opts = GaussianOptions {
        interactions: InteractionPolicy::Drop,
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let (st, rep) = GaussianState::from_circuit(c, &opts).unwrap();
    eprintln!(
        "su2: {} blocks, detect {:.4} s, detect + evolve {:.4} s",
        rep.blocks,
        rep.secs,
        t0.elapsed().as_secs_f64()
    );
    let mut site = vec![0usize; c.num_qubits]; // wire -> position on its chain
    for p in &rep.paths {
        for (r, &w) in p.iter().enumerate() {
            site[w] = r;
        }
    }
    let (mut s, mut tot) = (0.0, 0.0);
    for q in 0..c.num_qubits {
        let w = rep.order[st.mode_of_qubit(q)];
        let occ = 0.5 * (1.0 - st.expectation_z(q).unwrap());
        tot += occ;
        s += if site[w].is_multiple_of(2) { occ } else { -occ };
    }
    (s, tot, rep)
}

#[test]
fn su2_hadron_circuits_are_free_fermions_up_to_small_phases() {
    let (s_scv, n_scv, r) = stag(&su2("SCV"));
    assert!(r.free && !r.exact);
    assert_eq!(r.ordering, Some(Ordering::Paths));
    assert_eq!(r.paths.len(), 2);
    assert!(r.paths.iter().all(|p| p.len() == 60));
    assert_eq!(r.interactions.len(), 1200);
    assert!(r.interaction_max <= 0.0101, "{}", r.interaction_max);
    assert!(r.max_residual < 1e-12, "{}", r.max_residual);
    let (s_meson, n_meson, _) = stag(&su2("meson"));
    assert!((n_scv - 60.0).abs() < 1e-9 && (n_meson - 60.0).abs() < 1e-9);
    // research/data/su2-hadron/README.md: free-fermion values at step 20
    assert!((s_scv - (-2.641078)).abs() < 1e-6, "stag_SCV {s_scv}");
    let nf = s_meson - s_scv;
    eprintln!(
        "su2: stag_SCV = {s_scv:.9}, stag_meson = {s_meson:.9}, n_f = {nf:.9}, N = {n_scv:.9}"
    );
    assert!((nf - 0.118038).abs() < 1e-6, "n_f {nf}");
}
