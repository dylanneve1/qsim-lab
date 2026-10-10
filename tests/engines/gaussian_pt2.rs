//! Second-order perturbation theory in weak interaction phases on the
//! Gaussian engine (`engines::gaussian::pt2`, docs/ENGINE_GAUSSIAN.md §6):
//! random number-conserving circuits with weak `CPhase` interactions
//! against the exact state vector (`a0` at `λ = 0`, `a1`, `a2` against
//! finite differences in `λ`), the `λ³` scaling of the truncation error,
//! the refusal of circuits that do not conserve the particle number, and
//! the SU(2) hadron-dynamics tracker values of research/data/su2-hadron.

use qsim_lab::circuit::{Circuit, SimError};
use qsim_lab::engines::gaussian::pt2::{self, OneBody, Pt2Options};
use qsim_lab::engines::gaussian::{compile, DetectOptions};
use qsim_lab::engines::statevector::StateVectorF64;
use qsim_lab::gate::Gate;
use qsim_lab::Op;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

/// Basis change taking Z to `axis` (0 = X, 1 = Y), in time order, and back.
fn to_axis(axis: u8, q: usize) -> (Vec<Gate>, Vec<Gate>) {
    match axis {
        0 => (vec![Gate::H(q)], vec![Gate::H(q)]),
        _ => (vec![Gate::Sdg(q), Gate::H(q)], vec![Gate::H(q), Gate::S(q)]),
    }
}

/// `exp(-i t/2 P_a P_b)` with `P ∈ {X, Y}`, written with CNOTs.
fn pauli_pair(a: usize, b: usize, pa: u8, pb: u8, t: f64, flip: bool) -> Vec<Gate> {
    let (ia, oa) = to_axis(pa, a);
    let (ib, ob) = to_axis(pb, b);
    let mut v: Vec<Gate> = ia.into_iter().chain(ib).collect();
    let (c, tg) = if flip { (b, a) } else { (a, b) };
    v.extend([Gate::Cnot(c, tg), Gate::Rz(tg, t), Gate::Cnot(c, tg)]);
    v.extend(oa.into_iter().chain(ob));
    v
}

/// The hopping matchgate `exp(iθ (XX + YY)/2)` on `(a, b)`.
fn hop(a: usize, b: usize, theta: f64, flip: bool) -> Vec<Gate> {
    let mut v = pauli_pair(a, b, 0, 0, -theta, flip);
    v.extend(pauli_pair(a, b, 1, 1, -theta, !flip));
    v
}

#[derive(Clone, Copy, Debug)]
enum Item {
    G(Gate),
    /// `exp(i λ g n_a n_b)`
    Int(usize, usize, f64),
}

/// A random number-conserving circuit on the line `0..n`: hopping gates
/// (plain, with a complex phase, iSWAP), one-site phases, and `n_int`
/// weak interaction phases on random pairs. The initial basis state is a
/// random half-filling, prepared by `X` gates.
fn random_circuit(
    n: usize,
    depth: usize,
    n_int: usize,
    gscale: f64,
    rng: &mut StdRng,
) -> Vec<Item> {
    let mut v = Vec::new();
    let mut sites: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        sites.swap(i, rng.random_range(0..=i));
    }
    for &q in &sites[..n / 2] {
        v.push(Item::G(Gate::X(q)));
    }
    // interaction positions among the free steps
    let mut at: Vec<usize> = (0..n_int).map(|_| rng.random_range(0..depth)).collect();
    at.sort_unstable();
    let mut next = 0;
    for step in 0..depth {
        while next < at.len() && at[next] == step {
            let a = rng.random_range(0..n);
            let mut b = rng.random_range(0..n - 1);
            if b >= a {
                b += 1;
            }
            let g =
                gscale * rng.random_range(0.5..1.5) * if rng.random_bool(0.5) { 1.0 } else { -1.0 };
            v.push(Item::Int(a, b, g));
            next += 1;
        }
        let a = rng.random_range(0..n - 1);
        let (a, b) = if rng.random_bool(0.5) {
            (a, a + 1)
        } else {
            (a + 1, a)
        };
        let t = rng.random_range(-PI..PI);
        match rng.random_range(0..8) {
            0 => v.push(Item::G(Gate::Rz(a, t))),
            1 => v.push(Item::G(Gate::Phase(b, t))),
            2 => v.push(Item::G(if rng.random_bool(0.5) {
                Gate::ISwap(a, b)
            } else {
                Gate::ISwapdg(a, b)
            })),
            3 | 4 => {
                // complex hopping: conjugated by a one-site phase
                let f = rng.random_range(-PI..PI);
                v.push(Item::G(Gate::Rz(a, f)));
                v.extend(hop(a, b, t, rng.random_bool(0.5)).into_iter().map(Item::G));
                v.push(Item::G(Gate::Rz(a, -f)));
            }
            _ => v.extend(hop(a, b, t, rng.random_bool(0.5)).into_iter().map(Item::G)),
        }
    }
    v
}

fn build(items: &[Item], n: usize, lambda: f64) -> Circuit {
    let mut c = Circuit::new(n);
    for it in items {
        c.ops.push(Op::Gate(match *it {
            Item::G(g) => g,
            Item::Int(a, b, g) => Gate::CPhase(a, b, lambda * g),
        }));
    }
    c
}

/// Exact `⟨O⟩` of every observable (qubit indices) on the state vector.
fn exact(c: &Circuit, obs: &[OneBody]) -> Vec<f64> {
    let mut sv = StateVectorF64::new(c.num_qubits);
    for g in c.gates() {
        sv.apply_gate(g).unwrap();
    }
    let p: Vec<f64> = sv.amplitudes().iter().map(|a| a.norm_sqr()).collect();
    let n_q: Vec<f64> = (0..c.num_qubits)
        .map(|q| {
            p.iter()
                .enumerate()
                .filter(|(x, _)| (x >> q) & 1 == 1)
                .map(|(_, v)| v)
                .sum()
        })
        .collect();
    obs.iter()
        .map(|o| o.constant + o.density.iter().map(|&(q, x)| x * n_q[q]).sum::<f64>())
        .collect()
}

fn random_observables(n: usize, rng: &mut StdRng) -> Vec<OneBody> {
    let mut obs: Vec<OneBody> = (0..n).map(|q| OneBody::new().n(q, 1.0)).collect();
    obs.push(OneBody::new().z(rng.random_range(0..n), 1.0));
    let mut mix = OneBody::new();
    for q in 0..n {
        if rng.random_bool(0.6) {
            mix = if rng.random_bool(0.5) {
                mix.n(q, rng.random_range(-1.0..1.0))
            } else {
                mix.z(q, rng.random_range(-1.0..1.0))
            };
        }
    }
    obs.push(mix.n(0, 0.3));
    obs
}

fn norm(o: &OneBody) -> f64 {
    o.constant.abs() + o.density.iter().map(|x| x.1.abs()).sum::<f64>()
}

#[test]
fn coefficients_match_finite_differences_of_the_state_vector() {
    let mut rng = StdRng::seed_from_u64(0x9792);
    let mut done = 0;
    let mut tries = 0;
    let mut worst: f64 = 0.0;
    while done < 16 {
        tries += 1;
        assert!(tries < 200, "too many circuits rejected by the detector");
        let n = rng.random_range(3..=8);
        let n_int = rng.random_range(5..=20);
        let depth = rng.random_range(10..50);
        let items = random_circuit(n, depth, n_int, 0.02, &mut rng);
        let obs = random_observables(n, &mut rng);
        let run = match pt2::pt2_circuit(&build(&items, n, 1.0), &obs, &Pt2Options::default()) {
            Ok(r) => r,
            // a CPhase fused with a hop on the same pair: not free
            Err(SimError::NotSupported { .. }) => continue,
            Err(e) => panic!("{e}"),
        };
        assert!(!run.report.interactions.is_empty());
        let gsum: f64 = run.report.interactions.iter().map(|i| i.g.abs()).sum();
        // exact ⟨O⟩(λ) on a 5-point stencil
        let h = 0.02;
        let f: Vec<Vec<f64>> = (-2..=2)
            .map(|s| exact(&build(&items, n, s as f64 * h), &obs))
            .collect();
        for (i, o) in obs.iter().enumerate() {
            let r = run.results[i];
            let fd1 = (-f[4][i] + 8.0 * f[3][i] - 8.0 * f[1][i] + f[0][i]) / (12.0 * h);
            let fd2 = (-f[4][i] + 16.0 * f[3][i] - 30.0 * f[2][i] + 16.0 * f[1][i] - f[0][i])
                / (12.0 * h * h)
                / 2.0;
            let scale = norm(o).max(1e-300);
            let e0 = (r.a0 - f[2][i]).abs() / scale;
            let e1 = (r.a1 - fd1).abs() / (scale * gsum);
            let e2 = (r.a2 - fd2).abs() / (scale * gsum * gsum);
            worst = worst.max(e0).max(e1).max(e2);
            assert!(
                e0 < 1e-10 && e1 < 1e-6 && e2 < 1e-6,
                "circuit {done} (n={n}, {} phases) obs {i}: a0 {} vs {}, a1 {} vs {fd1}, a2 {} vs {fd2}",
                run.report.interactions.len(),
                r.a0,
                f[2][i],
                r.a1,
                r.a2
            );
            assert!(r.est_error >= 0.0 && r.light_cone <= run.report.interactions.len());
        }
        done += 1;
    }
    eprintln!("pt2: worst relative deviation {worst:.2e} ({tries} circuits drawn)");
}

#[test]
fn the_truncation_error_scales_as_lambda_cubed() {
    let mut rng = StdRng::seed_from_u64(0x9793);
    let n = 8;
    let (items, run) = loop {
        let items = random_circuit(n, 40, 15, 0.3, &mut rng);
        let obs = vec![
            OneBody::new().n(3, 1.0),
            OneBody::new().z(5, 1.0).n(1, -0.5),
        ];
        if let Ok(run) = pt2::pt2_circuit(&build(&items, n, 1.0), &obs, &Pt2Options::default()) {
            break (items, (run, obs));
        }
    };
    let (run, obs) = run;
    for (i, r) in run.results.iter().enumerate() {
        let mut errs = Vec::new();
        for s in 0..5 {
            let lam = 0.5f64.powi(s);
            let want = exact(&build(&items, n, lam), &obs)[i];
            let got = r.a0 + r.a1 * lam + r.a2 * lam * lam;
            errs.push((lam, (got - want).abs()));
        }
        eprintln!("pt2 scaling obs {i}: {errs:?}");
        for w in errs.windows(2) {
            let ((l0, e0), (l1, e1)) = (w[0], w[1]);
            assert!(e0 > 1e-13, "error {e0} at λ = {l0} lost in round-off");
            // λ³ (or faster): halving λ divides the error by >= ~8
            assert!(
                e1 <= e0 / 6.5,
                "obs {i}: error {e0} at λ = {l0}, {e1} at λ = {l1}"
            );
        }
    }
}

#[test]
fn circuits_that_do_not_conserve_the_particle_number_are_refused() {
    let mut c = Circuit::new(4);
    c.x(0).x(2);
    for g in hop(0, 1, 0.4, false)
        .into_iter()
        .chain(hop(1, 2, 0.3, true))
    {
        c.ops.push(Op::Gate(g));
    }
    c.cphase(0, 2, 0.05);
    // an XY pairing term: Gaussian, but it creates particle pairs
    for g in pauli_pair(2, 3, 0, 1, 0.3, false) {
        c.ops.push(Op::Gate(g));
    }
    let obs = [OneBody::new().n(1, 1.0)];
    match pt2::pt2_circuit(&c, &obs, &Pt2Options::default()) {
        Err(SimError::NotSupported { what }) => {
            assert!(what.contains("number-conserving"), "{what}")
        }
        other => panic!("{:?}", other.map(|r| r.results)),
    }
    // an X in the middle of the circuit is not part of the initial state
    let mut c = Circuit::new(3);
    c.x(0);
    for g in hop(0, 1, 0.4, false) {
        c.ops.push(Op::Gate(g));
    }
    c.x(1).cphase(0, 2, 0.05);
    assert!(matches!(
        pt2::pt2_circuit(&c, &obs, &Pt2Options::default()),
        Err(SimError::NotSupported { .. })
    ));
    // a non-Gaussian circuit
    let mut c = Circuit::new(3);
    c.h(0).cnot(0, 1).cphase(0, 2, 0.05);
    assert!(matches!(
        pt2::pt2_circuit(&c, &obs, &Pt2Options::default()),
        Err(SimError::NotSupported { .. })
    ));
}

#[test]
fn a_free_circuit_has_no_corrections_and_the_initial_layer_is_split_off() {
    let mut c = Circuit::new(4);
    c.x(1).x(3);
    for g in hop(1, 2, 0.7, false) {
        c.ops.push(Op::Gate(g));
    }
    let (rest, occ) = pt2::split_initial_x(&c);
    assert_eq!(occ, vec![false, true, false, true]);
    assert_eq!(rest.ops.len(), c.ops.len() - 2);
    let obs = [OneBody::new().n(2, 1.0), OneBody::new().z(0, 2.0)];
    let run = pt2::pt2_circuit(&c, &obs, &Pt2Options::default()).unwrap();
    let want = exact(&c, &obs);
    for (r, w) in run.results.iter().zip(&want) {
        assert!((r.a0 - w).abs() < 1e-12 && r.a1 == 0.0 && r.a2 == 0.0 && r.total == r.a0);
    }
}

// ---------------------------------------------------------------------------
// SU(2) hadron dynamics (research/data/su2-hadron, wick_pt2.py)

fn su2(name: &str) -> Circuit {
    let path = format!(
        "{}/tests/data/su2-hadron/x_100_{name}.qasm",
        env!("CARGO_MANIFEST_DIR")
    );
    let src = std::fs::read_to_string(&path).unwrap();
    qsim_lab::io::qasm::from_qasm(&src).unwrap()
}

/// PT2 of the staggered occupation `Σ_r (-1)^r (n_{r,0} + n_{r,1})` along
/// the two chains the detector finds, at the end of the circuit.
fn su2_stag(name: &str) -> pt2::Pt2Result {
    let c = su2(name);
    let t0 = std::time::Instant::now();
    let (rest, init) = pt2::split_initial_x(&c);
    let prog = compile(&rest, &DetectOptions::default());
    let rep = &prog.report;
    assert!(rep.free && rep.number_conserving, "{name}: {rep:?}");
    assert_eq!(rep.paths.len(), 2);
    assert_eq!(rep.interactions.len(), 1200);
    let mut site = vec![0usize; c.num_qubits];
    for p in &rep.paths {
        for (r, &w) in p.iter().enumerate() {
            site[w] = r;
        }
    }
    let occ: Vec<bool> = rep.order.iter().map(|&w| init[w]).collect();
    let mut stag = OneBody::new();
    for (k, &w) in rep.order.iter().enumerate() {
        stag = stag.n(k, if site[w].is_multiple_of(2) { 1.0 } else { -1.0 });
    }
    let r = pt2::pt2(&prog, &occ, &[stag], &Pt2Options::default()).unwrap()[0];
    eprintln!(
        "su2 {name}: a0 {:.9} a1 {:+.3e} a2 {:+.9} total {:.9} est_error {:.1e} light cone {} ({:.1} s)",
        r.a0,
        r.a1,
        r.a2,
        r.total,
        r.est_error,
        r.light_cone,
        t0.elapsed().as_secs_f64()
    );
    r
}

#[test]
#[ignore = "slow in debug builds (1200 interaction phases); run with --release --ignored"]
fn su2_hadron_pt2_reproduces_the_reference() {
    let scv = su2_stag("SCV");
    let meson = su2_stag("meson");
    // free values (tests/engines/gaussian.rs)
    assert!((scv.a0 - (-2.641078)).abs() < 1e-6, "{scv:?}");
    assert!((meson.a0 - scv.a0 - 0.118038).abs() < 1e-6);
    // research/data/su2-hadron/results_summary.json: PT2 at step 20
    let nf = meson.total - scv.total;
    eprintln!("su2: stag_SCV(20) = {:.9}, n_f(20) = {nf:.9}", scv.total);
    assert!(
        (scv.total - (-2.6238502)).abs() < 1e-6,
        "stag_SCV {}",
        scv.total
    );
    assert!((nf - 0.1168633).abs() < 1e-6, "n_f {nf}");
}
