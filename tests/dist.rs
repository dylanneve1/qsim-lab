//! Differential tests for the two-node distributed state vector
//! (`qsim_lab::engines::dist`) against the single-node reference state vector and the
//! independent audit reference (`tests/audit_common`).

mod audit_common;

use audit_common::{random_gate, RefSv};
use num_complex::{Complex, Complex64};
use proptest::prelude::*;
use qsim_lab::algorithms;
use qsim_lab::circuit::Circuit;
use qsim_lab::engines::dist::{
    chan_pair, fingerprint, handshake, needs_local, plan_circuit, DistConfig, DistPlan, DistState,
    DistStep, Link, PlanOptions, TcpLink,
};
use qsim_lab::engines::statevector::{Real, StateVector};
use qsim_lab::gate::Gate;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::net::TcpListener;
use std::time::Duration;

fn max_diff<T: Real>(a: &[Complex<T>], b: &[Complex<T>]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = *x - *y;
            (d.re.to_f64().powi(2) + d.im.to_f64().powi(2)).sqrt()
        })
        .fold(0.0, f64::max)
}

/// Random circuit over every gate type.
fn random_all_gates(n: usize, len: usize, rng: &mut StdRng) -> Circuit {
    let mut c = Circuit::new(n);
    for _ in 0..len {
        c.gate(random_any_gate(n, rng));
    }
    c
}

fn random_any_gate(n: usize, rng: &mut StdRng) -> Gate {
    let q = rng.random_range(0..n);
    let th = rng.random_range(-3.2..3.2);
    let other = |rng: &mut StdRng, not: &[usize]| loop {
        let x = rng.random_range(0..n);
        if !not.contains(&x) {
            break x;
        }
    };
    match rng.random_range(0..23) {
        0 => Gate::I(q),
        1 => Gate::H(q),
        2 => Gate::X(q),
        3 => Gate::Y(q),
        4 => Gate::Z(q),
        5 => Gate::S(q),
        6 => Gate::Sdg(q),
        7 => Gate::T(q),
        8 => Gate::Tdg(q),
        9 => Gate::Sx(q),
        10 => Gate::Sxdg(q),
        11 => Gate::Rx(q, th),
        12 => Gate::Ry(q, th),
        13 => Gate::Rz(q, th),
        14 => Gate::Phase(q, th),
        15 => Gate::U(q, th, th * 0.37 + 0.1, -th * 0.61),
        16 => Gate::Cnot(q, other(rng, &[q])),
        17 => Gate::Cz(q, other(rng, &[q])),
        18 => Gate::Swap(q, other(rng, &[q])),
        19 => Gate::ISwap(q, other(rng, &[q])),
        20 => Gate::ISwapdg(q, other(rng, &[q])),
        21 => Gate::CPhase(q, other(rng, &[q]), th),
        _ => {
            let b = other(rng, &[q]);
            let t = other(rng, &[q, b]);
            Gate::Ccx(q, b, t)
        }
    }
}

/// Runs `plan` on two nodes connected by `links`, gathers on node 0.
fn run_pair<T: Real, L: Link + 'static>(
    plan: &DistPlan,
    owner: &[u8],
    x: usize,
    links: (L, L),
    cfg: &DistConfig,
) -> Vec<Complex<T>> {
    let (mut l0, mut l1) = links;
    let fp = fingerprint(plan, owner, std::mem::size_of::<Complex<T>>());
    let (p1, o1, c1) = (plan.clone(), owner.to_vec(), cfg.clone());
    let h = std::thread::spawn(move || {
        handshake(&mut l1, 1, fp).unwrap();
        let mut st = DistState::<T>::new_basis(p1.n, p1.local_bits, 1, o1, &p1.initial_v2p, x, c1);
        st.run(&p1, &mut l1).unwrap();
        let nrm = st.norm_sqr(&mut l1).unwrap();
        assert!(st.gather(&mut l1).unwrap().is_none());
        nrm
    });
    handshake(&mut l0, 0, fp).unwrap();
    let mut st = DistState::<T>::new_basis(
        plan.n,
        plan.local_bits,
        0,
        owner.to_vec(),
        &plan.initial_v2p,
        x,
        cfg.clone(),
    );
    st.run(plan, &mut l0).unwrap();
    assert_eq!(st.layout(), &plan.final_v2p[..]);
    let nrm = st.norm_sqr(&mut l0).unwrap();
    let out = st.gather(&mut l0).unwrap().unwrap();
    let nrm1 = h.join().unwrap();
    assert!((nrm - nrm1).abs() < 1e-12);
    let direct: f64 = out.iter().map(|a| a.norm_sqr().to_f64()).sum();
    assert!(
        (nrm - direct).abs() < 1e-5,
        "norm {nrm} vs gathered {direct}"
    );
    out
}

fn small_cfg() -> DistConfig {
    DistConfig {
        // Small messages so exchanges are split into many blocks.
        msg_amps: 8,
        ..DistConfig::default()
    }
}

fn reference<T: Real>(c: &Circuit) -> Vec<Complex<T>> {
    let mut sv = StateVector::<T>::new(c.num_qubits);
    sv.apply_circuit(c).unwrap();
    sv.amplitudes().to_vec()
}

fn owners_for(g: usize, rng: &mut StdRng) -> Vec<Vec<u8>> {
    let r = 1usize << g;
    let mut v = vec![
        (0..r).map(|i| (i == r - 1) as u8).collect::<Vec<u8>>(), // one rank on node 1
        (0..r).map(|i| (i >= r / 2) as u8).collect(),            // symmetric split
        (0..r).map(|i| (i == 0) as u8).collect(),                // node 1 owns rank 0
    ];
    if r >= 4 {
        v.push((0..r).map(|i| (i.count_ones() & 1) as u8).collect()); // parity split
        v.push((0..r).map(|_| rng.random_range(0..2u8)).collect());
    }
    v.push(vec![0; r]); // everything on node 0 (no cross pairs)
    v
}

fn all_opts() -> Vec<PlanOptions> {
    let mut v = Vec::new();
    for free in [false, true] {
        for fold in [false, true] {
            for restore in [false, true] {
                v.push(PlanOptions {
                    free_initial_layout: free,
                    fold_swaps: fold,
                    restore_order: restore,
                });
            }
        }
    }
    v
}

fn check_circuit<T: Real>(c: &Circuit, l: usize, tol: f64, rng: &mut StdRng) {
    let want = reference::<T>(c);
    for owner in owners_for(c.num_qubits - l, rng) {
        for opts in all_opts() {
            let plan = plan_circuit(c, l, &opts).unwrap();
            let got = run_pair::<T, _>(&plan, &owner, 0, chan_pair(), &small_cfg());
            let d = max_diff(&got, &want);
            assert!(
                d <= tol,
                "n={} l={l} owner={owner:?} opts={opts:?}: max |dAmp| {d:e} (swaps {})",
                c.num_qubits,
                plan.swaps
            );
            if opts.restore_order {
                assert!(plan.final_v2p.iter().enumerate().all(|(v, &p)| v == p));
            }
        }
    }
}

#[test]
fn random_all_gate_types_f64() {
    let mut rng = StdRng::seed_from_u64(0xD157_0001);
    for n in [6usize, 7, 9, 11, 12] {
        for l in [n - 1, n - 2, n - 3, n - 4] {
            if l < 3 {
                continue;
            }
            for len in [20, 80] {
                let c = random_all_gates(n, len, &mut rng);
                check_circuit::<f64>(&c, l, 1e-12, &mut rng);
            }
        }
    }
}

#[test]
fn random_all_gate_types_f32() {
    let mut rng = StdRng::seed_from_u64(0xD157_0002);
    for n in [7usize, 10] {
        for l in [n - 1, n - 2, n - 3] {
            let c = random_all_gates(n, 50, &mut rng);
            check_circuit::<f32>(&c, l, 1e-5, &mut rng);
        }
    }
}

/// Builds a gate from a list of qubits.
type Proto = Box<dyn Fn(&[usize]) -> Gate>;

/// Every gate type with every qubit-role assignment over two global and two
/// local qubits (identity layout, so the globals really are global).
#[test]
fn every_gate_on_global_qubits() {
    let n = 7;
    let l = 5; // globals: 5, 6
    let pick = [0usize, 3, 5, 6];
    let mut rng = StdRng::seed_from_u64(0xD157_0003);
    let protos: Vec<Proto> = vec![
        Box::new(|q| Gate::I(q[0])),
        Box::new(|q| Gate::H(q[0])),
        Box::new(|q| Gate::X(q[0])),
        Box::new(|q| Gate::Y(q[0])),
        Box::new(|q| Gate::Z(q[0])),
        Box::new(|q| Gate::S(q[0])),
        Box::new(|q| Gate::Sdg(q[0])),
        Box::new(|q| Gate::T(q[0])),
        Box::new(|q| Gate::Tdg(q[0])),
        Box::new(|q| Gate::Sx(q[0])),
        Box::new(|q| Gate::Sxdg(q[0])),
        Box::new(|q| Gate::Rx(q[0], 0.7)),
        Box::new(|q| Gate::Ry(q[0], -1.1)),
        Box::new(|q| Gate::Rz(q[0], 2.3)),
        Box::new(|q| Gate::Phase(q[0], 0.9)),
        Box::new(|q| Gate::U(q[0], 0.4, 1.3, -0.8)),
        Box::new(|q| Gate::Cnot(q[0], q[1])),
        Box::new(|q| Gate::Cz(q[0], q[1])),
        Box::new(|q| Gate::Swap(q[0], q[1])),
        Box::new(|q| Gate::ISwap(q[0], q[1])),
        Box::new(|q| Gate::ISwapdg(q[0], q[1])),
        Box::new(|q| Gate::CPhase(q[0], q[1], 1.234)),
        Box::new(|q| Gate::Ccx(q[0], q[1], q[2])),
    ];
    let mut prep = Circuit::new(n);
    for q in 0..n {
        prep.gate(Gate::U(
            q,
            rng.random_range(0.1..3.0),
            rng.random_range(-3.0..3.0),
            rng.random_range(-3.0..3.0),
        ));
    }
    for q in 0..n - 1 {
        prep.gate(Gate::Cnot(q, q + 1));
    }
    let mut cases = 0;
    for proto in &protos {
        let arity = proto(&[0, 1, 2]).arity();
        let mut assign = vec![0usize; arity];
        let total = pick.len().pow(arity as u32);
        for code in 0..total {
            let mut c = code;
            for a in assign.iter_mut() {
                *a = pick[c % pick.len()];
                c /= pick.len();
            }
            let mut uniq = assign.clone();
            uniq.sort_unstable();
            uniq.dedup();
            if uniq.len() != arity {
                continue;
            }
            let mut circ = prep.clone();
            circ.gate(proto(&assign));
            circ.gate(Gate::H(0));
            let want = reference::<f64>(&circ);
            for owner in [vec![0u8, 0, 0, 1], vec![0, 1, 1, 0]] {
                for fold in [false, true] {
                    let opts = PlanOptions {
                        free_initial_layout: false,
                        fold_swaps: fold,
                        restore_order: true,
                    };
                    let plan = plan_circuit(&circ, l, &opts).unwrap();
                    let got = run_pair::<f64, _>(&plan, &owner, 0, chan_pair(), &small_cfg());
                    let d = max_diff(&got, &want);
                    assert!(
                        d < 1e-12,
                        "{:?} owner={owner:?} fold={fold}: {d:e}",
                        proto(&assign)
                    );
                    cases += 1;
                }
            }
        }
    }
    assert!(cases > 400, "only {cases} cases");
}

/// Plans keep global qubits out of non-diagonal roles.
#[test]
fn plan_respects_roles() {
    let mut rng = StdRng::seed_from_u64(0xD157_0004);
    for _ in 0..20 {
        let n = rng.random_range(6..12);
        let l = rng.random_range(3..n);
        let c = random_all_gates(n, 80, &mut rng);
        for opts in all_opts() {
            let plan = plan_circuit(&c, l, &opts).unwrap();
            for st in &plan.steps {
                match st {
                    DistStep::Local(gs) => {
                        for g in gs {
                            assert!(needs_local(g).iter().all(|&q| q < l), "{g:?} l={l}");
                            if let Gate::Swap(a, b) = g {
                                assert!(*a < l && *b < l);
                            }
                        }
                    }
                    DistStep::Swap { local, global } => {
                        assert!(*local < l && *global >= l && *global < n)
                    }
                    DistStep::Relabel(pairs) => {
                        assert!(pairs.iter().all(|&(a, b)| a < l && b < l && a != b))
                    }
                    DistStep::Rename { a, b } => assert!(a != b && *a < n && *b < n),
                }
            }
        }
    }
}

/// QFT on basis states vs the analytic result; the free layout + diagonal
/// specialisation needs exactly 2 swaps for 2 global qubits.
#[test]
fn qft_basis_states_analytic() {
    let n = 12;
    let l = 10;
    let c = algorithms::qft(n);
    let plan = plan_circuit(&c, l, &PlanOptions::default()).unwrap();
    assert_eq!(plan.swaps, 2, "QFT should need one swap per global qubit");
    let mut rng = StdRng::seed_from_u64(0xD157_0005);
    for _ in 0..4 {
        let x = rng.random_range(0..1usize << n);
        let got = run_pair::<f64, _>(&plan, &[0, 0, 0, 1], x, chan_pair(), &small_cfg());
        let nn = (1usize << n) as f64;
        for (k, a) in got.iter().enumerate() {
            let ang = 2.0 * std::f64::consts::PI * ((x * k) % (1 << n)) as f64 / nn;
            let want = Complex64::from_polar(nn.powf(-0.5), ang);
            assert!((a - want).norm() < 1e-12, "x={x} k={k}: {a} vs {want}");
        }
    }
}

#[test]
fn tcp_loopback_matches_reference() {
    let mut rng = StdRng::seed_from_u64(0xD157_0006);
    let n = 12;
    let c = random_all_gates(n, 120, &mut rng);
    let want = reference::<f32>(&c);
    for (l, owner) in [(11usize, vec![0u8, 1]), (9, vec![0, 0, 0, 0, 0, 0, 0, 1])] {
        let plan = plan_circuit(&c, l, &PlanOptions::default()).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let h = std::thread::spawn(move || TcpLink::accept(&listener).unwrap());
        let a = TcpLink::connect(addr, Duration::from_secs(5)).unwrap();
        let b = h.join().unwrap();
        let cfg = DistConfig {
            msg_amps: 64,
            ..DistConfig::default()
        };
        let got = run_pair::<f32, _>(&plan, &owner, 0, (a, b), &cfg);
        let d = max_diff(&got, &want);
        assert!(d < 1e-5, "l={l}: {d:e}");
    }
}

#[test]
fn handshake_rejects_mismatch() {
    let (mut a, mut b) = chan_pair();
    let h = std::thread::spawn(move || handshake(&mut b, 1, 2));
    assert!(handshake(&mut a, 0, 1).is_err());
    assert!(h.join().unwrap().is_err());
    let (mut a, mut b) = chan_pair();
    let h = std::thread::spawn(move || handshake(&mut b, 0, 7));
    assert!(handshake(&mut a, 0, 7).is_err());
    assert!(h.join().unwrap().is_err());
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 24, .. ProptestConfig::default() })]

    /// Audit-harness gates vs the independent reference, random shapes.
    #[test]
    fn prop_matches_audit_reference(seed in any::<u64>(), n in 5usize..11, g in 1usize..4, depth in 10usize..60) {
        let mut rng = StdRng::seed_from_u64(seed);
        let l = n.saturating_sub(g).max(3);
        let mut c = Circuit::new(n);
        let mut r = RefSv::new(n);
        for _ in 0..depth {
            let gate = random_gate(&mut rng, n, false, false);
            c.gate(gate);
            r.apply(&gate);
        }
        let gl = n - l;
        let owners = owners_for(gl, &mut rng);
        let owner = &owners[rng.random_range(0..owners.len())];
        let opts = PlanOptions {
            free_initial_layout: rng.random(),
            fold_swaps: rng.random(),
            restore_order: rng.random(),
        };
        let plan = plan_circuit(&c, l, &opts).unwrap();
        let got = run_pair::<f64, _>(&plan, owner, 0, chan_pair(), &small_cfg());
        let d = got.iter().zip(&r.a).map(|(x, y)| (x - y).norm()).fold(0.0, f64::max);
        prop_assert!(d < 1e-10, "n={} l={} owner={:?} {:?}: {:e}", n, l, owner, opts, d);
    }
}
