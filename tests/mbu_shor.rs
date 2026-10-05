//! Differential tests of the measurement-based oracles
//! (`Oracle::WindowedMbu`, `Oracle::WindowedMbuLookup`, `src/shor_mbu.rs`)
//! against the permutation oracle and the reversible windowed-opt oracle:
//! the sliced engine (sign word, asserted clean), the gate-by-gate sparse
//! state (genuine H + projection for every X-basis measurement, P = 1/2
//! asserted), measured integers up to 20 bits, and beyond 64 qubits.
use qsim_lab::algorithms::gcd;
use qsim_lab::shor::{self, Backend, Instance, Oracle};
use qsim_lab::shor_mbu::{self, MbuCounts, MbuLayout, MbuOp, MbuOpts, Outcomes};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn bases(n: u64, count: usize) -> Vec<u64> {
    (2..n).filter(|&a| gcd(a, n) == 1).take(count).collect()
}

fn dist<S: shor::OrderFindingState>(inst: &Instance, s: S) -> Vec<f64> {
    shor::semiclassical_distribution(inst, s, 1e-15)
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn mbu_distribution_matches_permutation() {
    for (n, count) in [(15u64, 4), (21, 3), (35, 1)] {
        for w in [1usize, 2, 3, 4] {
            for a in bases(n, count) {
                let perm = Instance::new(n, a, Oracle::Permutation);
                let d_perm = dist(&perm, shor::sparse_initial(&perm));
                for oracle in [Oracle::WindowedMbu(w), Oracle::WindowedMbuLookup(w)] {
                    let inst = Instance::new(n, a, oracle);
                    let d_sl = dist(&inst, shor::sliced::SlicedState::<f64>::new(&inst));
                    let d = max_diff(&d_perm, &d_sl);
                    assert!(d < 1e-12, "N={n} {oracle:?} a={a}: sliced vs perm {d:e}");
                    if n <= 21 && inst.qubits() <= 64 {
                        // every X-basis measurement as a real H + projection
                        let d_gbg = dist(&inst, shor::sparse_initial(&inst));
                        let d = max_diff(&d_perm, &d_gbg);
                        assert!(
                            d < 1e-12,
                            "N={n} {oracle:?} a={a}: gate-by-gate vs perm {d:e}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn mbu_measures_the_same_bits() {
    for (n, count) in [(15u64, 3), (143, 2), (1003, 1), (1_005_973, 1)] {
        for a in bases(n, count) {
            let perm = Instance::new(n, a, Oracle::Permutation);
            let opt = Instance::new(n, a, Oracle::WindowedOpt(4));
            let mbu = Instance::new(n, a, Oracle::WindowedMbu(4));
            let mbl = Instance::new(n, a, Oracle::WindowedMbuLookup(4));
            for seed in 0..2 {
                let r = |inst: &Instance, b: Backend| {
                    shor::order_finding(inst, b, &mut StdRng::seed_from_u64(seed))
                };
                let base = r(&perm, Backend::FusedSparse).measured;
                let ro = r(&opt, Backend::SlicedF64);
                let rm = r(&mbu, Backend::SlicedF64);
                let rl = r(&mbl, Backend::SlicedF64);
                assert_eq!(base, ro.measured, "opt N={n} a={a}");
                assert_eq!(base, rm.measured, "mbu N={n} a={a}");
                assert_eq!(base, rl.measured, "mbu-lookup N={n} a={a}");
                assert!(rm.measurements > 0 && rl.measurements > 0);
                assert_eq!(ro.measurements, 0);
                assert!(rm.toffoli_gates < ro.toffoli_gates);
                assert!(rl.toffoli_gates < ro.toffoli_gates);
                assert_eq!(rl.qubits, ro.qubits);
            }
        }
    }
}

#[test]
fn mbu_beyond_64_qubits_matches_permutation() {
    let n = 4093u64 * 4099;
    let a = (2..1000u64)
        .map(|g| qsim_lab::algorithms::pow_mod(g, 4 * 683, n))
        .find(|&a| a > 1 && gcd(a, n) == 1)
        .unwrap();
    let perm = Instance::new(n, a, Oracle::Permutation);
    for oracle in [
        Oracle::WindowedMbu(4),
        Oracle::WindowedMbuLookup(4),
        Oracle::WindowedMbu(5),
    ] {
        let inst = Instance::new(n, a, oracle);
        assert!(inst.qubits() > 64);
        for seed in 0..2 {
            let b = shor::order_finding(
                &perm,
                Backend::FusedSparse,
                &mut StdRng::seed_from_u64(seed),
            );
            let s =
                shor::order_finding(&inst, Backend::SlicedF64, &mut StdRng::seed_from_u64(seed));
            assert_eq!(b.measured, s.measured, "{oracle:?} seed={seed}");
        }
    }
}

/// Every forced outcome pattern (all 0: no fix-up ever; all 1: every
/// fix-up) and several random streams give the same exact block.
#[test]
fn mbu_block_is_outcome_independent() {
    for n_mod in [143u64, 1003, 4087] {
        let n = shor::work_bits(n_mod);
        for o in [MbuOpts::ALL, MbuOpts::LOOKUPS] {
            let lay = MbuLayout::new(n, 4, &o);
            let a = bases(n_mod, 1)[0];
            let want: Vec<u128> = (0..n_mod)
                .map(|x| 1 | (u128::from(x * a % n_mod) << 1))
                .collect();
            for (seed, mode) in [(0u64, 1u8), (0, 2), (1, 0), (2, 0), (3, 0)] {
                let ops =
                    shor_mbu::controlled_ua(&lay, a, n_mod, &o, &mut Outcomes::new(seed, mode));
                for x in 0..n_mod {
                    let (k, s) = shor_mbu::eval_on_key(&ops, 1 | (u128::from(x) << 1));
                    assert!(!s);
                    assert_eq!(k, want[x as usize], "N={n_mod} {o:?} mode={mode} x={x}");
                }
            }
        }
    }
}

/// T2(c) of `research/theory/theory-shor.md` on the measurement-based blocks: with
/// a two-branch input (control |+⟩, x a basis state) the state at **every**
/// op boundary (after each projection) is two distinct basis branches of
/// equal weight whose relative phase is ±1, so it is a stabilizer state
/// (ν = 0); the measurement phases are exactly the ±1 the theorem allows.
#[test]
fn t2_mbu_two_branch_boundaries_stay_stabilizer() {
    let mut boundaries = 0u64;
    let mut minus = 0u64;
    for n_mod in [15u64, 21, 143, 1003] {
        let n = shor::work_bits(n_mod);
        let a = bases(n_mod, 1)[0];
        let lay = MbuLayout::new(n, 4, &MbuOpts::ALL);
        let ops =
            shor_mbu::controlled_ua(&lay, a, n_mod, &MbuOpts::ALL, &mut Outcomes::new(n_mod, 0));
        for x in 0..n_mod.min(64) {
            let (mut u, mut v) = (u128::from(x) << 1, 1 | (u128::from(x) << 1));
            let (mut su, mut sv) = (false, false);
            for op in &ops {
                let (u2, s2) = shor_mbu::eval_on_key(std::slice::from_ref(op), u);
                let (v2, t2) = shor_mbu::eval_on_key(std::slice::from_ref(op), v);
                if let MbuOp::MeasX(q, _) = *op {
                    // measured qubit must not be the only difference
                    assert!(u ^ v != 1 << q, "X-measurement would merge the branches");
                }
                (u, v, su, sv) = (u2, v2, su ^ s2, sv ^ t2);
                assert_ne!(u, v);
                boundaries += 1;
                minus += u64::from(su != sv);
            }
            assert!(!su && !sv);
        }
    }
    eprintln!("T2(c) MBU: {boundaries} boundaries, relative phase −1 at {minus}");
    assert!(minus > 0, "the measurement phases were never exercised");
}

#[test]
fn mbu_counts_halve_toffolis() {
    // 20-bit instance of research/shor/superopt.md
    let n_mod = 1_005_973u64;
    let n = shor::work_bits(n_mod);
    let a = 2;
    let opt = qsim_lab::shor_superopt::controlled_ua(
        &qsim_lab::shor_window::WindowLayout::new(n, 4),
        a,
        n_mod,
        &qsim_lab::shor_superopt::Opts::ALL,
    );
    let (_, opt_tof) = qsim_lab::shor_ripple::gate_counts(&opt);
    let mut oc = Outcomes::new(7, 0);
    let c_all = MbuCounts::of(&shor_mbu::controlled_ua(
        &MbuLayout::new(n, 4, &MbuOpts::ALL),
        a,
        n_mod,
        &MbuOpts::ALL,
        &mut oc,
    ));
    let c_lk = MbuCounts::of(&shor_mbu::controlled_ua(
        &MbuLayout::new(n, 4, &MbuOpts::LOOKUPS),
        a,
        n_mod,
        &MbuOpts::LOOKUPS,
        &mut oc,
    ));
    eprintln!("opt CCX {opt_tof}; mbu {c_all:?}; mbu-lookup {c_lk:?}");
    assert!(c_all.toffoli * 100 < opt_tof * 55);
    assert!(c_lk.toffoli < opt_tof);
}
