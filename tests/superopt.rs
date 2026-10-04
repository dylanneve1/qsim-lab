//! Differential tests of the superoptimised windowed oracle
//! (`Oracle::WindowedOpt`, `src/shor_superopt.rs`) against the permutation
//! oracle and the unoptimised windowed oracle, through the sliced engine and
//! (where the state fits in u64 keys) gate by gate on the sparse state.
use qsim_lab::algorithms::gcd;
use qsim_lab::shor::{self, Backend, Instance, Oracle};
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
fn windowed_opt_distribution_matches_permutation() {
    for (n, count) in [(15u64, 4), (21, 3), (35, 1)] {
        for w in [1usize, 2, 3, 4] {
            for a in bases(n, count) {
                let perm = Instance::new(n, a, Oracle::Permutation);
                let opt = Instance::new(n, a, Oracle::WindowedOpt(w));
                let d_perm = dist(&perm, shor::sparse_initial(&perm));
                let d_sl = dist(&opt, shor::sliced::SlicedState::<f64>::new(&opt));
                let d = max_diff(&d_perm, &d_sl);
                assert!(d < 1e-12, "N={n} w={w} a={a}: sliced opt vs perm {d:e}");
                if n <= 21 && opt.qubits() <= 64 {
                    let d_gbg = dist(&opt, shor::sparse_initial(&opt));
                    let d = max_diff(&d_perm, &d_gbg);
                    assert!(
                        d < 1e-12,
                        "N={n} w={w} a={a}: gate-by-gate opt vs perm {d:e}"
                    );
                }
            }
        }
    }
}

#[test]
fn windowed_opt_measures_the_same_bits() {
    for (n, count) in [(15u64, 3), (143, 2), (1003, 1), (1_005_973, 1)] {
        for a in bases(n, count) {
            let perm = Instance::new(n, a, Oracle::Permutation);
            let win = Instance::new(n, a, Oracle::Windowed(4));
            let opt = Instance::new(n, a, Oracle::WindowedOpt(4));
            for seed in 0..2 {
                let r = |inst: &Instance, b: Backend| {
                    shor::order_finding(inst, b, &mut StdRng::seed_from_u64(seed)).measured
                };
                let base = r(&perm, Backend::FusedSparse);
                assert_eq!(base, r(&win, Backend::SlicedF64), "windowed N={n} a={a}");
                assert_eq!(base, r(&opt, Backend::SlicedF64), "opt N={n} a={a}");
            }
        }
    }
}

#[test]
fn windowed_opt_beyond_64_qubits_matches_permutation() {
    let n = 4093u64 * 4099;
    let a = (2..1000u64)
        .map(|g| qsim_lab::algorithms::pow_mod(g, 4 * 683, n))
        .find(|&a| a > 1 && gcd(a, n) == 1)
        .unwrap();
    let perm = Instance::new(n, a, Oracle::Permutation);
    for w in [3usize, 4, 5] {
        let opt = Instance::new(n, a, Oracle::WindowedOpt(w));
        assert!(opt.qubits() > 64);
        for seed in 0..2 {
            let r = |inst: &Instance, b: Backend| {
                shor::order_finding(inst, b, &mut StdRng::seed_from_u64(seed)).measured
            };
            assert_eq!(
                r(&perm, Backend::FusedSparse),
                r(&opt, Backend::SlicedF64),
                "w={w}"
            );
        }
    }
}

/// Fewer gates and Toffolis per run than the unoptimised windowed oracle.
#[test]
fn windowed_opt_counts_fewer_gates() {
    let (n, a) = (1_005_973u64, 980_062u64);
    let win = Instance::new(n, a, Oracle::Windowed(4));
    let opt = Instance::new(n, a, Oracle::WindowedOpt(4));
    let r0 = shor::order_finding(&win, Backend::SlicedF64, &mut StdRng::seed_from_u64(1));
    let r1 = shor::order_finding(&opt, Backend::SlicedF64, &mut StdRng::seed_from_u64(1));
    assert_eq!(r0.measured, r1.measured);
    assert!(
        r1.total_gates * 10 < r0.total_gates * 7,
        "{} vs {}",
        r1.total_gates,
        r0.total_gates
    );
    assert!(r1.toffoli_gates * 10 < r0.toffoli_gates * 6);
}
