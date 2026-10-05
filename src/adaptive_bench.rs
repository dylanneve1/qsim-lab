//! Benchmarks for [`crate::adaptive`] (`qsim adaptive ...`). Each call runs
//! one method on one circuit family and prints markdown rows, so that A/B
//! comparisons can be interleaved run by run through the swarm's bench lock.

use crate::adaptive::{self, AdaptiveOptions, CompressedState, Strategy};
use crate::bench::{clifford_t_family, skeleton_stabilizer};
use crate::circuit::Circuit;
use crate::pauli_path::{self, FrameOptions, PauliSum};
use crate::statevector::{StateVectorF64, MAX_STATE_BYTES};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::io::Write;
use std::time::Instant;

/// A circuit with a magic-rich *core* and a magic-sparse *wide* part:
///
/// 1. `t_core` rounds of (random Clifford block of `depth` layers on qubits
///    `0..core`, T/T† on a random core qubit). The x-span of these rotation
///    axes stays inside the core, so it saturates at `d = core` while the
///    Pauli-path term count keeps branching.
/// 2. `t_tail` rounds of (random Clifford block on all `n` qubits, T/T† on
///    a random qubit). Each of these generically adds one new direction, so
///    the active register ends at `core + t_tail` qubits, while the
///    Heisenberg sweep through them prunes every branch.
/// 3. A final random Clifford block on all `n` qubits.
pub fn two_phase(
    n: usize,
    core: usize,
    t_core: usize,
    t_tail: usize,
    depth: usize,
    seed: u64,
) -> Circuit {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut c = Circuit::new(n);
    let tee = |c: &mut Circuit, q: usize, rng: &mut StdRng| {
        if rng.random_bool(0.5) {
            c.t(q);
        } else {
            c.gate(crate::gate::Gate::Tdg(q));
        }
    };
    for _ in 0..t_core {
        let blk = Circuit::random_clifford(core, depth, &mut rng);
        let mut wide = Circuit::new(n);
        wide.append(&blk);
        c.append(&wide);
        let q = rng.random_range(0..core);
        tee(&mut c, q, &mut rng);
    }
    for _ in 0..t_tail {
        c.append(&Circuit::random_clifford(n, depth, &mut rng));
        let q = rng.random_range(0..n);
        tee(&mut c, q, &mut rng);
    }
    c.append(&Circuit::random_clifford(n, depth, &mut rng));
    c
}

fn secs(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}

fn flush() {
    let _ = std::io::stdout().flush();
}

/// Builds the circuit of a family by name.
pub fn family(name: &str, n: usize, t: usize, core: usize, t_tail: usize, depth: usize) -> Circuit {
    match name {
        // The README / research/performance/pauli.md family (seed 3, nested in t).
        "random" => clifford_t_family(n, depth, t, 3)(t),
        "two-phase" => two_phase(n, core, t, t_tail, depth, 5),
        // Cuccaro ripple-carry adder on `t` bits (n is ignored: 2t + 2).
        "adder" => crate::bench::cuccaro_adder(t),
        _ => panic!("unknown family {name:?} (random | two-phase | adder)"),
    }
}

/// `<O>` for one method; prints one markdown row.
/// Methods: `sv` (full state vector, f64), `legacy`, `frame`, `dense`,
/// `auto`, `switch:K`.
#[allow(clippy::too_many_arguments)]
pub fn expect_row(
    c: &Circuit,
    label: &str,
    obs: &PauliSum,
    method: &str,
    max_terms: usize,
    max_dense: usize,
    repeat: usize,
) -> f64 {
    let n = c.num_qubits;
    let mut best = f64::INFINITY;
    let mut out = String::new();
    for _ in 0..repeat.max(1) {
        let t0 = Instant::now();
        let res: Result<String, String> = match method {
            "sv" => {
                let bytes = (1u128 << n.min(100)) * 16;
                if bytes > MAX_STATE_BYTES {
                    Err(format!("does not fit ({} GiB)", bytes >> 30))
                } else {
                    let mut sv = StateVectorF64::new(n);
                    sv.apply_circuit(c).expect("valid circuit");
                    // single-term observables only in the benchmarks
                    let v = sv_expect(&sv, obs);
                    Ok(format!("{v:+.12e} | - | - | - | {} MiB", bytes >> 20))
                }
            }
            "legacy" => match pauli_path::expectation_legacy(c, obs, max_terms) {
                Ok((v, s)) => Ok(format!("{v:+.12e} | {} | - | - | -", s.peak_terms)),
                Err(e) => Err(e.to_string()),
            },
            "frame" => {
                let opt = FrameOptions {
                    max_terms,
                    ..FrameOptions::default()
                };
                match pauli_path::expectation_with(c, obs, &opt) {
                    Ok((v, s)) => Ok(format!("{v:+.12e} | {} | - | - | -", s.peak_terms)),
                    Err(e) => Err(e.to_string()),
                }
            }
            m => {
                let strategy = match m {
                    "dense" => Strategy::Dense,
                    "auto" => Strategy::Auto,
                    s if s.starts_with("switch:") => {
                        Strategy::SwitchAt(s[7..].parse().expect("switch:K"))
                    }
                    _ => panic!("unknown method {m:?}"),
                };
                let opt = AdaptiveOptions {
                    frame: FrameOptions {
                        max_terms,
                        ..FrameOptions::default()
                    },
                    strategy,
                    max_dense_qubits: max_dense,
                    ..AdaptiveOptions::default()
                };
                match adaptive::expectation(c, obs, &opt) {
                    Ok(r) => Ok(format!(
                        "{:+.12e} | {} | {} | {} | {} (frame {:.3} s + dense {:.3} s)",
                        r.value,
                        r.frame_stats.peak_terms,
                        r.switched_at
                            .map(|k| format!("k={k} ({} terms)", r.handover_terms))
                            .unwrap_or_else(|| "-".into()),
                        if r.switched_at.is_some() {
                            r.dense_qubits.to_string()
                        } else {
                            "-".into()
                        },
                        if r.switched_at.is_some() {
                            format!("{} MiB", (16u64 << r.dense_qubits) >> 20)
                        } else {
                            "-".into()
                        },
                        r.frame_secs,
                        r.dense_secs
                    )),
                    Err(e) => Err(e.to_string()),
                }
            }
        };
        best = best.min(secs(t0));
        match res {
            Ok(s) => out = s,
            Err(e) => {
                out = format!("aborted: {e} | - | - | - | -");
                break;
            }
        }
    }
    println!("| {label} | {method} | {out} | {best:.4} |");
    flush();
    best
}

fn sv_expect(sv: &StateVectorF64, obs: &PauliSum) -> f64 {
    // Use the compressed machinery's string form: apply each term's Paulis.
    let mut total = 0.0;
    for (s, c) in obs.terms_as_strings() {
        let mut phi = sv.clone();
        for (q, ch) in s.chars().enumerate() {
            let g = match ch {
                'X' => crate::gate::Gate::X(q),
                'Y' => crate::gate::Gate::Y(q),
                'Z' => crate::gate::Gate::Z(q),
                _ => continue,
            };
            phi.apply_gate(&g).expect("valid");
        }
        total += c * sv.inner(&phi).re;
    }
    total
}

pub fn expect_header() {
    println!("| circuit | method | value | peak terms | switch | dense qubits | dense memory | time (s) |");
    println!("|---|---|---|---|---|---|---|---|");
}

/// Sampler: build time, shot throughput, register size; optionally the
/// full state vector (f64) for comparison.
pub fn sample_row(c: &Circuit, label: &str, shots: usize, method: &str, max_dense: usize) {
    let mut rng = StdRng::seed_from_u64(17);
    let n = c.num_qubits;
    match method {
        "sv" => {
            let bytes = (1u128 << n.min(100)) * 16;
            if bytes > MAX_STATE_BYTES {
                println!("| {label} | sv | does not fit | - | - | - | - | - |");
                return;
            }
            let t0 = Instant::now();
            let mut sv = StateVectorF64::new(n);
            sv.apply_circuit(c).expect("valid");
            let t_build = secs(t0);
            let t1 = Instant::now();
            let s = sv.sample(shots, &mut rng);
            let t_shots = secs(t1);
            std::hint::black_box(s);
            println!(
                "| {label} | sv | {n} | {} MiB | {t_build:.4} | - | {t_shots:.4} | {:.3e} |",
                bytes >> 20,
                shots as f64 / t_shots
            );
        }
        "compressed" => {
            let t0 = Instant::now();
            let cs = match CompressedState::new(c, max_dense) {
                Ok(cs) => cs,
                Err(e) => {
                    println!("| {label} | compressed | aborted: {e} | - | - | - | - | - |");
                    return;
                }
            };
            let t_state = secs(t0);
            let d = cs.active_qubits();
            let t1 = Instant::now();
            let s = cs.sampler();
            let t_sampler = secs(t1);
            let t2 = Instant::now();
            let mut acc = 0u64;
            for _ in 0..shots {
                acc ^= s.sample_packed(&mut rng)[0];
            }
            let t_shots = secs(t2);
            std::hint::black_box(acc);
            println!(
                "| {label} | compressed | {d} | {} MiB | {t_state:.4} | {t_sampler:.4} (rand {} / dense {} / det {}, {} passes) | {t_shots:.4} | {:.3e} |",
                (16u64 << d) >> 20,
                s.stats.random_bits,
                s.stats.dense_bits,
                s.stats.determined,
                s.stats.dense_passes,
                shots as f64 / t_shots
            );
        }
        _ => panic!("unknown sampling method {method:?} (sv | compressed)"),
    }
    flush();
}

pub fn sample_header() {
    println!("| circuit | method | active qubits | memory | state (s) | sampler setup (s) | shots (s) | shots/s |");
    println!("|---|---|---|---|---|---|---|---|");
}

/// The observable for a family: a stabilizer of the circuit's Clifford
/// skeleton (non-zero generically) or `<Z_0>`.
pub fn observable(c: &Circuit, name: &str) -> PauliSum {
    match name {
        "stab" => skeleton_stabilizer(c, &[0]),
        "z0" => PauliSum::z_product(c.num_qubits, &[0]),
        "zlast" => PauliSum::z_product(c.num_qubits, &[c.num_qubits - 1]),
        // Z_2 Z_{n-2}: Z_b0 Z_b_top on the adder.
        "zz" => PauliSum::z_product(c.num_qubits, &[2, c.num_qubits - 2]),
        _ => panic!("unknown observable {name:?} (stab | z0 | zlast | zz)"),
    }
}
