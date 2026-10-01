//! A small timing harness used by `qsim bench`. Each benchmark prints a
//! markdown table row as soon as it is measured.

use crate::circuit::Circuit;
use crate::gate::Gate;
use crate::mps::Mps;
use crate::pauli_path::{self, PauliSum};
use crate::stabilizer::{tableau_bytes, Tableau};
use crate::statevector::{state_bytes, Real, StateVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::io::Write;
use std::time::Instant;

fn secs(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}

/// Human-readable byte count.
pub fn fmt_bytes(b: u128) -> String {
    let units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
    let mut v = b as f64;
    let mut u = 0;
    while v >= 1024.0 && u + 1 < units.len() {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{b} B")
    } else if b >= 1 << 70 {
        format!("2^{:.0} B", (b as f64).log2())
    } else {
        format!("{v:.1} {}", units[u])
    }
}

fn header(cols: &[&str]) {
    println!("| {} |", cols.join(" | "));
    println!("|{}", cols.iter().map(|_| "---|").collect::<String>());
}

fn row(cells: &[String]) {
    println!("| {} |", cells.join(" | "));
    let _ = std::io::stdout().flush();
}

/// GHZ on the state vector: allocation + H + (n-1) CNOTs, then 1000 shots.
pub fn sv_ghz<T: Real>(ns: &[usize], label: &str) {
    header(&[
        "n",
        "precision",
        "memory",
        "alloc (s)",
        "gates (s)",
        "1000 shots (s)",
        "total (s)",
    ]);
    let mut rng = StdRng::seed_from_u64(1);
    for &n in ns {
        let t0 = Instant::now();
        let mut s = match StateVector::<T>::try_new(n) {
            Ok(s) => s,
            Err(e) => {
                println!(
                    "| {n} | {label} | {} | skipped: {e} | | | |",
                    fmt_bytes(state_bytes::<T>(n))
                );
                continue;
            }
        };
        let t_alloc = secs(t0);
        let t1 = Instant::now();
        s.apply_circuit(&crate::algorithms::ghz(n)).expect("valid");
        let t_gates = secs(t1);
        let t2 = Instant::now();
        let shots = s.sample(1000, &mut rng);
        let t_shots = secs(t2);
        let all_ones = (1usize << n) - 1;
        assert!(
            shots.iter().all(|&x| x == 0 || x == all_ones),
            "GHZ check failed"
        );
        row(&[
            n.to_string(),
            label.to_string(),
            fmt_bytes(s.bytes() as u128),
            format!("{t_alloc:.3}"),
            format!("{t_gates:.3}"),
            format!("{t_shots:.3}"),
            format!("{:.3}", secs(t0)),
        ]);
    }
}

/// QFT (n H, n(n-1)/2 controlled phases, n/2 SWAPs) on a random basis state.
pub fn sv_qft<T: Real>(ns: &[usize], label: &str) {
    header(&["n", "precision", "gates", "time (s)", "time per gate (ms)"]);
    for &n in ns {
        let mut s = StateVector::<T>::basis_state(n, (0x5A5A_5A5A_usize) & ((1 << n) - 1));
        let c = crate::algorithms::qft(n);
        let t = Instant::now();
        s.apply_circuit(&c).expect("valid");
        let dt = secs(t);
        row(&[
            n.to_string(),
            label.to_string(),
            c.num_gates().to_string(),
            format!("{dt:.3}"),
            format!("{:.2}", 1e3 * dt / c.num_gates() as f64),
        ]);
    }
}

/// GHZ on the stabilizer tableau, then measure every qubit. For small `n`
/// also times plain one-qubit-at-a-time CHP measurement for comparison.
pub fn stab_ghz(ns: &[usize]) {
    header(&[
        "n",
        "tableau memory",
        "prepare (s)",
        "measure all (s)",
        "total (s)",
        "CHP qubit-by-qubit measure (s)",
    ]);
    let mut rng = StdRng::seed_from_u64(2);
    for &n in ns {
        let t0 = Instant::now();
        let mut t = match Tableau::try_new(n) {
            Ok(t) => t,
            Err(e) => {
                println!(
                    "| {n} | {} | skipped: {e} | | | |",
                    fmt_bytes(tableau_bytes(n))
                );
                continue;
            }
        };
        t.h(0);
        for q in 1..n {
            t.cnot(q - 1, q);
        }
        let t_prep = secs(t0);
        let seq = (n <= 5000).then(|| t.clone());
        let t1 = Instant::now();
        let bits = t.measure_all(&mut rng);
        let t_meas = secs(t1);
        let total = secs(t0);
        assert!(bits.iter().all(|&b| b == bits[0]), "GHZ check failed");
        let seq_time = seq.map_or("-".to_string(), |mut s| {
            let t2 = Instant::now();
            let b = s.measure_all_sequential(&mut rng);
            assert!(b.iter().all(|&x| x == b[0]));
            format!("{:.3}", secs(t2))
        });
        row(&[
            n.to_string(),
            fmt_bytes(t.bytes() as u128),
            format!("{t_prep:.3}"),
            format!("{t_meas:.3}"),
            format!("{total:.3}"),
            seq_time,
        ]);
    }
}

/// Clifford+T circuits on `n` qubits: starting from `|0...0>`, `t` rounds of
/// (random Clifford block of depth `depth`, T on a random qubit), then a
/// final Clifford block. The circuit for `t + 1` adds one round at the
/// start, so the backward propagation for `t` is a prefix of the one for
/// `t + 1`, which keeps the growth curve smooth. Reports the cost of
/// computing `<Z_0>` exactly by Pauli-path summation (for these scrambling
/// circuits the value itself is 0; correctness is covered by the tests,
/// which compare against the state vector).
pub fn clifford_t(n: usize, depth: usize, ts: &[usize], max_terms: usize) {
    let max_t = ts.iter().copied().max().unwrap_or(0);
    let mut rng = StdRng::seed_from_u64(3);
    let final_block = Circuit::random_clifford(n, depth, &mut rng);
    let rounds: Vec<(Circuit, usize)> = (0..max_t)
        .map(|_| {
            (
                Circuit::random_clifford(n, depth, &mut rng),
                rng.random_range(0..n),
            )
        })
        .collect();
    let build = |t: usize| {
        let mut c = Circuit::new(n);
        for (block, q) in rounds[..t].iter().rev() {
            c.append(block);
            c.t(*q);
        }
        c.append(&final_block);
        c
    };
    println!(
        "n = {n} qubits (a state vector would need {}); each round is a random \
         Clifford block of depth {depth} followed by one T gate.\n",
        fmt_bytes(state_bytes::<f32>(n))
    );
    header(&["T gates", "gates total", "Pauli terms (peak)", "time (s)"]);
    for &t in ts {
        let c = build(t);
        let obs = PauliSum::z_product(n, &[0]);
        let t0 = Instant::now();
        match pauli_path::expectation(&c, &obs, max_terms) {
            Ok((_, st)) => row(&[
                t.to_string(),
                c.num_gates().to_string(),
                st.peak_terms.to_string(),
                format!("{:.4}", secs(t0)),
            ]),
            Err(e) => {
                row(&[
                    t.to_string(),
                    c.num_gates().to_string(),
                    format!("aborted: {e}"),
                    format!("{:.4}", secs(t0)),
                ]);
                break;
            }
        }
    }
}

/// GHZ on the MPS backend.
pub fn mps_ghz(ns: &[usize]) {
    header(&["n", "max bond", "memory", "time (s)"]);
    for &n in ns {
        let t0 = Instant::now();
        let mut m = Mps::new(n, 64);
        m.apply_gate(&Gate::H(0)).expect("valid");
        for q in 1..n {
            m.apply_gate(&Gate::Cnot(q - 1, q)).expect("valid");
        }
        let dt = secs(t0);
        row(&[
            n.to_string(),
            m.max_bond_dim().to_string(),
            fmt_bytes(m.bytes() as u128),
            format!("{dt:.3}"),
        ]);
    }
}

/// Random brickwork circuits (random single-qubit rotations + CNOT layers
/// on neighbours) on the MPS backend: bond dimension and time per depth.
pub fn mps_random(n: usize, max_bond: usize, depths: &[usize]) {
    header(&[
        "n",
        "depth",
        "χ cap",
        "max bond",
        "fidelity est.",
        "memory",
        "time (s)",
    ]);
    for &d in depths {
        let mut rng = StdRng::seed_from_u64(4);
        let t0 = Instant::now();
        let mut m = Mps::new(n, max_bond);
        for layer in 0..d {
            for q in 0..n {
                m.apply_gate(&Gate::Ry(q, rng.random::<f64>() * std::f64::consts::PI))
                    .expect("valid");
                m.apply_gate(&Gate::Rz(q, rng.random::<f64>() * std::f64::consts::PI))
                    .expect("valid");
            }
            for q in (layer % 2..n.saturating_sub(1)).step_by(2) {
                m.apply_gate(&Gate::Cnot(q, q + 1)).expect("valid");
            }
        }
        let dt = secs(t0);
        row(&[
            n.to_string(),
            d.to_string(),
            max_bond.to_string(),
            m.max_bond_dim().to_string(),
            format!("{:.6}", m.fidelity_estimate()),
            fmt_bytes(m.bytes() as u128),
            format!("{dt:.3}"),
        ]);
    }
}

// ----- hybrid Schrödinger–Feynman ------------------------------------------

/// Resets the kernel's peak-RSS counter for this process (Linux; no-op
/// elsewhere), so `VmHWM` measures the next phase only.
fn reset_peak_rss() {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
}

fn status_kb(field: &str) -> u128 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(field))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<u128>().ok())
        })
        .unwrap_or(0)
}

/// Runs `f`, returning its result, wall time and the peak RSS growth above
/// the RSS at the start (bytes).
fn measured<R>(f: impl FnOnce() -> R) -> (R, f64, u128) {
    reset_peak_rss();
    let base = status_kb("VmRSS:");
    let t = Instant::now();
    let r = f();
    let dt = secs(t);
    let peak = status_kb("VmHWM:");
    (r, dt, peak.saturating_sub(base) * 1024)
}

fn max_diff(a: &[num_complex::Complex64], b: &[num_complex::Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

/// HSF vs the f64 state vector on two-block circuits with `k` crossing
/// gates: full output and a batch of `amps` amplitudes. Interleaved,
/// min of `reps`; also prints the accuracy of each HSF result.
pub fn hsf_crossover(n: usize, ks: &[usize], depth: usize, amps: usize, reps: usize, full: bool) {
    use crate::hsf::{two_block_circuit, HsfOptions, HybridSchrodingerFeynman};
    header(&[
        "n",
        "k",
        "paths",
        "SV (s)",
        "HSF full (s)",
        "HSF amps (s)",
        "SV/HSF full",
        "SV/HSF amps",
        "SV peak",
        "HSF full peak",
        "HSF amps peak",
        "max |Δ|",
    ]);
    for &k in ks {
        let mut rng = StdRng::seed_from_u64(1000 + k as u64);
        let c = two_block_circuit(n, n / 2, depth, k, false, &mut rng);
        let xs: Vec<usize> = (0..amps)
            .map(|_| rng.random_range(0..1usize << n))
            .collect();
        let h = HybridSchrodingerFeynman::auto(&c, HsfOptions::default()).expect("plan");
        let (mut t_sv, mut t_full, mut t_amp) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let (mut m_sv, mut m_full, mut m_amp) = (0, 0, 0);
        let mut err = 0.0f64;
        for _ in 0..reps {
            let (sv, dt, m) = measured(|| {
                let mut s = StateVector::<f64>::new(n);
                s.apply_circuit(&c).expect("valid");
                s
            });
            t_sv = t_sv.min(dt);
            m_sv = m_sv.max(m);
            if full {
                let (psi, dt, m) = measured(|| h.state_vector().expect("full output"));
                t_full = t_full.min(dt);
                m_full = m_full.max(m);
                err = err.max(max_diff(&psi, sv.amplitudes()));
            }
            let (a, dt, m) = measured(|| h.amplitudes(&xs).expect("amplitudes"));
            t_amp = t_amp.min(dt);
            m_amp = m_amp.max(m);
            let want: Vec<_> = xs.iter().map(|&x| sv.amplitude(x)).collect();
            err = err.max(max_diff(&a, &want));
        }
        row(&[
            n.to_string(),
            h.num_cut_gates().to_string(),
            h.num_paths().to_string(),
            format!("{t_sv:.3}"),
            if full {
                format!("{t_full:.3}")
            } else {
                "-".into()
            },
            format!("{t_amp:.4}"),
            if full {
                format!("{:.2}", t_sv / t_full)
            } else {
                "-".into()
            },
            format!("{:.1}", t_sv / t_amp),
            fmt_bytes(m_sv),
            if full { fmt_bytes(m_full) } else { "-".into() },
            fmt_bytes(m_amp),
            format!("{err:.1e}"),
        ]);
    }
}

/// HSF amplitude batches on circuits too large for the state vector.
pub fn hsf_big(ns: &[usize], ks: &[usize], depth: usize, amps: usize, middle: bool) {
    use crate::hsf::{cut_bits, two_block_circuit, HsfOptions, HybridSchrodingerFeynman};
    header(&[
        "n",
        "blocks",
        "planted k",
        "auto k",
        "paths",
        "amps",
        "time (s)",
        "peak RSS growth",
        "state vector would need",
    ]);
    for &n in ns {
        for &k in ks {
            let mut rng = StdRng::seed_from_u64(7 * n as u64 + k as u64);
            let c = two_block_circuit(n, n / 2, depth, k, middle, &mut rng);
            let xs: Vec<usize> = (0..amps)
                .map(|_| rng.random_range(0..1usize << n))
                .collect();
            let o = HsfOptions::default();
            let (h, t_plan, _) = measured(|| HybridSchrodingerFeynman::auto(&c, o.clone()));
            let h = h.expect("plan");
            let planted: Vec<bool> = (0..n).map(|q| q < n / 2).collect();
            let pk = cut_bits(&c, &planted, &o).expect("valid");
            let (r, dt, m) = measured(|| h.amplitudes(&xs));
            let (na, nb) = h.block_sizes();
            row(&[
                n.to_string(),
                format!("{na}+{nb}"),
                pk.to_string(),
                h.num_cut_gates().to_string(),
                h.num_paths().to_string(),
                amps.to_string(),
                match r {
                    Ok(_) => format!("{:.2} (+{t_plan:.2} plan)", dt),
                    Err(e) => format!("refused: {e}"),
                },
                fmt_bytes(m),
                fmt_bytes(state_bytes::<f64>(n)),
            ]);
        }
    }
}

/// A/B of the HSF design choices on one circuit (amplitude batch and full
/// output), interleaved, min of `reps`.
pub fn hsf_ablation(n: usize, k: usize, depth: usize, amps: usize, reps: usize, middle: bool) {
    use crate::hsf::{
        two_block_circuit, HsfOptions, HybridSchrodingerFeynman, LeafMode, SchmidtMode,
    };
    let mut rng = StdRng::seed_from_u64(42);
    let c = two_block_circuit(n, n / 2, depth, k, middle, &mut rng);
    let xs: Vec<usize> = (0..amps)
        .map(|_| rng.random_range(0..1usize << n))
        .collect();
    let part: Vec<bool> = (0..n).map(|q| q < n / 2).collect();
    let d = HsfOptions::default;
    let variants: Vec<(&str, HsfOptions)> = vec![
        ("default (rank-2, ASAP, auto leaf)", d()),
        ("ASAP scheduling off", HsfOptions { asap: false, ..d() }),
        (
            "leaf = forward",
            HsfOptions {
                leaf: LeafMode::Forward,
                ..d()
            },
        ),
        (
            "leaf = bra",
            HsfOptions {
                leaf: LeafMode::Bra,
                ..d()
            },
        ),
        ("1 worker", HsfOptions { threads: 1, ..d() }),
        ("2 workers", HsfOptions { threads: 2, ..d() }),
        (
            "first attempt's 4-term expansion",
            HsfOptions {
                schmidt: SchmidtMode::MatrixUnits,
                ..d()
            },
        ),
    ];
    header(&[
        "variant",
        "paths",
        "amps (s)",
        "vs default",
        "full (s)",
        "vs default",
    ]);
    let plans: Vec<_> = variants
        .iter()
        .map(|(_, o)| HybridSchrodingerFeynman::new(&c, &part, o.clone()).expect("plan"))
        .collect();
    let full_ok = n <= 22;
    let mut ta = vec![f64::INFINITY; plans.len()];
    let mut tf = vec![f64::INFINITY; plans.len()];
    let mut reference: Option<Vec<num_complex::Complex64>> = None;
    for _ in 0..reps {
        for (i, h) in plans.iter().enumerate() {
            let (a, dt, _) = measured(|| h.amplitudes(&xs).expect("amps"));
            ta[i] = ta[i].min(dt);
            match &reference {
                None => reference = Some(a),
                Some(r) => assert!(max_diff(r, &a) < 1e-12, "variant {i} disagrees"),
            }
            if full_ok {
                let (_, dt, _) = measured(|| h.state_vector().expect("full"));
                tf[i] = tf[i].min(dt);
            }
        }
    }
    for (i, (name, _)) in variants.iter().enumerate() {
        row(&[
            name.to_string(),
            plans[i].num_paths().to_string(),
            format!("{:.3}", ta[i]),
            format!("{:.2}x", ta[i] / ta[0]),
            if full_ok {
                format!("{:.3}", tf[i])
            } else {
                "-".into()
            },
            if full_ok {
                format!("{:.2}x", tf[i] / tf[0])
            } else {
                "-".into()
            },
        ]);
    }
}
