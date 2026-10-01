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

use crate::zx;

/// Measure ZX teleportation T-count reduction and simulation speedup.
pub fn zx_teleport(n: usize, depth: usize, ts: &[usize], max_terms: usize) {
    let max_t = ts.iter().copied().max().unwrap_or(0);
    let mut rng = StdRng::seed_from_u64(42);
    let final_block = Circuit::random_clifford(n, depth, &mut rng);
    let rounds: Vec<(Circuit, usize)> = (0..max_t)
        .map(|_| (Circuit::random_clifford(n, depth, &mut rng), rng.random_range(0..n)))
        .collect();
    let build = |t: usize| {
        let mut c = Circuit::new(n);
        for (block, q) in rounds[..t].iter() {
            c.append(block);
            c.t(*q);
        }
        c.append(&final_block);
        c
    };

    println!("ZX Teleportation benchmark: {n} qubits, random Clifford depth {depth} + T blocks");
    header(&["Original T", "ZX T", "Fusions", "Orig time (s)", "ZX time (s)", "Speedup"]);
    
    for &t in ts {
        let c = build(t);
        let obs = PauliSum::z_product(n, &[0]);
        
        let zx_res = zx::simplify(&c);
        let opt_c = zx_res.circuit;
        
        let t0 = Instant::now();
        let orig_run = pauli_path::expectation(&c, &obs, max_terms).map(|(_, st)| st.peak_terms);
        let orig_time = secs(t0);
        
        let t1 = Instant::now();
        let opt_run = pauli_path::expectation(&opt_c, &obs, max_terms).map(|(_, st)| st.peak_terms);
        let opt_time = secs(t1);

        let orig_time_str = if orig_run.is_ok() { format!("{:.4}", orig_time) } else { "aborted".to_string() };
        let opt_time_str = if opt_run.is_ok() { format!("{:.4}", opt_time) } else { "aborted".to_string() };
        let speedup = if orig_run.is_ok() && opt_run.is_ok() && opt_time > 0.0 {
            format!("{:.2}x", orig_time / opt_time)
        } else {
            "-".to_string()
        };

        row(&[
            c.t_count().to_string(),
            opt_c.t_count().to_string(),
            zx_res.stats.phase_fusions.to_string(),
            orig_time_str,
            opt_time_str,
            speedup,
        ]);
    }
}

/// Measure ZX teleportation T-count reduction on Toffoli-heavy circuits.
pub fn zx_toffoli(n: usize, toffolis: &[usize], max_terms: usize) {
    let max_toffoli = toffolis.iter().copied().max().unwrap_or(0);
    let mut rng = StdRng::seed_from_u64(42);
    let rounds: Vec<(usize, usize, usize)> = (0..max_toffoli)
        .map(|_| {
            let mut q = [0; 3];
            for i in 0..3 {
                loop {
                    let cand = rng.random_range(0..n);
                    if !q[..i].contains(&cand) {
                        q[i] = cand;
                        break;
                    }
                }
            }
            (q[0], q[1], q[2])
        })
        .collect();
    let build = |t: usize| {
        let mut c = Circuit::new(n);
        for &(a, b, c_q) in rounds[..t].iter() {
            c.ccx(a, b, c_q);
        }
        c
    };

    println!("ZX Teleportation benchmark: {n} qubits, random CCX blocks (1 CCX = 7 T gates)");
    header(&["Original T", "ZX T", "Fusions", "Orig time (s)", "ZX time (s)", "Speedup"]);
    
    for &t in toffolis {
        let c = build(t);
        let obs = PauliSum::z_product(n, &[0]);
        
        let zx_res = zx::simplify(&c);
        let opt_c = zx_res.circuit;
        
        let t0 = Instant::now();
        let orig_run = pauli_path::expectation(&c, &obs, max_terms).map(|(_, st)| st.peak_terms);
        let orig_time = secs(t0);
        
        let t1 = Instant::now();
        let opt_run = pauli_path::expectation(&opt_c, &obs, max_terms).map(|(_, st)| st.peak_terms);
        let opt_time = secs(t1);

        let orig_time_str = if orig_run.is_ok() { format!("{:.4}", orig_time) } else { "aborted".to_string() };
        let opt_time_str = if opt_run.is_ok() { format!("{:.4}", opt_time) } else { "aborted".to_string() };
        let speedup = if orig_run.is_ok() && opt_run.is_ok() && opt_time > 0.0 {
            format!("{:.2}x", orig_time / opt_time)
        } else {
            "-".to_string()
        };

        row(&[
            c.t_count().to_string(),
            opt_c.t_count().to_string(),
            zx_res.stats.phase_fusions.to_string(),
            orig_time_str,
            opt_time_str,
            speedup,
        ]);
    }
}
