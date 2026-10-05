//! T-count benchmark driver for `compile::todd` (notebook:
//! `research/compiler/todd.md`).
//!
//! ```text
//! cargo run --release --example todd_bench -- [--restarts R] [--seed S] [--passes P]
//!     [--no-todd] [--out DIR] [--csv FILE] FILE.qc...
//! ```
//!
//! For every `.qc` file: parse, optimise, verify the output exactly
//! against the input (path-sum canonical form), and print one row with the
//! original T-count, the repo's `compile::phase_fold` on the 7-T expansion,
//! slot phase folding, and the TODD result. `--out` writes each verified
//! output circuit as `.qc`.

use qsim_lab::compile::phase_fold;
use qsim_lab::compile::todd::{self, PhaseCircuit, ToddOptions};
use qsim_lab::io::qc::{parse_qc, to_qc};
use qsim_lab::Gate;
use std::io::Write;
use std::time::Instant;

fn t_gates(c: &qsim_lab::Circuit) -> usize {
    c.gates()
        .filter(|g| match g {
            Gate::T(_) | Gate::Tdg(_) => true,
            Gate::Phase(_, t) => {
                let k = (t / std::f64::consts::FRAC_PI_4).round();
                (k as i64).rem_euclid(2) == 1
            }
            _ => false,
        })
        .count()
}

fn main() {
    let mut opts = ToddOptions::default();
    let mut files = Vec::new();
    let mut out_dir: Option<String> = None;
    let mut csv: Option<String> = None;
    let mut hred = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--restarts" => opts.restarts = args.next().unwrap().parse().unwrap(),
            "--seed" => opts.seed = args.next().unwrap().parse().unwrap(),
            "--passes" => opts.reassign_passes = args.next().unwrap().parse().unwrap(),
            "--seconds" => opts.reassign_seconds = args.next().unwrap().parse().unwrap(),
            "--no-todd" => opts.todd = false,
            "--hred" => hred = true,
            "--out" => out_dir = Some(args.next().unwrap()),
            "--csv" => csv = Some(args.next().unwrap()),
            _ => files.push(a),
        }
    }
    let mut csv_file = csv.map(|p| {
        let mut f = std::fs::File::create(p).unwrap();
        writeln!(
            f,
            "circuit,qubits,hadamards,t_orig,t_repo_phasefold,t_slotfold,t_todd,cnots,seconds,verified,restarts,seed"
        )
        .unwrap();
        f
    });
    println!(
        "{:22} {:>4} {:>5} {:>6} {:>8} {:>8} {:>6} {:>8} {:>8}  verified",
        "circuit", "n", "H", "T", "repoPF", "slotPF", "TODD", "CNOT", "sec"
    );
    for f in &files {
        let name = std::path::Path::new(f)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let src = std::fs::read_to_string(f).unwrap();
        let qc = parse_qc(&src).unwrap_or_else(|e| panic!("{f}: {e}"));
        let pc0 = PhaseCircuit::from_qc(&qc);
        let mut pc = pc0.clone();
        let mut hred_check = String::new();
        if hred {
            let removed = pc.reduce_hadamards();
            if removed > 0 {
                // the Hadamard rewrite changes the path variables, so it is
                // checked semantically: exact basis-state simulation
                let n = pc.num_qubits;
                let inputs: Vec<u128> = if n <= 16 {
                    (0..(1u128 << n)).collect()
                } else {
                    let mut x: u128 = 0x9E37_79B9_7F4A_7C15;
                    (0..4096)
                        .map(|_| {
                            x ^= x << 13;
                            x ^= x >> 7;
                            x ^= x << 17;
                            x & ((1u128 << n) - 1)
                        })
                        .collect()
                };
                let ok = todd::verify::basis_equivalent(n, &pc0.to_vgates(), &pc.to_vgates(), &inputs);
                hred_check = format!(
                    " hred -{removed}H basis[{}{}]={}",
                    inputs.len(),
                    if n <= 16 { " all" } else { " sampled" },
                    if ok == Ok(0) { "ok" } else { "FAIL" }
                );
            }
        }
        let (expanded, _) = pc0.to_circuit();
        let repo_pf = t_gates(&phase_fold(&expanded).circuit);
        let t0 = Instant::now();
        let (out, rep) = todd::optimize(&pc, &opts);
        let secs = t0.elapsed().as_secs_f64();
        let ver = todd::verify_output(&pc, &out);
        let verified = match &ver {
            Ok(g) if *g == rep.global_phase => "yes".to_string(),
            Ok(g) => format!("PHASE-MISMATCH {g} vs {}", rep.global_phase),
            Err(e) => format!("NO: {e}"),
        };
        println!(
            "{:22} {:>4} {:>5} {:>6} {:>8} {:>8} {:>6} {:>8} {:>8.3}  {}{}",
            name,
            pc.num_qubits,
            rep.hadamards,
            rep.t_input,
            repo_pf,
            rep.t_folded,
            rep.t_output,
            rep.cnots,
            secs,
            verified,
            hred_check
        );
        if let Some(fh) = csv_file.as_mut() {
            writeln!(
                fh,
                "{},{},{},{},{},{},{},{},{:.3},{},{},{}",
                name,
                pc.num_qubits,
                rep.hadamards,
                rep.t_input,
                repo_pf,
                rep.t_folded,
                rep.t_output,
                rep.cnots,
                secs,
                verified == "yes",
                opts.restarts,
                opts.seed
            )
            .unwrap();
        }
        if let (Some(dir), Ok(_)) = (&out_dir, &ver) {
            std::fs::create_dir_all(dir).unwrap();
            let text = to_qc(&out).unwrap();
            std::fs::write(format!("{dir}/{name}.qc"), text).unwrap();
        }
    }
}
