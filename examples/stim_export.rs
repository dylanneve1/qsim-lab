//! Exports qsim-lab's surface-code circuit to .stim format and compares
//! per-detector firing rates and timing between qsim-lab's SymPhase sampler
//! and Stim.
//!
//! Usage: cargo run --release --example stim_export -- [d] [p] [shots] [mode]
//!
//! Modes:
//! - "export": write .stim circuit to stdout
//! - "rates": write SymPhase detector and observable firing rates to stdout
//! - "bench": benchmark SymPhase detector sampling, print time and rate to stdout

use qsim_lab::qec::surface::SurfaceCode;
use qsim_lab::stabilizer::symphase::SymPhaseSampler;
use qsim_lab::NoiseModel;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::io::Write;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let d: usize = args.get(1).map_or(3, |s| s.parse().unwrap());
    let p: f64 = args.get(2).map_or(0.003, |s| s.parse().unwrap());
    let shots: usize = args.get(3).map_or(100_000, |s| s.parse().unwrap());
    let mode = args.get(4).map_or("export", |s| s.as_str());

    let noise = NoiseModel::circuit_level(p, p);
    let sc = SurfaceCode::new(d, d);
    let circuit = sc.build_circuit();
    let num_z = sc.z_stabilizers.len();
    let num_x = sc.x_stabilizers.len();
    let num_anc = num_z + num_x;
    let total_q = SurfaceCode::total_qubits(d);
    let rounds = d;

    match mode {
        "export" => {
            // Export .stim circuit matching build_circuit() exactly
            let stdout = std::io::stdout();
            let mut out = stdout.lock();

            // qsim-lab starts all qubits at |0⟩ without explicit Reset ops
            // (no noise on the initial state). Stim needs R to establish the
            // initial reference, but no X_ERROR since there's no reset noise
            // in qsim-lab's round 0.
            let all_q: Vec<String> = (0..total_q).map(|q| q.to_string()).collect();
            writeln!(out, "R {}", all_q.join(" ")).unwrap();

            for r in 0..rounds {
                writeln!(out, "TICK").unwrap();

                // Reset ancillas (rounds > 0)
                if r > 0 {
                    let anc_z: Vec<String> = (0..num_z)
                        .map(|k| SurfaceCode::z_ancilla_idx(d, k).to_string())
                        .collect();
                    let anc_x: Vec<String> = (0..num_x)
                        .map(|k| SurfaceCode::x_ancilla_idx(d, k).to_string())
                        .collect();
                    let all_anc: Vec<String> = anc_z.iter().chain(anc_x.iter()).cloned().collect();
                    writeln!(out, "R {}", all_anc.join(" ")).unwrap();
                    if noise.p_reset > 0.0 {
                        writeln!(out, "X_ERROR({}) {}", noise.p_reset, all_anc.join(" ")).unwrap();
                    }
                }

                // H on X-ancillas
                {
                    let x_anc: Vec<String> = (0..num_x)
                        .map(|k| SurfaceCode::x_ancilla_idx(d, k).to_string())
                        .collect();
                    writeln!(out, "H {}", x_anc.join(" ")).unwrap();
                    if noise.p_1q > 0.0 {
                        writeln!(out, "DEPOLARIZE1({}) {}", noise.p_1q, x_anc.join(" ")).unwrap();
                    }
                }

                writeln!(out, "TICK").unwrap();

                // Z-checks: CNOT(data, ancilla) - sequential matching build_circuit()
                for (k, stab) in sc.z_stabilizers.iter().enumerate() {
                    let a = SurfaceCode::z_ancilla_idx(d, k);
                    for &dq in &stab.data_qubits {
                        writeln!(out, "CX {} {}", dq, a).unwrap();
                        if noise.p_2q > 0.0 {
                            writeln!(out, "DEPOLARIZE2({}) {} {}", noise.p_2q, dq, a).unwrap();
                        }
                    }
                }

                writeln!(out, "TICK").unwrap();

                // X-checks: CNOT(ancilla, data) - sequential matching build_circuit()
                for (k, stab) in sc.x_stabilizers.iter().enumerate() {
                    let a = SurfaceCode::x_ancilla_idx(d, k);
                    for &dq in &stab.data_qubits {
                        writeln!(out, "CX {} {}", a, dq).unwrap();
                        if noise.p_2q > 0.0 {
                            writeln!(out, "DEPOLARIZE2({}) {} {}", noise.p_2q, a, dq).unwrap();
                        }
                    }
                }

                writeln!(out, "TICK").unwrap();

                // H on X-ancillas before measurement
                {
                    let x_anc: Vec<String> = (0..num_x)
                        .map(|k| SurfaceCode::x_ancilla_idx(d, k).to_string())
                        .collect();
                    writeln!(out, "H {}", x_anc.join(" ")).unwrap();
                    if noise.p_1q > 0.0 {
                        writeln!(out, "DEPOLARIZE1({}) {}", noise.p_1q, x_anc.join(" ")).unwrap();
                    }
                }

                writeln!(out, "TICK").unwrap();

                // Measure Z-ancillas then X-ancillas
                {
                    let z_anc: Vec<String> = (0..num_z)
                        .map(|k| SurfaceCode::z_ancilla_idx(d, k).to_string())
                        .collect();
                    let x_anc: Vec<String> = (0..num_x)
                        .map(|k| SurfaceCode::x_ancilla_idx(d, k).to_string())
                        .collect();
                    if noise.p_meas > 0.0 {
                        writeln!(out, "MZ({}) {}", noise.p_meas, z_anc.join(" ")).unwrap();
                        writeln!(out, "MZ({}) {}", noise.p_meas, x_anc.join(" ")).unwrap();
                    } else {
                        writeln!(out, "MZ {}", z_anc.join(" ")).unwrap();
                        writeln!(out, "MZ {}", x_anc.join(" ")).unwrap();
                    }
                }

                // DETECTOR annotations for Z-detectors
                // Round 0: detector = ancilla measurement k alone
                // Round r > 0: detector = measurement r XOR measurement r-1 of same ancilla
                for k in 0..num_z {
                    let cur = -(num_anc as isize) + k as isize;
                    if r == 0 {
                        writeln!(out, "DETECTOR rec[{}]", cur).unwrap();
                    } else {
                        let prev = cur - num_anc as isize;
                        writeln!(out, "DETECTOR rec[{}] rec[{}]", cur, prev).unwrap();
                    }
                }
            }

            writeln!(out, "TICK").unwrap();

            // Final round: measure all data qubits in Z basis
            {
                let data_q: Vec<String> = (0..d * d).map(|q| q.to_string()).collect();
                if noise.p_meas > 0.0 {
                    writeln!(out, "MZ({}) {}", noise.p_meas, data_q.join(" ")).unwrap();
                } else {
                    writeln!(out, "MZ {}", data_q.join(" ")).unwrap();
                }
            }

            // Final-round detectors: parity of data qubits in each Z-stabilizer
            // XOR last ancilla measurement
            for (k, stab) in sc.z_stabilizers.iter().enumerate() {
                let last_anc = -((d * d) as isize) - num_anc as isize + k as isize;
                let mut rec_args: Vec<String> = stab
                    .data_qubits
                    .iter()
                    .map(|&dq| format!("rec[{}]", -((d * d) as isize) + dq as isize))
                    .collect();
                rec_args.push(format!("rec[{}]", last_anc));
                writeln!(out, "DETECTOR {}", rec_args.join(" ")).unwrap();
            }

            // Observable: parity of data qubits on column 0
            let obs_args: Vec<String> = (0..d)
                .map(|r| {
                    let dq = SurfaceCode::data_idx(d, r, 0);
                    format!("rec[{}]", -((d * d) as isize) + dq as isize)
                })
                .collect();
            writeln!(out, "OBSERVABLE_INCLUDE(0) {}", obs_args.join(" ")).unwrap();
        }
        "rates" => {
            let mut rng = StdRng::seed_from_u64(42);

            let sampler = SymPhaseSampler::new(&circuit, &noise).unwrap();
            let det_records = sc.detector_records();
            let obs_records = sc.observable_records();
            let all_parity_sets: Vec<Vec<usize>> = det_records
                .iter()
                .chain(std::iter::once(&obs_records))
                .cloned()
                .collect();
            let det_sampler = sampler.with_parities(&all_parity_sets);
            let num_dets = det_records.len();

            let batches = shots / 64;
            let actual_shots = batches * 64;
            let mut sym_counts = vec![0u64; num_dets + 1];
            let mut vals = vec![0u64; det_sampler.num_vars()];
            let mut out = vec![0u64; num_dets + 1];
            let t0 = Instant::now();
            for _ in 0..batches {
                det_sampler.sample_batch(&mut rng, &mut vals, &mut out);
                for (i, &w) in out.iter().enumerate() {
                    sym_counts[i] += w.count_ones() as u64;
                }
            }
            let sym_time = t0.elapsed().as_secs_f64();

            println!(
                "=== SymPhase detector rates (d={}, p={}, shots={}) ===",
                d, p, actual_shots
            );
            for (i, count) in sym_counts.iter().enumerate().take(num_dets) {
                println!("D{}: {:.6}", i, *count as f64 / actual_shots as f64);
            }
            println!(
                "OBS: {:.6}",
                sym_counts[num_dets] as f64 / actual_shots as f64
            );
            eprintln!(
                "SymPhase time: {:.4}s for {} shots = {:.0} shots/s",
                sym_time,
                actual_shots,
                actual_shots as f64 / sym_time
            );
        }
        "bench" => {
            let sampler = SymPhaseSampler::new(&circuit, &noise).unwrap();
            let det_records = sc.detector_records();
            let obs_records = sc.observable_records();
            let all_parity_sets: Vec<Vec<usize>> = det_records
                .iter()
                .chain(std::iter::once(&obs_records))
                .cloned()
                .collect();
            let det_sampler = sampler.with_parities(&all_parity_sets);

            let batches = shots / 64;
            let actual_shots = batches * 64;
            let mut rng = StdRng::seed_from_u64(42);
            let mut vals = vec![0u64; det_sampler.num_vars()];
            let mut out = vec![0u64; det_records.len() + 1];
            let t0 = Instant::now();
            for _ in 0..batches {
                det_sampler.sample_batch(&mut rng, &mut vals, &mut out);
            }
            let elapsed = t0.elapsed().as_secs_f64();
            println!(
                "d={} shots={} time={:.6} rate={:.0}",
                d,
                actual_shots,
                elapsed,
                actual_shots as f64 / elapsed
            );
        }
        _ => eprintln!("Unknown mode: {}", mode),
    }
}
