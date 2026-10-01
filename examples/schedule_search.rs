//! Automated search over CNOT schedules for rotated surface code syndrome extraction.
//!
//! Searches all 24 × 24 = 576 CNOT schedules (per check type), applies circuit-distance
//! filtering, evaluates survivors under unbiased (η = 1) and Z-biased (η = 10, 100) noise,
//! and runs held-out seed validation, fault-injection checks, and tableau cross-checks.

use qsim_lab::noise::NoiseModel;
use qsim_lab::qec::schedules::{
    fault_injection_distance_ok, wilson_score_interval, BiasedDemSamplerV2, Permutation, Schedule,
    ScheduledSurfaceCode,
};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

fn score_biased<R: RngCore>(
    sc: &ScheduledSurfaceCode,
    noise: &NoiseModel,
    eta: f64,
    shots: usize,
    rng: &mut R,
) -> (f64, usize, (f64, f64, f64)) {
    let sampler = BiasedDemSamplerV2::new(&sc.faults, noise, eta);
    let mut flags = Vec::new();
    let mut defects = Vec::new();
    let mut errors = 0usize;
    for _ in 0..shots {
        let raw = sampler.sample_into(rng, &mut flags, &mut defects);
        if raw ^ sc.decoder.decode(&defects) {
            errors += 1;
        }
    }
    let p_l = errors as f64 / shots.max(1) as f64;
    let ci = wilson_score_interval(errors, shots);
    (p_l, errors, ci)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut d = 3usize;
    let mut rounds = 3usize;
    let mut shots = 100_000usize;
    let mut seed = 42u64;
    let mut p_phys = 0.005f64;
    let mut validate = false;
    let mut interleaved = false;
    let mut cf_only = false;

    for arg in &args {
        if arg == "--validate" {
            validate = true;
        } else if arg == "--interleaved" {
            interleaved = true;
        } else if arg == "--cf-only" {
            cf_only = true;
        } else if let Some(v) = arg.strip_prefix("d=") {
            d = v.parse().unwrap_or(d);
        } else if let Some(v) = arg.strip_prefix("rounds=") {
            rounds = v.parse().unwrap_or(rounds);
        } else if let Some(v) = arg.strip_prefix("shots=") {
            shots = v.parse().unwrap_or(shots);
        } else if let Some(v) = arg.strip_prefix("seed=") {
            seed = v.parse().unwrap_or(seed);
        } else if let Some(v) = arg.strip_prefix("p=") {
            p_phys = v.parse().unwrap_or(p_phys);
        }
    }

    println!("================================================================================");
    println!("  Surface-Code Syndrome Extraction Schedule Search (d={d}, rounds={rounds})");
    println!("  p_phys={p_phys}, shots={shots}, seed={seed}, interleaved={interleaved}, cf_only={cf_only}, validate={validate}");
    println!("================================================================================\n");

    let t0 = Instant::now();

    // ──────────────────────────────────────────────────────────
    // Step 1: Distance filter on all 576 schedules
    // ──────────────────────────────────────────────────────────
    println!("--- Step 1: Distance and fault-injection filtering across schedules ---");
    let all_perms = Permutation::all();
    let mut dist_counts = std::collections::BTreeMap::new();
    let mut survivors_seq = Vec::new();
    let mut collision_free_total = 0;
    let mut collision_free_survivors = 0;
    let mut fi_pass_count = 0;

    for z_perm in &all_perms {
        for x_perm in &all_perms {
            let sched = Schedule {
                z_perm: *z_perm,
                x_perm: *x_perm,
                interleaved,
            };
            let is_cf = sched.is_collision_free();
            if is_cf {
                collision_free_total += 1;
            }
            if cf_only && !is_cf {
                continue;
            }

            let sc = ScheduledSurfaceCode::new(d, rounds, sched.clone());
            let fi_ok = fault_injection_distance_ok(&sc);
            if fi_ok {
                fi_pass_count += 1;
            }

            match sc.circuit_distance_and_mechanisms() {
                Some((dist, count)) => {
                    *dist_counts.entry(dist).or_insert(0usize) += 1;
                    if dist == d && fi_ok {
                        survivors_seq.push((sched.clone(), dist, count));
                        if is_cf {
                            collision_free_survivors += 1;
                        }
                    }
                }
                None => {
                    *dist_counts.entry(0usize).or_insert(0usize) += 1;
                }
            }
        }
    }

    let search_pool_size = if cf_only { 96 } else { 576 };
    println!(
        "Distance filter completed in {:.3}s",
        t0.elapsed().as_secs_f64()
    );
    println!("Circuit distance distribution across {search_pool_size} schedules:");
    for (&dist, &cnt) in &dist_counts {
        println!(
            "  distance = {dist}: {cnt} / {search_pool_size} schedules ({:.1}%)",
            100.0 * cnt as f64 / search_pool_size as f64
        );
    }
    println!(
        "Fault-injection distance check passed: {} / {}",
        fi_pass_count, search_pool_size
    );
    if !cf_only {
        println!(
            "Collision-free schedules (interleaved 4-step compatible): {} / 576",
            collision_free_total
        );
        println!(
            "Collision-free survivors with distance = {d} and FI passed: {} / {}",
            collision_free_survivors, collision_free_total
        );
    }
    println!(
        "Total surviving schedules with full distance={d} & FI passed: {} / {}\n",
        survivors_seq.len(),
        search_pool_size
    );

    // ──────────────────────────────────────────────────────────
    // Step 2: Scoring survivors on unbiased and biased noise
    // ──────────────────────────────────────────────────────────
    let noise = NoiseModel::circuit_level(p_phys, p_phys);
    let mut rng = StdRng::seed_from_u64(seed);

    struct EvaluatedSchedule {
        sched: Schedule,
        min_mechanisms: usize,
        p_l_eta1: f64,
        ci_eta1: (f64, f64, f64),
        p_l_eta10: f64,
        ci_eta10: (f64, f64, f64),
        p_l_eta100: f64,
        ci_eta100: (f64, f64, f64),
    }

    let t_eval = Instant::now();
    println!(
        "--- Step 2: Scoring {} survivors (shots={}, p={}) ---",
        survivors_seq.len(),
        shots,
        p_phys
    );

    let mut evaluated: Vec<EvaluatedSchedule> = Vec::with_capacity(survivors_seq.len());

    for (idx, (sched, _dist, count)) in survivors_seq.iter().enumerate() {
        let sc = ScheduledSurfaceCode::new(d, rounds, sched.clone());
        let (p1, _, ci1) = score_biased(&sc, &noise, 1.0, shots, &mut rng);
        let (p10, _, ci10) = score_biased(&sc, &noise, 10.0, shots, &mut rng);
        let (p100, _, ci100) = score_biased(&sc, &noise, 100.0, shots, &mut rng);

        evaluated.push(EvaluatedSchedule {
            sched: sched.clone(),
            min_mechanisms: *count,
            p_l_eta1: p1,
            ci_eta1: ci1,
            p_l_eta10: p10,
            ci_eta10: ci10,
            p_l_eta100: p100,
            ci_eta100: ci100,
        });

        if (idx + 1) % 20 == 0 || idx + 1 == survivors_seq.len() {
            println!(
                "  Evaluated {} / {} survivors ({:.1}s)",
                idx + 1,
                survivors_seq.len(),
                t_eval.elapsed().as_secs_f64()
            );
        }
    }

    // Sort by p_L at η = 100 ascending
    evaluated.sort_by(|a, b| a.p_l_eta100.partial_cmp(&b.p_l_eta100).unwrap());

    // Locate standard schedule
    let std_z = Permutation::standard_z();
    let std_x = Permutation::standard_x();
    let std_idx = evaluated
        .iter()
        .position(|e| e.sched.z_perm == std_z && e.sched.x_perm == std_x);

    println!("\nStandard schedule (Z={:?}, X={:?}):", std_z.0, std_x.0);
    if let Some(pos) = std_idx {
        let std_entry = &evaluated[pos];
        println!("  Rank at η=100: #{} of {}", pos + 1, evaluated.len());
        println!("  Min-weight failure paths: {}", std_entry.min_mechanisms);
        println!(
            "  p_L(η=1):   {:.5} (95% CI: [{:.5}, {:.5}])",
            std_entry.p_l_eta1, std_entry.ci_eta1.1, std_entry.ci_eta1.2
        );
        println!(
            "  p_L(η=10):  {:.5} (95% CI: [{:.5}, {:.5}])",
            std_entry.p_l_eta10, std_entry.ci_eta10.1, std_entry.ci_eta10.2
        );
        println!(
            "  p_L(η=100): {:.5} (95% CI: [{:.5}, {:.5}])",
            std_entry.p_l_eta100, std_entry.ci_eta100.1, std_entry.ci_eta100.2
        );
    } else {
        println!("  WARNING: Standard schedule was not among survivors!");
    }

    println!("\nTop 10 Schedules (sorted by p_L at η = 100):");
    println!("| Rank | Z-perm | X-perm | CF? | MinW Paths | p_L(η=1) | p_L(η=10) | p_L(η=100) | 95% CI (η=100) |");
    println!("|------|--------|--------|-----|------------|----------|-----------|------------|----------------|");
    for (i, e) in evaluated.iter().take(10).enumerate() {
        println!(
            "| {:>4} | {:?} | {:?} | {:<3} | {:>10} | {:>8.5} | {:>9.5} | {:>10.5} | [{:.5}, {:.5}] |",
            i + 1,
            e.sched.z_perm.0,
            e.sched.x_perm.0,
            if e.sched.is_collision_free() { "Yes" } else { "No" },
            e.min_mechanisms,
            e.p_l_eta1,
            e.p_l_eta10,
            e.p_l_eta100,
            e.ci_eta100.1,
            e.ci_eta100.2
        );
    }

    // Save CSV
    let csv_path = format!(
        "research/data/schedules/d{}_rounds{}_p{}_search.csv",
        d, rounds, p_phys
    );
    if let Ok(mut f) = File::create(&csv_path) {
        writeln!(f, "rank,z_perm,x_perm,collision_free,min_weight_paths,p_l_eta1,ci1_low,ci1_high,p_l_eta10,ci10_low,ci10_high,p_l_eta100,ci100_low,ci100_high").unwrap();
        for (i, e) in evaluated.iter().enumerate() {
            writeln!(
                f,
                "{},\"{:?}\",\"{:?}\",{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                i + 1,
                e.sched.z_perm.0,
                e.sched.x_perm.0,
                e.sched.is_collision_free(),
                e.min_mechanisms,
                e.p_l_eta1,
                e.ci_eta1.1,
                e.ci_eta1.2,
                e.p_l_eta10,
                e.ci_eta10.1,
                e.ci_eta10.2,
                e.p_l_eta100,
                e.ci_eta100.1,
                e.ci_eta100.2
            )
            .unwrap();
        }
        println!("\nFull search results written to {csv_path}");
    }

    // ──────────────────────────────────────────────────────────
    // Step 3: Validation on held-out seeds, fault injection, tableau
    // ──────────────────────────────────────────────────────────
    println!("\n--- Step 3: Held-out validation and cross-checks ---");
    let top_candidates: Vec<Schedule> = evaluated.iter().take(5).map(|e| e.sched.clone()).collect();

    let mut validation_schedules = top_candidates;
    // Ensure standard schedule is included in validation
    let std_sched = Schedule {
        z_perm: std_z,
        x_perm: std_x,
        interleaved,
    };
    if !validation_schedules.contains(&std_sched) {
        validation_schedules.push(std_sched.clone());
    }

    let val_seeds = [seed + 10007, seed + 20011, seed + 30013];
    let val_shots = shots.max(100_000);

    println!(
        "Validating {} candidate schedules across {} held-out seeds ({} shots/seed):",
        validation_schedules.len(),
        val_seeds.len(),
        val_shots
    );

    for (cand_idx, cand) in validation_schedules.iter().enumerate() {
        let is_std = *cand == std_sched;
        let label = if is_std { "Standard" } else { "Candidate" };
        let sc = ScheduledSurfaceCode::new(d, rounds, cand.clone());
        let fi_ok = fault_injection_distance_ok(&sc);
        let (dist, count) = sc.circuit_distance_and_mechanisms().unwrap();

        println!(
            "\n[{label} #{}] Z={:?}, X={:?} (CF={})",
            cand_idx + 1,
            cand.z_perm.0,
            cand.x_perm.0,
            cand.is_collision_free()
        );
        println!(
            "  Fault-injection distance check: {}",
            if fi_ok {
                "PASSED (all single faults correctable)"
            } else {
                "FAILED"
            }
        );
        println!(
            "  Circuit distance: {}, min-weight failure paths: {}",
            dist, count
        );

        for &eta in &[1.0, 10.0, 100.0] {
            let mut total_errs = 0usize;
            let mut total_shots = 0usize;
            for &s in &val_seeds {
                let mut v_rng = StdRng::seed_from_u64(s);
                let (_, errs, _) = score_biased(&sc, &noise, eta, val_shots, &mut v_rng);
                total_errs += errs;
                total_shots += val_shots;
            }
            let pooled_p = total_errs as f64 / total_shots as f64;
            let pooled_ci = wilson_score_interval(total_errs, total_shots);
            println!(
                "  η = {:<3}: p_L = {:.5} (pooled {} shots, 95% CI: [{:.5}, {:.5}])",
                eta, pooled_p, total_shots, pooled_ci.1, pooled_ci.2
            );
        }

        // Tableau simulation cross-check at d=3
        let tab_shots = 1000;
        let mut tab_rng = StdRng::seed_from_u64(seed + 9999);
        let (tab_rate, tab_errs) = sc.run_experiment_tableau(&noise, tab_shots, &mut tab_rng);
        let tab_ci = wilson_score_interval(tab_errs, tab_shots);
        println!(
            "  Tableau cross-check (η=1, {} shots): p_L = {:.4} (95% CI: [{:.4}, {:.4}])",
            tab_shots, tab_rate, tab_ci.1, tab_ci.2
        );
    }

    println!(
        "\nSearch and validation completed in {:.1}s",
        t0.elapsed().as_secs_f64()
    );
    println!("================================================================================");
}
