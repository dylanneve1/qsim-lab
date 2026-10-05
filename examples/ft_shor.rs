//! Fault-tolerant vs. unencoded Shor (N = 15) under circuit-level noise.
//!
//! ```text
//! ft_shor shor <mode> <level> <magic> <p> <shots> <seed> [a] [mask]
//!     mode  = enc | unenc | unenc-ccx
//!     magic = raw | ideal | dist      (enc only; dist = 15-to-1 model,
//!             eps_out = 35 eps_in^3 with eps_in measured by injection runs)
//!     mask  = bit mask over components prep,gate,ec,inject,meas (default 31)
//! ft_shor inject <level> <p> <trials> <seed> [postselect 0/1]
//! ```
//! Prints one `key=value` line per run.

use qsim_lab::ft::core::{Noise, ALL_COMPS, COMP_NAMES, N_COMP};
use qsim_lab::ft::logical::{inject_errors, Checked, Encoded, MagicMode, Unencoded};
use qsim_lab::ft::machine::FtConfig;
use qsim_lab::ft::shor::{
    ideal_distribution, ideal_qpe_distribution, run_shor15, run_shor21_compiled, NLOG15, NLOG21,
};
use std::time::Instant;

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a[1].as_str() {
        "inject" => {
            let level: usize = a[2].parse().unwrap();
            let p: f64 = a[3].parse().unwrap();
            let trials: u64 = a[4].parse().unwrap();
            let seed: u64 = a[5].parse().unwrap();
            let ps = a.get(6).map(|s| s != "0").unwrap_or(true);
            let (px, py, pz, eps, acc) = inject_errors(level, p, trials, seed, ps);
            println!(
                "kind=inject level={level} p={p:e} trials={trials} postselect={ps} pX={px:.6e} pY={py:.6e} pZ={pz:.6e} eps={eps:.6e} accept={acc:.5}"
            );
        }
        "shor" => {
            let mode = a[2].as_str();
            let level: usize = a[3].parse().unwrap();
            let magic = a[4].as_str();
            let p: f64 = a[5].parse().unwrap();
            let shots: u64 = a[6].parse().unwrap();
            let seed: u64 = a[7].parse().unwrap();
            // instance: a base for N = 15, or "21c" for the compiled N = 21, a = 4
            let inst: String = a.get(8).cloned().unwrap_or("7".into());
            let n21 = inst == "21c";
            let base: u64 = if n21 { 4 } else { inst.parse().unwrap() };
            let nlog = if n21 { NLOG21 } else { NLOG15 };
            let mask: u32 = a.get(9).map(|s| s.parse().unwrap()).unwrap_or(ALL_COMPS);
            let t = 3usize;
            let ideal = if n21 {
                ideal_qpe_distribution(3, t)
            } else {
                ideal_distribution(base, t)
            };
            let mut hist = [0u64; 8];
            let mut lfault = 0u64;
            let mut hist_f = [0u64; 8];
            let start = Instant::now();
            let mut eps_in = 0.0;
            let mut eps_out = 0.0;
            let mm = match magic {
                "raw" => MagicMode::Raw,
                "ideal" => MagicMode::Model(0.0),
                "dist" => {
                    let n = if level == 1 { 200_000 } else { 4_000 };
                    eps_in = inject_errors(level, p, n, seed ^ 0xABCD, true).3;
                    eps_out = 35.0 * eps_in.powi(3);
                    MagicMode::Model(eps_out)
                }
                _ => panic!("magic"),
            };
            let mut locs = 0u64;
            let mut faults = [0u64; N_COMP];
            let mut comp_locs = [0u64; N_COMP];
            let mut phys = (0u64, 0u64, 0u64, 0u64);
            let mut rejects = (0u64, 0u64);
            let mut qubits = 0usize;
            let mut tg = 0u64;
            for s in 0..shots {
                let sd = seed.wrapping_mul(1_000_003).wrapping_add(s);
                let mut noise = Noise::new(p, sd);
                noise.mask = mask;
                let y = match mode {
                    "enc" => {
                        let mut ce = Checked(Encoded::frame(
                            level,
                            nlog,
                            noise,
                            FtConfig::default(),
                            mm,
                            sd,
                        ));
                        let y = if n21 {
                            run_shor21_compiled(&mut ce, t)
                        } else {
                            run_shor15(&mut ce, base, t)
                        };
                        let e = ce.0;
                        if e.counts.logical_fault {
                            lfault += 1;
                            hist_f[y as usize] += 1;
                        }
                        locs += e.m.noise.loc;
                        for c in 0..N_COMP {
                            faults[c] += e.m.noise.faults[c];
                            comp_locs[c] += e.m.noise.locs[c];
                        }
                        let st = &e.m.stats;
                        phys.0 += st.phys_prep;
                        phys.1 += st.phys_1q;
                        phys.2 += st.phys_2q;
                        phys.3 += st.phys_meas;
                        rejects.0 += st.prep_rejects;
                        rejects.1 += st.inject_rejects;
                        qubits = qubits.max(e.m.phys_qubits());
                        tg += e.counts.t_gadgets;
                        y
                    }
                    "unenc" | "unenc-ccx" => {
                        let mut u = Unencoded::new(nlog, noise, mode == "unenc-ccx", sd);
                        let y = if n21 {
                            run_shor21_compiled(&mut u, t)
                        } else {
                            run_shor15(&mut u, base, t)
                        };
                        if u.noise.faults.iter().sum::<u64>() > 0 {
                            lfault += 1;
                            hist_f[y as usize] += 1;
                        }
                        locs += u.locations;
                        for (f, uf) in faults.iter_mut().zip(u.noise.faults.iter()).take(N_COMP) {
                            *f += *uf;
                        }
                        qubits = NLOG15;
                        y
                    }
                    _ => panic!("mode"),
                };
                hist[y as usize] += 1;
            }
            let n = shots as f64;
            // peak: |y/2^t - s/r| < 1/(2 r^2) for some s
            let rr: u64 = if n21 {
                3
            } else {
                ideal.iter().filter(|&&v| v > 0.0).count() as u64
            };
            let is_peak = |y: usize| {
                (0..rr).any(|s| {
                    ((y as f64) / 8.0 - s as f64 / rr as f64).abs() < 1.0 / (2.0 * (rr * rr) as f64)
                })
            };
            let peak: u64 = (0..8).filter(|&y| is_peak(y)).map(|y| hist[y]).sum();
            // y whose continued fraction gives r directly (s/r in lowest terms)
            let r = rr;
            let order: u64 = (0..8u64)
                .filter(|&y| {
                    is_peak(y as usize) && {
                        let s = ((y as f64) * r as f64 / 8.0).round() as u64 % r;
                        gcd(s, r) == 1
                    }
                })
                .map(|y| hist[y as usize])
                .sum();
            let tvd: f64 = 0.5
                * (0..8)
                    .map(|y| (hist[y] as f64 / n - ideal[y]).abs())
                    .sum::<f64>();
            let fs: Vec<String> = (0..N_COMP)
                .map(|c| {
                    format!(
                        "f_{}={:.4} l_{}={:.0}",
                        COMP_NAMES[c],
                        faults[c] as f64 / n,
                        COMP_NAMES[c],
                        comp_locs[c] as f64 / n
                    )
                })
                .collect();
            println!(
                "kind=shor inst={inst} mode={mode} level={level} magic={magic} p={p:e} shots={shots} seed={seed} a={base} mask={mask} \
                 peak={peak} P_peak={:.6} order={order} P_order={:.6} lfault={lfault} P_lfault={:.6} tvd={tvd:.6} hist={} hist_faulty={} ideal={} \
                 eps_in={eps_in:.4e} eps_out={eps_out:.4e} locs_per_shot={:.1} phys_qubits={qubits} \
                 prep_per_shot={:.1} g1_per_shot={:.1} g2_per_shot={:.1} meas_per_shot={:.1} \
                 prep_rej_per_shot={:.2} inj_rej_per_shot={:.3} t_gadgets_per_shot={:.2} {} secs={:.2}",
                peak as f64 / n,
                order as f64 / n,
                lfault as f64 / n,
                hist.iter().map(|h| h.to_string()).collect::<Vec<_>>().join(","),
                hist_f.iter().map(|h| h.to_string()).collect::<Vec<_>>().join(","),
                ideal.iter().map(|h| format!("{h:.10}")).collect::<Vec<_>>().join(","),
                locs as f64 / n,
                phys.0 as f64 / n,
                phys.1 as f64 / n,
                phys.2 as f64 / n,
                phys.3 as f64 / n,
                rejects.0 as f64 / n,
                rejects.1 as f64 / n,
                tg as f64 / n,
                fs.join(" "),
                start.elapsed().as_secs_f64()
            );
        }
        _ => panic!("usage: see source"),
    }
}
