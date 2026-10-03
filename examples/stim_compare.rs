//! Identical-circuit comparison with Stim (both directions).
//!
//! ```text
//! stim_compare export-surface <d> <p> <out.stim>
//!     qsim-lab's native SurfaceCode::new(d, d) memory circuit under
//!     NoiseModel::circuit_level(p, p), serialised op by op (stim_io::to_stim).
//! stim_compare sample <in.stim> <shots> <out.ptb64> [seed]
//!     parse the file, compile SymPhase, write detection events (+ observables
//!     appended) in Stim's ptb64 layout.
//! stim_compare sample-native <d> <p> <shots> <out.ptb64> [seed]
//!     same, but from the native circuit object (no file round trip).
//! stim_compare bench <in.stim> <shots> [reps]
//!     single thread: compile time, then sampling time with ptb64 output
//!     streamed to /dev/null (what Stim's sample_write does).
//! ```
use qsim_lab::qec::surface::SurfaceCode;
use qsim_lab::stabilizer::symphase::SymPhaseSampler;
use qsim_lab::stim_io::{parse_stim, to_stim};
use qsim_lab::{Circuit, NoiseModel};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::io::{BufWriter, Write};
use std::time::Instant;

fn compile(
    c: &Circuit,
    noise: &NoiseModel,
    dets: &[Vec<usize>],
    obs: &[Vec<usize>],
) -> SymPhaseSampler {
    let sets: Vec<Vec<usize>> = dets.iter().chain(obs.iter()).cloned().collect();
    let s = SymPhaseSampler::new(c, noise).expect("compile");
    let ds = s.with_parities(&sets);
    // deterministic detectors: no reference flips (Stim reports events relative to noiseless)
    assert!(
        ds.reference().iter().all(|&b| !b),
        "non-zero reference parity"
    );
    ds
}

fn sample_to<W: Write>(s: &SymPhaseSampler, shots: usize, seed: u64, w: &mut W) {
    let mut rng = StdRng::seed_from_u64(seed);
    let rows = s.num_measurements();
    let mut vals = vec![0u64; s.num_vars()];
    let mut out = vec![0u64; rows];
    let mut bytes = vec![0u8; rows * 8];
    let batches = shots.div_ceil(64);
    for _ in 0..batches {
        s.sample_batch(&mut rng, &mut vals, &mut out);
        for (k, word) in out.iter().enumerate() {
            bytes[8 * k..8 * k + 8].copy_from_slice(&word.to_le_bytes());
        }
        w.write_all(&bytes).unwrap();
    }
    w.flush().unwrap();
}

fn load(path: &str) -> qsim_lab::stim_io::StimProgram {
    parse_stim(&std::fs::read_to_string(path).expect("read")).expect("parse")
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a[1].as_str() {
        "export-surface" => {
            let d: usize = a[2].parse().unwrap();
            let p: f64 = a[3].parse().unwrap();
            let sc = SurfaceCode::new(d, d);
            let text = to_stim(
                &sc.build_circuit(),
                &NoiseModel::circuit_level(p, p),
                &sc.detector_records(),
                &[sc.observable_records()],
            )
            .unwrap();
            std::fs::write(&a[4], text).unwrap();
        }
        "sample" => {
            let prog = load(&a[2]);
            let shots: usize = a[3].parse().unwrap();
            let seed: u64 = a.get(5).map_or(1, |s| s.parse().unwrap());
            let s = compile(
                &prog.circuit,
                &prog.noise,
                &prog.detectors,
                &prog.observables,
            );
            let f = std::fs::File::create(&a[4]).unwrap();
            sample_to(&s, shots, seed, &mut BufWriter::with_capacity(1 << 20, f));
        }
        "sample-native" => {
            let d: usize = a[2].parse().unwrap();
            let p: f64 = a[3].parse().unwrap();
            let shots: usize = a[4].parse().unwrap();
            let seed: u64 = a.get(6).map_or(1, |s| s.parse().unwrap());
            let sc = SurfaceCode::new(d, d);
            let s = compile(
                &sc.build_circuit(),
                &NoiseModel::circuit_level(p, p),
                &sc.detector_records(),
                &[sc.observable_records()],
            );
            let f = std::fs::File::create(&a[5]).unwrap();
            sample_to(&s, shots, seed, &mut BufWriter::with_capacity(1 << 20, f));
        }
        "bench" => {
            let text = std::fs::read_to_string(&a[2]).unwrap();
            let shots: usize = a[3].parse().unwrap();
            let reps: usize = a.get(4).map_or(1, |s| s.parse().unwrap());
            let t0 = Instant::now();
            let prog = parse_stim(&text).unwrap();
            let t_parse = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let s = compile(
                &prog.circuit,
                &prog.noise,
                &prog.detectors,
                &prog.observables,
            );
            let t_compile = t1.elapsed().as_secs_f64();
            let mut best = f64::INFINITY;
            for r in 0..reps {
                let f = std::fs::OpenOptions::new()
                    .write(true)
                    .open("/dev/null")
                    .unwrap();
                let mut w = BufWriter::with_capacity(1 << 20, f);
                let t = Instant::now();
                sample_to(&s, shots, 1000 + r as u64, &mut w);
                best = best.min(t.elapsed().as_secs_f64());
            }
            println!(
                "parse={t_parse:.6} compile={t_compile:.6} sample_min={best:.6} shots={} dets={} vars={} nnz={}",
                shots.div_ceil(64) * 64,
                s.num_measurements(),
                s.num_vars(),
                s.nnz()
            );
        }
        "dem-support" => {
            // every distinct non-empty (detector/observable) signature of a
            // single fault outcome, one per line, rows space-separated
            let prog = load(&a[2]);
            let s = compile(
                &prog.circuit,
                &prog.noise,
                &prog.detectors,
                &prog.observables,
            );
            let mut cols: Vec<Vec<usize>> = vec![Vec::new(); s.num_vars()];
            for j in 0..s.num_measurements() {
                for &v in s.row(j) {
                    cols[v as usize].push(j);
                }
            }
            let mut sigs = std::collections::BTreeSet::new();
            for g in s.groups() {
                for (pat, _) in g.dist.outcomes() {
                    let mut sig = std::collections::BTreeSet::new();
                    for k in 0..g.dist.len() {
                        if pat >> k & 1 == 1 {
                            for &r in &cols[g.first as usize + k] {
                                if !sig.remove(&r) {
                                    sig.insert(r);
                                }
                            }
                        }
                    }
                    if !sig.is_empty() {
                        sigs.insert(sig.into_iter().collect::<Vec<_>>());
                    }
                }
            }
            for sig in sigs {
                println!(
                    "{}",
                    sig.iter()
                        .map(|r| r.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
        }
        m => panic!("unknown mode {m}"),
    }
}
