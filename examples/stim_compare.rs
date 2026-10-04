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
//! stim_compare sample-fast <in.stim> <shots> <out.ptb64> [seed]
//! stim_compare sample-native-fast <d> <p> <shots> <out.ptb64> [seed]
//!     as sample / sample-native, with the Poisson-hit FastSampler
//!     (1024-shot batches); optional 5th/6th argument after the seed:
//!     rng = wy (wyrand, default) | xo (Xoshiro256++).
//! stim_compare bench-fast <in.stim> <shots> [reps] [words]
//!     FastSampler only (Xoshiro256++ and wyrand), ptb64 to /dev/null.
//! ```
use qsim_lab::qec::surface::SurfaceCode;
use qsim_lab::stabilizer::fast_sampler::{FastSampler, WyRand};
use qsim_lab::stabilizer::symphase::SymPhaseSampler;
use qsim_lab::stim_io::{parse_stim, to_stim};
use qsim_lab::{Circuit, NoiseModel};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
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

/// Sparse path (only touched variable words cleared, column-wise evaluation),
/// generic RNG.
fn sample_to_sparse<W: Write, R: rand::Rng>(
    s: &SymPhaseSampler,
    cv: &qsim_lab::stabilizer::symphase::ColumnView,
    shots: usize,
    rng: &mut R,
    w: &mut W,
) {
    let rows = s.num_measurements();
    let mut vals = vec![0u64; s.num_vars()];
    let mut touched = Vec::new();
    let mut out = vec![0u64; rows];
    let mut bytes = vec![0u8; rows * 8];
    for _ in 0..shots.div_ceil(64) {
        s.sample_vars_sparse(rng, &mut vals, &mut touched);
        s.eval_sparse(cv, &vals, &touched, &mut out);
        for (k, word) in out.iter().enumerate() {
            bytes[8 * k..8 * k + 8].copy_from_slice(&word.to_le_bytes());
        }
        w.write_all(&bytes).unwrap();
    }
    w.flush().unwrap();
}

const FAST_WORDS: usize = 16;

/// Poisson-hit FastSampler, `words` 64-shot groups per batch, ptb64 out.
fn sample_to_fast<W: Write, R: rand::RngCore>(
    f: &FastSampler,
    shots: usize,
    words: usize,
    rng: &mut R,
    w: &mut W,
) {
    let mut out = vec![0u64; f.stride() * words];
    let mut bytes = Vec::with_capacity(f.rows() * words * 8);
    let groups = shots.div_ceil(64);
    let mut done = 0;
    while done < groups {
        f.sample_batch(rng, &mut out);
        let take = words.min(groups - done);
        bytes.clear();
        f.ptb64(&out, take, &mut bytes);
        w.write_all(&bytes).unwrap();
        done += take;
    }
    w.flush().unwrap();
}

fn devnull() -> BufWriter<std::fs::File> {
    BufWriter::with_capacity(
        1 << 20,
        std::fs::OpenOptions::new()
            .write(true)
            .open("/dev/null")
            .unwrap(),
    )
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
        "sample-fast" | "sample-native-fast" => {
            let (s, rest) = if a[1] == "sample-fast" {
                let prog = load(&a[2]);
                (
                    compile(
                        &prog.circuit,
                        &prog.noise,
                        &prog.detectors,
                        &prog.observables,
                    ),
                    &a[3..],
                )
            } else {
                let d: usize = a[2].parse().unwrap();
                let p: f64 = a[3].parse().unwrap();
                let sc = SurfaceCode::new(d, d);
                (
                    compile(
                        &sc.build_circuit(),
                        &NoiseModel::circuit_level(p, p),
                        &sc.detector_records(),
                        &[sc.observable_records()],
                    ),
                    &a[4..],
                )
            };
            let shots: usize = rest[0].parse().unwrap();
            let seed: u64 = rest.get(2).map_or(1, |s| s.parse().unwrap());
            let f = FastSampler::new(&s);
            let file = std::fs::File::create(&rest[1]).unwrap();
            let mut w = BufWriter::with_capacity(1 << 20, file);
            match rest.get(3).map_or("wy", |s| s.as_str()) {
                "wy" => {
                    // splitmix the seed so nearby seeds give unrelated streams
                    let mut rng = WyRand(rand::rngs::SmallRng::seed_from_u64(seed).next_u64());
                    sample_to_fast(&f, shots, FAST_WORDS, &mut rng, &mut w)
                }
                "xo" => {
                    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
                    sample_to_fast(&f, shots, FAST_WORDS, &mut rng, &mut w)
                }
                r => panic!("unknown rng {r}"),
            }
        }
        "bench-fast" => {
            let text = std::fs::read_to_string(&a[2]).unwrap();
            let shots: usize = a[3].parse().unwrap();
            let reps: usize = a.get(4).map_or(1, |s| s.parse().unwrap());
            let words: usize = a.get(5).map_or(FAST_WORDS, |s| s.parse().unwrap());
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
            let f = FastSampler::new(&s);
            let t_compile = t1.elapsed().as_secs_f64();
            // variants: blocked table path (default), unblocked table path,
            // column-by-column path; Xoshiro256++ (SmallRng) and wyrand
            let mut fu = f.clone();
            fu.set_blocked(false);
            let mut fc = f.clone();
            fc.force_column_path();
            let mut fw32 = f.clone();
            fw32.set_narrow(false);
            let variants: [(&str, &FastSampler); 4] = [
                ("blocked", &f),
                ("blocked32", &fw32),
                ("unblocked", &fu),
                ("column", &fc),
            ];
            let mut best = [[f64::INFINITY; 2]; 4];
            for r in 0..reps {
                for (i, (_, v)) in variants.iter().enumerate() {
                    let mut w = devnull();
                    let mut rng = rand::rngs::SmallRng::seed_from_u64(4000 + r as u64);
                    let t = Instant::now();
                    sample_to_fast(v, shots, words, &mut rng, &mut w);
                    best[i][0] = best[i][0].min(t.elapsed().as_secs_f64());
                    let mut w = devnull();
                    let mut rng = WyRand(5000 + r as u64);
                    let t = Instant::now();
                    sample_to_fast(v, shots, words, &mut rng, &mut w);
                    best[i][1] = best[i][1].min(t.elapsed().as_secs_f64());
                }
            }
            let mut line = format!("parse={t_parse:.6} compile={t_compile:.6}");
            for (i, (name, _)) in variants.iter().enumerate() {
                line += &format!(
                    " {name}_xoshiro={:.6} {name}_wyrand={:.6}",
                    best[i][0], best[i][1]
                );
            }
            println!(
                "{line} fast_min={:.6} words={words} shots={} dets={} classes_dense={:?} hits_per_shot={:.2} xors_per_hit={:.2}",
                best[0][0],
                shots.div_ceil(64) * 64,
                f.rows(),
                f.layout(),
                f.hit_stats().0,
                f.hit_stats().1
            );
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
            let t2 = Instant::now();
            let cv = s.column_view();
            let t_cv = t2.elapsed().as_secs_f64();
            let (mut best, mut best_sp, mut best_sp_small) =
                (f64::INFINITY, f64::INFINITY, f64::INFINITY);
            for r in 0..reps {
                let mut w = devnull();
                let t = Instant::now();
                sample_to(&s, shots, 1000 + r as u64, &mut w);
                best = best.min(t.elapsed().as_secs_f64());
                let mut w = devnull();
                let mut rng = StdRng::seed_from_u64(2000 + r as u64);
                let t = Instant::now();
                sample_to_sparse(&s, &cv, shots, &mut rng, &mut w);
                best_sp = best_sp.min(t.elapsed().as_secs_f64());
                let mut w = devnull();
                let mut rng = rand::rngs::SmallRng::seed_from_u64(3000 + r as u64);
                let t = Instant::now();
                sample_to_sparse(&s, &cv, shots, &mut rng, &mut w);
                best_sp_small = best_sp_small.min(t.elapsed().as_secs_f64());
            }
            // sample_min: original dense path, StdRng (ChaCha12); sparse_*: sparse
            // draw + column-wise evaluation (bit-identical output for the same RNG stream)
            println!(
                "parse={t_parse:.6} compile={t_compile:.6} colview={t_cv:.6} sample_min={best:.6} sparse_min={best_sp:.6} sparse_smallrng_min={best_sp_small:.6} shots={} dets={} vars={} nnz={}",
                shots.div_ceil(64) * 64,
                s.num_measurements(),
                s.num_vars(),
                s.nnz()
            );
        }
        "profile" => {
            // split sampling time: drawing the variables vs the sparse GF(2) evaluation
            let prog = load(&a[2]);
            let shots: usize = a[3].parse().unwrap();
            let s = compile(
                &prog.circuit,
                &prog.noise,
                &prog.detectors,
                &prog.observables,
            );
            let mut rng = StdRng::seed_from_u64(5);
            let mut vals = vec![0u64; s.num_vars()];
            let mut out = vec![0u64; s.num_measurements()];
            let batches = shots.div_ceil(64);
            let coins = s
                .groups()
                .iter()
                .filter(|g| matches!(g.dist, qsim_lab::stabilizer::symphase::VarDist::Coin))
                .count();
            let t = Instant::now();
            for _ in 0..batches {
                s.sample_vars(&mut rng, &mut vals);
            }
            let tv = t.elapsed().as_secs_f64();
            let mut acc = 0u64;
            let t = Instant::now();
            for _ in 0..batches {
                s.eval(&vals, &mut out);
                acc ^= out[0];
            }
            let te = t.elapsed().as_secs_f64();
            let gate_ops = prog
                .circuit
                .ops
                .iter()
                .filter(|o| {
                    matches!(
                        o,
                        qsim_lab::Op::Gate(_) | qsim_lab::Op::Measure(_) | qsim_lab::Op::Reset(_)
                    )
                })
                .count();
            let noise_ops = prog.circuit.ops.len() - gate_ops;
            println!(
                "sample_vars={tv:.4} eval={te:.4} vars={} groups={} coin_groups={coins} nnz={} rows={} circuit_ops={} noise_ops={} ({acc})",
                s.num_vars(),
                s.groups().len(),
                s.nnz(),
                s.num_measurements(),
                gate_ops,
                noise_ops
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
