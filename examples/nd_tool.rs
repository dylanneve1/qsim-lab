//! Data plumbing for the neural-decoder study (`research/neural-decoder.md`).
//!
//! ```text
//! nd_tool color-export <d> <rounds> <cnot|uniform> <p> <kf|tri|sched-file> <prefix> [x]
//!     writes <prefix>.stim (to_stim of the native circuit), <prefix>.dem (our circuit DEM,
//!     tab format "p \t obs \t dets", header "#x <X-type dets>" and "# detectors N") and
//!     <prefix>.meta (one line per detector: index plaquette_x plaquette_y round is_x colour).
//! nd_tool stream <in.stim> <seed> [shots]
//!     FastSampler (wyrand) detection events + observables in Stim's ptb64 layout on stdout,
//!     1024-shot blocks; shots = 0 or absent streams forever (training data).
//! nd_tool bposd <dem> <shots.ptb64> <pred.out> [threads] [osd_order] [zonly]
//!     our BP+OSD-CS (50 iterations, min-sum 0.625) on every shot of a ptb64 file
//!     (rows = detectors + 1 observable); writes one predicted-observable byte per shot and
//!     prints a JSON summary. zonly = decode with the non-#x detectors only (as color_ler).
//! ```
use qsim_lab::qec::bposd::{BpOsd, DecodeStats, DemMatrix};
use qsim_lab::qec::color::{
    circuit_dem, ColorCode, ColorNoise, ColorSchedule, KF_SCHEDULE, TRI_OPTIMAL,
};
use qsim_lab::stabilizer::fast_sampler::{FastSampler, WyRand};
use qsim_lab::stabilizer::symphase::SymPhaseSampler;
use qsim_lab::stim_io::{parse_stim, to_stim};
use rand::{RngCore, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::time::Instant;

fn schedule(cc: &ColorCode, spec: &str) -> ColorSchedule {
    match spec {
        "kf" => cc.uniform_schedule(KF_SCHEDULE),
        "tri" => cc.uniform_schedule([TRI_OPTIMAL; 3]),
        path => std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let v: Vec<u8> = l.split_whitespace().map(|t| t.parse().unwrap()).collect();
                [v[0], v[1], v[2], v[3], v[4], v[5]]
            })
            .collect(),
    }
}

struct Dem {
    nd: usize,
    xdet: Vec<bool>,
    ents: Vec<(f64, u64, Vec<u32>)>,
}

fn read_dem(path: &str) -> Dem {
    let text = std::fs::read_to_string(path).unwrap();
    let mut nd = 0;
    let mut xs = Vec::new();
    let mut ents = Vec::new();
    for l in text.lines() {
        if let Some(r) = l.strip_prefix("#x") {
            xs = r
                .split_whitespace()
                .map(|t| t.parse::<usize>().unwrap())
                .collect();
        } else if l.starts_with('#') {
            nd = l.split_whitespace().last().unwrap().parse().unwrap();
        } else if !l.trim().is_empty() {
            let f: Vec<&str> = l.split('\t').collect();
            let ds = f[2]
                .split_whitespace()
                .map(|t| t.parse().unwrap())
                .collect();
            ents.push((f[0].parse().unwrap(), f[1].parse().unwrap(), ds));
        }
    }
    let mut xdet = vec![false; nd];
    for x in xs {
        xdet[x] = true;
    }
    Dem { nd, xdet, ents }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a[1].as_str() {
        "color-export" => {
            let d: usize = a[2].parse().unwrap();
            let rounds: usize = a[3].parse().unwrap();
            let p: f64 = a[5].parse().unwrap();
            let noise = match a[4].as_str() {
                "cnot" => ColorNoise::Cnot(p),
                "uniform" => ColorNoise::Uniform(p),
                k => panic!("{k}"),
            };
            let cc = ColorCode::new(d);
            let s = schedule(&cc, &a[6]);
            assert!(cc.collisions(&s).is_empty(), "schedule has collisions");
            let x_basis = a.get(8).is_some_and(|b| b == "x");
            let m = cc.memory_basis(&s, rounds, noise, x_basis);
            let pre = &a[7];
            let t = to_stim(&m.circuit, &m.noise, &m.detectors, &m.observables).unwrap();
            std::fs::write(format!("{pre}.stim"), t).unwrap();
            let dem = circuit_dem(&m.circuit, &m.noise, &m.detectors, &m.observables);
            let mut out = BufWriter::new(std::fs::File::create(format!("{pre}.dem")).unwrap());
            let xd: Vec<String> = m
                .detector_info
                .iter()
                .enumerate()
                .filter(|(_, i)| i.1)
                .map(|(k, _)| k.to_string())
                .collect();
            writeln!(out, "#x {}", xd.join(" ")).unwrap();
            writeln!(out, "# detectors {}", m.detectors.len()).unwrap();
            for e in dem {
                let ds: Vec<String> = e.detectors.iter().map(|x| x.to_string()).collect();
                writeln!(out, "{:e}\t{}\t{}", e.p, e.observables, ds.join(" ")).unwrap();
            }
            let mut meta = BufWriter::new(std::fs::File::create(format!("{pre}.meta")).unwrap());
            for (k, &(pi, isx, r)) in m.detector_info.iter().enumerate() {
                let pl = &cc.plaquettes[pi];
                writeln!(meta, "{k} {} {} {r} {} {}", pl.x, pl.y, isx as u8, pl.color).unwrap();
            }
        }
        "stream" => {
            let prog = parse_stim(&std::fs::read_to_string(&a[2]).unwrap()).unwrap();
            let seed: u64 = a[3].parse().unwrap();
            let shots: usize = a.get(4).map_or(0, |s| s.parse().unwrap());
            let sets: Vec<Vec<usize>> = prog
                .detectors
                .iter()
                .chain(prog.observables.iter())
                .cloned()
                .collect();
            let s = SymPhaseSampler::new(&prog.circuit, &prog.noise)
                .expect("compile")
                .with_parities(&sets)
                .relative_to_reference();
            let f = FastSampler::new(&s);
            let mut rng = WyRand(rand::rngs::SmallRng::seed_from_u64(seed).next_u64());
            let words = 16usize;
            let mut out = vec![0u64; f.stride() * words];
            let mut bytes = Vec::with_capacity(f.rows() * words * 8);
            let mut w = BufWriter::with_capacity(1 << 20, std::io::stdout().lock());
            let blocks = shots.div_ceil(64);
            let mut done = 0usize;
            loop {
                if shots > 0 && done >= blocks {
                    break;
                }
                f.sample_batch(&mut rng, &mut out);
                let take = if shots > 0 {
                    words.min(blocks - done)
                } else {
                    words
                };
                bytes.clear();
                f.ptb64(&out, take, &mut bytes);
                if w.write_all(&bytes).is_err() {
                    return; // reader closed the pipe
                }
                done += take;
            }
            w.flush().unwrap();
        }
        "bposd" => {
            let dem = read_dem(&a[2]);
            let threads: usize = a.get(5).map_or(1, |s| s.parse().unwrap());
            let order: usize = a.get(6).map_or(10, |s| s.parse().unwrap());
            let zonly = a.get(7).is_some_and(|s| s == "zonly");
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .unwrap();
            let keep: Vec<usize> = (0..dem.nd).filter(|&i| !zonly || !dem.xdet[i]).collect();
            let mut map = vec![u32::MAX; dem.nd];
            for (k, &i) in keep.iter().enumerate() {
                map[i] = k as u32;
            }
            let mut merged: HashMap<(Vec<u32>, u64), f64> = HashMap::new();
            for (p, ob, ds) in &dem.ents {
                let mut zs: Vec<u32> = ds
                    .iter()
                    .filter_map(|&i| (map[i as usize] != u32::MAX).then_some(map[i as usize]))
                    .collect();
                zs.sort();
                if zs.is_empty() {
                    continue;
                }
                let v = merged.entry((zs, *ob)).or_insert(0.0);
                *v = *v * (1.0 - p) + p * (1.0 - *v);
            }
            let mut ents: Vec<_> = merged.into_iter().collect();
            ents.sort_by(|x, y| x.0.cmp(&y.0));
            let dm = DemMatrix {
                num_detectors: keep.len(),
                cols: ents.iter().map(|e| e.0 .0.clone()).collect(),
                obs: ents.iter().map(|e| e.0 .1).collect(),
                p: ents.iter().map(|e| e.1).collect(),
            };
            let dec = BpOsd::new(dm, 50, 0.625, order);
            let raw = std::fs::read(&a[3]).unwrap();
            let rows = dem.nd + 1;
            let words: Vec<u64> = raw
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            assert_eq!(words.len() % rows, 0);
            let blocks = words.len() / rows;
            let t0 = Instant::now();
            let res: Vec<(Vec<u8>, u64, DecodeStats)> = (0..blocks)
                .into_par_iter()
                .with_min_len(16)
                .map(|b| {
                    let blk = &words[b * rows..(b + 1) * rows];
                    let mut sc = dec.scratch();
                    let mut st = DecodeStats::default();
                    let mut fired: Vec<Vec<u32>> = vec![Vec::new(); 64];
                    for (k, &i) in keep.iter().enumerate() {
                        let mut x = blk[i];
                        while x != 0 {
                            fired[x.trailing_zeros() as usize].push(k as u32);
                            x &= x - 1;
                        }
                    }
                    let mut pred = vec![0u8; 64];
                    let mut fails = 0;
                    for sh in 0..64 {
                        let pr = (dec.decode(&fired[sh], &mut sc, &mut st) & 1) as u8;
                        pred[sh] = pr;
                        fails += (pr as u64 != (blk[dem.nd] >> sh & 1)) as u64;
                    }
                    (pred, fails, st)
                })
                .collect();
            let mut out = BufWriter::new(std::fs::File::create(&a[4]).unwrap());
            let (mut fails, mut conv, mut osd) = (0u64, 0u64, 0u64);
            for (p, f, st) in &res {
                out.write_all(p).unwrap();
                fails += f;
                conv += st.bp_converged;
                osd += st.osd_calls;
            }
            println!(
                "{{\"decoder\":\"bposd\",\"shots\":{},\"fails\":{fails},\"dets\":{},\"mechanisms\":{},\"osd_order\":{order},\"bp_converged\":{conv},\"osd_calls\":{osd},\"decode_s\":{:.2},\"threads\":{threads}}}",
                blocks * 64, keep.len(), dec.num_mechanisms(), t0.elapsed().as_secs_f64()
            );
        }
        c => panic!("unknown command {c}"),
    }
}
