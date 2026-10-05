//! Logical error rate of a colour-code memory under a given schedule:
//! SymPhase sampling + BP+OSD-CS decoding of the Z sector of the
//! circuit-derived DEM (X and Z sectors decoded independently, as usual for
//! CSS codes).
//!
//! ```text
//! color_ler <d> <rounds> <cnot|uniform> <p> <schedule-spec> <shots> <seed> [threads] [osd_order]
//! (schedule spec: see `qec::color::parse_schedule_spec`; flags supported)
//! ```
use qsim_lab::engines::stabilizer::symphase::SymPhaseSampler;
use qsim_lab::qec::bposd::{BpOsd, DecodeStats, DemMatrix};
use qsim_lab::qec::color::{circuit_dem, parse_schedule_spec, ColorCode, ColorNoise};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d: usize = a[1].parse().unwrap();
    let rounds: usize = a[2].parse().unwrap();
    let p: f64 = a[4].parse().unwrap();
    let noise = match a[3].as_str() {
        "cnot" => ColorNoise::Cnot(p),
        "uniform" => ColorNoise::Uniform(p),
        k => panic!("{k}"),
    };
    let shots: usize = a[6].parse().unwrap();
    let seed: u64 = a[7].parse().unwrap();
    let threads: usize = a.get(8).map_or(1, |s| s.parse().unwrap());
    let order: usize = a.get(9).map_or(10, |s| s.parse().unwrap());
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    let cc = ColorCode::new(d);
    let (s, flags) = parse_schedule_spec(&cc, &a[5]);
    assert!(cc.collisions(&s).is_empty(), "schedule has collisions");
    // BASIS=x: X-basis memory (decode the X-type sector)
    let x_basis = std::env::var("BASIS").is_ok_and(|b| b == "x");
    let m = cc.memory_flagged(&s, &flags, rounds, noise, x_basis);
    let t0 = Instant::now();
    // Z sector
    // FULL=1: decode with all detectors (X and Z type; keeps Y correlations), else Z sector only
    let full = std::env::var("FULL").is_ok();
    let zdet: Vec<usize> = (0..m.detectors.len())
        .filter(|&i| full || m.detector_info[i].1 == x_basis)
        .collect();
    let mut zmap = vec![u32::MAX; m.detectors.len()];
    for (k, &i) in zdet.iter().enumerate() {
        zmap[i] = k as u32;
    }
    let dem = circuit_dem(&m.circuit, &m.noise, &m.detectors, &m.observables);
    let mut merged: HashMap<(Vec<u32>, u64), f64> = HashMap::new();
    for e in &dem {
        let zs: Vec<u32> = e
            .detectors
            .iter()
            .filter_map(|&i| {
                let z = zmap[i as usize];
                (z != u32::MAX).then_some(z)
            })
            .collect();
        if zs.is_empty() {
            assert_eq!(e.observables, 0, "undetectable logical mechanism");
            continue;
        }
        let v = merged.entry((zs, e.observables)).or_insert(0.0);
        *v = *v * (1.0 - e.p) + e.p * (1.0 - *v);
    }
    let mut ents: Vec<_> = merged.into_iter().collect();
    ents.sort_by(|x, y| x.0.cmp(&y.0));
    let dm = DemMatrix {
        num_detectors: zdet.len(),
        cols: ents.iter().map(|e| e.0 .0.clone()).collect(),
        obs: ents.iter().map(|e| e.0 .1).collect(),
        p: ents.iter().map(|e| e.1).collect(),
    };
    let dec = BpOsd::new(dm, 50, 0.625, order);
    let sets: Vec<Vec<usize>> = zdet
        .iter()
        .map(|&i| m.detectors[i].clone())
        .chain(m.observables.iter().cloned())
        .collect();
    let smp = SymPhaseSampler::new(&m.circuit, &m.noise)
        .unwrap()
        .with_parities(&sets);
    let t_setup = t0.elapsed().as_secs_f64();
    let nz = zdet.len();
    let batches = shots.div_ceil(64);
    let chunk = 64usize; // batches per task
    let tasks: Vec<usize> = (0..batches.div_ceil(chunk)).collect();
    let t1 = Instant::now();
    let res: Vec<(u64, u64, DecodeStats)> = tasks
        .par_iter()
        .map(|&t| {
            let mut rng =
                StdRng::seed_from_u64(seed.wrapping_mul(1_000_003).wrapping_add(t as u64));
            let mut vals = vec![0u64; smp.num_vars()];
            let mut out = vec![0u64; nz + 1];
            let mut sc = dec.scratch();
            let mut st = DecodeStats::default();
            let (mut fails, mut n) = (0u64, 0u64);
            let mut fired: Vec<Vec<u32>> = vec![Vec::new(); 64];
            let nb = chunk.min(batches - t * chunk);
            for _ in 0..nb {
                smp.sample_batch(&mut rng, &mut vals, &mut out);
                for f in fired.iter_mut() {
                    f.clear();
                }
                for (r, &w) in out[..nz].iter().enumerate() {
                    let mut x = w;
                    while x != 0 {
                        fired[x.trailing_zeros() as usize].push(r as u32);
                        x &= x - 1;
                    }
                }
                for (sh, f) in fired.iter().enumerate() {
                    let actual = out[nz] >> sh & 1;
                    let pred = dec.decode(f, &mut sc, &mut st) & 1;
                    fails += (pred != actual) as u64;
                    n += 1;
                }
            }
            (fails, n, st)
        })
        .collect();
    let (mut fails, mut n, mut st) = (0u64, 0u64, DecodeStats::default());
    for (f, k, s) in res {
        fails += f;
        n += k;
        st.bp_converged += s.bp_converged;
        st.osd_calls += s.osd_calls;
    }
    let pl = fails as f64 / n as f64;
    let z = 1.96f64;
    let den = 1.0 + z * z / n as f64;
    let c = (pl + z * z / (2.0 * n as f64)) / den;
    let h = z * (pl * (1.0 - pl) / n as f64 + z * z / (4.0 * (n * n) as f64)).sqrt() / den;
    let per_round = |x: f64| (1.0 - (1.0 - 2.0 * x).max(0.0).powf(1.0 / rounds as f64)) / 2.0;
    println!(
        "{{\"d\":{d},\"rounds\":{rounds},\"noise\":\"{}\",\"p\":{p},\"schedule\":\"{}\",\"shots\":{n},\"fails\":{fails},\"p_L\":{pl:.6e},\"ci95\":[{:.6e},{:.6e}],\"p_L_round\":{:.6e},\"ci95_round\":[{:.6e},{:.6e}],\"mechanisms_z\":{},\"z_detectors\":{nz},\"bp_converged\":{},\"osd_calls\":{},\"osd_order\":{order},\"setup_s\":{t_setup:.2},\"decode_s\":{:.2},\"threads\":{threads}}}",
        a[3], a[5], c - h, c + h, per_round(pl), per_round(c - h), per_round(c + h),
        dec.num_mechanisms(), st.bp_converged, st.osd_calls, t1.elapsed().as_secs_f64()
    );
}
