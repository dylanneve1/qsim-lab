//! Colour-code schedule experiments (see research/qec-r4.md).
//!
//! ```text
//! color_search layout <d>
//!     plaquettes: index x y color weight data-per-position(-1 absent)
//! color_search dem <d> <rounds> <cnot|uniform> <p> <schedule> [out]
//!     circuit-derived DEM: "p<TAB>obsmask<TAB>det det ..." per distinct signature
//! color_search export <d> <rounds> <cnot|uniform> <p> <schedule> <out.stim>
//! color_search collisions <d> <schedule>
//! ```
//! `<schedule>`: `kf`, `tri`, or a file with one line per plaquette
//! `t_a t_b t_c t_d t_e t_f` (absent positions: anything, e.g. 0).
use qsim_lab::qec::color::{
    circuit_dem, ColorCode, ColorNoise, ColorSchedule, KF_SCHEDULE, TRI_OPTIMAL,
};
use qsim_lab::stim_io::to_stim;
use std::io::Write;

fn schedule(cc: &ColorCode, spec: &str) -> ColorSchedule {
    match spec {
        "kf" => cc.uniform_schedule(KF_SCHEDULE),
        "tri" => cc.uniform_schedule([TRI_OPTIMAL; 3]),
        path => {
            let text = std::fs::read_to_string(path).expect("schedule file");
            let s: ColorSchedule = text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| {
                    let v: Vec<u8> = l.split_whitespace().map(|t| t.parse().unwrap()).collect();
                    [v[0], v[1], v[2], v[3], v[4], v[5]]
                })
                .collect();
            assert_eq!(s.len(), cc.plaquettes.len(), "schedule lines != plaquettes");
            s
        }
    }
}

fn noise(kind: &str, p: f64) -> ColorNoise {
    match kind {
        "cnot" => ColorNoise::Cnot(p),
        "uniform" => ColorNoise::Uniform(p),
        k => panic!("noise {k}"),
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d: usize = a[2].parse().unwrap();
    let cc = ColorCode::new(d);
    match a[1].as_str() {
        "layout" => {
            for (i, p) in cc.plaquettes.iter().enumerate() {
                let dq: Vec<String> = p
                    .data
                    .iter()
                    .map(|q| q.map_or("-1".to_string(), |q| q.to_string()))
                    .collect();
                println!(
                    "{i} {} {} {} {} {}",
                    p.x,
                    p.y,
                    p.color,
                    p.weight(),
                    dq.join(" ")
                );
            }
        }
        "collisions" => {
            let s = schedule(&cc, &a[3]);
            for c in cc.collisions(&s) {
                println!("{c}");
            }
        }
        "dem" | "export" => {
            let rounds: usize = a[3].parse().unwrap();
            let p: f64 = a[5].parse().unwrap();
            let s = schedule(&cc, &a[6]);
            let m = cc.memory(&s, rounds, noise(&a[4], p));
            if a[1] == "export" {
                let t = to_stim(&m.circuit, &m.noise, &m.detectors, &m.observables).unwrap();
                std::fs::write(&a[7], t).unwrap();
                return;
            }
            let dem = circuit_dem(&m.circuit, &m.noise, &m.detectors, &m.observables);
            let mut out: Box<dyn Write> = match a.get(7) {
                Some(f) => Box::new(std::io::BufWriter::new(std::fs::File::create(f).unwrap())),
                None => Box::new(std::io::BufWriter::new(std::io::stdout().lock())),
            };
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
        }
        "distance" => {
            // exact Z-memory circuit distance (Z sector + twin certification), noisy-CNOT model
            let rounds: usize = a[3].parse().unwrap();
            let s = schedule(&cc, &a[4]);
            let cap: u64 = a.get(5).map_or(1_000_000, |x| x.parse().unwrap());
            let node_limit: u64 = a.get(6).map_or(u64::MAX, |x| x.parse().unwrap());
            let noise_kind = a.get(7).map_or("cnot", |x| x.as_str());
            let t = std::time::Instant::now();
            let m = cc.memory(&s, rounds, noise(noise_kind, 0.001));
            let dem = circuit_dem(&m.circuit, &m.noise, &m.detectors, &m.observables);
            let mut zmap = vec![u32::MAX; m.detectors.len()];
            let mut nz = 0u32;
            for (i, inf) in m.detector_info.iter().enumerate() {
                if !inf.1 {
                    zmap[i] = nz;
                    nz += 1;
                }
            }
            let mut keys: Vec<(Vec<u32>, bool)> = Vec::new();
            let mut pure = std::collections::HashSet::new();
            let mut seen = std::collections::HashSet::new();
            for e in &dem {
                let zs: Vec<u32> = e.detectors.iter().filter_map(|&i| {
                    let z = zmap[i as usize];
                    (z != u32::MAX).then_some(z)
                }).collect();
                let ob = e.observables & 1 == 1;
                if zs.is_empty() && !ob {
                    continue;
                }
                let k = (zs.clone(), ob);
                if zs.len() == e.detectors.len() {
                    pure.insert(k.clone());
                }
                if seen.insert(k.clone()) {
                    keys.push(k);
                }
            }
            let dets: Vec<Vec<u32>> = keys.iter().map(|k| k.0.clone()).collect();
            let obs: Vec<bool> = keys.iter().map(|k| k.1).collect();
            let r = qsim_lab::qec::distance::min_logical(nz as usize, &dets, &obs, 4 * d, cap, node_limit);
            let certified = r.example.iter().all(|&j| pure.contains(&keys[j]));
            // describe the example: per mechanism, the plaquettes (and rounds) of its Z detectors
            let zinfo: Vec<(usize, usize)> = m.detector_info.iter().filter(|i| !i.1).map(|i| (i.0, i.2)).collect();
            let ex: Vec<String> = r.example.iter().map(|&j| {
                let v: Vec<String> = keys[j].0.iter().map(|&z| format!("{}@{}", zinfo[z as usize].0, zinfo[z as usize].1)).collect();
                format!("[{}{}]", v.join(" "), if keys[j].1 { " L" } else { "" })
            }).collect();
            println!(
                "{{\"d\":{d},\"rounds\":{rounds},\"schedule\":\"{}\",\"noise\":\"{noise_kind}\",\"distance\":{},\"count\":{},\"count_capped\":{},\"certified\":{certified},\"mechanisms\":{},\"z_detectors\":{nz},\"nodes\":{},\"seconds\":{:.3},\"example\":\"{}\"}}",
                a[4],
                r.weight.map_or("null".to_string(), |w| w.to_string()),
                r.count,
                r.count >= cap,
                keys.len(),
                r.nodes,
                t.elapsed().as_secs_f64(),
                ex.join(" ")
            );
        }
        m => panic!("mode {m}"),
    }
}
