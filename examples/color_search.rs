//! Colour-code schedule experiments (see research/qec/qec-r4.md).
//!
//! ```text
//! color_search layout <d>
//!     plaquettes: index x y color weight data-per-position(-1 absent)
//! color_search dem <d> <rounds> <cnot|uniform> <p> <schedule> [out]
//!     circuit-derived DEM: "p<TAB>obsmask<TAB>det det ..." per distinct signature
//! color_search export <d> <rounds> <cnot|uniform> <p> <schedule> <out.stim> [x]
//! color_search collisions <d> <schedule>
//! color_search resources <d> <schedule>
//!     qubits, CNOT layers and CNOTs per round, flag slots
//! ```
//! `<schedule>`: see [`parse_schedule_spec`]: `kf`, `tri`, or a file with one
//! line per plaquette `t_a t_b t_c t_d t_e t_f [F]` (absent positions:
//! anything, e.g. 0; `F` = flag qubit); suffix `+bflags` flags all
//! boundary-touching plaquettes.
use qsim_lab::qec::color::{
    circuit_dem, parse_schedule_spec, ColorCode, ColorNoise, ColorSchedule,
};
use qsim_lab::stim_io::to_stim;
use std::io::Write;

fn schedule(cc: &ColorCode, spec: &str) -> (ColorSchedule, Vec<bool>) {
    parse_schedule_spec(cc, spec)
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
        "resources" => {
            let (s, f) = schedule(&cc, &a[3]);
            let r = cc.resources(&s, &f);
            let slots: Vec<String> = cc
                .flag_slots(&s, &f)
                .iter()
                .enumerate()
                .filter_map(|(i, sl)| sl.map(|(a, b)| format!("{i}:{a}/{b}")))
                .collect();
            println!(
                "{{\"d\":{d},\"schedule\":\"{}\",\"data\":{},\"aux\":{},\"flags\":{},\"qubits\":{},\"cnot_layers_per_round\":{},\"cnots_per_round\":{},\"flag_slots\":\"{}\"}}",
                a[3], r.data, r.aux, r.flags, r.data + r.aux + r.flags, r.cnot_layers, r.cnots, slots.join(" ")
            );
        }
        "collisions" => {
            let (s, _) = schedule(&cc, &a[3]);
            for c in cc.collisions(&s) {
                println!("{c}");
            }
        }
        "dem" | "export" => {
            let rounds: usize = a[3].parse().unwrap();
            let p: f64 = a[5].parse().unwrap();
            let (s, f) = schedule(&cc, &a[6]);
            // optional 9th argument "x": X-basis memory
            let x_basis = a.get(8).is_some_and(|b| b == "x");
            let m = cc.memory_flagged(&s, &f, rounds, noise(&a[4], p), x_basis);
            if a[1] == "export" {
                let t = to_stim(&m.circuit, &m.noise, &m.detectors, &m.observables).unwrap();
                std::fs::write(&a[7], t).unwrap();
                // sidecar: per detector "plaquette is_x_type round is_flag"
                let info: String = m
                    .detector_info
                    .iter()
                    .zip(&m.flag_detector)
                    .map(|(i, &f)| format!("{} {} {} {}\n", i.0, i.1 as u8, i.2, f as u8))
                    .collect();
                std::fs::write(format!("{}.info", a[7]), info).unwrap();
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
            let (s, f) = schedule(&cc, &a[4]);
            let cap: u64 = a.get(5).map_or(1_000_000, |x| x.parse().unwrap());
            let node_limit: u64 = a.get(6).map_or(u64::MAX, |x| x.parse().unwrap());
            let noise_kind = a.get(7).map_or("cnot", |x| x.as_str());
            let t = std::time::Instant::now();
            let x_basis = a.get(8).is_some_and(|b| b == "x");
            let m = cc.memory_flagged(&s, &f, rounds, noise(noise_kind, 0.001), x_basis);
            let dem = circuit_dem(&m.circuit, &m.noise, &m.detectors, &m.observables);
            let mut zmap = vec![u32::MAX; m.detectors.len()];
            let mut nz = 0u32;
            for (i, inf) in m.detector_info.iter().enumerate() {
                if inf.1 == x_basis {
                    zmap[i] = nz;
                    nz += 1;
                }
            }
            let mut keys: Vec<(Vec<u32>, bool)> = Vec::new();
            let mut pure = std::collections::HashSet::new();
            let mut seen = std::collections::HashSet::new();
            for e in &dem {
                let zs: Vec<u32> = e
                    .detectors
                    .iter()
                    .filter_map(|&i| {
                        let z = zmap[i as usize];
                        (z != u32::MAX).then_some(z)
                    })
                    .collect();
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
            let r = qsim_lab::qec::distance::min_logical(
                nz as usize,
                &dets,
                &obs,
                4 * d,
                cap,
                node_limit,
            );
            let certified = r.example.iter().all(|&j| pure.contains(&keys[j]));
            // describe the example: per mechanism, the plaquettes (and rounds) of its Z detectors
            let zinfo: Vec<(String, usize)> = m
                .detector_info
                .iter()
                .zip(&m.flag_detector)
                .filter(|(i, _)| i.1 == x_basis)
                .map(|(i, &fl)| {
                    (
                        if fl {
                            format!("f{}", i.0)
                        } else {
                            i.0.to_string()
                        },
                        i.2,
                    )
                })
                .collect();
            let ex: Vec<String> = r
                .example
                .iter()
                .map(|&j| {
                    let v: Vec<String> = keys[j]
                        .0
                        .iter()
                        .map(|&z| format!("{}@{}", zinfo[z as usize].0, zinfo[z as usize].1))
                        .collect();
                    format!("[{}{}]", v.join(" "), if keys[j].1 { " L" } else { "" })
                })
                .collect();
            if std::env::var("DUMP_ALL").is_ok() {
                for sol in &r.all {
                    let mut pl: Vec<String> = sol
                        .iter()
                        .flat_map(|&j| keys[j].0.iter().map(|&z| zinfo[z as usize].0.clone()))
                        .collect();
                    pl.sort_unstable();
                    pl.dedup();
                    let desc: Vec<String> = sol
                        .iter()
                        .map(|&j| {
                            let v: Vec<String> = keys[j]
                                .0
                                .iter()
                                .map(|&z| {
                                    format!("{}@{}", zinfo[z as usize].0, zinfo[z as usize].1)
                                })
                                .collect();
                            format!("[{}{}]", v.join(" "), if keys[j].1 { " L" } else { "" })
                        })
                        .collect();
                    eprintln!("plaquettes {:?} :: {}", pl, desc.join(" "));
                }
            }
            let mut involved: Vec<String> = r
                .all
                .iter()
                .flat_map(|sol| {
                    sol.iter()
                        .flat_map(|&j| keys[j].0.iter().map(|&z| zinfo[z as usize].0.clone()))
                })
                .collect();
            involved.sort_unstable();
            involved.dedup();

            println!(
                "{{\"d\":{d},\"rounds\":{rounds},\"involved\":[{}],\"schedule\":\"{}\",\"noise\":\"{noise_kind}\",\"distance\":{},\"count\":{},\"count_capped\":{},\"certified\":{certified},\"mechanisms\":{},\"z_detectors\":{nz},\"nodes\":{},\"seconds\":{:.3},\"example\":\"{}\"}}",
                involved.iter().map(|x| format!("\"{x}\"")).collect::<Vec<_>>().join(","),
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
