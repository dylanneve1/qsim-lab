//! Exact minimum-weight logical of a DEM given on stdin (used by the global
//! colour-code schedule search, `research/data/colour-global/`).
//!
//! Protocol (repeatable, so one process serves many queries):
//! ```text
//! P <num_detectors> <max_weight> <count_cap> <num_mechanisms>
//! <obs 0|1> <det> <det> ...        (one line per mechanism)
//! ```
//! Answer, one line each:
//! ```text
//! R <weight|none> <count> <nodes> <num_listed>
//! L <mech> <mech> ...              (each listed minimum-weight logical)
//! ```
use qsim_lab::qec::distance::min_logical;
use std::io::{BufRead, Write};

fn main() {
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let out = std::io::stdout();
    let mut out = std::io::BufWriter::new(out.lock());
    while let Some(Ok(head)) = lines.next() {
        let h: Vec<&str> = head.split_whitespace().collect();
        if h.is_empty() {
            continue;
        }
        assert_eq!(h[0], "P", "expected a problem header");
        let nd: usize = h[1].parse().unwrap();
        let maxw: usize = h[2].parse().unwrap();
        let cap: u64 = h[3].parse().unwrap();
        let nm: usize = h[4].parse().unwrap();
        let mut dets = Vec::with_capacity(nm);
        let mut obs = Vec::with_capacity(nm);
        for _ in 0..nm {
            let l = lines.next().unwrap().unwrap();
            let mut it = l.split_whitespace();
            obs.push(it.next().unwrap() == "1");
            let mut v: Vec<u32> = it.map(|x| x.parse().unwrap()).collect();
            v.sort_unstable();
            dets.push(v);
        }
        let r = min_logical(nd, &dets, &obs, maxw, cap, u64::MAX);
        writeln!(
            out,
            "R {} {} {} {}",
            r.weight.map_or("none".to_string(), |w| w.to_string()),
            r.count,
            r.nodes,
            r.all.len()
        )
        .unwrap();
        for l in &r.all {
            let s: Vec<String> = l.iter().map(|x| x.to_string()).collect();
            writeln!(out, "L {}", s.join(" ")).unwrap();
        }
        out.flush().unwrap();
    }
}
