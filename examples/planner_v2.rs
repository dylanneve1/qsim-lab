//! Driver for the Planner v2 study (research/planner-v2.md).
//!
//! ```text
//! planner_v2 req ENGINE SPEC SEED [MEM]   -> evolve once, then time every read-out:
//!     e (Z on every qubit), a1 / a1k (1 / 1000 amplitudes), prep (MPS canonical
//!     form, HSF full output, compressed sampler), s1 / s1k / s100k (shots)
//! planner_v2 plan VARIANT REQ SPEC SEED [MEM]  -> plan + execute in process
//!     VARIANT: v2 | v2nc (no cache) | v1 | rule (old hand rules); REQ: e s1 s1k s100k a1 a1k
//! planner_v2 feat FILE              -> per line "SPEC SEED": every planner feature
//!     (all tiers forced), the time of each tier, and the v1 / rule choices
//! ```

use qsim_lab::compile::plan::AdaptiveRule;
use qsim_lab::mps_cost::{replay, BondSource, Estimator};
use qsim_lab::planner::{
    self, mps_readout_units, mps_work_log2, predict_secs, quick_features, Engine, PlanFeatures,
    PlanRequest, PlannerConfig, Prepared,
};
use qsim_lab::simulability::{self, build, Spec};
use qsim_lab::Circuit;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

const REQS: [&str; 6] = ["e", "s1", "s1k", "s100k", "a1", "a1k"];

fn request(name: &str, n: usize) -> PlanRequest {
    match name {
        "e" => PlanRequest::Expectation((0..n).collect()),
        "s1" => PlanRequest::Samples(1),
        "s1k" => PlanRequest::Samples(1000),
        "s100k" => PlanRequest::Samples(100_000),
        "a1" => PlanRequest::Amplitudes(1),
        "a1k" => PlanRequest::Amplitudes(1000),
        _ => panic!("unknown request {name}"),
    }
}

fn xs_for(n: usize, m: usize, seed: u64) -> Vec<u128> {
    let mut rng = StdRng::seed_from_u64(seed ^ 0xa5a5);
    let mask = if n >= 128 {
        u128::MAX
    } else {
        (1u128 << n) - 1
    };
    (0..m).map(|_| rng.random::<u128>() & mask).collect()
}

fn parity_sum(it: impl Iterator<Item = (u128, f64)>, mask: u128) -> f64 {
    it.map(|(x, p)| {
        if (x & mask).count_ones() & 1 == 1 {
            -p
        } else {
            p
        }
    })
    .sum()
}

fn j(x: Option<f64>) -> String {
    x.map_or("null".into(), |v| format!("{v:.7}"))
}

/// HSF amplitudes only (set-up, then the path sums for 1 and 1000 basis
/// states); the full-output time of the expectation sweep says nothing about
/// them, so they are timed separately (process timeout = censoring).
fn cmd_hsfamp(c: &Circuit, mem: u128, seed: u64) -> String {
    let n = c.num_qubits;
    let cfg = PlannerConfig {
        mem_bytes: mem,
        ..PlannerConfig::default()
    };
    let t0 = Instant::now();
    let mut p = match planner::prepare(Engine::Hsf, c, &cfg, None) {
        Ok(Some(p)) => p,
        Ok(None) => return "{\"ok\":false,\"error\":\"aborted\"}".into(),
        Err(err) => {
            return format!(
                "{{\"ok\":false,\"error\":\"{}\"}}",
                format!("{err:?}").replace('"', "'")
            )
        }
    };
    let evolve = t0.elapsed().as_secs_f64();
    let mut ta = [None, None];
    for (i, m) in [1usize, 1000].into_iter().enumerate() {
        let xs = xs_for(n, m, seed);
        let t = Instant::now();
        if p.amplitudes(&xs).is_ok() {
            ta[i] = Some(t.elapsed().as_secs_f64());
        }
    }
    let paths = match &p {
        Prepared::Hsf(h, _) => h.num_paths() as f64,
        _ => 0.0,
    };
    format!(
        "{{\"ok\":true,\"hsf_paths\":{paths},\"evolve\":{evolve:.7},\"a1\":{},\"a1k\":{}}}",
        j(ta[0]),
        j(ta[1])
    )
}

fn cmd_req(engine: &str, c: &Circuit, mem: u128, seed: u64) -> String {
    if engine == "hsfamp" {
        return cmd_hsfamp(c, mem, seed);
    }
    let n = c.num_qubits;
    let e = Engine::from_name(engine).expect("engine");
    let cfg = PlannerConfig {
        mem_bytes: mem,
        ..PlannerConfig::default()
    };
    let t0 = Instant::now();
    let p = planner::prepare(e, c, &cfg, Some(12.0));
    let evolve = t0.elapsed().as_secs_f64();
    let mut p = match p {
        Ok(Some(p)) => p,
        Ok(None) => {
            return format!("{{\"ok\":false,\"error\":\"aborted\",\"evolve\":{evolve:.6}}}")
        }
        Err(err) => {
            return format!(
                "{{\"ok\":false,\"error\":\"{}\",\"evolve\":{evolve:.6}}}",
                format!("{err:?}").replace('"', "'")
            )
        }
    };
    let mask: u128 = if n >= 128 {
        u128::MAX
    } else {
        (1u128 << n) - 1
    };
    let all: Vec<usize> = (0..n).collect();
    let mut size = String::new();
    // expectation read-out (where the state is directly readable)
    let te = Instant::now();
    let ev: Option<f64> = match &mut p {
        Prepared::Sv(sv) => Some(parity_sum(
            sv.amplitudes()
                .iter()
                .enumerate()
                .map(|(x, a)| (x as u128, a.norm_sqr())),
            mask,
        )),
        Prepared::Sparse(s) => {
            size = format!("\"nnz\":{},\"peak_nnz\":{},", s.nnz(), s.peak_nnz());
            Some(parity_sum(
                s.iter().map(|(x, a)| (u128::from(x), a.norm_sqr())),
                mask,
            ))
        }
        Prepared::Mps(m) => {
            let b = m.bond_dims();
            let (cu, su, au) = mps_readout_units(&b);
            size = format!(
                "\"max_bond\":{},\"canon_u\":{cu:.6e},\"shot_u\":{su:.6e},\"amp_u\":{au:.6e},",
                m.max_bond_dim()
            );
            Some(m.expectation_z_product(&all))
        }
        Prepared::Compressed(Some(st), _) => {
            size = format!("\"d\":{},", st.active_qubits());
            Some(st.expectation(&qsim_lab::pauli_path::PauliSum::z_product(n, &all)))
        }
        _ => None,
    };
    let t_e = ev.map(|_| te.elapsed().as_secs_f64());
    // amplitudes (HSF: from the path sum, before any full output exists)
    let mut ta = [None, None];
    let mut amp0 = None;
    for (i, m) in [1usize, 1000].into_iter().enumerate() {
        let xs = xs_for(n, m, seed);
        let t = Instant::now();
        // MPS amplitudes one chunk at a time, censored after 6 s; the HSF
        // path sum is one batch (its cost is per batch, not per amplitude)
        let chunk = if matches!(p, Prepared::Mps(_)) { 50 } else { m };
        let mut ok = true;
        for (k, part) in xs.chunks(chunk).enumerate() {
            match p.amplitudes(part) {
                Ok(a) => {
                    if i == 0 && k == 0 {
                        amp0 = Some(a[0]);
                    }
                }
                Err(_) => {
                    ok = false;
                    break;
                }
            }
            if t.elapsed().as_secs_f64() > 6.0 {
                ok = false;
                break;
            }
        }
        if ok {
            ta[i] = Some(t.elapsed().as_secs_f64());
        }
    }
    // sampling
    let t = Instant::now();
    let prep_ok = p.prepare_sampling().is_ok();
    let t_prep = t.elapsed().as_secs_f64();
    let mut ts = [None, None, None];
    let mut first = None;
    if prep_ok {
        let mut rng = StdRng::seed_from_u64(seed);
        for (i, s) in [1usize, 1000, 100_000].into_iter().enumerate() {
            let t = Instant::now();
            if let Ok(Some(v)) = p.samples(s, &mut rng, Some((t, 6.0))) {
                ts[i] = Some(t.elapsed().as_secs_f64());
                if i == 0 {
                    first = v.first().copied();
                }
            }
        }
    }
    if let Prepared::Hsf(h, _) = &p {
        size = format!("\"hsf_paths\":{},", h.num_paths());
    }
    // HSF expectation from the full output (prep)
    let t_e = match (&p, t_e) {
        (Prepared::Hsf(_, Some(sv)), None) => {
            let t = Instant::now();
            let _ = parity_sum(
                sv.amplitudes()
                    .iter()
                    .enumerate()
                    .map(|(x, a)| (x as u128, a.norm_sqr())),
                mask,
            );
            Some(t_prep + t.elapsed().as_secs_f64())
        }
        _ => t_e,
    };
    format!(
        "{{\"ok\":true,{size}\"evolve\":{evolve:.7},\"e\":{},\"a1\":{},\"a1k\":{},\"prep\":{},\"s1\":{},\"s1k\":{},\"s100k\":{},\"value\":{},\"amp0\":{},\"sample0\":{}}}",
        j(t_e),
        j(ta[0]),
        j(ta[1]),
        if prep_ok { format!("{t_prep:.7}") } else { "null".into() },
        j(ts[0]),
        j(ts[1]),
        j(ts[2]),
        ev.map_or("null".into(), |v| format!("{v:.12e}")),
        amp0.map_or("null".into(), |a| format!("[{:.12e},{:.12e}]", a.re, a.im)),
        first.map_or("null".into(), |x| format!("\"{x}\"")),
    )
}

/// The old hand rule (pipeline before v2): samples -> tableau if Clifford,
/// compressed state if `AdaptiveRule` accepts, else the state vector;
/// amplitudes -> state vector.
fn rule_engine(c: &Circuit, req: &PlanRequest) -> Engine {
    match req {
        PlanRequest::Amplitudes(_) => Engine::StateVector,
        _ => {
            if c.is_clifford() {
                Engine::Tableau
            } else if AdaptiveRule::default().accepts(c).is_some() {
                Engine::Compressed
            } else {
                Engine::StateVector
            }
        }
    }
}

fn cmd_plan(variant: &str, req_name: &str, c: &Circuit, mem: u128, seed: u64) -> String {
    let n = c.num_qubits;
    let req = request(req_name, n);
    // `force-ENGINE`: the oracle, i.e. the engine measured best for this
    // request in the read-out session, run with no planning and no
    // speculation (same execution path), interleaved with the planner runs
    // so both see the same machine load.
    let forced = variant
        .strip_prefix("force-")
        .map(|e| Engine::from_name(e).expect("engine"));
    let mut cfg = match variant {
        "v1" | "rule" => PlannerConfig::v1(),
        _ if forced.is_some() => PlannerConfig {
            speculate: 0.0,
            cache: false,
            ..PlannerConfig::default()
        },
        "v2" => PlannerConfig::default(),
        "v2nc" => PlannerConfig {
            cache: false,
            ..PlannerConfig::default()
        },
        _ => panic!("variant"),
    };
    cfg.mem_bytes = mem;
    // the dataset compares state engines: no certificate (as `planx`)
    cfg.use_certificate = false;
    let mut rng = StdRng::seed_from_u64(seed);
    let t0 = Instant::now();
    let (plan, rule) =
        if forced.is_some() || (variant == "rule" && !matches!(req, PlanRequest::Expectation(_))) {
            let e = forced.unwrap_or_else(|| rule_engine(c, &req));
            if forced.is_some() {
                // the oracle: no planning at all
                let p = planner::Plan {
                    engine: e,
                    ranked: vec![(e, 0.0)],
                    features: PlanFeatures::default(),
                    plan_secs: 0.0,
                    solved: None,
                    probe: None,
                    stage_secs: [0.0; 4],
                    cached: false,
                };
                (p, true)
            } else {
                let mut p = planner::plan(
                    c,
                    &PlanRequest::Expectation(vec![]),
                    &PlannerConfig {
                        tiered: true,
                        voi: f64::INFINITY,
                        cache: false,
                        ..cfg
                    },
                )
                .expect("plan");
                p.engine = e;
                p.ranked = vec![(e, 0.0)];
                (p, true)
            }
        } else {
            (planner::plan(c, &req, &cfg).expect("plan"), false)
        };
    let plan_secs = t0.elapsed().as_secs_f64();
    let res: Result<(Engine, Vec<Engine>), String> = match &req {
        PlanRequest::Expectation(obs) => planner::execute_expectation(&plan, c, obs, &cfg)
            .map(|r| (r.engine, r.aborted))
            .map_err(|e| format!("{e:?}")),
        PlanRequest::Samples(s) => planner::execute_samples(&plan, c, *s, &mut rng, &cfg)
            .map(|r| (r.engine, r.aborted))
            .map_err(|e| format!("{e:?}")),
        PlanRequest::Amplitudes(m) => {
            let xs = xs_for(n, *m, seed);
            planner::execute_amplitudes(&plan, c, &xs, &cfg)
                .map(|r| (r.engine, r.aborted))
                .map_err(|e| format!("{e:?}"))
        }
    };
    let secs = t0.elapsed().as_secs_f64();
    let st = plan.stage_secs;
    match res {
        Ok((e, ab)) => format!(
            "{{\"ok\":true,\"secs\":{secs:.7},\"plan_secs\":{plan_secs:.7},\"chosen\":\"{}\",\"engine\":\"{}\",\"aborted\":[{}],\"tier\":{},\"stages\":[{:.7},{:.7},{:.7},{:.7}],\"rule\":{rule}}}",
            plan.engine.name(),
            e.name(),
            ab.iter().map(|x| format!("\"{}\"", x.name())).collect::<Vec<_>>().join(","),
            plan.features.tier,
            st[0], st[1], st[2], st[3]
        ),
        Err(err) => format!(
            "{{\"ok\":false,\"secs\":{secs:.7},\"chosen\":\"{}\",\"error\":\"{}\"}}",
            plan.engine.name(),
            err.replace('"', "'")
        ),
    }
}

fn cmd_feat(spec: &str, seed: u64) -> String {
    let c = build(&Spec::parse(spec).expect("spec"), seed).expect("build");
    let n = c.num_qubits;
    let cfg = PlannerConfig {
        use_certificate: false,
        ..PlannerConfig::default()
    };
    let t = Instant::now();
    let q = quick_features(&c);
    let t_q = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let all: Vec<usize> = (0..n).collect();
    let mut base = simulability::Features {
        n,
        gates: q.gates,
        ..Default::default()
    };
    let gates: Vec<qsim_lab::Gate> = c.gates().copied().collect();
    base.sup = simulability::support_bound(n, &gates);
    let t_sup = t.elapsed().as_secs_f64();
    let prof = qsim_lab::adaptive::active_dimension_profile(&c).expect("frame");
    let t_t1a = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let obs_zero = qsim_lab::adaptive::z_product_vanishes(&c, &all).expect("cert");
    let t_cert = t.elapsed().as_secs_f64();
    let g = q.gates.max(1) as f64;
    base.sparse_l = g.log2() + base.sup as f64;
    base.sv_l = g.log2() + n as f64;
    base.rotations = prof.len();
    base.d = prof.last().copied().unwrap_or(0);
    base.dense_l = if prof.is_empty() {
        0.0
    } else {
        let m = *prof.iter().max().unwrap() as f64;
        m + prof
            .iter()
            .map(|&d| (d as f64 - m).exp2())
            .sum::<f64>()
            .log2()
    };
    let t = Instant::now();
    let r = replay(&c, BondSource::Bound(Estimator::Best)).expect("replay");
    let t_mps = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let hsf_ok = n >= 2 && simulability::add_hsf_features(&c, &mut base).is_ok();
    let t_hsf = t.elapsed().as_secs_f64();
    let (cu, su, au) = mps_readout_units(&r.final_bonds);
    let f = PlanFeatures {
        mps_r: mps_work_log2(&r.stats, &cfg.model),
        mps_max_bond: r.max_bond,
        clifford: q.clifford,
        mps_bonds: r.final_bonds.clone(),
        quick: q.clone(),
        tier: 3,
        base: base.clone(),
    };
    let mut choices = Vec::new();
    for rq in REQS {
        let req = request(rq, n);
        let v1 = planner::plan(
            &c,
            &req,
            &PlannerConfig {
                use_certificate: false,
                ..PlannerConfig::v1()
            },
        )
        .map(|p| p.engine.name())
        .unwrap_or("none");
        let p2 = planner::plan(
            &c,
            &req,
            &PlannerConfig {
                use_certificate: false,
                cache: false,
                ..PlannerConfig::default()
            },
        );
        let v2 = p2.as_ref().map(|p| p.engine.name()).unwrap_or("none");
        let v2tier = p2.as_ref().map(|p| p.features.tier).unwrap_or(0);
        let rule = rule_engine(&c, &req).name();
        let preds: Vec<String> = [
            Engine::StateVector,
            Engine::Sparse,
            Engine::Mps,
            Engine::Hsf,
            Engine::Compressed,
            Engine::Tableau,
        ]
        .iter()
        .map(|&e| {
            format!(
                "\"{}\":{:.6e}",
                e.name(),
                predict_secs(e, &f, &req, &cfg.model)
            )
        })
        .collect();
        choices.push(format!(
            "\"{rq}\":{{\"v1\":\"{v1}\",\"v2\":\"{v2}\",\"v2tier\":{v2tier},\"rule\":\"{rule}\",\"pred\":{{{}}}}}",
            preds.join(",")
        ));
    }
    format!(
        "{{\"spec\":\"{spec}\",\"seed\":{seed},\"n\":{n},\"gates\":{},\"g2\":{},\"clifford\":{},\"branching\":{},\"rot_q\":{},\"mps_steps\":{},\"sup\":{},\"sparse_l\":{:.4},\"sv_l\":{:.4},\"rotations\":{},\"d\":{},\"dense_l\":{:.4},\"obs_zero\":{obs_zero},\"mps_r\":{:.4},\"mps_max_bond\":{},\"mps_canon_u\":{cu:.6e},\"mps_shot_u\":{su:.6e},\"mps_amp_u\":{au:.6e},\"hsf_ok\":{hsf_ok},\"hsf_l\":{:.4},\"hsf_keff\":{},\"hsf_k\":{},\"hsf_na\":{},\"hsf_nb\":{},\"t_quick\":{t_q:.7},\"t_tier1\":{t_t1a:.7},\"t_sup\":{t_sup:.7},\"t_cert\":{t_cert:.7},\"t_mps\":{t_mps:.7},\"t_hsf\":{t_hsf:.7},\"choices\":{{{}}}}}",
        q.gates, q.g2, q.clifford, q.branching, q.rotations, q.mps_steps, base.sup, base.sparse_l, base.sv_l,
        base.rotations, base.d, base.dense_l, f.mps_r, r.max_bond, base.hsf_l, base.hsf_keff, base.hsf_k,
        base.hsf_na, base.hsf_nb, choices.join(",")
    )
}

fn perturb(c: &Circuit, rng: &mut StdRng) -> Circuit {
    use qsim_lab::gate::is_multiple_of_half_pi as half;
    use qsim_lab::Gate::*;
    let mut d = Circuit::new(c.num_qubits);
    let mut f = |t: f64| {
        if half(t) {
            t
        } else {
            t * (0.5 + rng.random::<f64>())
        }
    };
    for g in c.gates() {
        d.gate(match *g {
            Rx(q, t) => Rx(q, f(t)),
            Ry(q, t) => Ry(q, f(t)),
            Rz(q, t) => Rz(q, f(t)),
            Phase(q, t) => Phase(q, f(t)),
            U(q, a, b, cc) => U(q, f(a), f(b), f(cc)),
            CPhase(a, b, t) => {
                let u = f(t);
                CPhase(a, b, if half(t / 2.0) { t } else { u })
            }
            g => g,
        });
    }
    d
}

fn cmd_cachedemo(c0: &Circuit, reqn: &str, seed: u64) -> String {
    let n = c0.num_qubits;
    let req = request(reqn, n);
    planner::clear_cache();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::new();
    for cache in [false, true] {
        let cfg = PlannerConfig {
            cache,
            use_certificate: false,
            ..PlannerConfig::default()
        };
        let mut secs = Vec::new();
        let mut hits = 0;
        for _ in 0..20 {
            let c = perturb(c0, &mut rng);
            let p = planner::plan(&c, &req, &cfg).expect("plan");
            hits += usize::from(p.cached);
            secs.push(p.plan_secs);
        }
        let first = secs[0];
        let mut rest = secs[1..].to_vec();
        rest.sort_by(f64::total_cmp);
        out.push(format!(
            "\"{}\":{{\"first\":{first:.7},\"median_rest\":{:.7},\"hits\":{hits}}}",
            if cache { "cache" } else { "nocache" },
            rest[rest.len() / 2]
        ));
    }
    planner::clear_cache();
    format!("{{{}}}", out.join(","))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mem = |i: usize| -> u128 {
        args.get(i)
            .map(|s| s.parse().expect("mem"))
            .unwrap_or(1 << 30)
    };
    match args.get(1).map(String::as_str) {
        Some("req") => {
            let c = build(
                &Spec::parse(&args[3]).expect("spec"),
                args[4].parse().unwrap(),
            )
            .expect("build");
            println!(
                "{}",
                cmd_req(&args[2], &c, mem(5), args[4].parse().unwrap())
            );
        }
        Some("plan") => {
            let seed: u64 = args[5].parse().unwrap();
            let c = build(&Spec::parse(&args[4]).expect("spec"), seed).expect("build");
            println!("{}", cmd_plan(&args[2], &args[3], &c, mem(6), seed));
        }
        Some("cachedemo") => {
            // a parameter sweep: the same circuit structure with new angles
            let seed: u64 = args[3].parse().unwrap();
            let c0 = build(&Spec::parse(&args[2]).expect("spec"), seed).expect("build");
            let reqn = args.get(4).map(String::as_str).unwrap_or("e");
            println!("{}", cmd_cachedemo(&c0, reqn, seed));
        }
        Some("feat") => {
            let text = std::fs::read_to_string(&args[2]).expect("file");
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                let mut it = line.split_whitespace();
                let spec = it.next().unwrap();
                let seed: u64 = it.next().unwrap_or("1").parse().unwrap();
                println!("{}", cmd_feat(spec, seed));
            }
        }
        _ => {
            eprintln!("usage: planner_v2 req ENGINE SPEC SEED [MEM] | plan VARIANT REQ SPEC SEED [MEM] | feat FILE");
            std::process::exit(2);
        }
    }
}
