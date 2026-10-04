//! Magic atlas CLI (research/magic-atlas.md).
//!
//! ```text
//! magic_atlas profile SPEC [SEED] [PROFILE_CSV]   # one JSON line; optional per-checkpoint CSV
//! magic_atlas verify SPEC [SEED]                  # cstate & factored vs state vector (n <= 24)
//! magic_atlas time ENGINE SPEC [SEED] [MAXQ]      # ENGINE = sv | cstate | factored; one JSON line
//! magic_atlas pauli SPEC SEED PAULI...            # factored-state <P> for Pauli strings (any n)
//! magic_atlas magic SPEC SEED CSV [MAXCK]         # ground-truth nullity / SRE vs d, f along the circuit (n <= 13)
//! ```
use qsim_lab::adaptive::CompressedState;
use qsim_lab::circuit::Circuit;
use qsim_lab::gate::Gate;
use qsim_lab::magic_atlas::{families, profile, state_magic, AtlasOptions, FactoredState};
use qsim_lab::StateVectorF64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::io::Write;
use std::time::Instant;

fn sv_of(c: &Circuit) -> StateVectorF64 {
    let mut sv = StateVectorF64::new(c.num_qubits);
    for g in c.gates() {
        sv.apply_gate(g).unwrap();
    }
    sv
}

fn sv_pauli(sv: &StateVectorF64, p: &[u8]) -> f64 {
    let mut phi = sv.clone();
    for (q, &ch) in p.iter().enumerate() {
        match ch {
            b'X' => phi.apply_gate(&Gate::X(q)).unwrap(),
            b'Y' => phi.apply_gate(&Gate::Y(q)).unwrap(),
            b'Z' => phi.apply_gate(&Gate::Z(q)).unwrap(),
            _ => {}
        }
    }
    sv.inner(&phi).re
}

fn pauli_xz(p: &[u8]) -> (Vec<bool>, Vec<bool>) {
    let x = p.iter().map(|&c| c == b'X' || c == b'Y').collect();
    let z = p.iter().map(|&c| c == b'Z' || c == b'Y').collect();
    (x, z)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("");
    let seed = |i: usize| -> u64 { args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1) };
    match cmd {
        "profile" => {
            let spec = &args[2];
            let s = seed(3);
            let tb = Instant::now();
            let c = families::build(spec, s).unwrap();
            let build_secs = tb.elapsed().as_secs_f64();
            let n = c.num_qubits;
            let opts = AtlasOptions {
                checkpoints: if n > 600 { 16 } else { 64 },
                ..Default::default()
            };
            let p = profile(&c, &opts).unwrap();
            println!(
                "{{\"spec\":\"{spec}\",\"seed\":{s},\"n\":{},\"gates\":{},\"lowered\":{},\"twoq\":{},\"toffolis\":{},\"rotations\":{},\"t_count\":{},\"d\":{},\"f\":{},\"log2_work\":{:.3},\"log2_work_f\":{:.3},\"e_stab_max\":{},\"e_stab_final\":{},\"e_bound_max\":{},\"support\":{},\"secs\":{:.4},\"build_secs\":{:.4}}}",
                p.n,
                p.gates,
                p.lowered,
                p.two_qubit,
                p.toffolis,
                p.rotations,
                p.t_count,
                p.d,
                p.f,
                p.log2_work,
                p.log2_work_f,
                p.e_stab_max,
                p.checkpoints.last().map_or(0, |c| c.e_stab),
                p.e_bound_max,
                p.support.unwrap_or(0),
                p.secs,
                build_secs
            );
            if let Some(path) = args.get(4) {
                let mut f = std::fs::File::create(path).unwrap();
                writeln!(f, "kind,gate,rotations,t_count,d,f,e_stab").unwrap();
                for ck in &p.checkpoints {
                    writeln!(
                        f,
                        "ck,{},{},{},{},{},{}",
                        ck.gate, ck.rotations, ck.t_count, ck.d, ck.f, ck.e_stab
                    )
                    .unwrap();
                }
                // per-rotation d / f profile, downsampled to <= 2000 rows
                let m = p.d_prof.len();
                let step = m.div_ceil(2000).max(1);
                for j in (0..m).step_by(step).chain(m.checked_sub(1)) {
                    writeln!(
                        f,
                        "rot,{},{},,{},{},",
                        p.rot_gate[j],
                        j + 1,
                        p.d_prof[j],
                        p.f_prof[j]
                    )
                    .unwrap();
                }
            }
        }
        "verify" => {
            let spec = &args[2];
            let s = seed(3);
            let c = families::build(spec, s).unwrap();
            let n = c.num_qubits;
            assert!(n <= 24, "verify needs n <= 24");
            let t0 = Instant::now();
            let sv = sv_of(&c);
            let sv_secs = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let cs = CompressedState::new(&c, 24).unwrap();
            let cs_secs = t1.elapsed().as_secs_f64();
            let full = cs.to_statevector();
            let fid = sv.inner(&full).norm();
            let t2 = Instant::now();
            let fs = FactoredState::new(&c, 24).unwrap();
            let fs_secs = t2.elapsed().as_secs_f64();
            let da = fs.dense_active();
            let ca = cs.active_amplitudes();
            // same frame, so equal up to a global phase (the factored engine
            // drops empty-support rotations, i.e. global phases)
            let ov: num_complex::Complex64 = da.iter().zip(ca).map(|(a, b)| a.conj() * b).sum();
            let fdiff = (1.0 - ov.norm()).abs();
            let t3 = Instant::now();
            let fr = FactoredState::with_recycling(&c, 24, 16).unwrap();
            let fr_secs = t3.elapsed().as_secs_f64();
            let mut rng = StdRng::seed_from_u64(s ^ 0x5eed);
            let mut pmax: f64 = 0.0;
            let mut nonzero = 0;
            for k in 0..40 {
                // half random Paulis, half low-weight
                let p: Vec<u8> = (0..n)
                    .map(|_| {
                        let r = if k % 2 == 0 { 4 } else { 12 };
                        [b'X', b'Y', b'Z', b'I'][rng.random_range(0..r).min(3)]
                    })
                    .collect();
                let (x, z) = pauli_xz(&p);
                let a = fs.pauli_expectation(&x, &z);
                let b = sv_pauli(&sv, &p);
                pmax = pmax.max((fr.pauli_expectation(&x, &z) - b).abs());
                if b.abs() > 1e-9 {
                    nonzero += 1;
                }
                pmax = pmax.max((a - b).abs());
            }
            println!(
                "{{\"spec\":\"{spec}\",\"seed\":{s},\"n\":{n},\"d\":{},\"f\":{},\"f_recycled\":{},\"absorbed\":{},\"recycled_secs\":{fr_secs:.4},\"infidelity\":{:.3e},\"factored_vs_cstate\":{:.3e},\"pauli_maxerr\":{:.3e},\"pauli_nonzero\":{nonzero},\"sv_secs\":{sv_secs:.4},\"cstate_secs\":{cs_secs:.4},\"factored_secs\":{fs_secs:.4}}}",
                cs.active_qubits(),
                fs.stats.f,
                fr.stats.f,
                fr.stats.absorbed,
                (1.0 - fid).abs(),
                fdiff,
                pmax
            );
            assert!((1.0 - fid).abs() < 1e-10 && fdiff < 1e-10 && pmax < 1e-10);
        }
        "time" => {
            let engine = args[2].as_str();
            let spec = &args[3];
            let s = seed(4);
            let maxq: usize = args.get(5).and_then(|v| v.parse().ok()).unwrap_or(30);
            let c = families::build(spec, s).unwrap();
            let n = c.num_qubits;
            let t0 = Instant::now();
            let (extra, d) = match engine {
                "sv" => {
                    let r = qsim_lab::simulability::run_engine("sv", &c, 1u128 << 34).unwrap();
                    (format!("\"value\":{:.12}", r.value), n)
                }
                "cstate" => {
                    let cs = CompressedState::new(&c, maxq).unwrap();
                    (
                        format!(
                            "\"element_ops\":{},\"compile_secs\":{:.4},\"evolve_secs\":{:.4}",
                            cs.stats.element_ops, cs.stats.compile_secs, cs.stats.evolve_secs
                        ),
                        cs.active_qubits(),
                    )
                }
                "factored" => {
                    let fs = FactoredState::new(&c, maxq).unwrap();
                    (
                        format!(
                            "\"f\":{},\"factors\":{},\"element_ops\":{},\"compile_secs\":{:.4},\"evolve_secs\":{:.4}",
                            fs.stats.f,
                            fs.stats.factors,
                            fs.stats.element_ops,
                            fs.stats.compile_secs,
                            fs.stats.evolve_secs
                        ),
                        fs.stats.d,
                    )
                }
                "recycled" => {
                    let fs = FactoredState::with_recycling(&c, maxq, 20).unwrap();
                    let lmax = fs.stats.live.iter().map(|x| x.0).max().unwrap_or(0);
                    (
                        format!(
                            "\"f\":{},\"live_max\":{lmax},\"absorbed\":{},\"element_ops\":{},\"compile_secs\":{:.4},\"evolve_secs\":{:.4},\"recycle_secs\":{:.4}",
                            fs.stats.f,
                            fs.stats.absorbed,
                            fs.stats.element_ops,
                            fs.stats.compile_secs,
                            fs.stats.evolve_secs,
                            fs.stats.recycle_secs
                        ),
                        fs.stats.d,
                    )
                }
                e => panic!("unknown engine {e}"),
            };
            let secs = t0.elapsed().as_secs_f64();
            println!(
                "{{\"engine\":\"{engine}\",\"spec\":\"{spec}\",\"seed\":{s},\"n\":{n},\"d\":{d},\"secs\":{secs:.5},{extra}}}"
            );
        }
        "pauli" => {
            let spec = &args[2];
            let s = seed(3);
            let c = families::build(spec, s).unwrap();
            let t0 = Instant::now();
            let rec: usize = std::env::var("RECYCLE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let fs = FactoredState::with_recycling(&c, 30, rec).unwrap();
            let secs = t0.elapsed().as_secs_f64();
            println!(
                "# n={} gates={} t_count={} d={} f={} absorbed={} factors={} rotations={} element_ops={} secs={secs:.3}",
                c.num_qubits,
                c.num_gates(),
                c.t_count(),
                fs.stats.d,
                fs.stats.f,
                fs.stats.absorbed,
                fs.stats.factors,
                fs.stats.rotations,
                fs.stats.element_ops
            );
            for p in &args[4..] {
                let (x, z) = pauli_xz(p.as_bytes());
                println!("{p} {:.12}", fs.pauli_expectation(&x, &z));
            }
        }
        "magic" => {
            let spec = &args[2];
            let s = seed(3);
            let path = &args[4];
            let maxck: usize = args.get(5).and_then(|v| v.parse().ok()).unwrap_or(200);
            let c = families::build(spec, s).unwrap();
            let n = c.num_qubits;
            assert!(n <= 14, "magic needs n <= 14");
            let p = profile(
                &c,
                &AtlasOptions {
                    checkpoints: 0,
                    entanglement: false,
                    cut: None,
                    support: false,
                },
            )
            .unwrap();
            let gates: Vec<Gate> = c.gates().copied().collect();
            let ng = gates.len();
            let every = ng.div_ceil(maxck).max(1);
            let mut f = std::fs::File::create(path).unwrap();
            let fr = FactoredState::with_recycling(&c, 24, 16).unwrap();
            writeln!(f, "gate,rotations,d,f,live,live_max,nullity,m2").unwrap();
            let mut sv = StateVectorF64::new(n);
            let mut j = 0usize; // rotations so far
            let (mut dk, mut fk) = (0u32, 0u32);
            let m0 = state_magic(sv.amplitudes());
            writeln!(f, "-1,0,0,0,0,0,{:.6},{:.6}", m0.nullity, m0.m2).unwrap();
            let mut lmax_all = 0u32;
            let (mut numax, mut m2max) = (0.0f64, 0.0f64);
            for (gi, g) in gates.iter().enumerate() {
                sv.apply_gate(g).unwrap();
                while j < p.rot_gate.len() && p.rot_gate[j] as usize <= gi {
                    dk = p.d_prof[j];
                    fk = fk.max(p.f_prof[j]);
                    j += 1;
                }
                if (gi + 1) % every == 0 || gi + 1 == ng {
                    let m = state_magic(sv.amplitudes());
                    numax = numax.max(m.nullity);
                    m2max = m2max.max(m.m2);
                    let (lv, lm) = fr.stats.live[gi];
                    lmax_all = lmax_all.max(lv);
                    assert!(
                        m.nullity <= lv as f64 + 1e-6,
                        "nullity {} > live {lv}",
                        m.nullity
                    );
                    writeln!(
                        f,
                        "{gi},{j},{dk},{fk},{lv},{lm},{:.6},{:.6}",
                        m.nullity, m.m2
                    )
                    .unwrap();
                }
            }
            let mf = state_magic(sv.amplitudes());
            println!(
                "{{\"spec\":\"{spec}\",\"n\":{n},\"gates\":{ng},\"rotations\":{},\"t_count\":{},\"d\":{},\"f\":{},\"live_max\":{lmax_all},\"f_recycled\":{},\"nullity_max\":{numax:.4},\"nullity_final\":{:.4},\"m2_max\":{m2max:.4},\"m2_final\":{:.4}}}",
                p.rotations, p.t_count, p.d, p.f, fr.stats.f, mf.nullity, mf.m2
            );
        }
        _ => {
            eprintln!("usage: magic_atlas profile|verify|time|pauli ...");
            std::process::exit(2);
        }
    }
}
