//! Low-magic chemistry driver (research/simulability/lowmagic-chem.md).
//!
//! ```text
//! lowmagic_chem profile PROG [RANK_CAP]          # d, f, W_d (+ branching rank) of a program
//! lowmagic_chem energy FCIDUMP PROG [SWEEPS] [MAX_D] [NOPT]  # exact energy (+ Rotosolve of params < NOPT)
//!     (LANCZOS=k: also the lowest energy of H projected on the same 2^d register, k Lanczos steps)
//! lowmagic_chem check FCIDUMP PROG               # n <= 22: compressed vs state vector, filtered vs full H
//! ```
//! Programs and FCIDUMPs are written by `research/data/lowmagic-chem/chem.py`.
//! Every command prints one JSON line.
use qsim_lab::chem::{self, Fcidump, Program, Span};
use qsim_lab::magic_atlas::{self, AtlasOptions, FactoredState};
use qsim_lab::stab_rank::RankState;
use qsim_lab::StateVector;
use std::time::Instant;

fn read(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let arg = |i: usize, d: usize| a.get(i).map_or(d, |s| s.parse().unwrap());
    match a.get(1).map(|s| s.as_str()) {
        Some("profile") => profile(&a[2], arg(3, 0)),
        Some("energy") => energy(&a[2], &a[3], arg(4, 0), arg(5, 26), arg(6, usize::MAX)),
        Some("check") => check(&a[2], &a[3]),
        _ => eprintln!("usage: see the source header"),
    }
}

fn json_list<T: std::fmt::Display>(v: &[T]) -> String {
    let s: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    format!("[{}]", s.join(","))
}

fn profile(prog_path: &str, rank_cap: usize) {
    let prog = Program::parse(&read(prog_path)).unwrap();
    let c = prog.circuit(None);
    let opts = AtlasOptions {
        checkpoints: 0,
        entanglement: false,
        cut: None,
        support: false,
    };
    let ap = magic_atlas::profile(&c, &opts).unwrap();
    let span = Span::from_program(&prog);
    // d after each rotation, subsampled to <= 200 points
    let step = (ap.d_prof.len() / 200).max(1);
    let dsub: Vec<u32> = ap.d_prof.iter().step_by(step).copied().collect();
    let sat_rot = ap
        .d_prof
        .iter()
        .position(|&x| x as usize == ap.d)
        .map_or(0, |i| i + 1);
    let mut rank = String::from("null");
    if rank_cap > 0 {
        let mut rs = RankState::new(c.num_qubits);
        rs.max_terms = rank_cap;
        let t0 = Instant::now();
        let ok = rs.run(&c);
        rank = format!(
            "{{\"ok\":{},\"r_max\":{},\"r_final\":{},\"secs\":{:.4}}}",
            ok,
            rs.stats.max_r,
            rs.rank(),
            t0.elapsed().as_secs_f64()
        );
    }
    println!(
        "{{\"prog\":\"{}\",\"n\":{},\"gates\":{},\"rotations\":{},\"d\":{},\"sat_rot\":{},\"f\":{},\"span_dim\":{},\"log2_work\":{:.3},\"log2_work_f\":{:.3},\"secs\":{:.4},\"d_prof_step\":{},\"d_prof\":{},\"rank\":{}}}",
        prog_path,
        ap.n,
        ap.gates,
        ap.rotations,
        ap.d,
        sat_rot,
        ap.f,
        span.dim(),
        ap.log2_work,
        ap.log2_work_f,
        ap.secs,
        step,
        json_list(&dsub),
        rank
    );
}

fn energy(fd_path: &str, prog_path: &str, sweeps: usize, max_d: usize, nopt: usize) {
    let fd = Fcidump::parse(&read(fd_path)).unwrap();
    let prog = Program::parse(&read(prog_path)).unwrap();
    assert_eq!(prog.n, fd.qubits(), "program and FCIDUMP disagree on qubits");
    let span = Span::from_program(&prog);
    let filt = prog.x_preserving();
    assert!(filt || prog.n <= 40, "non-x-preserving program: full H only for n <= 40");
    let (h, hs) = chem::jw_hamiltonian(&fd, if filt { Some(&span) } else { None }, 1e-12);
    let t0 = Instant::now();
    let (e0, st) = chem::energy(&prog, None, &h, max_d).unwrap();
    let t_e0 = t0.elapsed().as_secs_f64();
    let d = st.active_qubits();
    let nnz = st
        .active_amplitudes()
        .iter()
        .filter(|a| a.norm_sqr() > 1e-24)
        .count();
    let evolve = st.stats.evolve_secs;
    drop(st);
    let (mut eopt, mut theta, mut hist, mut evals, mut t_opt) =
        (e0, prog.params.clone(), vec![e0], 0usize, 0.0);
    if sweeps > 0 && !prog.params.is_empty() {
        let t1 = Instant::now();
        let (th, hi, ev) = chem::rotosolve(&prog, &h, sweeps, 1e-9, max_d, nopt).unwrap();
        t_opt = t1.elapsed().as_secs_f64();
        eopt = *hi.last().unwrap();
        theta = th;
        hist = hi;
        evals = ev;
    }
    // best state in the same register (Lanczos), at the final parameters
    let lanczos: usize = std::env::var("LANCZOS")
        .ok()
        .map_or(0, |v| v.parse().unwrap());
    let (mut e_reg, mut reg_iters, mut reg_secs) = (f64::NAN, 0usize, 0.0);
    if lanczos > 0 {
        let t2 = Instant::now();
        let (_, st) = chem::energy(&prog, Some(&theta), &h, max_d).unwrap();
        let (hist, _) = chem::register_ground(&st, &h, lanczos, 1e-9);
        e_reg = *hist.last().unwrap();
        reg_iters = hist.len();
        reg_secs = t2.elapsed().as_secs_f64();
    }
    println!(
        "{{\"fcidump\":\"{}\",\"prog\":\"{}\",\"n\":{},\"rotations\":{},\"params\":{},\"d\":{},\"nnz\":{},\"span_dim\":{},\"filtered\":{},\"monomials_total\":{},\"monomials_kept\":{},\"pauli_terms\":{},\"ham_secs\":{:.3},\"e_init\":{:.10},\"eval_secs\":{:.4},\"evolve_secs\":{:.4},\"e_opt\":{:.10},\"opt_secs\":{:.3},\"evals\":{},\"e_reg\":{:.10},\"reg_iters\":{},\"reg_secs\":{:.3},\"hist\":{},\"theta\":{}}}",
        fd_path,
        prog_path,
        prog.n,
        prog.rotations(),
        prog.params.len(),
        d,
        nnz,
        span.dim(),
        filt,
        hs.monomials_total,
        hs.monomials_kept,
        hs.pauli_terms,
        hs.secs,
        e0,
        t_e0,
        evolve,
        eopt,
        t_opt,
        evals,
        e_reg,
        reg_iters,
        reg_secs,
        json_list(&hist.iter().map(|x| format!("{x:.10}")).collect::<Vec<_>>()),
        json_list(&theta.iter().map(|x| format!("{x:.12}")).collect::<Vec<_>>()),
    );
}

fn check(fd_path: &str, prog_path: &str) {
    let fd = Fcidump::parse(&read(fd_path)).unwrap();
    let prog = Program::parse(&read(prog_path)).unwrap();
    let n = prog.n;
    assert!(n <= 22);
    let c = prog.circuit(None);
    let mut sv = StateVector::<f64>::new(n);
    for g in c.gates() {
        sv.apply_gate(g).unwrap();
    }
    let (hfull, _) = chem::jw_hamiltonian(&fd, None, 0.0);
    let e_sv = chem::sv_expectation(sv.amplitudes(), &hfull);
    let span = Span::from_program(&prog);
    let filt = prog.x_preserving();
    let (hf, _) = chem::jw_hamiltonian(&fd, if filt { Some(&span) } else { None }, 0.0);
    let (e_cs, st) = chem::energy(&prog, None, &hf, 30).unwrap();
    let e_cs_full = st.expectation(&hfull);
    let v = st.to_statevector();
    let ov: num_complex::Complex64 = v
        .amplitudes()
        .iter()
        .zip(sv.amplitudes())
        .map(|(a, b)| a.conj() * b)
        .sum();
    let fs = FactoredState::new(&c, 30).unwrap();
    // HF determinant energy (Slater–Condon) for the program's X gates
    let occ: Vec<usize> = prog
        .ops
        .iter()
        .filter_map(|o| match o {
            chem::POp::Clifford(qsim_lab::Gate::X(q)) => Some(*q),
            _ => None,
        })
        .collect();
    println!(
        "{{\"n\":{},\"d\":{},\"f\":{},\"full_terms\":{},\"filtered_terms\":{},\"e_sv\":{:.12},\"e_cs\":{:.12},\"e_cs_fullH\":{:.12},\"infidelity\":{:.3e},\"e_det_slater_condon\":{:.12}}}",
        n,
        st.active_qubits(),
        fs.stats.f,
        hfull.num_terms(),
        hf.num_terms(),
        e_sv,
        e_cs,
        e_cs_full,
        1.0 - ov.norm(),
        fd.determinant_energy(&occ)
    );
}
