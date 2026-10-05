//! Executable checks for research/theory/stabrank5.md: χ(|T⟩^{⊗5}) = χ(|H⟩^{⊗5}) = 6.
//!
//! The engine is the standalone research/data/stabrank5/stabrank5.rs (std only), which includes
//! research/data/stabrank-lower/stabrank.rs. The theorem rests on a complete search whose
//! completeness argument is in the write-up: (1) all optimal 4-term and all minimal 5-term
//! decompositions of H^{⊗4}, enumerated with the Galois lemma; (2) the lift of each to H^{⊗5}
//! (restriction of the last qubit, Pauli-orbit completions, meet in the middle).
//!
//! The fast tests check every ingredient on instances with known answers, check the integrity of
//! the stored certificate (the 14181 orbit representatives of the 2,662,464 minimal 5-term
//! decompositions of H^{⊗4}, `research/data/stabrank5/m5_H4_reps.u16`) and lift a sample of it.
//! The ignored tests re-run the whole n = 5 search from the certificate (a few CPU-minutes) and
//! regenerate the certificate (about half a CPU-hour):
//! `cargo test --release --test stabrank5 -- --ignored`.
#[allow(dead_code, unused_imports, unused_variables, unused_mut, clippy::all)]
#[path = "../../research/data/stabrank5/stabrank5.rs"]
mod stabrank5;

use stabrank5::*;
use std::collections::HashSet;

const CERT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/research/data/stabrank5/m5_H4_reps.u16"
);
const H4_LIST: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/research/data/stabrank-lower/dec_H4_k4.txt"
);

/// Index sets of a decomposition list written by stabrank.rs (`idx:re,im` per term), sorted.
fn read_list(path: &str) -> Vec<Vec<u32>> {
    let mut v: Vec<Vec<u32>> = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| {
            let mut s: Vec<u32> = l
                .split_whitespace()
                .map(|p| p.split(':').next().unwrap().parse().unwrap())
                .collect();
            s.sort();
            s
        })
        .collect();
    v.sort();
    v
}

/// Lifts found from the given representatives, rotated back to the frame of psi^{⊗n} and expanded
/// under the n-qubit symmetry group (n <= 4), as sorted index sets.
fn expand_found(kind: &str, n: usize, found: &[Vec<Vec<C>>], bra: usize) -> Vec<Vec<u32>> {
    let t1 = Table::new(1);
    let u = cliffords1()
        .into_iter()
        .find(|m| apply1(m, &t1.states[bra])[0].abs() > 1.0 - 1e-9)
        .unwrap();
    let udag: M2 = [cj(u[0]), cj(u[2]), cj(u[1]), cj(u[3])];
    let tn = Table::new(n);
    let p = psi1(kind);
    let psin = tensor_power(&p, n);
    let gn = sym_generators(&tn, &p);
    let ds: Vec<Dec> = found
        .iter()
        .map(|f| {
            let mut ix: Vec<u32> = f
                .iter()
                .map(|v| tn.lookup(&apply_local(&udag, n - 1, v)).unwrap())
                .collect();
            ix.sort();
            check_set(&tn, &psin, &ix).unwrap()
        })
        .collect();
    let mut out: Vec<Vec<u32>> = expand(&tn, &psin, &ds, &gn)
        .into_iter()
        .map(|d| d.idx)
        .collect();
    out.sort();
    out
}

/// Algorithm 1 at level n-1 -> n with restriction bra `bra`: degenerate lifts of the optimal
/// representatives plus type-I lifts of the minimal ones.
fn algorithm1(
    kind: &str,
    n: usize,
    lv: &Level,
    opt: &[Vec<u32>],
    min: &[Vec<u32>],
    bra: usize,
) -> Vec<Vec<Vec<C>>> {
    let p = psi1(kind);
    let last = if bra == 0 {
        p
    } else {
        let t1 = Table::new(1);
        let u = cliffords1()
            .into_iter()
            .find(|m| apply1(m, &t1.states[bra])[0].abs() > 1.0 - 1e-9)
            .unwrap();
        apply1(&u, &p)
    };
    let cx = LiftCtx::with_last(&p, n, &last);
    let mut found = vec![];
    for s in opt {
        let tops: Vec<&[C]> = s
            .iter()
            .map(|&x| lv.t.states[x as usize].as_slice())
            .collect();
        found.extend(lift_degenerate(&cx, &lv.t, &tops, &coefs(lv, s)).0);
    }
    for s in min {
        let tops: Vec<&[C]> = s
            .iter()
            .map(|&x| lv.t.states[x as usize].as_slice())
            .collect();
        found.extend(lift_type1(&cx, &tops, &coefs(lv, s)));
    }
    found
}

#[test]
fn completions_are_the_pauli_orbit() {
    // (|0>A + |1>z)/sqrt2 is a stabilizer state iff z = w X^v Z^u A, w in {±1, ±i}: compare with
    // the is_stabilizer-based completions() of stabrank.rs on every 2- and 3-qubit state.
    for n in [2usize, 3] {
        let t = Table::new(n);
        for a in &t.states {
            let mut kb: Vec<Vec<i64>> = bottoms(a, n).iter().map(|z| key_of(z)).collect();
            let mut kc: Vec<Vec<i64>> = completions(&t, a)
                .iter()
                .map(|(x, y)| key_of(&y.iter().map(|&z| csc(z, 1.0 / x)).collect::<Vec<C>>()))
                .collect();
            kb.sort();
            kc.sort();
            assert_eq!(kb, kc);
            assert_eq!(kb.len(), 1 + 4 * (1 << n));
        }
    }
}

#[test]
fn galois_search_reproduces_the_known_lists() {
    // H^3, 4 terms: 42261 minimal decompositions in 1467 orbits (direct search, stabrank-lower.md)
    let lv3 = Level::new("H", 3);
    let (r3, tot3, _) = decs(&lv3, 4);
    assert_eq!((tot3, r3.len()), (42261, 1467));
    // H^4, 4 terms: exactly the 449 optimal decompositions of stabrank-lower.md
    let lv4 = Level::new("H", 4);
    let (r4, tot4, _) = decs(&lv4, 4);
    assert_eq!((tot4, r4.len()), (449, 19));
    let ds: Vec<Dec> = r4
        .iter()
        .map(|s| check_set(&lv4.t, &lv4.psi, s).unwrap())
        .collect();
    let mut all: Vec<Vec<u32>> = expand(&lv4.t, &lv4.psi, &ds, &lv4.gens)
        .into_iter()
        .map(|d| d.idx)
        .collect();
    all.sort();
    assert_eq!(all, read_list(H4_LIST));
    // Galois lemma: psi_perp^{⊗4} lies in the span of every one of them
    for s in &all {
        let vs: Vec<&[C]> = s
            .iter()
            .map(|&i| lv4.t.states[i as usize].as_slice())
            .collect();
        assert!(lsq(&vs, &lv4.psip).1 < 1e-10);
    }
    // and H^4 has no 3-term decomposition (old projective-hash search)
    let mut st = SearchStats {
        w_count: 0,
        cand_count: 0,
        verified: 0,
    };
    let g16 = lv4.group.clone();
    assert!(search(&lv4.t, &lv4.psi, 3, &lv4.orbit, Some(&g16), &mut st).is_empty());
}

#[test]
fn both_algorithms_reproduce_the_449_decompositions_of_h4() {
    // the n = 5 machinery run one level down, where the answer is known (stabrank-lower.md)
    let lv = Level::new("H", 3);
    let (opt, _, _) = decs(&lv, 3);
    let (min, _, _) = decs(&lv, 4);
    let known = read_list(H4_LIST);
    // Algorithm 1: restriction of the last qubit by <0| and, independently, by <+i|
    for bra in [0usize, 4] {
        let f = algorithm1("H", 4, &lv, &opt, &min, bra);
        assert_eq!(expand_found("H", 4, &f, bra), known, "bra #{bra}");
    }
    // Algorithm 2: degenerate restrictions for every bra orbit (|0>, |1>, |+i>) plus the
    // all-type-I case by the ℓ1-bucketed glue_from() of stabrank.rs
    let p = psi1("H");
    let mut f = vec![];
    for bra in [0usize, 1, 4] {
        let g = algorithm1("H", 4, &lv, &opt, &[], bra);
        let t1 = Table::new(1);
        let u = cliffords1()
            .into_iter()
            .find(|m| apply1(m, &t1.states[bra])[0].abs() > 1.0 - 1e-9)
            .unwrap();
        let udag: M2 = [cj(u[0]), cj(u[2]), cj(u[1]), cj(u[3])];
        for terms in g {
            f.push(
                terms
                    .iter()
                    .map(|v| apply_local(&udag, 3, v))
                    .collect::<Vec<_>>(),
            );
        }
    }
    let reps: Vec<Dec> = min
        .iter()
        .map(|s| check_set(&lv.t, &lv.psi, s).unwrap())
        .collect();
    let bfull = expand(&lv.t, &lv.psi, &reps, &lv.gens);
    for (terms, _) in glue_from(&lv.t, &reps, &bfull, &p, 4) {
        f.push(terms);
    }
    assert_eq!(expand_found("H", 4, &f, 0), known);
}

#[test]
fn certificate_is_the_orbit_list_of_minimal_five_term_decompositions_of_h4() {
    let lv = Level::new("H", 4);
    let reps = read_reps_u16(CERT, 5);
    assert_eq!(reps.len(), 14181);
    let set: HashSet<&Vec<u32>> = reps.iter().collect();
    assert_eq!(set.len(), reps.len());
    let mut total = 0usize;
    let (mut l1min, mut l1max) = (f64::MAX, 0f64);
    for r in &reps {
        // canonical (least image under the 768-element symmetry group), hence pairwise inequivalent
        let (c, stab) = canon_set(r, &lv.group);
        assert_eq!(&c, r);
        total += lv.group.len() / stab;
        // a minimal decomposition: independent, psi in the span, every coefficient non-zero ...
        let d = check_set(&lv.t, &lv.psi, r).expect("not a minimal decomposition");
        let l1: f64 = d.coef.iter().map(|z| z.abs()).sum();
        l1min = l1min.min(l1);
        l1max = l1max.max(l1);
        // ... whose span also contains psi_perp^{⊗4} (Galois lemma)
        let vs: Vec<&[C]> = r
            .iter()
            .map(|&i| lv.t.states[i as usize].as_slice())
            .collect();
        assert!(lsq(&vs, &lv.psip).1 < 1e-10);
    }
    assert_eq!(total, 2_662_464);
    assert!((l1min - 1.621320).abs() < 1e-6 && (l1max - 6.535534).abs() < 1e-6);
    // lift a deterministic sample (every 97th representative) to H^{⊗5}: nothing
    let cx = LiftCtx::new(&psi1("H"), 5);
    for r in reps.iter().step_by(97) {
        let tops: Vec<&[C]> = r
            .iter()
            .map(|&x| lv.t.states[x as usize].as_slice())
            .collect();
        assert!(lift_type1(&cx, &tops, &coefs(&lv, r)).is_empty());
    }
}

#[test]
#[ignore]
fn h5_has_no_five_term_decomposition() {
    // Algorithm 1 in full: degenerate <0|-restrictions (19 optimal representatives) and type-I
    // lifts of all 14181 certificate representatives.  About 12 CPU-minutes in release mode.
    let lv = Level::new("H", 4);
    let (opt, tot, _) = decs(&lv, 4);
    assert_eq!((tot, opt.len()), (449, 19));
    let min = read_reps_u16(CERT, 5);
    let found = lift_all("H", 5, &lv, &opt, &min);
    assert!(found.is_empty(), "a 5-term decomposition of H^5 exists");
}

#[test]
#[ignore]
fn certificate_regenerates() {
    // the Galois-pair search re-derives the certificate exactly (about half a CPU-hour)
    let lv = Level::new("H", 4);
    let (reps, total, _) = decs(&lv, 5);
    assert_eq!(total, 2_662_464);
    assert_eq!(reps, read_reps_u16(CERT, 5));
}
