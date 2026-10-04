//! Executable checks for research/stabrank-lower.md.
//!
//! The engine is the standalone file research/data/stabrank-lower/stabrank.rs (std only), included
//! here as a module.  Every negative claim ("no decomposition with k terms") comes from a complete
//! search whose completeness argument is in the write-up; the tests below re-run those searches,
//! cross-check them against naive enumeration where that is feasible, and would fail if a claimed
//! bound were false.
//!
//! Fast tests run by default.  The H^{⊗5} test needs the 4-term decompositions of H^{⊗4} (about an
//! hour single-threaded) and is #[ignore]d: `cargo test --release --test stabrank_lower -- --ignored`.
#[allow(dead_code, unused_imports, unused_variables, unused_mut, clippy::all)]
#[path = "../research/data/stabrank-lower/stabrank.rs"]
mod stabrank;

use stabrank::*;
use std::collections::HashSet;

fn all_decs(kind: &str, n: usize, k: usize) -> (Table, Vec<Dec>) {
    let p = psi1(kind);
    let t = Table::new(n);
    let psi = tensor_power(&p, n);
    let gens = sym_generators(&t, &p);
    let orb = orbits(t.states.len(), &gens);
    let group = group_closure(&gens, 1000);
    let mut st = SearchStats { w_count: 0, cand_count: 0, verified: 0 };
    let reps = search(&t, &psi, k, &orb, group.as_ref(), &mut st);
    let all = expand(&t, &psi, &reps, &gens);
    (t, all)
}

fn brute_count(kind: &str, n: usize, k: usize) -> usize {
    let p = psi1(kind);
    let t = Table::new(n);
    let psi = tensor_power(&p, n);
    let m = t.states.len() as u32;
    let mut cnt = 0;
    let mut idx = vec![0u32; k];
    fn rec(t: &Table, psi: &[C], m: u32, idx: &mut Vec<u32>, pos: usize, start: u32, cnt: &mut usize) {
        if pos == idx.len() {
            if check_set(t, psi, idx).is_some() {
                *cnt += 1;
            }
            return;
        }
        for v in start..m {
            idx[pos] = v;
            rec(t, psi, m, idx, pos + 1, v + 1, cnt);
        }
    }
    rec(&t, &psi, m, &mut idx, 0, 0, &mut cnt);
    cnt
}

/// Glue: from all rank-k decompositions at n-1 to all rank-k decompositions at n (as sets of
/// canonical vector keys).  Valid when chi(psi^{⊗(n-1)}) = k (plateau lemma).
fn glue_sets(t: &Table, d: &[Dec], kind: &str, n: usize) -> (Vec<(Vec<Vec<C>>, Vec<C>)>, usize) {
    let g = glue(t, d, &psi1(kind), n);
    let mut sets: HashSet<Vec<Vec<i64>>> = HashSet::new();
    for (terms, _) in &g {
        let mut ks: Vec<Vec<i64>> = terms.iter().map(|v| key_of(&canonical(v))).collect();
        ks.sort();
        sets.insert(ks);
    }
    let c = sets.len();
    (g, c)
}

#[test]
fn stabilizer_state_counts() {
    for n in 1..=4 {
        assert_eq!(enum_states(n).len() as u64, n_stab(n));
    }
    assert_eq!(n_stab(5), 2_423_520);
    // the enumerated states pass the independent membership test, and a magic state fails it
    let t = Table::new(3);
    assert!(t.states.iter().all(|v| is_stabilizer(v)));
    assert!(!is_stabilizer(&tensor_power(&psi1("H"), 3)));
}

#[test]
fn search_matches_naive_enumeration() {
    for kind in ["H", "F"] {
        for k in 2..=4 {
            let (_, d) = all_decs(kind, 2, k);
            assert_eq!(d.len(), brute_count(kind, 2, k), "{kind}^2 k={k}");
        }
        let (_, d) = all_decs(kind, 3, 2);
        assert_eq!(d.len(), 0);
    }
    // known totals at n=2 (naive enumeration, cross-checked in the binary as well)
    assert_eq!(all_decs("H", 2, 3).1.len(), 788);
    assert_eq!(all_decs("F", 2, 3).1.len(), 1071);
}

#[test]
fn exact_small_ranks() {
    // chi(H^1)=chi(H^2)=2, chi(H^3)=3, chi(F^2)=2, chi(F^3)=3, with the complete lists
    assert_eq!(all_decs("H", 2, 2).1.len(), 1);
    assert_eq!(all_decs("F", 2, 2).1.len(), 3);
    assert_eq!(all_decs("H", 3, 3).1.len(), 16);
    assert_eq!(all_decs("F", 3, 3).1.len(), 72);
}

#[test]
fn plateau_gluing_reproduces_known_and_proves_face_state_bound() {
    // H: chi(H^1)=2 -> gluing finds exactly the unique rank-2 decomposition of H^2, none for H^3
    let (t1, d1) = all_decs("H", 1, 2);
    let (g2, c2) = glue_sets(&t1, &d1, "H", 2);
    assert_eq!(c2, 1);
    let t2 = Table::new(2);
    let psi2 = tensor_power(&psi1("H"), 2);
    let d2: Vec<Dec> = {
        let mut seen = HashSet::new();
        let mut v = vec![];
        for (terms, _) in &g2 {
            let mut s: Vec<u32> = terms.iter().map(|x| t2.lookup(x).unwrap()).collect();
            s.sort();
            if seen.insert(s.clone()) {
                v.push(check_set(&t2, &psi2, &s).unwrap());
            }
        }
        v
    };
    assert_eq!(glue_sets(&t2, &d2, "H", 3).1, 0); // => chi(H^3) >= 3, agreeing with the search

    // F: rank-3 decompositions of F^3 (72, exhaustive) glue to exactly 9 of F^4 ...
    let (t3, d3) = all_decs("F", 3, 3);
    let (g4, c4) = glue_sets(&t3, &d3, "F", 4);
    assert_eq!(c4, 9);
    // ... which agrees with a direct exhaustive search at n=4 (independent method)
    let (t4, d4direct) = all_decs("F", 4, 3);
    assert_eq!(d4direct.len(), 9);
    // every term is 1-uniform (plateau lemma) and the two lists coincide
    let psi4 = tensor_power(&psi1("F"), 4);
    let mut glued: HashSet<Vec<u32>> = HashSet::new();
    for (terms, _) in &g4 {
        for v in terms {
            assert!(min_restriction_ok(v, 4, 1));
        }
        let mut s: Vec<u32> = terms.iter().map(|x| t4.lookup(x).unwrap()).collect();
        s.sort();
        glued.insert(s);
    }
    let direct: HashSet<Vec<u32>> = d4direct.iter().map(|d| d.idx.clone()).collect();
    assert_eq!(glued, direct);
    // F^5: no rank-3 decomposition  =>  chi(F^{⊗5}) >= 4   (new; previously 3 <= chi <= 6)
    let d4: Vec<Dec> = d4direct.iter().map(|d| check_set(&t4, &psi4, &d.idx).unwrap()).collect();
    assert_eq!(glue_sets(&t4, &d4, "F", 5).1, 0);
}

#[test]
fn no_two_uniform_stabilizer_states_on_3_or_4_qubits() {
    // => by the plateau lemma, chi(psi^{⊗3}) > chi(psi) = 2 and chi(psi^{⊗4}) > chi(psi^{⊗2})
    //    for EVERY non-stabilizer single-qubit psi.
    for n in [3usize, 4] {
        let t = Table::new(n);
        assert!(t.states.iter().all(|v| !min_restriction_ok(v, n, 2)), "n={n}");
    }
    // sanity: 1-uniform states exist on 2 qubits (Bell states)
    let t = Table::new(2);
    assert!(t.states.iter().any(|v| min_restriction_ok(v, 2, 1)));
}

#[test]
fn chi_three_copies_at_least_three_for_random_magic_states() {
    // direct check of the corollary on a few arbitrary single-qubit states
    let t = Table::new(3);
    for (th, ph) in [(0.3f64, 0.7f64), (1.1, 2.9), (0.05, 0.0), (2.0, 1.3)] {
        let p = [C::new((th / 2.).cos(), 0.), C::new((th / 2.).sin() * ph.cos(), (th / 2.).sin() * ph.sin())];
        let psi = tensor_power(&p, 3);
        let gens = sym_generators(&t, &p);
        let orb = orbits(t.states.len(), &gens);
        let mut st = SearchStats { w_count: 0, cand_count: 0, verified: 0 };
        assert!(search(&t, &psi, 2, &orb, None, &mut st).is_empty());
    }
}

#[test]
#[ignore]
fn h4_needs_four_terms() {
    // chi(H^{⊗4}) >= 4 (Labib-Russo 2026 certified this independently)
    assert!(all_decs("H", 4, 3).1.is_empty());
}
