//! Pins the exact parameters of published two-block codes, computed with
//! `qec::bicycle` (GF(2) rank for k, exact branch-and-bound for d).
//!
//! Sources: Bravyi et al., Nature 627, 778 (2024) Table 3 (BB codes);
//! Liang et al., arXiv:2503.03827 Tables 5-8 (cyclic GB codes); Wang &
//! Mueller, arXiv:2408.10001 (coprime BB). See
//! `research/data/code-discovery/literature.md`.
use qsim_lab::qec::bicycle::{logical_masks, DistanceOpts, TwoBlockCode};

fn check(l: usize, m: usize, a: &str, b: &str, n: usize, k: usize, d: usize) {
    let c = TwoBlockCode::parse(l, m, a, b);
    assert_eq!(c.n(), n);
    assert_eq!(c.k(), k, "k of {a} | {b}");
    let r = c.distance(&DistanceOpts::default());
    assert_eq!(r.exact(), Some(d), "d of {a} | {b}: {r:?}");
    // both CSS distances agree for abelian two-block codes
    let (hx, hz) = (c.hx(), c.hz());
    assert_eq!(logical_masks(&hz, &hx).1, k);
}

#[test]
fn bravyi_et_al_bb_codes() {
    check(6, 6, "x^3+y+y^2", "y^3+x+x^2", 72, 12, 6);
    check(15, 3, "x^9+y+y^2", "1+x^2+x^7", 90, 8, 10);
    check(9, 6, "x^3+y+y^2", "y^3+x+x^2", 108, 8, 10);
    check(12, 6, "x^3+y+y^2", "y^3+x+x^2", 144, 12, 12);
}

/// [[288,12,18]]: about 5 s in release, much longer in debug.
#[test]
#[ignore]
fn bravyi_et_al_288() {
    check(12, 12, "x^3+y^2+y^7", "y^3+x+x^2", 288, 12, 18);
}

#[test]
fn liang_et_al_cyclic_gb_codes() {
    check(36, 1, "1+x^2+x^7", "1+x+x^11", 72, 4, 10);
    check(63, 1, "1+x^12+x^23", "1+x+x^8", 126, 12, 10);
    check(105, 1, "1+x^11+x^27", "1+x+x^19", 210, 14, 12);
}

#[test]
fn both_css_distances_equal_on_gross_code() {
    let c = TwoBlockCode::parse(12, 6, "x^3+y+y^2", "y^3+x+x^2");
    let (dx, dz) = c.distances_both(&DistanceOpts::default());
    assert_eq!(dx.exact(), Some(12));
    assert_eq!(dz.exact(), Some(12));
}
