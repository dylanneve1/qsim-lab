//! Round-4 audit helpers (shared by `tests/audit/audit_phasefold.rs` and
//! `tests/audit/audit_repeat.rs`): the naive reference of `tests/audit_common`
//! extended to every gate, and an exact *instrument* comparison that
//! enumerates every branch of measurements, resets and Pauli flips.
#![allow(dead_code, clippy::needless_range_loop)]

use crate::audit_common::{cx, RefSv};
use num_complex::Complex64 as C;
use qsim_lab::{Circuit, Gate, Op};
use rand::rngs::StdRng;
use rand::Rng;
use std::collections::BTreeMap;

/// Applies any gate on the reference (adds the gates `RefSv::m1` lacks).
pub fn apply(s: &mut RefSv, g: &Gate) {
    let one_q = |s: &mut RefSv, q: usize, m: [[C; 2]; 2]| {
        let old = s.a.clone();
        for i in 0..s.a.len() {
            let r = (i >> q) & 1;
            let (i0, i1) = (i & !(1 << q), i | (1 << q));
            s.a[i] = m[r][0] * old[i0] + m[r][1] * old[i1];
        }
    };
    let e = |t: f64| cx(t.cos(), t.sin());
    match *g {
        Gate::I(_) => {}
        Gate::Sx(q) => one_q(
            s,
            q,
            [[cx(0.5, 0.5), cx(0.5, -0.5)], [cx(0.5, -0.5), cx(0.5, 0.5)]],
        ),
        Gate::Sxdg(q) => one_q(
            s,
            q,
            [[cx(0.5, -0.5), cx(0.5, 0.5)], [cx(0.5, 0.5), cx(0.5, -0.5)]],
        ),
        Gate::U(q, th, ph, la) => {
            let (c, sn) = ((th / 2.0).cos(), (th / 2.0).sin());
            one_q(
                s,
                q,
                [[cx(c, 0.0), -e(la) * sn], [e(ph) * sn, e(ph + la) * c]],
            )
        }
        Gate::ISwap(a, b) | Gate::ISwapdg(a, b) => {
            let ph = if matches!(g, Gate::ISwap(..)) {
                cx(0.0, 1.0)
            } else {
                cx(0.0, -1.0)
            };
            let old = s.a.clone();
            for i in 0..s.a.len() {
                let (ba, bb) = ((i >> a) & 1, (i >> b) & 1);
                let j = (i & !(1 << a) & !(1 << b)) | (ba << b) | (bb << a);
                s.a[j] = if ba != bb { ph * old[i] } else { old[i] };
            }
        }
        _ => s.apply(g),
    }
}

/// Branch tree of a circuit: key = outcome sequence of every stochastic
/// op (measure 0/1, reset 2/3, flip 4/5), value = unnormalised state.
pub fn branches(c: &Circuit, init: &RefSv) -> BTreeMap<Vec<u8>, Vec<C>> {
    branches_capped(c, init, usize::MAX).expect("uncapped")
}

/// [`branches`], or `None` once more than `cap` leaves would be produced
/// (the tree is exponential in the number of random outcomes).
pub fn branches_capped(c: &Circuit, init: &RefSv, cap: usize) -> Option<BTreeMap<Vec<u8>, Vec<C>>> {
    struct Out {
        map: BTreeMap<Vec<u8>, Vec<C>>,
        cap: usize,
        overflow: bool,
    }
    fn go(ops: &[Op], mut s: RefSv, mut key: Vec<u8>, rec: Vec<bool>, out: &mut Out) {
        if out.overflow {
            return;
        }
        for (k, op) in ops.iter().enumerate() {
            let rest = &ops[k + 1..];
            let proj = |s: &RefSv, q: usize, b: bool| -> RefSv {
                let mut t = s.clone();
                for (i, x) in t.a.iter_mut().enumerate() {
                    if ((i >> q) & 1 == 1) != b {
                        *x = cx(0.0, 0.0);
                    }
                }
                t
            };
            let norm = |s: &RefSv| s.a.iter().map(|x| x.norm_sqr()).sum::<f64>();
            match *op {
                Op::Gate(g) => apply(&mut s, &g),
                Op::Measure(q) => {
                    for b in [false, true] {
                        let t = proj(&s, q, b);
                        if norm(&t) < 1e-24 {
                            continue;
                        }
                        let (mut k2, mut r2) = (key.clone(), rec.clone());
                        k2.push(b as u8);
                        r2.push(b);
                        go(rest, t, k2, r2, out);
                    }
                    return;
                }
                Op::Reset(q) => {
                    for b in [false, true] {
                        let mut t = proj(&s, q, b);
                        if norm(&t) < 1e-24 {
                            continue;
                        }
                        if b {
                            apply(&mut t, &Gate::X(q));
                        }
                        let mut k2 = key.clone();
                        k2.push(2 + b as u8);
                        go(rest, t, k2, rec.clone(), out);
                    }
                    return;
                }
                Op::ClassicControlled {
                    gate,
                    meas_index,
                    target_value,
                } => {
                    if rec[meas_index] == target_value {
                        apply(&mut s, &gate);
                    }
                }
                Op::XFlip(q, p) | Op::YFlip(q, p) | Op::ZFlip(q, p) => {
                    let g = match *op {
                        Op::XFlip(..) => Gate::X(q),
                        Op::YFlip(..) => Gate::Y(q),
                        _ => Gate::Z(q),
                    };
                    let mut t = s.clone();
                    apply(&mut t, &g);
                    let (a, b) = ((1.0 - p).sqrt(), p.sqrt());
                    s.a.iter_mut().for_each(|x| *x *= a);
                    t.a.iter_mut().for_each(|x| *x *= b);
                    let mut k2 = key.clone();
                    k2.push(5);
                    go(rest, t, k2, rec.clone(), out);
                    key.push(4);
                }
                _ => unreachable!("not generated"),
            }
        }
        if out.map.len() >= out.cap {
            out.overflow = true;
            return;
        }
        out.map.insert(key, s.a);
    }
    let mut out = Out {
        map: BTreeMap::new(),
        cap,
        overflow: false,
    };
    go(&c.ops, init.clone(), Vec::new(), Vec::new(), &mut out);
    (!out.overflow).then_some(out.map)
}

/// max over branches of |a - e^{iφ} b|.
pub fn compare(a: &Circuit, b: &Circuit, phase: f64, init: &RefSv) -> f64 {
    let (ba, bb) = (branches(a, init), branches(b, init));
    let ph = C::from_polar(1.0, phase);
    let zero = vec![cx(0.0, 0.0); init.a.len()];
    let mut worst = 0.0f64;
    for k in ba.keys().chain(bb.keys()) {
        let x = ba.get(k).unwrap_or(&zero);
        let y = bb.get(k).unwrap_or(&zero);
        for (u, v) in x.iter().zip(y) {
            worst = worst.max((u - ph * v).norm());
        }
    }
    worst
}

/// A random normalised state (reference only, no library code).
pub fn random_state(n: usize, rng: &mut StdRng) -> RefSv {
    let mut s = RefSv::new(n);
    let mut nn = 0.0;
    for x in s.a.iter_mut() {
        *x = cx(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0));
        nn += x.norm_sqr();
    }
    let k = 1.0 / nn.sqrt();
    s.a.iter_mut().for_each(|x| *x *= k);
    s
}

pub fn basis(n: usize, col: usize) -> RefSv {
    let mut s = RefSv::new(n);
    s.a[0] = cx(0.0, 0.0);
    s.a[col] = cx(1.0, 0.0);
    s
}

/// Exact distribution of measurement records (one bool per `Measure`, in
/// program order) from the branch tree.
pub fn record_distribution(c: &Circuit) -> BTreeMap<Vec<bool>, f64> {
    record_distribution_capped(c, usize::MAX).expect("uncapped")
}

/// [`record_distribution`], or `None` if the branch tree has more than
/// `cap` leaves.
pub fn record_distribution_capped(c: &Circuit, cap: usize) -> Option<BTreeMap<Vec<bool>, f64>> {
    let mut out = BTreeMap::new();
    for (k, v) in branches_capped(c, &RefSv::new(c.num_qubits), cap)? {
        let rec: Vec<bool> = k.iter().filter(|&&b| b < 2).map(|&b| b == 1).collect();
        *out.entry(rec).or_insert(0.0) += v.iter().map(|x| x.norm_sqr()).sum::<f64>();
    }
    Some(out)
}
