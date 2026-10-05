//! Entanglement of `|ψ> = C (|φ>_A ⊗ |0>)` across a physical cut, exactly,
//! from the stabilizer frame plus the compressed register.
//!
//! Let `G = <S_j : j inactive>` (stabilizers of ψ) and, for a region `R`,
//! `H_R` = Paulis supported on `R` commuting with `G`. `H_R` contains
//! `G_R = G ∩ Paulis(R)` (`g` generators) and maps onto a logical Pauli
//! group `L_R` on the register (`k` generators, symplectic rank `2a`,
//! centre `b = k − 2a`). Up to a Clifford on `R`,
//! `ρ_R = |0><0|^{⊗g} ⊗ σ ⊗ (I/2)^{⊗(|R|−g−a−b)}` where `σ` is the state of
//! `φ` on the `a` logical pairs with the `b` central logicals dephased. So
//!
//! `S_α(ρ_R) = |R| − g − a − b + S_α(σ)`,  `0 ≤ S_α(σ) ≤ a + b`.
//!
//! The frame part is polynomial (GF(2) elimination); `S_2(σ)` is computed
//! from `φ` after a register Clifford that maps `L_R` to standard form.

use super::{dense_cnot, dense_cz, dense_h, dense_s, LPauli, Monitored};
use num_complex::Complex64 as C64;

#[derive(Clone, Debug)]
pub struct CutEntropy {
    pub size: usize,
    /// generators of G restricted to the region
    pub g: usize,
    /// logical symplectic pairs and central logicals
    pub a: usize,
    pub b: usize,
    /// `|R| − g − a − b` and that `+ a + b` (bounds on every Rényi entropy).
    pub lower: f64,
    pub upper: f64,
    /// exact Rényi-2 entropy (bits), when the register is tracked and small.
    pub s2: Option<f64>,
}

fn getb(v: &[u64], i: usize) -> bool {
    v[i >> 6] >> (i & 63) & 1 == 1
}
fn setb(v: &mut [u64], i: usize) {
    v[i >> 6] |= 1 << (i & 63);
}

/// Entropy of the physical qubits with `region[q] == true`.
pub fn cut_entropy(sim: &Monitored, region: &[bool], max_cost_log2: u32) -> CutEntropy {
    let n = sim.n;
    let tab = &sim.tab;
    let d = sim.d();
    let mut inactive = Vec::with_capacity(n - d);
    {
        let mut is_act = vec![false; n];
        for &c in &sim.active {
            is_act[c] = true;
        }
        for (j, &act) in is_act.iter().enumerate() {
            if !act {
                inactive.push(j);
            }
        }
    }
    let nc = inactive.len();
    let ncol = nc + 2 * d;
    let cw = ncol.div_ceil(64).max(1);
    let qs: Vec<usize> = (0..n).filter(|&q| region[q]).collect();
    let size = qs.len();
    // rows: X_q then Z_q for q in region
    let mut rows: Vec<Vec<u64>> = Vec::with_capacity(2 * size);
    for &q in &qs {
        for kind in 0..2 {
            let mut r = vec![0u64; cw];
            // X_q anticommutes with row iff row.z[q]; Z_q iff row.x[q]
            let f = |row: usize| -> bool {
                if kind == 0 {
                    tab.zbit(row, q) == 1
                } else {
                    tab.xbit(row, q) == 1
                }
            };
            for (k, &j) in inactive.iter().enumerate() {
                if f(n + j) {
                    setb(&mut r, k);
                }
            }
            for (p, &j) in sim.active.iter().enumerate() {
                if f(n + j) {
                    setb(&mut r, nc + p); // logical x on p
                }
                if f(j) {
                    setb(&mut r, nc + d + p); // logical z on p
                }
            }
            rows.push(r);
        }
    }
    // eliminate on the constraint columns
    let mut rank = 0;
    for col in 0..nc {
        let Some(pr) = (rank..rows.len()).find(|&i| getb(&rows[i], col)) else {
            continue;
        };
        rows.swap(rank, pr);
        let piv = rows[rank].clone();
        for r in rows.iter_mut().skip(rank + 1) {
            if getb(r, col) {
                for (a, b) in r.iter_mut().zip(&piv) {
                    *a ^= b;
                }
            }
        }
        rank += 1;
    }
    let h = rows.len() - rank;
    // logical parts of the kernel rows
    let mut logi: Vec<Vec<u64>> = rows[rank..]
        .iter()
        .map(|r| {
            let mut v = vec![0u64; (2 * d).div_ceil(64).max(1)];
            for i in 0..2 * d {
                if getb(r, nc + i) {
                    setb(&mut v, i);
                }
            }
            v
        })
        .collect();
    let mut k = 0;
    for col in 0..2 * d {
        let Some(pr) = (k..logi.len()).find(|&i| getb(&logi[i], col)) else {
            continue;
        };
        logi.swap(k, pr);
        let piv = logi[k].clone();
        for (i, r) in logi.iter_mut().enumerate() {
            if i != k && getb(r, col) {
                for (a, b) in r.iter_mut().zip(&piv) {
                    *a ^= b;
                }
            }
        }
        k += 1;
    }
    logi.truncate(k);
    let g = h - k;
    // symplectic rank of the logical group
    let sym = |u: &[u64], v: &[u64]| -> bool {
        let mut s = false;
        for p in 0..d {
            s ^= getb(u, p) & getb(v, d + p);
            s ^= getb(u, d + p) & getb(v, p);
        }
        s
    };
    let mut m: Vec<Vec<bool>> = (0..k)
        .map(|i| (0..k).map(|j| sym(&logi[i], &logi[j])).collect())
        .collect();
    let mut r2 = 0;
    for col in 0..k {
        let Some(pr) = (r2..k).find(|&i| m[i][col]) else {
            continue;
        };
        m.swap(r2, pr);
        let piv = m[r2].clone();
        for (i, row) in m.iter_mut().enumerate() {
            if i != r2 && row[col] {
                for (a, b) in row.iter_mut().zip(&piv) {
                    *a ^= b;
                }
            }
        }
        r2 += 1;
    }
    let a = r2 / 2;
    let b = k - 2 * a;
    let lower = (size - g - a - b) as f64;
    let upper = lower + (a + b) as f64;
    let mut s2 = None;
    if let Some(amp) = &sim.amp {
        if a + b == 0 {
            s2 = Some(lower);
        } else if d <= 34 {
            let basis: Vec<LPauli> = logi
                .iter()
                .map(|v| {
                    let mut p = LPauli { x: 0, z: 0, r: 0 };
                    for q in 0..d {
                        if getb(v, q) {
                            p.x |= 1 << q;
                        }
                        if getb(v, d + q) {
                            p.z |= 1 << q;
                        }
                    }
                    p
                })
                .collect();
            if let Some(v) = sigma_s2(amp, d, basis, max_cost_log2) {
                s2 = Some(lower + v);
            }
        }
    }
    CutEntropy {
        size,
        g,
        a,
        b,
        lower,
        upper,
        s2,
    }
}

fn anti(p: &LPauli, q: &LPauli) -> bool {
    ((p.x & q.z).count_ones() + (p.z & q.x).count_ones()) % 2 == 1
}

/// Applies a register gate to φ and conjugates the remaining logicals.
enum G {
    H(usize),
    S(usize),
    Cx(usize, usize),
    Cz(usize, usize),
}

fn apply(g: &G, phi: &mut [C64], list: &mut [LPauli]) {
    match *g {
        G::H(q) => dense_h(phi, q),
        G::S(q) => dense_s(phi, q, false),
        G::Cx(c, t) => dense_cnot(phi, c, t),
        G::Cz(a, b) => dense_cz(phi, a, b),
    }
    for p in list.iter_mut() {
        match *g {
            G::H(q) => p.h(q),
            G::S(q) => p.s(q),
            G::Cx(c, t) => p.cnot(c, t),
            G::Cz(a, b) => p.cz(a, b),
        }
    }
}

/// Reduces `list[i]` to a single `Z_u` (its support only); returns `u`.
fn to_z(i: usize, phi: &mut [C64], list: &mut [LPauli]) -> usize {
    let p = list[i];
    if p.x != 0 {
        let u = p.x.trailing_zeros() as usize;
        let mut xs = p.x & !(1 << u);
        while xs != 0 {
            let t = xs.trailing_zeros() as usize;
            xs &= xs - 1;
            apply(&G::Cx(u, t), phi, list);
        }
        let mut zs = list[i].z & !(1 << u);
        while zs != 0 {
            let t = zs.trailing_zeros() as usize;
            zs &= zs - 1;
            apply(&G::Cz(u, t), phi, list);
        }
        if list[i].z >> u & 1 == 1 {
            apply(&G::S(u), phi, list);
        }
        apply(&G::H(u), phi, list);
        debug_assert!(list[i].x == 0 && list[i].z == 1 << u);
        u
    } else {
        let u = p.z.trailing_zeros() as usize;
        let mut zs = p.z & !(1 << u);
        while zs != 0 {
            let t = zs.trailing_zeros() as usize;
            zs &= zs - 1;
            apply(&G::Cx(t, u), phi, list);
        }
        debug_assert!(list[i].x == 0 && list[i].z == 1 << u);
        u
    }
}

/// `S_2(σ)`: σ = state of φ on the logical pairs, central logicals dephased.
fn sigma_s2(amp: &[C64], d: usize, mut list: Vec<LPauli>, max_cost_log2: u32) -> Option<f64> {
    let mut phi = amp.to_vec();
    let mut pairs = Vec::new();
    let mut centre = Vec::new();
    // pairs
    loop {
        let mut found = None;
        'f: for i in 0..list.len() {
            for j in i + 1..list.len() {
                if anti(&list[i], &list[j]) {
                    found = Some((i, j));
                    break 'f;
                }
            }
        }
        let Some((i, j)) = found else { break };
        let u = to_z(i, &mut phi, &mut list);
        // f has x_u = 1; make it X_u keeping Z_u
        let f = list[j];
        let mut xs = f.x & !(1 << u);
        while xs != 0 {
            let t = xs.trailing_zeros() as usize;
            xs &= xs - 1;
            apply(&G::Cx(u, t), &mut phi, &mut list);
        }
        let mut zs = list[j].z & !(1 << u);
        while zs != 0 {
            let t = zs.trailing_zeros() as usize;
            zs &= zs - 1;
            apply(&G::Cz(u, t), &mut phi, &mut list);
        }
        if list[j].z >> u & 1 == 1 {
            apply(&G::S(u), &mut phi, &mut list);
        }
        debug_assert!(list[j].x == 1 << u && list[j].z == 0 && list[i].z == 1 << u);
        let (e, fp) = (list[i], list[j]);
        // clean the rest off qubit u
        let mut rest: Vec<LPauli> = Vec::new();
        for (k, p) in list.iter().enumerate() {
            if k == i || k == j {
                continue;
            }
            let mut p = *p;
            if p.x >> u & 1 == 1 {
                p.x ^= fp.x;
                p.z ^= fp.z;
            }
            if p.z >> u & 1 == 1 {
                p.x ^= e.x;
                p.z ^= e.z;
            }
            rest.push(p);
        }
        list = rest;
        pairs.push(u);
    }
    // centre (mutually commuting)
    while !list.is_empty() {
        let u = to_z(0, &mut phi, &mut list);
        let e = list[0];
        let mut rest = Vec::new();
        for p in &list[1..] {
            let mut p = *p;
            if p.z >> u & 1 == 1 {
                p.z ^= e.z;
            }
            debug_assert!(p.x >> u & 1 == 0);
            if p.x != 0 || p.z != 0 {
                rest.push(p);
            }
        }
        list = rest;
        centre.push(u);
    }
    let a = pairs.len();
    let b = centre.len();
    let rr = d - a - b;
    let side = a.min(rr) as u32;
    if d as u32 + side > max_cost_log2 {
        return None;
    }
    let pm: u64 = pairs.iter().fold(0, |m, &u| m | 1 << u);
    let cm: u64 = centre.iter().fold(0, |m, &u| m | 1 << u);
    let rest_bits: Vec<usize> = (0..d).filter(|&q| (pm | cm) >> q & 1 == 0).collect();
    let pext = |y: u64, bits: &[usize]| -> usize {
        let mut o = 0usize;
        for (i, &q) in bits.iter().enumerate() {
            o |= ((y >> q & 1) as usize) << i;
        }
        o
    };
    let cb: Vec<usize> = centre.clone();
    let pb: Vec<usize> = pairs.clone();
    let mut mats = vec![vec![C64::new(0.0, 0.0); (1 << a) * (1 << rr)]; 1 << b];
    for (y, &v) in phi.iter().enumerate() {
        let y = y as u64;
        let c = pext(y, &cb);
        let p = pext(y, &pb);
        let r = pext(y, &rest_bits);
        mats[c][p * (1 << rr) + r] = v;
    }
    let (np, nr) = (1usize << a, 1usize << rr);
    let mut tr2 = 0.0f64;
    for m in &mats {
        if a <= rr {
            for p in 0..np {
                for p2 in 0..np {
                    let mut s = C64::new(0.0, 0.0);
                    for r in 0..nr {
                        s += m[p * nr + r] * m[p2 * nr + r].conj();
                    }
                    tr2 += s.norm_sqr();
                }
            }
        } else {
            for r in 0..nr {
                for r2 in 0..nr {
                    let mut s = C64::new(0.0, 0.0);
                    for p in 0..np {
                        s += m[p * nr + r] * m[p * nr + r2].conj();
                    }
                    tr2 += s.norm_sqr();
                }
            }
        }
    }
    Some(-tr2.log2())
}
