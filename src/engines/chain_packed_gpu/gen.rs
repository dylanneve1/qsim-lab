//! Generator of the sub-stage kernel (WGSL) and its op programs.
//!
//! A workgroup runs the ops of one cache block (`2^sl <= MAXN` amplitudes).
//! Each thread keeps `EPT = 2^rb` amplitudes in registers: thread `tid`
//! holds the buffer indices `j = pdep(tid, !R) | pdep(i, R)`, `i < EPT`,
//! for a set `R` of `rb` buffer bits chosen per cache block (the most
//! targeted ones). Ops whose targets all lie in `R` (and every diagonal
//! group) run on the registers with no barrier; the others run on the
//! workgroup buffer, which the registers are written to and read back from
//! around each run of such ops. The arithmetic of every op is exactly the
//! CPU kernel's (same f32 `fma`/`mul` sequence per element); only where the
//! data lives changes. Register code is fully unrolled (constant indices),
//! so it stays in registers.

use crate::engines::blocked::gpu_export::{GpuOp, GpuSubStage};
use std::fmt::Write;

/// Program words before the records: count, `!R` mask, then `pdep(i, R)`
/// for `i < EPT`.
pub const HDR: usize = 2;

/// Record kinds.
mod kind {
    pub const U1: u32 = 1;
    pub const SWAP: u32 = 2;
    pub const PAIR: u32 = 3;
    pub const DIAG: u32 = 4;
    pub const RU1: u32 = 11;
    pub const RSWAP: u32 = 12;
    pub const RPAIR: u32 = 13;
    pub const RDIAG: u32 = 14;
    pub const TO_SHARED: u32 = 20;
    pub const TO_REGS: u32 = 21;
}

const NONE: u32 = u32::MAX;

/// WGSL of the sub-stage kernel for `2^rb` registers per thread
/// (`MAXN`, `SWG` = `MAXN >> rb`, `EPT` substituted by the caller's
/// constants).
pub fn substage_wgsl(rb: u32) -> String {
    let ept = 1usize << rb;
    let d0 = HDR;
    let mut s = String::new();
    let w = &mut s;
    writeln!(w, "const EPT: u32 = {ept}u;").unwrap();
    writeln!(w, "const RB: u32 = {rb}u;").unwrap();
    writeln!(w, "var<workgroup> sb: array<vec2<f32>, MAXN>;").unwrap();
    writeln!(w).unwrap();
    let vty = format!("ptr<function, array<vec2<f32>, {ept}>>");
    // register U1 on register bit b
    for b in 0..rb {
        writeln!(
            w,
            "fn ru1_{b}(v: {vty}, tb: u32, mo: u32, uk: u32, cin: u32) {{"
        )
        .unwrap();
        for i0 in 0..ept {
            if i0 >> b & 1 == 1 {
                continue;
            }
            let i1 = i0 | 1 << b;
            writeln!(
                w,
                "    if (((tb | pw({}u)) & cin) == cin) {{
        let x = (*v)[{i0}];
        let y = (*v)[{i1}];
        if (uk == 0u) {{
            (*v)[{i0}] = y;
            (*v)[{i1}] = x;
        }} else {{
            let r = mat(mo, uk == 1u, x, y);
            (*v)[{i0}] = r.xy;
            (*v)[{i1}] = r.zw;
        }}
    }}",
                d0 + i0
            )
            .unwrap();
        }
        writeln!(w, "}}\n").unwrap();
    }
    // register swap of register bits a < b
    for a in 0..rb {
        for b in a + 1..rb {
            writeln!(w, "fn rsw_{a}_{b}(v: {vty}) {{").unwrap();
            for i in 0..ept {
                if i >> a & 1 == 1 && i >> b & 1 == 0 {
                    let k = i ^ (1 << a) ^ (1 << b);
                    writeln!(
                        w,
                        "    {{ let x = (*v)[{i}]; (*v)[{i}] = (*v)[{k}]; (*v)[{k}] = x; }}"
                    )
                    .unwrap();
                }
            }
            writeln!(w, "}}\n").unwrap();
        }
    }
    // register pair on register bits a < b
    for a in 0..rb {
        for b in a + 1..rb {
            writeln!(
                w,
                "fn rpair_{a}_{b}(v: {vty}, m1: u32, r1: bool, m2: u32, r2: bool, cx: u32) {{"
            )
            .unwrap();
            for i in 0..ept {
                if i >> a & 1 == 1 || i >> b & 1 == 1 {
                    continue;
                }
                let (i1, i2, i3) = (i | 1 << a, i | 1 << b, i | 1 << a | 1 << b);
                writeln!(
                    w,
                    "    {{
        let b01 = mat(m1, r1, (*v)[{i}], (*v)[{i1}]);
        let b23 = mat(m1, r1, (*v)[{i2}], (*v)[{i3}]);
        let c02 = mat(m2, r2, b01.xy, b23.xy);
        let c13 = mat(m2, r2, b01.zw, b23.zw);
        var c1 = c13.xy;
        var c2 = c02.zw;
        var c3 = c13.zw;
        if (cx == 1u) {{
            let t = c1;
            c1 = c3;
            c3 = t;
        }} else if (cx == 2u) {{
            let t = c2;
            c2 = c3;
            c3 = t;
        }}
        (*v)[{i}] = c02.xy;
        (*v)[{i1}] = c1;
        (*v)[{i2}] = c2;
        (*v)[{i3}] = c3;
    }}"
                )
                .unwrap();
            }
            writeln!(w, "}}\n").unwrap();
        }
    }
    // register diagonal group
    writeln!(
        w,
        "fn rdiag(v: {vty}, tb: u32, cmask: u32, cpat: u32, lb: u32, n: u32, toff: u32) {{
    let nlo = 1u << lb;
    let nhi = n >> lb;"
    )
    .unwrap();
    for i in 0..ept {
        writeln!(
            w,
            "    {{ let j = tb | pw({}u); if ((j & cmask) == cpat) {{ (*v)[{i}] = dfac((*v)[{i}], j, lb, nlo, nhi, toff); }} }}",
            d0 + i
        )
        .unwrap();
    }
    writeln!(w, "}}\n").unwrap();
    // register <-> workgroup buffer and global memory
    writeln!(w, "fn to_sb(v: {vty}, tb: u32) {{").unwrap();
    for i in 0..ept {
        writeln!(w, "    sb[tb | pw({}u)] = (*v)[{i}];", d0 + i).unwrap();
    }
    writeln!(w, "}}\n").unwrap();
    writeln!(w, "fn from_sb(v: {vty}, tb: u32) {{").unwrap();
    for i in 0..ept {
        writeln!(w, "    (*v)[{i}] = sb[tb | pw({}u)];", d0 + i).unwrap();
    }
    writeln!(w, "}}\n").unwrap();
    writeln!(w, "fn gaddr(j: u32, lowm: u32) -> u32 {{\n    return (j & lowm) | pdep(j >> P.lowb, P.himask);\n}}\n").unwrap();
    writeln!(w, "fn ld_glob(v: {vty}, tb: u32, g0: u32, lowm: u32) {{").unwrap();
    for i in 0..ept {
        writeln!(
            w,
            "    (*v)[{i}] = work[g0 | gaddr(tb | pw({}u), lowm)];",
            d0 + i
        )
        .unwrap();
    }
    writeln!(w, "}}\n").unwrap();
    writeln!(w, "fn st_glob(v: {vty}, tb: u32, g0: u32, lowm: u32) {{").unwrap();
    for i in 0..ept {
        writeln!(
            w,
            "    work[g0 | gaddr(tb | pw({}u), lowm)] = (*v)[{i}];",
            d0 + i
        )
        .unwrap();
    }
    writeln!(w, "}}\n").unwrap();

    // dispatch helpers (runtime register-bit numbers -> unrolled code)
    writeln!(
        w,
        "fn ru1(v: {vty}, b: u32, tb: u32, mo: u32, uk: u32, cin: u32) {{"
    )
    .unwrap();
    for b in 0..rb {
        writeln!(w, "    if (b == {b}u) {{ ru1_{b}(v, tb, mo, uk, cin); }}").unwrap();
    }
    writeln!(w, "}}\n").unwrap();
    writeln!(w, "fn rsw(v: {vty}, a: u32, b: u32) {{").unwrap();
    for a in 0..rb {
        for b in a + 1..rb {
            writeln!(w, "    if (a == {a}u && b == {b}u) {{ rsw_{a}_{b}(v); }}").unwrap();
        }
    }
    writeln!(w, "}}\n").unwrap();
    writeln!(
        w,
        "fn rpair(v: {vty}, a: u32, b: u32, m1: u32, r1: bool, m2: u32, r2: bool, cx: u32) {{"
    )
    .unwrap();
    for a in 0..rb {
        for b in a + 1..rb {
            writeln!(
                w,
                "    if (a == {a}u && b == {b}u) {{ rpair_{a}_{b}(v, m1, r1, m2, r2, cx); }}"
            )
            .unwrap();
        }
    }
    writeln!(w, "}}\n").unwrap();

    w.push_str(KERNEL);
    s
}

/// The sub-stage entry point (uses the generated helpers above).
const KERNEL: &str = r#"
@compute @workgroup_size(SWG)
fn substage(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let w = wid.y * P.nx + wid.x;
    if (w >= P.total) {
        return;
    }
    let per = 1u << (P.l - P.sl);
    let gi = w / per;
    let cc = w % per;
    let n = 1u << P.sl;
    let base = pdep(cc, P.outmask);
    let cblk = P.c0 + gi;
    // register index of the cache block (64-bit: lo, hi)
    let rlo = (cblk << P.l) | base;
    var rhi = 0u;
    if (P.l > 0u) {
        rhi = cblk >> (32u - P.l);
    }
    let g0 = (gi << P.l) | base;
    let lowm = (1u << P.lowb) - 1u;
    let act = lid < (n >> RB);
    let tb = pdep(lid, pw(1u));
    var v: array<vec2<f32>, EPT>;
    if (act) {
        ld_glob(&v, tb, g0, lowm);
    }
    let nrec = pw(0u);
    var pc = 2u + EPT;
    for (var rec = 0u; rec < nrec; rec++) {
        let kind = pw(pc);
        if (kind == 11u) {
            // register U1: b, kind, cin, cout lo, cout hi, m[8]
            let colo = pw(pc + 4u);
            let cohi = pw(pc + 5u);
            if (act && (rlo & colo) == colo && (rhi & cohi) == cohi) {
                ru1(&v, pw(pc + 1u), tb, pc + 6u, pw(pc + 2u), pw(pc + 3u));
            }
            pc += 14u;
        } else if (kind == 13u) {
            // register pair: a, b, real1, real2, cx, m1[8], m2[8]
            if (act) {
                rpair(&v, pw(pc + 1u), pw(pc + 2u), pc + 6u, pw(pc + 3u) != 0u, pc + 14u,
                      pw(pc + 4u) != 0u, pw(pc + 5u));
            }
            pc += 22u;
        } else if (kind == 12u) {
            if (act) {
                rsw(&v, pw(pc + 1u), pw(pc + 2u));
            }
            pc += 3u;
        } else if (kind == 4u || kind == 14u) {
            // diagonal group: cmask, cpat, lb, nconds, voff, conds[4 each]
            let cmask = pw(pc + 1u);
            let cpat = pw(pc + 2u);
            let lb = pw(pc + 3u);
            let nc = pw(pc + 4u);
            let voff = pw(pc + 5u);
            var pat = 0u;
            for (var k = 0u; k < nc; k++) {
                let q = pc + 6u + 4u * k;
                if ((rlo & pw(q)) == pw(q + 2u) && (rhi & pw(q + 1u)) == pw(q + 3u)) {
                    pat = pat | (1u << k);
                }
            }
            let toff = bitcast<u32>(tables[voff + pat]);
            if (kind == 14u) {
                if (act && toff != NONE) {
                    rdiag(&v, tb, cmask, cpat, lb, n, toff);
                }
            } else {
                if (toff != NONE) {
                    let nlo = 1u << lb;
                    let nhi = n >> lb;
                    for (var j = lid; j < n; j += SWG) {
                        if ((j & cmask) == cpat) {
                            sb[j] = dfac(sb[j], j, lb, nlo, nhi, toff);
                        }
                    }
                }
                workgroupBarrier();
            }
            pc += 6u + 4u * nc;
        } else if (kind == 20u) {
            if (act) {
                to_sb(&v, tb);
            }
            workgroupBarrier();
            pc += 1u;
        } else if (kind == 21u) {
            if (act) {
                from_sb(&v, tb);
            }
            workgroupBarrier();
            pc += 1u;
        } else if (kind == 1u) {
            // U1 on the workgroup buffer: t, kind, cin, cout lo, cout hi, m[8]
            let t = pw(pc + 1u);
            let uk = pw(pc + 2u);
            let cin = pw(pc + 3u);
            let colo = pw(pc + 4u);
            let cohi = pw(pc + 5u);
            if ((rlo & colo) == colo && (rhi & cohi) == cohi) {
                let s = 1u << t;
                for (var p = lid; p < n / 2u; p += SWG) {
                    let i = ins0(p, t);
                    if ((i & cin) == cin) {
                        let x = sb[i];
                        let y = sb[i | s];
                        if (uk == 0u) {
                            sb[i] = y;
                            sb[i | s] = x;
                        } else {
                            let r = mat(pc + 6u, uk == 1u, x, y);
                            sb[i] = r.xy;
                            sb[i | s] = r.zw;
                        }
                    }
                }
            }
            workgroupBarrier();
            pc += 14u;
        } else if (kind == 2u) {
            let a = pw(pc + 1u);
            let b = pw(pc + 2u);
            let sa = 1u << a;
            let sbb = 1u << b;
            for (var p = lid; p < n / 4u; p += SWG) {
                let i = ins0(ins0(p, a), b);
                let x = sb[i | sa];
                sb[i | sa] = sb[i | sbb];
                sb[i | sbb] = x;
            }
            workgroupBarrier();
            pc += 3u;
        } else {
            // kind 3: pair on the workgroup buffer
            let t1 = pw(pc + 1u);
            let t2 = pw(pc + 2u);
            let r1 = pw(pc + 3u) != 0u;
            let r2 = pw(pc + 4u) != 0u;
            let cx = pw(pc + 5u);
            let s1 = 1u << t1;
            let s2 = 1u << t2;
            for (var p = lid; p < n / 4u; p += SWG) {
                let i = ins0(ins0(p, t1), t2);
                let b01 = mat(pc + 6u, r1, sb[i], sb[i | s1]);
                let b23 = mat(pc + 6u, r1, sb[i | s2], sb[i | s1 | s2]);
                let c02 = mat(pc + 14u, r2, b01.xy, b23.xy);
                let c13 = mat(pc + 14u, r2, b01.zw, b23.zw);
                var c1 = c13.xy;
                var c2 = c02.zw;
                var c3 = c13.zw;
                if (cx == 1u) {
                    let t = c1;
                    c1 = c3;
                    c3 = t;
                } else if (cx == 2u) {
                    let t = c2;
                    c2 = c3;
                    c3 = t;
                }
                sb[i] = c02.xy;
                sb[i | s1] = c1;
                sb[i | s2] = c2;
                sb[i | s1 | s2] = c3;
            }
            workgroupBarrier();
            pc += 22u;
        }
    }
    if (act) {
        st_glob(&v, tb, g0, lowm);
    }
}
"#;

fn pdep(x: usize, mut mask: usize) -> usize {
    let (mut out, mut x) = (0, x);
    while mask != 0 && x != 0 {
        let b = mask & mask.wrapping_neg();
        if x & 1 == 1 {
            out |= b;
        }
        x >>= 1;
        mask &= mask - 1;
    }
    out
}

/// Non-diagonal target bits of an op.
fn targets(op: &GpuOp) -> u32 {
    match op {
        GpuOp::U1 { t, .. } => 1 << t,
        GpuOp::Swap { a, b } => (1 << a) | (1 << b),
        GpuOp::Pair { t1, t2, .. } => (1 << t1) | (1 << t2),
        GpuOp::Diag(_) => 0,
    }
}

/// Encodes the ops of a cache block into a program for `2^rb` registers
/// per thread, appending the diagonal tables to `tables` (as f32 bits).
/// Returns the program words and the number of register-local ops.
pub fn encode_sub(s: &GpuSubStage, rb: u32, tables: &mut Vec<u32>) -> (Vec<u32>, usize) {
    let rb = rb as usize;
    assert!(s.l >= rb, "cache block smaller than the register tile");
    let ept = 1usize << rb;
    // register bits: the most targeted buffer bits
    let mut hist = vec![0usize; s.l];
    for op in &s.ops {
        let m = targets(op);
        for (b, h) in hist.iter_mut().enumerate() {
            *h += (m >> b & 1) as usize;
        }
    }
    let mut order: Vec<usize> = (0..s.l).collect();
    order.sort_by_key(|&b| (std::cmp::Reverse(hist[b]), std::cmp::Reverse(b)));
    let rmask: u32 = order.iter().take(rb).fold(0, |m, &b| m | 1 << b);
    let rank = |t: u32| (rmask & ((1u32 << t) - 1)).count_ones();
    let full = ((1u64 << s.l) - 1) as u32;
    let mut w = vec![0u32, full & !rmask];
    w.extend((0..ept).map(|i| pdep(i, rmask as usize) as u32));
    let mut nrec = 0u32;
    let mut local = 0usize;
    let fb = |m: &[f32; 8]| m.map(f32::to_bits);
    let mut shared = false;
    // whether the next non-diagonal op from `k` needs the workgroup buffer
    let needs = |op: &GpuOp| targets(op) & !rmask != 0;
    let next_needs = |k: usize| {
        s.ops[k..]
            .iter()
            .find(|o| !matches!(o, GpuOp::Diag(_)))
            .is_some_and(needs)
    };
    for (k, op) in s.ops.iter().enumerate() {
        let want_shared = match op {
            GpuOp::Diag(_) => shared && next_needs(k + 1),
            _ => needs(op),
        };
        if want_shared != shared {
            w.push(if want_shared {
                kind::TO_SHARED
            } else {
                kind::TO_REGS
            });
            nrec += 1;
            shared = want_shared;
        }
        if !shared {
            local += 1;
        }
        match op {
            GpuOp::U1 {
                t,
                m,
                kind: uk,
                cin,
                cout,
            } => {
                let (k1, tt) = if shared {
                    (kind::U1, *t)
                } else {
                    (kind::RU1, rank(*t))
                };
                w.extend([k1, tt, *uk as u32, *cin, *cout as u32, (*cout >> 32) as u32]);
                w.extend(fb(m));
                nrec += 1;
            }
            GpuOp::Swap { a, b } => {
                if shared {
                    w.extend([kind::SWAP, *a, *b]);
                } else {
                    w.extend([kind::RSWAP, rank(*a), rank(*b)]);
                }
                nrec += 1;
            }
            GpuOp::Pair {
                t1,
                m1,
                real1,
                t2,
                m2,
                real2,
                cx,
            } => {
                let (k1, a, b) = if shared {
                    (kind::PAIR, *t1, *t2)
                } else {
                    (kind::RPAIR, rank(*t1), rank(*t2))
                };
                w.extend([k1, a, b, *real1 as u32, *real2 as u32, *cx as u32]);
                w.extend(fb(m1));
                w.extend(fb(m2));
                nrec += 1;
            }
            GpuOp::Diag(gs) => {
                for g in gs {
                    let voff = tables.len() as u32;
                    tables.resize(tables.len() + g.variants.len(), NONE);
                    for (i, v) in g.variants.iter().enumerate() {
                        if let Some(t) = v {
                            tables[voff as usize + i] = tables.len() as u32;
                            for part in [&t.lor, &t.loi, &t.hr, &t.hi] {
                                tables.extend(part.iter().map(|x| x.to_bits()));
                            }
                        }
                    }
                    let k1 = if shared { kind::DIAG } else { kind::RDIAG };
                    w.extend([k1, g.cmask, g.cpat, g.lb, g.conds.len() as u32, voff]);
                    for &(om, op) in &g.conds {
                        w.extend([om as u32, (om >> 32) as u32, op as u32, (op >> 32) as u32]);
                    }
                    nrec += 1;
                }
            }
        }
    }
    if shared {
        w.push(kind::TO_REGS);
        nrec += 1;
    }
    w[0] = nrec;
    (w, local)
}

/// WGSL of the plain sub-stage kernel (`reg_bits = 0`): the whole cache
/// block in workgroup memory, one barrier per op (`WG` threads).
pub fn substage_shared_wgsl() -> String {
    SHARED_KERNEL.to_string()
}

const SHARED_KERNEL: &str = r#"
var<workgroup> sre: array<f32, MAXN>;
var<workgroup> sim: array<f32, MAXN>;

fn ld(i: u32) -> vec2<f32> {
    return vec2<f32>(sre[i], sim[i]);
}

fn st(i: u32, v: vec2<f32>) {
    sre[i] = v.x;
    sim[i] = v.y;
}

@compute @workgroup_size(WG)
fn substage(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let w = wid.y * P.nx + wid.x;
    if (w >= P.total) {
        return;
    }
    let per = 1u << (P.l - P.sl);
    let gi = w / per;
    let cc = w % per;
    let n = 1u << P.sl;
    let base = pdep(cc, P.outmask);
    let cblk = P.c0 + gi;
    // register index of the sub-block (64-bit: lo, hi)
    let rlo = (cblk << P.l) | base;
    var rhi = 0u;
    if (P.l > 0u) {
        rhi = cblk >> (32u - P.l);
    }
    let boff = gi << P.l;
    let lowm = (1u << P.lowb) - 1u;
    let glid = (lid & lowm) | pdep(lid >> P.lowb, P.himask);
    for (var j = lid; j < n; j += WG) {
        // gaddr is bitwise linear: gaddr(lid + k WG) = gaddr(lid) | gaddr(k WG)
        let a = work[boff | base | glid | pw(1u + j / WG)];
        sre[j] = a.x;
        sim[j] = a.y;
    }
    workgroupBarrier();
    var pc = 1u + MAXN / WG;
    let nops = pw(0u);
    for (var op = 0u; op < nops; op++) {
        let kind = pw(pc);
        if (kind == 1u) {
            // U1: t, kind, cin, cout lo, cout hi, m[8]
            let t = pw(pc + 1u);
            let uk = pw(pc + 2u);
            let cin = pw(pc + 3u);
            let colo = pw(pc + 4u);
            let cohi = pw(pc + 5u);
            if ((rlo & colo) == colo && (rhi & cohi) == cohi) {
                let s = 1u << t;
                for (var p = lid; p < n / 2u; p += WG) {
                    let i = ins0(p, t);
                    if ((i & cin) == cin) {
                        let x = ld(i);
                        let y = ld(i | s);
                        if (uk == 0u) {
                            st(i, y);
                            st(i | s, x);
                        } else {
                            let r = mat(pc + 6u, uk == 1u, x, y);
                            st(i, r.xy);
                            st(i | s, r.zw);
                        }
                    }
                }
            }
            pc += 14u;
        } else if (kind == 2u) {
            // Swap a < b
            let a = pw(pc + 1u);
            let b = pw(pc + 2u);
            let sa = 1u << a;
            let sb = 1u << b;
            for (var p = lid; p < n / 4u; p += WG) {
                let i = ins0(ins0(p, a), b);
                let x = ld(i | sa);
                st(i | sa, ld(i | sb));
                st(i | sb, x);
            }
            pc += 3u;
        } else if (kind == 3u) {
            // Pair: t1, t2, real1, real2, cx, m1[8], m2[8]
            let t1 = pw(pc + 1u);
            let t2 = pw(pc + 2u);
            let r1 = pw(pc + 3u) != 0u;
            let r2 = pw(pc + 4u) != 0u;
            let cx = pw(pc + 5u);
            let s1 = 1u << t1;
            let s2 = 1u << t2;
            for (var p = lid; p < n / 4u; p += WG) {
                let i = ins0(ins0(p, t1), t2);
                let b01 = mat(pc + 6u, r1, ld(i), ld(i | s1));
                let b23 = mat(pc + 6u, r1, ld(i | s2), ld(i | s1 | s2));
                let c02 = mat(pc + 14u, r2, b01.xy, b23.xy);
                let c13 = mat(pc + 14u, r2, b01.zw, b23.zw);
                var c1 = c13.xy;
                var c2 = c02.zw;
                var c3 = c13.zw;
                if (cx == 1u) {
                    let tmp = c1;
                    c1 = c3;
                    c3 = tmp;
                } else if (cx == 2u) {
                    let tmp = c2;
                    c2 = c3;
                    c3 = tmp;
                }
                st(i, c02.xy);
                st(i | s1, c1);
                st(i | s2, c2);
                st(i | s1 | s2, c3);
            }
            pc += 22u;
        } else {
            // Diagonal group: cmask, cpat, lb, nconds, voff, conds[4 each]
            let cmask = pw(pc + 1u);
            let cpat = pw(pc + 2u);
            let lb = pw(pc + 3u);
            let nc = pw(pc + 4u);
            let voff = pw(pc + 5u);
            var pat = 0u;
            for (var k = 0u; k < nc; k++) {
                let q = pc + 6u + 4u * k;
                if ((rlo & pw(q)) == pw(q + 2u) && (rhi & pw(q + 1u)) == pw(q + 3u)) {
                    pat = pat | (1u << k);
                }
            }
            let toff = bitcast<u32>(tables[voff + pat]);
            if (toff != NONE) {
                let nlo = 1u << lb;
                let nhi = n >> lb;
                let lom = nlo - 1u;
                for (var j = lid; j < n; j += WG) {
                    if ((j & cmask) == cpat) {
                        let x = j & lom;
                        let h = j >> lb;
                        let lr = tables[toff + x];
                        let li = tables[toff + nlo + x];
                        let hr = tables[toff + 2u * nlo + h];
                        let hi = tables[toff + 2u * nlo + nhi + h];
                        let fr = fma(lr, hr, -(li * hi));
                        let fi = fma(lr, hi, li * hr);
                        let xr = sre[j];
                        let xi = sim[j];
                        sre[j] = fma(xr, fr, -(xi * fi));
                        sim[j] = fma(xr, fi, xi * fr);
                    }
                }
            }
            pc += 6u + 4u * nc;
        }
        workgroupBarrier();
    }
    for (var j = lid; j < n; j += WG) {
        work[boff | base | glid | pw(1u + j / WG)] = vec2<f32>(sre[j], sim[j]);
    }
}

"#;

/// Encodes the ops of a cache block for [`substage_shared_wgsl`].
pub fn encode_sub_shared(
    s: &GpuSubStage,
    tables: &mut Vec<u32>,
    maxn: usize,
    wg: usize,
) -> Vec<u32> {
    let mut w = vec![0u32];
    // gaddr(k WG) for k < MAXN / WG (the kernel adds gaddr(lid))
    let lowb = s.inner_mask.trailing_ones() as usize;
    let himask = s.inner_mask >> lowb << lowb;
    let lowm = (1usize << lowb) - 1;
    w.extend((0..maxn / wg).map(|k| {
        let j = k * wg;
        ((j & lowm) | pdep(j >> lowb, himask)) as u32
    }));
    let mut nops = 0u32;
    let fb = |m: &[f32; 8]| m.map(f32::to_bits);
    for op in &s.ops {
        match op {
            GpuOp::U1 {
                t,
                m,
                kind,
                cin,
                cout,
            } => {
                w.extend([
                    1,
                    *t,
                    *kind as u32,
                    *cin,
                    *cout as u32,
                    (*cout >> 32) as u32,
                ]);
                w.extend(fb(m));
                nops += 1;
            }
            GpuOp::Swap { a, b } => {
                w.extend([2, *a, *b]);
                nops += 1;
            }
            GpuOp::Pair {
                t1,
                m1,
                real1,
                t2,
                m2,
                real2,
                cx,
            } => {
                w.extend([3, *t1, *t2, *real1 as u32, *real2 as u32, *cx as u32]);
                w.extend(fb(m1));
                w.extend(fb(m2));
                nops += 1;
            }
            GpuOp::Diag(gs) => {
                for g in gs {
                    // variant offsets, then the tables of each variant
                    let voff = tables.len() as u32;
                    tables.resize(tables.len() + g.variants.len(), NONE);
                    for (i, v) in g.variants.iter().enumerate() {
                        if let Some(t) = v {
                            tables[voff as usize + i] = tables.len() as u32;
                            for part in [&t.lor, &t.loi, &t.hr, &t.hi] {
                                tables.extend(part.iter().map(|x| x.to_bits()));
                            }
                        }
                    }
                    w.extend([4, g.cmask, g.cpat, g.lb, g.conds.len() as u32, voff]);
                    for &(om, op) in &g.conds {
                        w.extend([om as u32, (om >> 32) as u32, op as u32, (op >> 32) as u32]);
                    }
                    nops += 1;
                }
            }
        }
    }
    w[0] = nops;
    w
}
