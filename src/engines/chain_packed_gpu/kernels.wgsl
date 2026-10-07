// Kernels of the packed chain-sweep GPU backend (chain_packed_gpu).
// Every floating-point step mirrors the CPU code it reproduces; see the
// Rust module docs. `MAXN` (sub-stage buffer amplitudes) and `WG`
// (workgroup size) are substituted by the host.

const MAXN: u32 = __MAXN__u;
const WG: u32 = __WG__u;
const NONE: u32 = 0xffffffffu;

struct Params {
    // gathered-block bits, blocks in this chunk, first block index
    l: u32,
    g: u32,
    c0: u32,
    fresh: u32,
    // :h exponent bases (in, out)
    base_in: i32,
    base_out: i32,
    // format
    bits: u32,
    maxv: u32,
    block: u32,
    wpb: u32,
    // sub-stage geometry: buffer bits, contiguous low inner bits, the
    // remaining inner mask (above the low bits), outer mask within the
    // gathered block, ops start word in `prog`
    sl: u32,
    lowb: u32,
    himask: u32,
    outmask: u32,
    // workgroups per row of the 2D dispatch, total work items
    nx: u32,
    total: u32,
    // codec table offsets: dec rows, thresholds
    dec_off: u32,
    thr_off: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> pk: array<u32>;
@group(0) @binding(1) var<storage, read_write> cd: array<u32>;
@group(0) @binding(2) var<storage, read_write> work: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> codec: array<f32>;
@group(0) @binding(4) var<storage, read_write> stats: array<atomic<i32>, 4>;
@group(0) @binding(5) var<uniform> P: Params;
@group(0) @binding(6) var<uniform> prog: array<vec4<u32>, 4096>;
@group(0) @binding(7) var<storage, read> tables: array<f32>;

fn pw(i: u32) -> u32 {
    return prog[i >> 2u][i & 3u];
}

fn pf(i: u32) -> f32 {
    return bitcast<f32>(pw(i));
}

fn pdep(x0: u32, mask0: u32) -> u32 {
    var out = 0u;
    var x = x0;
    var m = mask0;
    loop {
        if (m == 0u || x == 0u) {
            break;
        }
        let b = m & (~m + 1u);
        if ((x & 1u) != 0u) {
            out = out | b;
        }
        x = x >> 1u;
        m = m & (m - 1u);
    }
    return out;
}

fn pow2f(k: i32) -> f32 {
    return bitcast<f32>(u32(k + 127) << 23u);
}

fn item(wid: vec3<u32>, lid: u32) -> u32 {
    return (wid.y * P.nx + wid.x) * WG + lid;
}

// ----- unpack: one thread per scale block --------------------------------

@compute @workgroup_size(WG)
fn unpack(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let kb = item(wid, lid);
    if (kb >= P.total) {
        return;
    }
    let a0 = kb * P.block;
    if (P.fresh != 0u) {
        for (var j = 0u; j < P.block; j++) {
            work[a0 + j] = vec2<f32>(0.0, 0.0);
        }
        if (kb == 0u && P.c0 == 0u) {
            work[0] = vec2<f32>(1.0, 0.0);
        }
        return;
    }
    let code = (cd[kb >> 1u] >> ((kb & 1u) * 16u)) & 0xffffu;
    if (code == 0u) {
        for (var j = 0u; j < P.block; j++) {
            work[a0 + j] = vec2<f32>(0.0, 0.0);
        }
        return;
    }
    let e = i32(code >> 10u) - 1 + P.base_in;
    let mant = code & 0x3ffu;
    let k = e - 10;
    var sc = 0.0;
    var bad = 0;
    if (k >= -126 && k <= 127) {
        sc = pow2f(k);
    } else {
        bad = 1;
    }
    let w = 2u * P.maxv + 1u;
    let row = P.dec_off + mant * w;
    let wbase = kb * P.wpb;
    let bits = P.bits;
    let mask = (1u << bits) - 1u;
    for (var j = 0u; j < P.block; j++) {
        var v = vec2<f32>(0.0, 0.0);
        for (var c = 0u; c < 2u; c++) {
            let bp = (2u * j + c) * bits;
            let wi = wbase + (bp >> 5u);
            let off = bp & 31u;
            var u = pk[wi] >> off;
            if (off + bits > 32u) {
                u = u | (pk[wi + 1u] << (32u - off));
            }
            u = u & mask;
            let d = codec[row + u];
            let x = d * sc;
            if (d != 0.0 && abs(x) < 1.17549435e-38) {
                bad = 1;
            }
            v[c] = x;
        }
        work[a0 + j] = v;
    }
    if (bad != 0) {
        atomicAdd(&stats[3], 1);
    }
}

// ----- sub-stage: the ops of one cache block in workgroup memory ---------

var<workgroup> sre: array<f32, MAXN>;
var<workgroup> sim: array<f32, MAXN>;

fn ins0(p: u32, t: u32) -> u32 {
    let lo = p & ((1u << t) - 1u);
    return ((p >> t) << (t + 1u)) | lo;
}

// (m0 + i m0i)(xr + i xi) + (m1 + i m1i)(yr + i yi), CPU `cmul2`
fn cmul2(m0r: f32, m0i: f32, m1r: f32, m1i: f32, xr: f32, xi: f32, yr: f32, yi: f32) -> vec2<f32> {
    let re = fma(m0r, xr, fma(-m0i, xi, fma(m1r, yr, -(m1i * yi))));
    let im = fma(m0r, xi, fma(m0i, xr, fma(m1r, yi, m1i * yr)));
    return vec2<f32>(re, im);
}

// `m` (8 words at `mo`: re of m00 m01 m10 m11, then im) on (x, y);
// returns (x', y') as (x'.re, x'.im, y'.re, y'.im). CPU `mat_apply`.
fn mat(mo: u32, real: bool, x: vec2<f32>, y: vec2<f32>) -> vec4<f32> {
    let m0r = pf(mo);
    let m1r = pf(mo + 1u);
    let m2r = pf(mo + 2u);
    let m3r = pf(mo + 3u);
    if (real) {
        return vec4<f32>(
            fma(m0r, x.x, m1r * y.x),
            fma(m0r, x.y, m1r * y.y),
            fma(m2r, x.x, m3r * y.x),
            fma(m2r, x.y, m3r * y.y)
        );
    }
    let m0i = pf(mo + 4u);
    let m1i = pf(mo + 5u);
    let m2i = pf(mo + 6u);
    let m3i = pf(mo + 7u);
    let a = cmul2(m0r, m0i, m1r, m1i, x.x, x.y, y.x, y.y);
    let b = cmul2(m2r, m2i, m3r, m3i, x.x, x.y, y.x, y.y);
    return vec4<f32>(a, b);
}

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
    for (var j = lid; j < n; j += WG) {
        let a = work[boff | base | (j & lowm) | pdep(j >> P.lowb, P.himask)];
        sre[j] = a.x;
        sim[j] = a.y;
    }
    workgroupBarrier();
    var pc = 1u;
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
        work[boff | base | (j & lowm) | pdep(j >> P.lowb, P.himask)] = vec2<f32>(sre[j], sim[j]);
    }
}

// ----- pack: one thread per pair of scale blocks -------------------------

var<workgroup> wmax: atomic<i32>;

// CPU `int_step` for a `:h` format as (e, mant - 1024); see `step_parts`.
fn step_parts(m: f32, maxv: u32) -> vec2<i32> {
    let b = bitcast<u32>(m);
    let ex = (b >> 23u) & 0xffu;
    let fr = b & 0x7fffffu;
    var mm = fr;
    var ee = -149;
    if (ex != 0u) {
        mm = fr | 0x800000u;
        ee = i32(ex) - 150;
    }
    let sh = countLeadingZeros(mm);
    mm = mm << sh;
    ee = ee - i32(sh);
    let q = mm / maxv;
    let r = mm % maxv;
    let bl = 32u - countLeadingZeros(q);
    let s = bl - 11u;
    var mant = q >> s;
    if ((q & ((1u << s) - 1u)) != 0u || r != 0u) {
        mant = mant + 1u;
    }
    var e = i32(bl) - 1 + ee;
    if (mant == 2048u) {
        mant = 1024u;
        e = e + 1;
    }
    return vec2<i32>(e, i32(mant) - 1024);
}

fn pack_block(kb: u32) -> u32 {
    let a0 = kb * P.block;
    var m = 0.0;
    for (var j = 0u; j < P.block; j++) {
        let v = work[a0 + j];
        m = max(m, max(abs(v.x), abs(v.y)));
    }
    var code = 0u;
    var zero = true;
    var mant = 0u;
    var k = 0;
    if (m != 0.0 && m <= 3.40282347e38) {
        let sp = step_parts(m, P.maxv);
        let e = sp.x;
        atomicMax(&wmax, e);
        let ec = e - P.base_out + 1;
        if (ec < 1) {
            atomicAdd(&stats[1], 1);
        } else if (ec > 63) {
            atomicAdd(&stats[2], 1);
        } else {
            code = (u32(ec) << 10u) | u32(sp.y);
            zero = false;
            mant = u32(sp.y);
            k = -(e - 10);
        }
    }
    var k1 = k;
    var k2 = 0;
    if (k > 126) {
        k1 = 126;
        k2 = k - 126;
    }
    if (k1 < -126 || k1 > 126 || k2 < -126 || k2 > 126) {
        atomicAdd(&stats[3], 1);
    }
    let s1 = pow2f(clamp(k1, -126, 126));
    let s2 = pow2f(clamp(k2, -126, 126));
    let maxv = P.maxv;
    let trow = P.thr_off + mant * maxv;
    let bits = P.bits;
    let wbase = kb * P.wpb;
    var acc = 0u;
    var na = 0u;
    var wi = wbase;
    for (var j = 0u; j < P.block; j++) {
        let v = work[a0 + j];
        for (var c = 0u; c < 2u; c++) {
            let x = v[c];
            var q = 0u;
            if (!zero) {
                let a = abs(x) * s1 * s2;
                // number of thresholds <= a (ascending): binary search
                var lo = 0u;
                var hi = maxv;
                loop {
                    if (lo >= hi) {
                        break;
                    }
                    let mid = (lo + hi) / 2u;
                    if (a >= codec[trow + mid]) {
                        lo = mid + 1u;
                    } else {
                        hi = mid;
                    }
                }
                if (x < 0.0) {
                    q = maxv - lo;
                } else {
                    q = maxv + lo;
                }
            } else {
                q = maxv;
            }
            acc = acc | (q << na);
            na = na + bits;
            if (na >= 32u) {
                pk[wi] = acc;
                wi = wi + 1u;
                na = na - 32u;
                if (na > 0u) {
                    acc = q >> (bits - na);
                } else {
                    acc = 0u;
                }
            }
        }
    }
    return code;
}

@compute @workgroup_size(WG)
fn pack(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    if (lid == 0u) {
        atomicStore(&wmax, -2147483647 - 1);
    }
    workgroupBarrier();
    let t = item(wid, lid);
    if (2u * t < P.total) {
        let c0 = pack_block(2u * t);
        var c1 = 0u;
        if (2u * t + 1u < P.total) {
            c1 = pack_block(2u * t + 1u);
        }
        cd[t] = c0 | (c1 << 16u);
    }
    workgroupBarrier();
    if (lid == 0u) {
        let m = atomicLoad(&wmax);
        if (m != -2147483647 - 1) {
            atomicMax(&stats[0], m);
        }
    }
}
