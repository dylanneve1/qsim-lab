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
    // 1: `:h` 16-bit scale codes; 0: f32 steps (k table at `koff`)
    half: u32,
    koff: u32,
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


// ----- exact integer emulation of the f32-step formats (see `exact32`) ---

fn mul32(a: u32, b: u32) -> vec2<u32> {
    // (lo, hi)
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
    let lo = (p00 & 0xffffu) | (mid << 16u);
    let hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
    return vec2<u32>(lo, hi);
}

fn bitlen(x: u32) -> u32 {
    return 32u - countLeadingZeros(x);
}

fn bitlen3(p: vec3<u32>) -> u32 {
    if (p.z != 0u) {
        return 64u + bitlen(p.z);
    }
    if (p.y != 0u) {
        return 32u + bitlen(p.y);
    }
    return bitlen(p.x);
}

fn shr3(p: vec3<u32>, d: u32) -> vec3<u32> {
    let w = d / 32u;
    let s = d % 32u;
    var out = vec3<u32>(0u, 0u, 0u);
    for (var i = 0u; i < 3u; i++) {
        let j = i + w;
        if (j < 3u) {
            var v = p[j] >> s;
            if (s > 0u && j + 1u < 3u) {
                v = v | (p[j + 1u] << (32u - s));
            }
            out[i] = v;
        }
    }
    return out;
}

fn bit3(p: vec3<u32>, i: u32) -> bool {
    if (i >= 96u) {
        return false;
    }
    return ((p[i / 32u] >> (i % 32u)) & 1u) == 1u;
}

fn low3(p: vec3<u32>, n: u32) -> bool {
    var any = false;
    for (var i = 0u; i < 3u; i++) {
        let lo = 32u * i;
        if (n >= lo + 32u) {
            any = any || (p[i] != 0u);
        } else if (n > lo) {
            any = any || ((p[i] & ((1u << (n - lo)) - 1u)) != 0u);
        }
    }
    return any;
}

fn rne_shift3(p: vec3<u32>, d: u32) -> vec3<u32> {
    if (d == 0u) {
        return p;
    }
    var q = shr3(p, d);
    if (bit3(p, d - 1u) && (low3(p, d - 1u) || (q.x & 1u) == 1u)) {
        q.x = q.x + 1u;
        if (q.x == 0u) {
            q.y = q.y + 1u;
            if (q.y == 0u) {
                q.z = q.z + 1u;
            }
        }
    }
    return q;
}

// (sm, es) of `(m / maxv) as f32`: step = sm * 2^es
fn step32(m: f32, maxv: u32) -> vec2<i32> {
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
    let s = bitlen(q) - 24u;
    var sm = q >> s;
    let half = 1u << (s - 1u);
    let low = q & ((1u << s) - 1u);
    if (low > half || (low == half && (r != 0u || (sm & 1u) == 1u))) {
        sm = sm + 1u;
    }
    var es = i32(s) + ee;
    if (sm == 0x1000000u) {
        sm = 0x800000u;
        es = es + 1;
    }
    return vec2<i32>(i32(sm), es);
}

// fl64(1/sm) = R * 2^(adj - 76): returns (R lo, R hi, adj)
fn recip(sm: u32) -> vec3<u32> {
    var rem = 16u;
    var qh = 0u;
    var ql = 0u;
    for (var i = 0u; i < 9u; i++) {
        rem = rem << 8u;
        let d = rem / sm;
        rem = rem - d * sm;
        qh = (qh << 8u) | (ql >> 24u);
        ql = (ql << 8u) | d;
    }
    let twice = rem << 1u;
    if (twice > sm || (twice == sm && (ql & 1u) == 1u)) {
        ql = ql + 1u;
        if (ql == 0u) {
            qh = qh + 1u;
        }
    }
    if (qh == 0x200000u) {
        return vec3<u32>(0u, 0x100000u, 1u);
    }
    return vec3<u32>(ql, qh, 0u);
}

fn enc_q(x: f32, r: vec3<u32>, es: i32, maxv: u32) -> i32 {
    let b = bitcast<u32>(x);
    let ax = b & 0x7fffffffu;
    if (ax == 0u) {
        return 0;
    }
    let ex = ax >> 23u;
    var xm = ax;
    var xe = -149;
    if (ex != 0u) {
        xm = (ax & 0x7fffffu) | 0x800000u;
        xe = i32(ex) - 150;
    }
    let a = mul32(xm, r.x);
    let c = mul32(xm, r.y);
    let p1 = a.y + c.x;
    var carry = 0u;
    if (p1 < a.y) {
        carry = 1u;
    }
    let p = vec3<u32>(a.x, p1, c.y + carry);
    let t = xe - 76 + i32(r.z) - es;
    let bp = bitlen3(p);
    var d1 = 0u;
    if (bp > 53u) {
        d1 = bp - 53u;
    }
    let p53 = rne_shift3(p, d1);
    let f = -(t + i32(d1));
    var n = 0u;
    if (f <= 0) {
        n = maxv;
    } else if (f < 64) {
        let q = rne_shift3(p53, u32(f));
        if (q.y != 0u || q.z != 0u) {
            n = maxv;
        } else {
            n = min(q.x, maxv);
        }
    }
    if ((b >> 31u) == 1u) {
        return -i32(n);
    }
    return i32(n);
}

// (q * fl64(1/fl64(1/step))) as f32; w = 1 when outside the normal range
fn dec_v(q: i32, sm: u32, es: i32, k: i32) -> vec2<u32> {
    if (q == 0) {
        return vec2<u32>(0u, 0u);
    }
    let aq = u32(abs(q));
    let a = aq * sm;
    var hi = a >> 3u;
    var lo = a << 29u;
    let t = aq * u32(abs(k));
    if (k >= 0) {
        let l2 = lo + t;
        if (l2 < lo) {
            hi = hi + 1u;
        }
        lo = l2;
    } else {
        if (lo < t) {
            hi = hi - 1u;
        }
        lo = lo - t;
    }
    let n = vec3<u32>(lo, hi, 0u);
    var d1 = 0u;
    let bn = bitlen3(n);
    if (bn > 53u) {
        d1 = bn - 53u;
    }
    let n53 = rne_shift3(n, d1);
    var d2 = 0u;
    let b53 = bitlen3(n53);
    if (b53 > 24u) {
        d2 = b53 - 24u;
    }
    var m = rne_shift3(n53, d2).x;
    var dd = i32(d1 + d2);
    if (m == 0x1000000u) {
        m = 0x800000u;
        dd = dd + 1;
    }
    let bl = bitlen(m);
    let e = i32(bl) - 1 + dd + es - 29;
    let field = (m << (24u - bl)) & 0x7fffffu;
    var bad = 0u;
    if (e < -126 || e > 127) {
        bad = 1u;
    }
    var sign = 0u;
    if (q < 0) {
        sign = 0x80000000u;
    }
    return vec2<u32>(sign | (u32(clamp(e + 127, 1, 254)) << 23u) | field, bad);
}

// same as dec_v, cheaply (see `exact32::dec_fast`)
fn dec_fast(q: i32, sm: u32, es: i32, k: i32) -> vec2<u32> {
    if (q == 0) {
        return vec2<u32>(0u, 0u);
    }
    let aq = u32(abs(q));
    let a = aq * sm;
    let bl = bitlen(a);
    var e = i32(bl) - 1 + es;
    var m = 0u;
    if (bl > 24u) {
        let d = bl - 24u;
        let low = a & ((1u << d) - 1u);
        let half = 1u << (d - 1u);
        let t = aq * u32(abs(k));
        var up = false;
        if (low != half) {
            up = low > half;
        } else if (t > half) {
            up = k > 0;
        } else {
            up = ((a >> d) & 1u) == 1u;
        }
        m = (a >> d) + select(0u, 1u, up);
        if (m == 0x1000000u) {
            m = 0x800000u;
            e = e + 1;
        }
    } else {
        m = a << (24u - bl);
    }
    var bad = 0u;
    if (e < -126 || e > 127) {
        bad = 1u;
    }
    var sign = 0u;
    if (q < 0) {
        sign = 0x80000000u;
    }
    return vec2<u32>(sign | (u32(clamp(e + 127, 1, 254)) << 23u) | (m & 0x7fffffu), bad);
}

// same as enc_q: f32 fast path away from ties and the clamp (see
// `exact32::enc_fast`; `rinv` is any reciprocal of the step within a few ulp)
fn enc_fast(x: f32, rinv: f32, r: vec3<u32>, es: i32, maxv: u32) -> i32 {
    let margin = 1.0 / 4096.0;
    let y = abs(x) * rinv;
    let fm = f32(maxv);
    var n = 0u;
    if (y > fm + 0.5 + margin) {
        n = maxv;
    } else {
        let f = y - floor(y);
        if (abs(f - 0.5) > margin && y < fm + 0.5 - margin) {
            n = u32(floor(y + 0.5));
        } else {
            return enc_q(x, r, es, maxv);
        }
    }
    if ((bitcast<u32>(x) >> 31u) == 1u) {
        return -i32(n);
    }
    return i32(n);
}

fn get_int(wbase: u32, i: u32, bits: u32) -> u32 {
    let bp = i * bits;
    let wi = wbase + (bp >> 5u);
    let off = bp & 31u;
    var u = pk[wi] >> off;
    if (off + bits > 32u) {
        u = u | (pk[wi + 1u] << (32u - off));
    }
    return u & ((1u << bits) - 1u);
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
    if (P.half == 0u) {
        let sc = cd[kb];
        if (sc == 0u) {
            for (var j = 0u; j < P.block; j++) {
                work[a0 + j] = vec2<f32>(0.0, 0.0);
            }
            return;
        }
        let ex = (sc >> 23u) & 0xffu;
        let sm = (sc & 0x7fffffu) | 0x800000u;
        let es = i32(ex) - 150;
        let ki = sm - 0x800000u;
        let kw = bitcast<u32>(codec[P.koff + (ki >> 4u)]);
        let k = i32((kw >> ((ki & 15u) * 2u)) & 3u) - 1;
        var badf = 0u;
        if (ex == 0u) {
            badf = 1u;
        }
        let wb = kb * P.wpb;
        for (var j = 0u; j < P.block; j++) {
            let re = dec_fast(i32(get_int(wb, 2u * j, P.bits)) - i32(P.maxv), sm, es, k);
            let im = dec_fast(i32(get_int(wb, 2u * j + 1u, P.bits)) - i32(P.maxv), sm, es, k);
            badf = badf | re.y | im.y;
            work[a0 + j] = vec2<f32>(bitcast<f32>(re.x), bitcast<f32>(im.x));
        }
        if (badf != 0u) {
            atomicAdd(&stats[3], 1);
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

// ----- sub-stage: the ops of one cache block --------------------------------

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

// diagonal factor of buffer index j from the tables at `toff`, applied to z
fn dfac(z: vec2<f32>, j: u32, lb: u32, nlo: u32, nhi: u32, toff: u32) -> vec2<f32> {
    let x = j & (nlo - 1u);
    let h = j >> lb;
    let lr = tables[toff + x];
    let li = tables[toff + nlo + x];
    let hr = tables[toff + 2u * nlo + h];
    let hi = tables[toff + 2u * nlo + nhi + h];
    let fr = fma(lr, hr, -(li * hi));
    let fi = fma(lr, hi, li * hr);
    return vec2<f32>(fma(z.x, fr, -(z.y * fi)), fma(z.x, fi, z.y * fr));
}

__SUBSTAGE__

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

fn pack_block32(kb: u32, m: f32) -> u32 {
    let a0 = kb * P.block;
    let maxv = P.maxv;
    var sc = 0u;
    var sm = 0u;
    var es = 0;
    if (m != 0.0 && m <= 3.40282347e38) {
        let st = step32(m, maxv);
        sm = u32(st.x);
        es = st.y;
        let e = es + 23 + 127;
        if (e >= 1 && e <= 254) {
            sc = (u32(e) << 23u) | (sm & 0x7fffffu);
        } else {
            atomicAdd(&stats[3], 1);
        }
    }
    var r = vec3<u32>(0u, 0u, 0u);
    var rinv = 0.0;
    if (sc != 0u) {
        r = recip(sm);
        rinv = 1.0 / bitcast<f32>(sc);
    }
    let bits = P.bits;
    var acc = 0u;
    var na = 0u;
    var wi = kb * P.wpb;
    for (var j = 0u; j < P.block; j++) {
        let v = work[a0 + j];
        for (var c = 0u; c < 2u; c++) {
            var qi = 0;
            if (sc != 0u) {
                qi = enc_fast(v[c], rinv, r, es, maxv);
            }
            let q = u32(qi + i32(maxv));
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
    return sc;
}

fn pack_block(kb: u32) -> u32 {
    let a0 = kb * P.block;
    var m = 0.0;
    for (var j = 0u; j < P.block; j++) {
        let v = work[a0 + j];
        m = max(m, max(abs(v.x), abs(v.y)));
    }
    if (P.half == 0u) {
        return pack_block32(kb, m);
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
        if (P.half != 0u) {
            cd[t] = c0 | (c1 << 16u);
        } else {
            cd[2u * t] = c0;
            if (2u * t + 1u < P.total) {
                cd[2u * t + 1u] = c1;
            }
        }
    }
    workgroupBarrier();
    if (lid == 0u) {
        let m = atomicLoad(&wmax);
        if (m != -2147483647 - 1) {
            atomicMax(&stats[0], m);
        }
    }
}
