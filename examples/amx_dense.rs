//! AMX experiment for dense k-qubit fusion (k = 5..7), a time-boxed study for
//! `research/performance/avx512.md` (AMX section; raw data in
//! `research/data/avx512/amx/`). Not used by the engine.
//!
//! A fused `2^k x 2^k` unitary is applied to an n-qubit complex f32 state as a
//! real GEMM `Y = X R^T`: the targets are the k lowest qubits of an
//! interleaved (re, im) state, so each group of `D = 2^k` amplitudes is one
//! row of `X` (2D reals) and `R` is the 2D x 2D real form of the unitary.
//!
//! * `amx-bf16x3`: Intel AMX `TDPBF16PS` with an error-free split of every
//!   f32 of both operands into three bf16 parts (hi + mid + lo = x exactly)
//!   and the six partial products with i + j <= 2 kept, accumulated in f32
//!   tiles (`core::arch::asm!`: ldtilecfg / tileloadd / tdpbf16ps /
//!   tilestored); the f32 -> 3 x bf16 conversion of the state is inside the
//!   timed region. Every TDPBF16PS / TILELOADD is followed by 16 NOPs
//!   (`PAD`): back to back they issue at about half rate on this machine.
//! * `amx-bf16x2`: two parts (hi + mid), products hi*hi, hi*mid, mid*hi:
//!   half the tile work, about 2^-16 relative error (not f32 accuracy).
//! * `amx-bf16x1`: one bf16 product, no split (an upper bound on AMX speed;
//!   about three significant digits, far from the f32 engine's 1e-5).
//! * `avx512-f32`: AVX-512 f32 FMA GEMM on the same layout (6 x 64 register
//!   block, packed B panels), the competitor at equal accuracy.
//! * `scalar-f32`: plain f32 complex mat-vec (accuracy reference point only).
//!
//! All are compared with an f64 reference. Standalone, std only:
//!
//! ```text
//! rustc --edition 2021 -C opt-level=3 -C target-cpu=native examples/amx_dense.rs -o amx_dense
//! amx_dense acc                                   # accuracy table
//! amx_dense micro                                 # AMX / FMA throughput
//! amx_dense speed <n> <threads> <reps> [k..] [only=<method>]   # Gamp/s
//! ```
//!
//! The AMX tile registers are invisible to the compiler. No other code
//! touches them, so the tile config (loaded once per thread and kernel call)
//! and the accumulator tiles persist between the `asm!` blocks of a call.

#![allow(clippy::needless_range_loop)]
// AVX-512 intrinsics are stable since Rust 1.89 (the toolchain is pinned to
// 1.93); Cargo.toml's rust-version still says 1.80.
#![allow(clippy::incompatible_msrv)]
// Only the scalar path exists off x86_64.
#![cfg_attr(not(target_arch = "x86_64"), allow(dead_code, unused))]

use std::ops::{Add, Mul, Sub};
use std::time::Instant;

// ----- small helpers --------------------------------------------------------

/// splitmix64 with Box-Muller Gaussians.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Cx {
    re: f64,
    im: f64,
}

impl Add for Cx {
    type Output = Cx;
    fn add(self, o: Cx) -> Cx {
        Cx {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
}

impl Sub for Cx {
    type Output = Cx;
    fn sub(self, o: Cx) -> Cx {
        Cx {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }
}

impl Mul for Cx {
    type Output = Cx;
    fn mul(self, o: Cx) -> Cx {
        Cx {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
}

impl Cx {
    fn conj(self) -> Cx {
        Cx {
            re: self.re,
            im: -self.im,
        }
    }
    fn norm2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

// ----- unitaries (f64, row-major) ------------------------------------------

/// Haar-random `d x d` unitary: Gram-Schmidt (twice) on complex Gaussian
/// columns; the implied R factor has a positive diagonal, so this is Haar.
fn haar(d: usize, rng: &mut Rng) -> Vec<Cx> {
    let mut cols: Vec<Vec<Cx>> = Vec::with_capacity(d);
    for _ in 0..d {
        let mut v: Vec<Cx> = (0..d)
            .map(|_| Cx {
                re: rng.gauss(),
                im: rng.gauss(),
            })
            .collect();
        for _ in 0..2 {
            for q in &cols {
                let dot = q
                    .iter()
                    .zip(&v)
                    .fold(Cx::default(), |s, (a, b)| s + a.conj() * *b);
                for (vi, qi) in v.iter_mut().zip(q) {
                    *vi = *vi - dot * *qi;
                }
            }
        }
        let nrm = v.iter().map(|z| z.norm2()).sum::<f64>().sqrt();
        for z in &mut v {
            z.re /= nrm;
            z.im /= nrm;
        }
        cols.push(v);
    }
    let mut u = vec![Cx::default(); d * d];
    for (c, col) in cols.iter().enumerate() {
        for (r, z) in col.iter().enumerate() {
            u[r * d + c] = *z;
        }
    }
    u
}

/// What a fusion pass produces from a brickwork of random 2-qubit gates:
/// the product of `depth` layers of Haar 4x4 unitaries on local qubits
/// (q, q+1), alternating offsets.
fn brick_fused(k: usize, depth: usize, rng: &mut Rng) -> Vec<Cx> {
    let d = 1usize << k;
    let mut m = vec![Cx::default(); d * d];
    for i in 0..d {
        m[i * d + i] = Cx { re: 1.0, im: 0.0 };
    }
    for layer in 0..depth {
        for q in (layer % 2..k - 1).step_by(2) {
            let g = haar(4, rng);
            for c in 0..d {
                for i in 0..d {
                    if (i >> q) & 3 != 0 {
                        continue;
                    }
                    let idx = [i, i | (1 << q), i | (2 << q), i | (3 << q)];
                    let v = idx.map(|j| m[j * d + c]);
                    for (r, &j) in idx.iter().enumerate() {
                        m[j * d + c] = (0..4).fold(Cx::default(), |s, t| s + g[r * 4 + t] * v[t]);
                    }
                }
            }
        }
    }
    m
}

/// Real 2D x 2D form of a complex D x D matrix acting on interleaved
/// (re, im) vectors, rounded to f32.
fn real_form(u: &[Cx], d: usize) -> Vec<f32> {
    let d2 = 2 * d;
    let mut r = vec![0f32; d2 * d2];
    for i in 0..d {
        for j in 0..d {
            let z = u[i * d + j];
            r[2 * i * d2 + 2 * j] = z.re as f32;
            r[2 * i * d2 + 2 * j + 1] = -z.im as f32;
            r[(2 * i + 1) * d2 + 2 * j] = z.im as f32;
            r[(2 * i + 1) * d2 + 2 * j + 1] = z.re as f32;
        }
    }
    r
}

// ----- bf16 ------------------------------------------------------------------

/// f32 -> bf16 bits, round to nearest even (finite inputs).
fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    (b.wrapping_add(0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

fn bf16_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Error-free split x = hi + mid + lo (24 = 8 + 8 + 8 significand bits).
fn split3(x: f32) -> [u16; 3] {
    let h = bf16(x);
    let r = x - bf16_f32(h);
    let m = bf16(r);
    let s = r - bf16_f32(m);
    [h, m, bf16(s)]
}

/// One 64-byte line: 32 bf16 (an AMX tile row), 64-byte aligned.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Line([u16; 32]);

/// B = R^T split into `parts` bf16 parts, as AMX tiles in VNNI layout:
/// tile (kc, nt, p) holds K rows `kc*32..+32` (as 16 pair-rows) and N
/// columns `nt*16..+16`; tile index `((kc * NT + nt) * parts + p)`.
fn amx_pack_b(r: &[f32], d2: usize, parts: usize) -> Vec<Line> {
    let (kcn, ntn) = (d2 / 32, d2 / 16);
    let mut out = vec![Line([0; 32]); kcn * ntn * parts * 16];
    for kc in 0..kcn {
        for nt in 0..ntn {
            for p in 0..parts {
                let base = ((kc * ntn + nt) * parts + p) * 16;
                for kk in 0..16 {
                    for j in 0..32 {
                        let n = nt * 16 + j / 2;
                        let k = kc * 32 + 2 * kk + (j % 2);
                        // B[k][n] = R[n][k]
                        // parts 1 and 2 are the leading parts of the 3-split
                        // (hi = RNE(x), mid = RNE(x - hi))
                        out[base + kk].0[j] = split3(r[n * d2 + k])[p];
                    }
                }
            }
        }
    }
    out
}

/// B = R^T in panels of 64 columns for the AVX-512 kernel:
/// `out[(nb / 64) * d2 * 64 + k * 64 + j] = R[nb + j][k]`.
fn pack_bt(r: &[f32], d2: usize) -> Vec<f32> {
    let mut out = vec![0f32; d2 * d2];
    for nb in (0..d2).step_by(64) {
        for k in 0..d2 {
            for j in 0..64 {
                out[(nb / 64) * d2 * 64 + k * 64 + j] = r[(nb + j) * d2 + k];
            }
        }
    }
    out
}

// ----- reference and scalar paths -------------------------------------------

/// y = U x per group, in f64.
fn apply_ref(u: &[Cx], d: usize, x: &mut [Cx]) {
    let mut tmp = vec![Cx::default(); d];
    for g in x.chunks_exact_mut(d) {
        for (r, t) in tmp.iter_mut().enumerate() {
            *t = (0..d).fold(Cx::default(), |s, c| s + u[r * d + c] * g[c]);
        }
        g.copy_from_slice(&tmp);
    }
}

/// Plain f32 complex mat-vec per group (separate multiply and add).
fn apply_scalar(x: &mut [f32], ur: &[f32], ui: &[f32], d: usize) {
    let mut tmp = vec![0f32; 2 * d];
    for g in x.chunks_exact_mut(2 * d) {
        for r in 0..d {
            let (mut sr, mut si) = (0f32, 0f32);
            for c in 0..d {
                let (a, b) = (ur[r * d + c], ui[r * d + c]);
                let (xr, xi) = (g[2 * c], g[2 * c + 1]);
                sr = sr + a * xr - b * xi;
                si = si + a * xi + b * xr;
            }
            tmp[2 * r] = sr;
            tmp[2 * r + 1] = si;
        }
        g.copy_from_slice(&tmp);
    }
}

// ----- x86_64 kernels ----------------------------------------------------------

/// `PAD`: 16 NOPs after every TDPBF16PS / TILELOADD of the kernel: issued back to
/// back, AMX instructions run at about half rate on this machine (`micro`
/// shows 13.6-14.8 ns per TDPBF16PS back to back, 7.2-7.5 ns with 16 NOPs
/// between); padding made the bf16x3 kernel 1.2-1.6x faster.
macro_rules! PAD {
    () => {
        ".rept 16\nnop\n.endr"
    };
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::Line;
    use std::arch::asm;
    use std::arch::x86_64::*;

    /// Whether the CPU has AVX-512F.
    pub fn has_avx512() -> bool {
        is_x86_feature_detected!("avx512f")
    }

    /// AVX-512 BF16 + AMX-TILE/BF16 (CPUID 7.0 EDX bits 22, 24), and the OS
    /// grants the tile data state (arch_prctl ARCH_REQ_XCOMP_PERM).
    pub fn amx_ready() -> bool {
        if !(is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512bf16"))
        {
            return false;
        }
        // SAFETY: cpuid leaf 7 exists on every CPU with AVX-512.
        let r = unsafe { __cpuid_count(7, 0) };
        if (r.edx >> 22) & 1 == 0 || (r.edx >> 24) & 1 == 0 {
            return false;
        }
        let ret: i64;
        // SAFETY: arch_prctl(ARCH_REQ_XCOMP_PERM = 0x1023, XFEATURE_XTILEDATA
        // = 18) only changes this process's permitted xsave features.
        unsafe {
            asm!("syscall", inlateout("rax") 158i64 => ret, in("rdi") 0x1023u64,
                 in("rsi") 18u64, lateout("rcx") _, lateout("r11") _, options(nostack));
        }
        ret == 0
    }

    /// AVX-512 f32 GEMM `X <- X R^T` on rows of `d2` floats (an even
    /// number of rows, d2 % 64 == 0); `bt` from `pack_bt`; `tmp` holds 6
    /// rows. Blocks of 6 rows x 64 columns (24 zmm accumulators, 4 B loads
    /// and 6 broadcasts per 24 FMAs), then a 4- or 2-row tail (rows are
    /// powers of two here, so the tail is 2 or 4 rows).
    ///
    /// # Safety
    /// The CPU must support AVX-512F.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn gemm_avx512(x: &mut [f32], bt: &[f32], d2: usize, tmp: &mut [f32]) {
        let rows = x.len() / d2;
        assert!(rows & 1 == 0 && d2 & 63 == 0 && tmp.len() >= 6 * d2 && bt.len() == d2 * d2);
        let mut mb = 0;
        while mb + 6 <= rows {
            gemm_rows::<6>(x.as_mut_ptr().add(mb * d2), bt, d2, tmp);
            mb += 6;
        }
        while mb + 4 <= rows {
            gemm_rows::<4>(x.as_mut_ptr().add(mb * d2), bt, d2, tmp);
            mb += 4;
        }
        while mb + 2 <= rows {
            gemm_rows::<2>(x.as_mut_ptr().add(mb * d2), bt, d2, tmp);
            mb += 2;
        }
    }

    /// `MR` rows starting at `xp`, all 64-column panels, through `tmp`.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn gemm_rows<const MR: usize>(xp: *mut f32, bt: &[f32], d2: usize, tmp: &mut [f32]) {
        let tp = tmp.as_mut_ptr();
        for nb in (0..d2).step_by(64) {
            let mut acc = [[_mm512_setzero_ps(); 4]; MR];
            let mut bp = bt.as_ptr().add((nb / 64) * d2 * 64);
            for k in 0..d2 {
                let b = [
                    _mm512_loadu_ps(bp),
                    _mm512_loadu_ps(bp.add(16)),
                    _mm512_loadu_ps(bp.add(32)),
                    _mm512_loadu_ps(bp.add(48)),
                ];
                for m in 0..MR {
                    let a = _mm512_set1_ps(*xp.add(m * d2 + k));
                    for j in 0..4 {
                        acc[m][j] = _mm512_fmadd_ps(a, b[j], acc[m][j]);
                    }
                }
                bp = bp.add(64);
            }
            for m in 0..MR {
                for j in 0..4 {
                    _mm512_storeu_ps(tp.add(m * d2 + nb + 16 * j), acc[m][j]);
                }
            }
        }
        std::ptr::copy_nonoverlapping(tp, xp, MR * d2);
    }

    /// bf16 bits of 32 lanes -> two f32 vectors (exact).
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn widen(v: __m512i) -> (__m512, __m512) {
        let a = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(v));
        let b = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64::<1>(v));
        (
            _mm512_castsi512_ps(_mm512_slli_epi32::<16>(a)),
            _mm512_castsi512_ps(_mm512_slli_epi32::<16>(b)),
        )
    }

    /// Converts `len` floats into `parts` (1 or 3) bf16 arrays, element by
    /// element (round to nearest even; with 3 parts hi + mid + lo == x).
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512bf16")]
    unsafe fn split_block(
        src: *const f32,
        len: usize,
        dst: *mut u16,
        part_len: usize,
        parts: usize,
    ) {
        let mut i = 0;
        while i < len {
            let x0 = _mm512_loadu_ps(src.add(i));
            let x1 = _mm512_loadu_ps(src.add(i + 16));
            let h = std::mem::transmute::<__m512bh, __m512i>(_mm512_cvtne2ps_pbh(x1, x0));
            _mm512_storeu_si512(dst.add(i) as *mut _, h);
            if parts >= 2 {
                let (h0, h1) = widen(h);
                let (r0, r1) = (_mm512_sub_ps(x0, h0), _mm512_sub_ps(x1, h1));
                let m = std::mem::transmute::<__m512bh, __m512i>(_mm512_cvtne2ps_pbh(r1, r0));
                _mm512_storeu_si512(dst.add(part_len + i) as *mut _, m);
                if parts == 3 {
                    let (m0, m1) = widen(m);
                    let l = std::mem::transmute::<__m512bh, __m512i>(_mm512_cvtne2ps_pbh(
                        _mm512_sub_ps(r1, m1),
                        _mm512_sub_ps(r0, m0),
                    ));
                    _mm512_storeu_si512(dst.add(2 * part_len + i) as *mut _, l);
                }
            }
            i += 32;
        }
    }

    /// Which parts of the AMX kernel run (the partial ones attribute time).
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Stage {
        /// Conversion and tile work: the real kernel.
        Full,
        /// Only the f32 -> bf16 conversion of the state blocks.
        ConvertOnly,
        /// Only the tile loads, products and stores (stale bf16 buffers).
        TilesOnly,
    }

    /// The 64-byte tile configuration: palette 1, tmm0..7 all 16 rows x 64 B.
    #[repr(C, align(64))]
    struct TileCfg([u8; 64]);

    /// Loads the tile configuration on the calling thread.
    ///
    /// # Safety
    /// `amx_ready()` must have returned true.
    pub unsafe fn tile_config() {
        let mut c = TileCfg([0; 64]);
        c.0[0] = 1;
        for t in 0..8 {
            c.0[16 + 2 * t] = 64;
            c.0[48 + t] = 16;
        }
        asm!("ldtilecfg [{}]", in(reg) &c, options(nostack, readonly));
    }

    /// Releases the tile state of the calling thread.
    ///
    /// # Safety
    /// `tile_config` must have run on this thread.
    pub unsafe fn tile_release() {
        asm!("tilerelease", options(nostack, nomem));
    }

    /// Throughput micro-benchmarks behind the speed table (all register- or
    /// L1-resident, one thread): core clock from a dependent register add
    /// chain, then ns and core cycles per TDPBF16PS (4 independent
    /// accumulators), per 1 KiB tileloadd / tilestored, per (tileloadd +
    /// TDPBF16PS) when the load targets a tile the product does not read,
    /// and per 512-bit FMA (8 independent chains).
    ///
    /// # Safety
    /// `amx_ready()` must have returned true.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn micro() {
        use std::time::Instant;
        let iters = 200_000usize;
        let per = |t: Instant, ops: usize| t.elapsed().as_secs_f64() * 1e9 / ops as f64;
        let mut x = 0u64;
        let t = Instant::now();
        for _ in 0..iters {
            asm!(".rept 64", "add {x}, {y}", ".endr", x = inout(reg) x, y = in(reg) 1u64,
                 options(nostack, nomem));
        }
        let ghz = 1.0 / per(t, iters * 64);
        let buf = vec![Line([0x3f80; 32]); 128];
        let mut out = vec![Line([0; 32]); 128];
        let (p, q) = (buf.as_ptr() as *const u16, out.as_mut_ptr() as *mut u16);
        tile_config();
        asm!("tileloadd tmm4, [{p} + {s}*1]", "tileloadd tmm5, [{p} + {s}*1]",
             "tileloadd tmm6, [{p} + {s}*1]", "tileloadd tmm7, [{p} + {s}*1]",
             "tilezero tmm0", "tilezero tmm1", "tilezero tmm2", "tilezero tmm3",
             p = in(reg) p, s = in(reg) 64usize, options(nostack, readonly));
        let t = Instant::now();
        for _ in 0..iters {
            asm!(
                "tdpbf16ps tmm0, tmm4, tmm6",
                "tdpbf16ps tmm1, tmm4, tmm7",
                "tdpbf16ps tmm2, tmm5, tmm6",
                "tdpbf16ps tmm3, tmm5, tmm7",
                options(nostack, nomem)
            );
        }
        let dp = per(t, 4 * iters);
        let t = Instant::now();
        for _ in 0..iters {
            asm!(
                "tdpbf16ps tmm0, tmm4, tmm6",
                PAD!(),
                "tdpbf16ps tmm1, tmm4, tmm7",
                PAD!(),
                "tdpbf16ps tmm2, tmm5, tmm6",
                PAD!(),
                "tdpbf16ps tmm3, tmm5, tmm7",
                PAD!(),
                options(nostack, nomem)
            );
        }
        let dp_pad = per(t, 4 * iters);
        let t = Instant::now();
        for _ in 0..iters {
            asm!("tileloadd tmm6, [{p} + {s}*1]", "tileloadd tmm7, [{p} + {s}*1 + 1024]",
                 "tileloadd tmm6, [{p} + {s}*1 + 2048]", "tileloadd tmm7, [{p} + {s}*1 + 3072]",
                 p = in(reg) p, s = in(reg) 64usize, options(nostack, readonly));
        }
        let ld = per(t, 4 * iters);
        let t = Instant::now();
        for _ in 0..iters {
            asm!("tilestored [{q} + {s}*1], tmm0", "tilestored [{q} + {s}*1 + 1024], tmm1",
                 "tilestored [{q} + {s}*1 + 2048], tmm2", "tilestored [{q} + {s}*1 + 3072], tmm3",
                 q = in(reg) q, s = in(reg) 64usize, options(nostack));
        }
        let st = per(t, 4 * iters);
        let t = Instant::now();
        for _ in 0..iters {
            asm!("tileloadd tmm6, [{p} + {s}*1]", "tdpbf16ps tmm0, tmm4, tmm5",
                 "tileloadd tmm7, [{p} + {s}*1 + 1024]", "tdpbf16ps tmm1, tmm4, tmm5",
                 p = in(reg) p, s = in(reg) 64usize, options(nostack, readonly));
        }
        let ldp = per(t, 2 * iters);
        tile_release();
        let one = _mm512_set1_ps(1.0);
        let c = _mm512_set1_ps(1e-7);
        let mut acc = [one; 8];
        let t = Instant::now();
        for _ in 0..iters * 8 {
            for a in acc.iter_mut() {
                *a = _mm512_fmadd_ps(*a, one, c);
            }
        }
        let fma = per(t, iters * 64);
        let mut sink = [0f32; 16];
        _mm512_storeu_ps(sink.as_mut_ptr(), acc[0]);
        println!(
            "core clock (dependent add chain): {ghz:.2} GHz  (check {x} {})",
            sink[0]
        );
        println!("| op | ns | core cycles | MAC/ns |");
        println!("|---|---|---|---|");
        println!(
            "| TDPBF16PS 16x16x32 (4 accumulators, back to back) | {dp:.2} | {:.1} | {:.0} |",
            dp * ghz,
            8192.0 / dp
        );
        println!(
            "| TDPBF16PS, 16 NOPs between | {dp_pad:.2} | {:.1} | {:.0} |",
            dp_pad * ghz,
            8192.0 / dp_pad
        );
        println!("| tileloadd 1 KiB (L1) | {ld:.2} | {:.1} | |", ld * ghz);
        println!("| tilestored 1 KiB (L1) | {st:.2} | {:.1} | |", st * ghz);
        println!(
            "| tileloadd + TDPBF16PS (independent tiles) | {ldp:.2} | {:.1} | |",
            ldp * ghz
        );
        println!(
            "| vfmadd231ps zmm (8 chains) | {fma:.3} | {:.2} | {:.0} |",
            fma * ghz,
            16.0 / fma
        );
        println!(
            "| implied 3-part (6 products) AMX peak, padded | {:.2} per 8192 MAC | | {:.0} |",
            6.0 * dp_pad,
            8192.0 / (6.0 * dp_pad)
        );
    }

    /// `X <- X R^T` on AMX: per block of 32 rows, the rows are split into
    /// bf16 parts (`a`: 3 parts x 32 rows x d2 bf16), then every pair of
    /// 16-column N tiles accumulates 2 x 2 C tiles (tmm0..3) over the K
    /// chunks: A tiles in tmm4/5 (rows 0-15 / 16-31), B tiles in tmm6/7.
    /// `parts` = 3: hi*hi, hi*mid, hi*lo, mid*mid, mid*hi, lo*hi (24
    /// TDPBF16PS and 16 tile loads per K chunk); 2: hi*hi, mid*hi, hi*mid
    /// (12 and 10); 1: hi*hi only (4 and 4).
    ///
    /// The bf16 rows alternate between two buffers, so the conversion of
    /// the next block can overlap the tile work of the current one.
    ///
    /// # Safety
    /// `amx_ready()` returned true and `tile_config()` ran on this thread;
    /// rows % 32 == 0, d2 % 32 == 0, `b` from `amx_pack_b(.., parts)`,
    /// `a.len() * 32 >= 2 * 3 * 32 * d2`.
    #[target_feature(enable = "avx512f,avx512bw,avx512bf16")]
    pub unsafe fn gemm_amx(
        x: &mut [f32],
        b: &[Line],
        d2: usize,
        parts: usize,
        a: &mut [Line],
        mode: Stage,
    ) {
        let rows = x.len() / d2;
        let (kcn, ntn) = (d2 / 32, d2 / 16);
        assert!(rows & 31 == 0 && d2 & 31 == 0 && a.len() * 32 >= 2 * 3 * 32 * d2);
        assert_eq!(b.len(), kcn * ntn * parts * 16);
        let part_len = 32 * d2;
        let abuf = a.as_mut_ptr() as *mut u16;
        let bp0 = b.as_ptr() as *const u16;
        let astride = d2 * 2;
        let cstride = d2 * 4;
        for mb in (0..rows).step_by(32) {
            let xp = x.as_mut_ptr().add(mb * d2);
            let ap = abuf.add(((mb / 32) & 1) * 3 * part_len);
            if mode != Stage::TilesOnly {
                split_block(xp, 32 * d2, ap, part_len, parts);
            }
            if mode == Stage::ConvertOnly {
                continue;
            }
            for np in 0..ntn / 2 {
                asm!(
                    "tilezero tmm0",
                    "tilezero tmm1",
                    "tilezero tmm2",
                    "tilezero tmm3",
                    options(nostack, nomem)
                );
                for kc in 0..kcn {
                    let a0 = ap.add(kc * 32) as *const u16;
                    let bp = bp0.add((kc * ntn + 2 * np) * parts * 512);
                    if parts == 3 {
                        asm!(
                            // hi * hi
                            "tileloadd tmm4, [{ah0} + {s}*1]",
                            PAD!(),
                            "tileloadd tmm5, [{ah1} + {s}*1]",
                            PAD!(),
                            "tileloadd tmm6, [{b} + {bs}*1]",
                            PAD!(),
                            "tileloadd tmm7, [{b} + {bs}*1 + 3072]",
                            PAD!(),
                            "tdpbf16ps tmm0, tmm4, tmm6",
                            PAD!(),
                            "tdpbf16ps tmm1, tmm4, tmm7",
                            PAD!(),
                            "tdpbf16ps tmm2, tmm5, tmm6",
                            PAD!(),
                            "tdpbf16ps tmm3, tmm5, tmm7",
                            PAD!(),
                            // hi * mid
                            "tileloadd tmm6, [{b} + {bs}*1 + 1024]",
                            PAD!(),
                            "tileloadd tmm7, [{b} + {bs}*1 + 4096]",
                            PAD!(),
                            "tdpbf16ps tmm0, tmm4, tmm6",
                            PAD!(),
                            "tdpbf16ps tmm1, tmm4, tmm7",
                            PAD!(),
                            "tdpbf16ps tmm2, tmm5, tmm6",
                            PAD!(),
                            "tdpbf16ps tmm3, tmm5, tmm7",
                            PAD!(),
                            // hi * lo
                            "tileloadd tmm6, [{b} + {bs}*1 + 2048]",
                            PAD!(),
                            "tileloadd tmm7, [{b} + {bs}*1 + 5120]",
                            PAD!(),
                            "tdpbf16ps tmm0, tmm4, tmm6",
                            PAD!(),
                            "tdpbf16ps tmm1, tmm4, tmm7",
                            PAD!(),
                            "tdpbf16ps tmm2, tmm5, tmm6",
                            PAD!(),
                            "tdpbf16ps tmm3, tmm5, tmm7",
                            PAD!(),
                            // mid * mid
                            "tileloadd tmm4, [{am0} + {s}*1]",
                            PAD!(),
                            "tileloadd tmm5, [{am1} + {s}*1]",
                            PAD!(),
                            "tileloadd tmm6, [{b} + {bs}*1 + 1024]",
                            PAD!(),
                            "tileloadd tmm7, [{b} + {bs}*1 + 4096]",
                            PAD!(),
                            "tdpbf16ps tmm0, tmm4, tmm6",
                            PAD!(),
                            "tdpbf16ps tmm1, tmm4, tmm7",
                            PAD!(),
                            "tdpbf16ps tmm2, tmm5, tmm6",
                            PAD!(),
                            "tdpbf16ps tmm3, tmm5, tmm7",
                            PAD!(),
                            // mid * hi
                            "tileloadd tmm6, [{b} + {bs}*1]",
                            PAD!(),
                            "tileloadd tmm7, [{b} + {bs}*1 + 3072]",
                            PAD!(),
                            "tdpbf16ps tmm0, tmm4, tmm6",
                            PAD!(),
                            "tdpbf16ps tmm1, tmm4, tmm7",
                            PAD!(),
                            "tdpbf16ps tmm2, tmm5, tmm6",
                            PAD!(),
                            "tdpbf16ps tmm3, tmm5, tmm7",
                            PAD!(),
                            // lo * hi
                            "tileloadd tmm4, [{al0} + {s}*1]",
                            PAD!(),
                            "tileloadd tmm5, [{al1} + {s}*1]",
                            PAD!(),
                            "tdpbf16ps tmm0, tmm4, tmm6",
                            PAD!(),
                            "tdpbf16ps tmm1, tmm4, tmm7",
                            PAD!(),
                            "tdpbf16ps tmm2, tmm5, tmm6",
                            PAD!(),
                            "tdpbf16ps tmm3, tmm5, tmm7",
                            PAD!(),
                            ah0 = in(reg) a0,
                            ah1 = in(reg) a0.add(16 * d2),
                            am0 = in(reg) a0.add(part_len),
                            am1 = in(reg) a0.add(part_len + 16 * d2),
                            al0 = in(reg) a0.add(2 * part_len),
                            al1 = in(reg) a0.add(2 * part_len + 16 * d2),
                            b = in(reg) bp,
                            s = in(reg) astride,
                            bs = in(reg) 64usize,
                            options(nostack, readonly),
                        );
                    } else if parts == 2 {
                        asm!(
                            // hi * hi
                            "tileloadd tmm4, [{ah0} + {s}*1]",
                            PAD!(),
                            "tileloadd tmm5, [{ah1} + {s}*1]",
                            PAD!(),
                            "tileloadd tmm6, [{b} + {bs}*1]",
                            PAD!(),
                            "tileloadd tmm7, [{b} + {bs}*1 + 2048]",
                            PAD!(),
                            "tdpbf16ps tmm0, tmm4, tmm6",
                            PAD!(),
                            "tdpbf16ps tmm1, tmm4, tmm7",
                            PAD!(),
                            "tdpbf16ps tmm2, tmm5, tmm6",
                            PAD!(),
                            "tdpbf16ps tmm3, tmm5, tmm7",
                            PAD!(),
                            // mid * hi
                            "tileloadd tmm4, [{am0} + {s}*1]",
                            PAD!(),
                            "tileloadd tmm5, [{am1} + {s}*1]",
                            PAD!(),
                            "tdpbf16ps tmm0, tmm4, tmm6",
                            PAD!(),
                            "tdpbf16ps tmm1, tmm4, tmm7",
                            PAD!(),
                            "tdpbf16ps tmm2, tmm5, tmm6",
                            PAD!(),
                            "tdpbf16ps tmm3, tmm5, tmm7",
                            PAD!(),
                            // hi * mid
                            "tileloadd tmm4, [{ah0} + {s}*1]",
                            PAD!(),
                            "tileloadd tmm5, [{ah1} + {s}*1]",
                            PAD!(),
                            "tileloadd tmm6, [{b} + {bs}*1 + 1024]",
                            PAD!(),
                            "tileloadd tmm7, [{b} + {bs}*1 + 3072]",
                            PAD!(),
                            "tdpbf16ps tmm0, tmm4, tmm6",
                            PAD!(),
                            "tdpbf16ps tmm1, tmm4, tmm7",
                            PAD!(),
                            "tdpbf16ps tmm2, tmm5, tmm6",
                            PAD!(),
                            "tdpbf16ps tmm3, tmm5, tmm7",
                            PAD!(),
                            ah0 = in(reg) a0,
                            ah1 = in(reg) a0.add(16 * d2),
                            am0 = in(reg) a0.add(part_len),
                            am1 = in(reg) a0.add(part_len + 16 * d2),
                            b = in(reg) bp,
                            s = in(reg) astride,
                            bs = in(reg) 64usize,
                            options(nostack, readonly),
                        );
                    } else {
                        asm!(
                            "tileloadd tmm4, [{ah0} + {s}*1]",
                            PAD!(),
                            "tileloadd tmm5, [{ah1} + {s}*1]",
                            PAD!(),
                            "tileloadd tmm6, [{b} + {bs}*1]",
                            PAD!(),
                            "tileloadd tmm7, [{b} + {bs}*1 + 1024]",
                            PAD!(),
                            "tdpbf16ps tmm0, tmm4, tmm6",
                            PAD!(),
                            "tdpbf16ps tmm1, tmm4, tmm7",
                            PAD!(),
                            "tdpbf16ps tmm2, tmm5, tmm6",
                            PAD!(),
                            "tdpbf16ps tmm3, tmm5, tmm7",
                            PAD!(),
                            ah0 = in(reg) a0,
                            ah1 = in(reg) a0.add(16 * d2),
                            b = in(reg) bp,
                            s = in(reg) astride,
                            bs = in(reg) 64usize,
                            options(nostack, readonly),
                        );
                    }
                }
                let c0 = xp.add(np * 32);
                asm!(
                    "tilestored [{c0} + {s}*1], tmm0",
                    "tilestored [{c0} + {s}*1 + 64], tmm1",
                    "tilestored [{c1} + {s}*1], tmm2",
                    "tilestored [{c1} + {s}*1 + 64], tmm3",
                    c0 = in(reg) c0,
                    c1 = in(reg) c0.add(16 * d2),
                    s = in(reg) cstride,
                    options(nostack),
                );
            }
        }
    }
}

// ----- drivers -------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Method {
    Scalar,
    Avx512,
    Amx3,
    Amx2,
    Amx1,
    /// Diagnostic: only the 3-part conversion of `Amx3`.
    Cvt3,
    /// Diagnostic: only the tile work of `Amx3` (no conversion).
    Tiles3,
}

impl Method {
    fn name(self) -> &'static str {
        match self {
            Method::Scalar => "scalar-f32",
            Method::Avx512 => "avx512-f32",
            Method::Amx3 => "amx-bf16x3",
            Method::Amx2 => "amx-bf16x2",
            Method::Amx1 => "amx-bf16x1",
            Method::Cvt3 => "(amx3 convert only)",
            Method::Tiles3 => "(amx3 tiles only)",
        }
    }
}

/// Everything a method needs for one unitary.
struct Prepared {
    d: usize,
    ur: Vec<f32>,
    ui: Vec<f32>,
    bt: Vec<f32>,
    b3: Vec<Line>,
    b2: Vec<Line>,
    b1: Vec<Line>,
}

fn prepare(u: &[Cx], d: usize) -> Prepared {
    let r = real_form(u, d);
    Prepared {
        d,
        ur: u.iter().map(|z| z.re as f32).collect(),
        ui: u.iter().map(|z| z.im as f32).collect(),
        bt: pack_bt(&r, 2 * d),
        b3: amx_pack_b(&r, 2 * d, 3),
        b2: amx_pack_b(&r, 2 * d, 2),
        b1: amx_pack_b(&r, 2 * d, 1),
    }
}

/// Applies the unitary once to `x` (interleaved f32) with `threads` threads.
fn apply(m: Method, p: &Prepared, x: &mut [f32], threads: usize) {
    let d2 = 2 * p.d;
    let rows = x.len() / d2;
    // whole blocks of 32 rows per thread (the AMX block size)
    let per = ((rows / threads.max(1)) / 32 * 32).max(32) * d2;
    let work = |chunk: &mut [f32]| match m {
        Method::Scalar => apply_scalar(chunk, &p.ur, &p.ui, p.d),
        #[cfg(target_arch = "x86_64")]
        Method::Avx512 => {
            let mut tmp = vec![0f32; 6 * d2];
            // SAFETY: only selected when `x86::has_avx512()`.
            unsafe { x86::gemm_avx512(chunk, &p.bt, d2, &mut tmp) }
        }
        #[cfg(target_arch = "x86_64")]
        Method::Amx3 | Method::Amx2 | Method::Amx1 | Method::Cvt3 | Method::Tiles3 => {
            let (parts, b) = match m {
                Method::Amx2 => (2, &p.b2),
                Method::Amx1 => (1, &p.b1),
                _ => (3, &p.b3),
            };
            let stage = match m {
                Method::Cvt3 => x86::Stage::ConvertOnly,
                Method::Tiles3 => x86::Stage::TilesOnly,
                _ => x86::Stage::Full,
            };
            let mut a = vec![Line([0; 32]); 2 * 3 * d2];
            // SAFETY: only selected when `x86::amx_ready()`; config and
            // release bracket the kernel on this thread.
            unsafe {
                x86::tile_config();
                x86::gemm_amx(chunk, b, d2, parts, &mut a, stage);
                x86::tile_release();
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        _ => unreachable!("x86_64 only"),
    };
    if threads <= 1 || per >= x.len() {
        work(x);
    } else {
        let work = &work;
        std::thread::scope(|s| {
            for chunk in x.chunks_mut(per) {
                s.spawn(move || work(chunk));
            }
        });
    }
}

fn random_state(n: usize, rng: &mut Rng) -> Vec<Cx> {
    let mut v: Vec<Cx> = (0..1usize << n)
        .map(|_| Cx {
            re: rng.gauss(),
            im: rng.gauss(),
        })
        .collect();
    let nrm = v.iter().map(|z| z.norm2()).sum::<f64>().sqrt();
    for z in &mut v {
        z.re /= nrm;
        z.im /= nrm;
    }
    v
}

fn to_f32(v: &[Cx]) -> Vec<f32> {
    v.iter().flat_map(|z| [z.re as f32, z.im as f32]).collect()
}

fn available() -> Vec<Method> {
    let mut ms = vec![Method::Scalar];
    #[cfg(target_arch = "x86_64")]
    {
        if x86::has_avx512() {
            ms.push(Method::Avx512);
        } else {
            println!("# no AVX-512F: avx512 kernel skipped");
        }
        if x86::amx_ready() {
            ms.push(Method::Amx3);
            ms.push(Method::Amx2);
            ms.push(Method::Amx1);
            ms.push(Method::Cvt3);
            ms.push(Method::Tiles3);
        } else {
            println!("# no AMX-BF16 / AVX512-BF16 / tile permission: AMX kernels skipped");
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    println!("# not x86_64: only the scalar path runs");
    ms
}

/// Max |dAmp| and 2-norm error against the f64 reference, after `apps`
/// applications, for every method; random (spread) and basis (all norm in
/// one group: the largest amplitudes) inputs.
fn accuracy() {
    let ms = available();
    println!("| k | unitary | input | n | apps | method | max abs err | 2-norm err | norm^2 - 1 |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for k in [5usize, 6, 7] {
        let d = 1usize << k;
        for kind in ["brick", "haar"] {
            let mut rng = Rng(1000 + k as u64);
            let u = if kind == "brick" {
                brick_fused(k, k, &mut rng)
            } else {
                haar(d, &mut rng)
            };
            let p = prepare(&u, d);
            for input in ["random", "basis"] {
                let n = if input == "random" { 16 } else { k + 5 };
                let x0: Vec<Cx> = if input == "random" {
                    random_state(n, &mut rng)
                } else {
                    let mut v = vec![Cx::default(); 1 << n];
                    v[0] = Cx { re: 1.0, im: 0.0 };
                    v
                };
                let x0f = to_f32(&x0);
                for apps in [1usize, 20] {
                    // reference from the f32 input, in f64
                    let mut r: Vec<Cx> = x0f
                        .chunks_exact(2)
                        .map(|c| Cx {
                            re: c[0] as f64,
                            im: c[1] as f64,
                        })
                        .collect();
                    for _ in 0..apps {
                        apply_ref(&u, d, &mut r);
                    }
                    for &m in ms
                        .iter()
                        .filter(|m| !matches!(m, Method::Cvt3 | Method::Tiles3))
                    {
                        let mut x = x0f.clone();
                        for _ in 0..apps {
                            apply(m, &p, &mut x, 1);
                        }
                        let (mut mx, mut s2, mut nn) = (0f64, 0f64, 0f64);
                        for (c, z) in x.chunks_exact(2).zip(&r) {
                            let e = Cx {
                                re: c[0] as f64 - z.re,
                                im: c[1] as f64 - z.im,
                            }
                            .norm2();
                            mx = mx.max(e.sqrt());
                            s2 += e;
                            nn += (c[0] as f64).powi(2) + (c[1] as f64).powi(2);
                        }
                        println!(
                            "| {k} | {kind} | {input} | {n} | {apps} | {} | {mx:.2e} | {:.2e} | {:.1e} |",
                            m.name(),
                            s2.sqrt(),
                            nn - 1.0
                        );
                    }
                }
            }
        }
    }
}

fn speed(n: usize, threads: usize, reps: usize, ks: &[usize], only: Option<&str>) {
    let ms: Vec<Method> = available()
        .into_iter()
        .filter(|&m| m != Method::Scalar && only.is_none_or(|o| m.name() == o))
        .collect();
    println!("| k | n | threads | method | min s | Gamp/s | vs avx512 | all s |");
    println!("|---|---|---|---|---|---|---|---|");
    for &k in ks {
        let d = 1usize << k;
        let mut rng = Rng(77 + k as u64);
        let u = brick_fused(k, k, &mut rng);
        let p = prepare(&u, d);
        let mut x = to_f32(&random_state(n, &mut rng));
        // warm-up (page faults, thread start, AMX first-use trap)
        for &m in &ms {
            apply(m, &p, &mut x, threads);
        }
        let mut times = vec![Vec::new(); ms.len()];
        for _ in 0..reps {
            for (i, &m) in ms.iter().enumerate() {
                let t = Instant::now();
                apply(m, &p, &mut x, threads);
                times[i].push(t.elapsed().as_secs_f64());
            }
        }
        let best: Vec<f64> = times
            .iter()
            .map(|t| t.iter().copied().fold(f64::INFINITY, f64::min))
            .collect();
        let base = ms
            .iter()
            .position(|&m| m == Method::Avx512)
            .map(|i| best[i]);
        for (i, &m) in ms.iter().enumerate() {
            let rel = base
                .map(|b| format!("{:.2}x", b / best[i]))
                .unwrap_or_default();
            println!(
                "| {k} | {n} | {threads} | {} | {:.5} | {:.3} | {rel} | {} |",
                m.name(),
                best[i],
                (1u64 << n) as f64 / best[i] / 1e9,
                times[i]
                    .iter()
                    .map(|t| format!("{t:.5}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(|s| s.as_str()) {
        Some("acc") => accuracy(),
        Some("micro") => {
            #[cfg(target_arch = "x86_64")]
            if x86::amx_ready() {
                // SAFETY: AMX, AVX-512F and the tile permission were detected.
                unsafe { x86::micro() }
                return;
            }
            println!("# micro needs x86_64 with AMX-BF16 and AVX-512");
        }
        Some("speed") if a.len() >= 5 => {
            let n: usize = a[2].parse().expect("n");
            let threads: usize = a[3].parse().expect("threads");
            let reps: usize = a[4].parse().expect("reps");
            // optional trailing `only=<method name>`: time one method per
            // process (AMX lowers the clock for a while after it runs, which
            // would bias an interleaved AVX-512 measurement)
            let only = a[5..].iter().find_map(|s| s.strip_prefix("only="));
            let ks: Vec<usize> = a[5..]
                .iter()
                .filter(|s| !s.starts_with("only="))
                .map(|s| s.parse().expect("k"))
                .collect();
            let ks = if ks.is_empty() { vec![5, 6, 7] } else { ks };
            assert!(
                ks.iter().all(|&k| (5..=7).contains(&k) && n >= k + 5),
                "k in 5..=7 and n >= k + 5"
            );
            speed(n, threads, reps, &ks, only);
        }
        _ => {
            eprintln!("usage: amx_dense acc | micro | speed <n> <threads> <reps> [k ...]");
            std::process::exit(2);
        }
    }
}
