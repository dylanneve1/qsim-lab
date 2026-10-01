//! Single-thread microbenchmark of in-cache 2x2 kernels (AoS vs SoA).
use num_complex::Complex;
use std::hint::black_box;
use std::time::Instant;

type C = Complex<f32>;

#[inline(never)]
fn aos_ops(buf: &mut [C], t: usize, m: &[C; 4]) {
    let s = 1 << t;
    for ch in buf.chunks_exact_mut(2 * s) {
        let (lo, hi) = ch.split_at_mut(s);
        for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
            let (x, y) = (*a, *b);
            *a = m[0] * x + m[1] * y;
            *b = m[2] * x + m[3] * y;
        }
    }
}

#[inline(never)]
fn aos_explicit(buf: &mut [C], t: usize, m: &[C; 4]) {
    let s = 1 << t;
    let [m0, m1, m2, m3] = *m;
    for ch in buf.chunks_exact_mut(2 * s) {
        let (lo, hi) = ch.split_at_mut(s);
        for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
            let (xr, xi, yr, yi) = (a.re, a.im, b.re, b.im);
            a.re = m0.re * xr - m0.im * xi + m1.re * yr - m1.im * yi;
            a.im = m0.re * xi + m0.im * xr + m1.re * yi + m1.im * yr;
            b.re = m2.re * xr - m2.im * xi + m3.re * yr - m3.im * yi;
            b.im = m2.re * xi + m2.im * xr + m3.re * yi + m3.im * yr;
        }
    }
}

#[inline(never)]
fn soa(re: &mut [f32], im: &mut [f32], t: usize, m: &[C; 4]) {
    let s = 1 << t;
    let [m0, m1, m2, m3] = *m;
    for (cr, ci) in re.chunks_exact_mut(2 * s).zip(im.chunks_exact_mut(2 * s)) {
        let (lr, hr) = cr.split_at_mut(s);
        let (li, hi) = ci.split_at_mut(s);
        for (((ar, ai), br), bi) in lr
            .iter_mut()
            .zip(li.iter_mut())
            .zip(hr.iter_mut())
            .zip(hi.iter_mut())
        {
            let (xr, xi, yr, yi) = (*ar, *ai, *br, *bi);
            *ar = m0.re * xr - m0.im * xi + m1.re * yr - m1.im * yi;
            *ai = m0.re * xi + m0.im * xr + m1.re * yi + m1.im * yr;
            *br = m2.re * xr - m2.im * xi + m3.re * yr - m3.im * yi;
            *bi = m2.re * xi + m2.im * xr + m3.re * yi + m3.im * yr;
        }
    }
}

#[inline(never)]
fn diag_rows(re: &mut [f32], im: &mut [f32], lr: &[f32], li: &[f32], hi: &[C]) {
    let w = lr.len();
    for (h, hv) in hi.iter().enumerate() {
        let ar = &mut re[h * w..(h + 1) * w];
        let ai = &mut im[h * w..(h + 1) * w];
        for k in 0..w {
            let fr = lr[k] * hv.re - li[k] * hv.im;
            let fi = lr[k] * hv.im + li[k] * hv.re;
            let (xr, xi) = (ar[k], ai[k]);
            ar[k] = xr * fr - xi * fi;
            ai[k] = xr * fi + xi * fr;
        }
    }
}

/// Small target t < 3: process aligned groups of 8 with fixed index maps.
#[inline(never)]
fn soa_small<const T: usize>(re: &mut [f32], im: &mut [f32], m: &[C; 4]) {
    let s = 1 << T;
    // lo indices within a group of 8
    let lo: [usize; 4] = std::array::from_fn(|k| ((k >> T) << (T + 1)) | (k & (s - 1)));
    let [m0, m1, m2, m3] = *m;
    for (gr, gi) in re.chunks_exact_mut(8).zip(im.chunks_exact_mut(8)) {
        let xr: [f32; 4] = std::array::from_fn(|k| gr[lo[k]]);
        let xi: [f32; 4] = std::array::from_fn(|k| gi[lo[k]]);
        let yr: [f32; 4] = std::array::from_fn(|k| gr[lo[k] + s]);
        let yi: [f32; 4] = std::array::from_fn(|k| gi[lo[k] + s]);
        for k in 0..4 {
            gr[lo[k]] = m0.re * xr[k] - m0.im * xi[k] + m1.re * yr[k] - m1.im * yi[k];
            gi[lo[k]] = m0.re * xi[k] + m0.im * xr[k] + m1.re * yi[k] + m1.im * yr[k];
            gr[lo[k] + s] = m2.re * xr[k] - m2.im * xi[k] + m3.re * yr[k] - m3.im * yi[k];
            gi[lo[k] + s] = m2.re * xi[k] + m2.im * xr[k] + m3.re * yi[k] + m3.im * yr[k];
        }
    }
}

fn main() {
    let n = 1usize << 15;
    let reps = 400;
    let h = std::f32::consts::FRAC_1_SQRT_2;
    let m = [
        C::new(h, 0.1),
        C::new(h, -0.2),
        C::new(h, 0.3),
        C::new(-h, 0.0),
    ];
    let mut a: Vec<C> = (0..n).map(|i| C::new(i as f32 * 1e-6, 0.5)).collect();
    let mut re: Vec<f32> = a.iter().map(|z| z.re).collect();
    let mut im: Vec<f32> = a.iter().map(|z| z.im).collect();
    println!("| t | aos ops ns/amp | aos explicit | soa |");
    for t in [0usize, 1, 2, 3, 5, 10] {
        let mut best = [f64::INFINITY; 3];
        for _ in 0..3 {
            let t0 = Instant::now();
            for _ in 0..reps {
                aos_ops(black_box(&mut a), t, &m);
            }
            best[0] = best[0].min(t0.elapsed().as_secs_f64());
            let t0 = Instant::now();
            for _ in 0..reps {
                aos_explicit(black_box(&mut a), t, &m);
            }
            best[1] = best[1].min(t0.elapsed().as_secs_f64());
            let t0 = Instant::now();
            for _ in 0..reps {
                soa(black_box(&mut re), black_box(&mut im), t, &m);
            }
            best[2] = best[2].min(t0.elapsed().as_secs_f64());
        }
        let k = 1e9 / (reps * n) as f64;
        println!(
            "| {t} | {:.3} | {:.3} | {:.3} |",
            best[0] * k,
            best[1] * k,
            best[2] * k
        );
    }
    let lr: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).cos()).collect();
    let li: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
    let hi: Vec<C> = (0..n / 256)
        .map(|i| C::from_polar(1.0, i as f32 * 0.1))
        .collect();
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t0 = Instant::now();
        for _ in 0..reps {
            diag_rows(black_box(&mut re), black_box(&mut im), &lr, &li, &hi);
        }
        best = best.min(t0.elapsed().as_secs_f64());
    }
    for t in 0..3 {
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t0 = Instant::now();
            for _ in 0..reps {
                match t {
                    0 => soa_small::<0>(black_box(&mut re), black_box(&mut im), &m),
                    1 => soa_small::<1>(black_box(&mut re), black_box(&mut im), &m),
                    _ => soa_small::<2>(black_box(&mut re), black_box(&mut im), &m),
                }
            }
            best = best.min(t0.elapsed().as_secs_f64());
        }
        println!(
            "soa small t={t}: {:.3} ns/amp",
            best * 1e9 / (reps * n) as f64
        );
    }
    println!("diag rows: {:.3} ns/amp", best * 1e9 / (reps * n) as f64);
    black_box((&a, &re, &im));
}
