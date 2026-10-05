// Standalone probe behind amx-issue-spacing.md (x86_64 with AMX only):
//   rustc --edition 2021 -C opt-level=3 micro_spacing.rs -o micro_spacing && taskset -c 4 ./micro_spacing
// Requests the AMX tile permission (arch_prctl), then times TDPBF16PS back to back and with
// register adds / NOPs between them.
use std::arch::asm;
use std::time::Instant;
#[repr(C, align(64))]
struct Cfg([u8; 64]);
#[repr(C, align(64))]
struct Buf([u16; 16 * 32 * 8]);
fn main() {
    let r: i64;
    unsafe { asm!("syscall", inlateout("rax") 158i64 => r, in("rdi") 0x1023u64, in("rsi") 18u64, lateout("rcx") _, lateout("r11") _, options(nostack)); }
    assert_eq!(r, 0);
    let mut c = Cfg([0; 64]);
    c.0[0] = 1;
    for t in 0..8 { c.0[16 + 2 * t] = 64; c.0[48 + t] = 16; }
    let buf = Box::new(Buf([0x3f80u16; 16 * 32 * 8]));
    unsafe { asm!("ldtilecfg [{}]", in(reg) &c, options(nostack)); }
    let p = buf.0.as_ptr();
    unsafe { asm!("tileloadd tmm4, [{p} + {s}*1]", "tileloadd tmm5, [{p} + {s}*1]", "tileloadd tmm6, [{p} + {s}*1]", "tileloadd tmm7, [{p} + {s}*1]",
        "tilezero tmm0", "tilezero tmm1", "tilezero tmm2", "tilezero tmm3", p = in(reg) p, s = in(reg) 64usize, options(nostack)); }
    let iters = 200_000u64;
    let mut x = 0u64;
    for rep in 0..3 {
        let t = Instant::now();
        for _ in 0..iters { unsafe { asm!("tdpbf16ps tmm0, tmm4, tmm6", "tdpbf16ps tmm1, tmm4, tmm7", "tdpbf16ps tmm2, tmm5, tmm6", "tdpbf16ps tmm3, tmm5, tmm7", options(nostack, nomem)); } }
        let a = t.elapsed().as_secs_f64() / (4 * iters) as f64 * 1e9;
        let t = Instant::now();
        for _ in 0..iters { unsafe { asm!("tdpbf16ps tmm0, tmm4, tmm6", ".rept 8", "add {x}, {y}", ".endr", "tdpbf16ps tmm1, tmm4, tmm7", ".rept 8", "add {x}, {y}", ".endr", "tdpbf16ps tmm2, tmm5, tmm6", ".rept 8", "add {x}, {y}", ".endr", "tdpbf16ps tmm3, tmm5, tmm7", ".rept 8", "add {x}, {y}", ".endr", x = inout(reg) x, y = in(reg) 1u64, options(nostack, nomem)); } }
        let b = t.elapsed().as_secs_f64() / (4 * iters) as f64 * 1e9;
        let t = Instant::now();
        for _ in 0..iters { unsafe { asm!("tdpbf16ps tmm0, tmm4, tmm6", ".rept 2", "add {x}, {y}", ".endr", "tdpbf16ps tmm1, tmm4, tmm7", ".rept 2", "add {x}, {y}", ".endr", "tdpbf16ps tmm2, tmm5, tmm6", ".rept 2", "add {x}, {y}", ".endr", "tdpbf16ps tmm3, tmm5, tmm7", ".rept 2", "add {x}, {y}", ".endr", x = inout(reg) x, y = in(reg) 1u64, options(nostack, nomem)); } }
        let cc = t.elapsed().as_secs_f64() / (4 * iters) as f64 * 1e9;
        let t = Instant::now();
        for _ in 0..iters { unsafe { asm!("tdpbf16ps tmm0, tmm4, tmm6", ".rept 16", "nop", ".endr", "tdpbf16ps tmm1, tmm4, tmm7", ".rept 16", "nop", ".endr", "tdpbf16ps tmm2, tmm5, tmm6", ".rept 16", "nop", ".endr", "tdpbf16ps tmm3, tmm5, tmm7", ".rept 16", "nop", ".endr", options(nostack, nomem)); } }
        let d = t.elapsed().as_secs_f64() / (4 * iters) as f64 * 1e9;
        // 8 dps per asm block back to back (more unrolled)
        let t = Instant::now();
        for _ in 0..iters / 2 { unsafe { asm!(".rept 2", "tdpbf16ps tmm0, tmm4, tmm6", "tdpbf16ps tmm1, tmm4, tmm7", "tdpbf16ps tmm2, tmm5, tmm6", "tdpbf16ps tmm3, tmm5, tmm7", ".endr", options(nostack, nomem)); } }
        let e = t.elapsed().as_secs_f64() / (4 * iters) as f64 * 1e9;
        println!("rep {rep}: ns/dp: back-to-back {a:.2} | +8 adds {b:.2} | +2 adds {cc:.2} | +16 nops {d:.2} | unrolled x2 {e:.2}");
    }
    unsafe { asm!("tilerelease", options(nostack, nomem)); }
    println!("{x}");
}
