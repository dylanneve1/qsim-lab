//! The GPU pipeline of the packed chain sweep (`chain_packed_gpu`) against
//! the CPU packed path (`chain_packed::run_packed`): bit for bit, on the
//! IBM doped-Clifford circuit (tail windows), with direct and nested
//! (cache-blocked) stage plans. The CPU emulation of the GPU pipeline is
//! always tested; the GPU itself with `--features wgpu` when an adapter is
//! available (`QSIM_WGPU_REQUIRE=1` makes a missing adapter a failure).

use num_complex::Complex64;
use qsim_lab::circuit::Circuit;
use qsim_lab::engines::blocked::{avx512_available, fuse_1q, BlockConfig, Stage};
use qsim_lab::engines::chain_lowprec::LowPrec;
use qsim_lab::engines::chain_packed::{packed_stages, run_packed};
use qsim_lab::engines::chain_packed_gpu::{cfg_exportable, emulate_packed};
use qsim_lab::engines::chain_sweep::{self, compile, ChainCircuit, SweepPlan};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

const QASM: &str = include_str!("../../research/chain-sweep/nq70_depth70_checks27_doped.qasm");

/// The FMA-tier config the GPU reproduces, with nested cache blocks of
/// `2^nested_bits` amplitudes.
fn gpu_cfg(nested_bits: usize) -> BlockConfig {
    BlockConfig {
        block_bytes: 8 << nested_bits,
        fuse_1q: false,
        avx512: false,
        dense_fusion: 0,
        l1_tile_bytes: 0,
        ..BlockConfig::default()
    }
}

struct Case {
    plan: SweepPlan,
    stages: Vec<Stage>,
}

fn cases(lo: usize, hi: usize, k: usize, l_minus: usize, slots: usize, fuse: bool) -> Vec<Case> {
    let full = Circuit::from_qasm(QASM).unwrap();
    let c = chain_sweep::truncate_window(&full, 70, lo, hi);
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let mut rng = StdRng::seed_from_u64(5 + lo as u64);
    (0..k)
        .map(|_| {
            let x: u128 = (0..70).fold(0, |a, i| a | ((rng.random_bool(0.5) as u128) << i));
            let plan = compile(&cc, x, &HashMap::new());
            let ops = if fuse {
                fuse_1q(&plan.ops, plan.width, false)
            } else {
                plan.ops.clone()
            };
            let stages = packed_stages(&ops, plan.width, plan.width - l_minus, slots);
            Case { plan, stages }
        })
        .collect()
}

#[test]
fn emulated_gpu_pipeline_is_bit_exact_with_cpu_packed() {
    let fmts = [
        "int4:b16:h",
        "int5:b16:h",
        "int6:b64",
        "int4:b16",
        "int8:b32",
    ];
    // nested plans (cache blocks of 2^6 and 2^8 amplitudes inside the
    // gathered blocks) on the last 28 layers, fused and unfused
    for (fuse, nb) in [(false, 6), (true, 8)] {
        let cfg = gpu_cfg(nb);
        assert!(cfg_exportable(&cfg));
        for case in cases(42, 70, 3, 4, 3, fuse) {
            assert!(case.plan.width >= 13);
            for f in fmts {
                let lp = LowPrec::parse(f).unwrap();
                let cpu = run_packed(&case.plan, &case.stages, &lp, &cfg).unwrap();
                let emu = emulate_packed(&case.plan, &case.stages, &lp, &cfg).unwrap();
                assert_eq!(emu.amp, cpu.amp, "{f} fuse={fuse} nb={nb}");
                assert_eq!((emu.underflow, emu.overflow, emu.inexact), (0, 0, 0));
                assert_eq!(emu.store_bytes, cpu.store_bytes);
                assert!(emu.amp != Complex64::new(0.0, 0.0));
            }
        }
    }
    // direct plans (the stage runs as one block): FMA tier only without
    // AVX-512 (`run_prepared_on_block` picks AVX-512 when available)
    if !avx512_available() {
        let cfg = gpu_cfg(16);
        for case in cases(46, 70, 2, 4, 3, true) {
            for f in ["int4:b16:h", "int6:b32:h"] {
                let lp = LowPrec::parse(f).unwrap();
                let cpu = run_packed(&case.plan, &case.stages, &lp, &cfg).unwrap();
                let emu = emulate_packed(&case.plan, &case.stages, &lp, &cfg).unwrap();
                assert_eq!(emu.amp, cpu.amp, "{f} direct");
            }
        }
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn gpu_is_bit_exact_with_cpu_packed() {
    use qsim_lab::engines::chain_packed_gpu::gpu::{GpuOptions, GpuSweeper};
    let require = std::env::var_os("QSIM_WGPU_REQUIRE").is_some();
    let nb = 8;
    let cfg = gpu_cfg(nb);
    let opts = GpuOptions {
        nested_bits: nb,
        chunk_amps: 1 << 13,
        ..GpuOptions::default()
    };
    let gpu = match GpuSweeper::new(&opts) {
        Ok(g) => g,
        Err(e) => {
            assert!(!require, "no GPU adapter: {e}");
            eprintln!("skipping: no GPU adapter ({e})");
            return;
        }
    };
    eprintln!("adapter: {}", gpu.adapter_info());
    for fuse in [false, true] {
        for case in cases(42, 70, 2, 4, 3, fuse) {
            for f in ["int4:b16:h", "int6:b64", "int5:b16"] {
                let lp = LowPrec::parse(f).unwrap();
                let cpu = run_packed(&case.plan, &case.stages, &lp, &cfg).unwrap();
                let g = gpu.run(&case.plan, &case.stages, &lp, &cfg).unwrap();
                assert_eq!(g.amp, cpu.amp, "{f} fuse={fuse}");
                assert_eq!((g.underflow, g.overflow, g.inexact), (0, 0, 0));
            }
        }
    }
}
