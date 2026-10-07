# Packed chain sweep on a GPU (wgpu: Vulkan / DX12 / Metal)

Branch `exp/packed-wgpu`. Code: `src/engines/chain_packed_gpu.rs` (host side, codec, CPU emulation),
`src/engines/chain_packed_gpu/{gpu.rs,gen.rs,kernels.wgsl}` (wgpu driver, generated sub-stage kernel,
WGSL), `src/engines/blocked/gpu_export.rs` (export of the CPU's compiled plan), feature `wgpu`.
Entry points: `GpuSweeper::run` / `run_into`, the `GpuPacked` `SweepBackend` (`chain_sweep run --gpu`),
`chain_sweep packedgpu` (timing, `--cpu` bit-exact check, `--count` plan statistics).

## Design

- The packed register (`intB:bN` or `intB:bN:h`) stays in host RAM in the `PackedStore` layout. Each
  memory pass streams it through the GPU in chunks of whole gathered blocks:
  host gather of the packed runs into mapped staging → copy to VRAM → `unpack` → one `substage`
  dispatch per cache-blocked stage → `pack` → copy back → host scatter. `--inflight` chunks are in
  flight (separate buffers), so host copies, PCIe and kernels overlap. All dispatches are short
  (~10 ms on the 4090), well under the Windows TDR limit.
- **The ops are the CPU's own plan.** `chain_packed::gpu_stage_plan` builds the same `StageExec` as
  `PackedStore::run_stage` and exports its compiled cache-blocked stages (`gpu_export`): U1 (X / real
  / complex kernels), swaps, pairs (two 1q gates + folded CNOT), diagonal groups. The f32
  per-element formulas of the FMA kernel tier are short fixed `fma`/`mul` sequences, and the shader
  repeats them literally. The only f64 part of the CPU kernels, the diagonal groups' product tables,
  is precomputed on the host for every activity pattern of the group's outer-conditioned terms
  (at most 2 such terms at D = 70).
- **The format arithmetic is f64 on the CPU and has no f64 on the GPU** (Metal has none; Vulkan's is
  slow). It is reproduced exactly with f32/u32:
  - `:h` step: `m / maxv` rounded up to 11 bits, in integer arithmetic (`step_parts`). Encode and
    decode use tables over the 1024 step mantissas: thresholds of `round(x·(1/step))` and decoded
    values `q·(1/(1/step))` at a normalised step, moved to the block's exponent by exact power-of-two
    scaling.
  - f32 steps (`int6:b64`, `int4:b64`): `(m/maxv) as f32` by integer division with sticky rounding
    (`exact32::step32`). Encode: an f32 fast path `|x|·(1/step)` away from rounding ties and the clamp,
    otherwise the exact emulation of `fl64(x · fl64(1/step))` and round-half-even (53-bit reciprocal by
    long division, 77-bit product, double rounding). Decode `fl32(fl64(q·fl64(1/fl64(1/step))))`: a
    2-bit table k (2 MB) with `fl64(1/fl64(1/s)) = s + k ulp` for all 2^23 mantissas, then integer
    rounding of `q·s` where k only breaks exact ties (`dec_fast`).
- Sub-stage kernel: one workgroup per cache block (2^nb amplitudes, 32 KiB at nb = 12) in workgroup
  memory, 256 threads, one barrier per op. Op programs live in a 64 KiB uniform buffer. A second,
  generated kernel keeps 2^rb amplitudes per thread in registers (`--rb`, ops on the register bits
  need no barrier). It is bit-exact too, but slower on both GPUs tried, so `--rb 0` is the default.
- Workgroup-memory zero-initialisation is switched off (it made the kernel 20× slower on Metal), and
  the shader module is created without runtime bounds checks.

## Bit-exactness

The reference is `chain_packed::run_packed` with the same stages and the config the GPU reproduces
(`chain_packed_gpu::gpu_cfg(nb)`: FMA kernel tier, i.e. no AVX-512 kernels, no dense fusion, no L1
tiling). The CPU result itself depends on the kernel tier: the AVX-512 tier rounds differently
("agrees to rounding error"), so a GPU bit-exact with every tier is impossible. Evidence:

- `cargo test --features wgpu --test chain_packed_gpu`:
  - CPU emulation of the GPU pipeline == `run_packed`, bit for bit: int4:b16:h, int5:b16:h, int6:b64,
    int4:b16, int8:b32. Doped-circuit tail windows (last 28 layers, width ≥ 13), nested 2^6 and 2^8
    cache blocks, fused and unfused 1q gates, plus the direct (unnested) plan.
  - GPU == `run_packed`, bit for bit: int4:b16:h, int6:b64, int5:b16, fused and unfused, `rb` = 0, 2,
    4, 6. Passes on Apple M1 Pro (Metal).
  - Unit tests: the integer step / reciprocal / encode / decode emulation equals the CPU's f64
    arithmetic on 1.5M random and exact-tie cases per format (`exact32_matches_cpu_f64`,
    `step_parts_match_int_step`, `codec_matches_cpu_rounding`).
- `chain_sweep packedgpu --cpu` (full sweep, amplitude bits compared):

| device | case | GPU | CPU (`run_packed`, same config) | bit-exact |
|---|---|---|---|---|
| RTX 4090, Vulkan, driver 591.86 | D = 56 tail, width 28, 71 passes, int4:b64 | 6.1 s (rb 0), 6.7 s (rb 4) | 172 s / 170 s | yes / yes |
| Apple M1 Pro, Metal | D = 56 tail, width 28, 71 passes, int6:b64 | 85.7 s (before later kernel fixes) | 159.6 s | yes |
| Apple M1 Pro, Metal | D = 48 tail, width 24, 1 pass, int6:b64 | 52 s (before the zero-init fix) | 22.7 s | yes |

## Timings on the RTX 4090 (Paweł's PC: Ryzen 9 9950X3D, 64 GB DDR5, PCIe 4.0)

`--l 26 --slots 14 --nb 12 --rb 0 --chunk 27 --inflight 3`:

| case | register | passes | time | per pass |
|---|---|---|---|---|
| D = 62 tail, int4:b64 | 2^31 (2.1 GiB) | 71 | 45.6 s (rb 0), 50.7 s (rb 4), 53.6 s (rb 3) | 0.67 s median |
| D = 70, int4:b64 | 2^35 (34 GiB) | 71 | ~10.5 s/pass measured over passes 1–60 (645 s); the PC was switched off before pass 71 | ≈ 12.5 min/sweep |

For comparison, the CPU packed path on the same PC (AVX-512 tier, its own pilot) takes 52.8 min per
D = 70 sweep (44.6 s/pass). The GPU is ~4.2× faster. At D = 70 the process held 37.1 GB working set /
41.9 GB private (store 36.5 GB). `--chunk 26 --inflight 2` should cut the private bytes to ≈ 39.5 GB
(not measured).

Where the time goes (Metal profile, D = 54 tail, dropping op kinds, results wrong, timing only): pairs
~45 %, diagonal groups ~25 %, U1 ~15 %, everything else (load/store, unpack/pack, transfers) ~20 %.
The kernel is bound by per-op work in workgroup memory (one barrier per op, 8 pairs per thread per
op), far from the 4090's DRAM or PCIe limits. A sub-stage moves 512 GiB at ~250 GB/s effective.

## What remains

- Profile on the 4090 (`QSIM_GPU_SKIP`, `QSIM_GPU_NOOPS`, `QSIM_GPU_DROP` knobs exist) and speed up
  the op kernel. Options: warp-shuffle exchanges for low bits; fusing runs of ops between barriers;
  fusing unpack/pack into the first/last sub-stage (−17 % VRAM traffic); fewer sub-stages with
  `--nslots 12` (3.0 instead of 4.9 per pass, but a different CPU-reference plan).
- Keep part of the register resident in VRAM (≤ 20 GB) to cut host RAM and PCIe traffic. This also
  lets int6:b64 (50 GiB) fit next to the OS in 64 GB.
- Measure `--chunk 26 --inflight 2` (host RAM) and `--l 22` on the 4090.
