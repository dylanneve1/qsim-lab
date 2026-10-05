# AMX bf16x3 for large dense fusion (time-boxed experiment)

Driver: [`examples/amx_dense.rs`](../../../../examples/amx_dense.rs) (std only; not used by the engine).
Notebook: [`research/performance/avx512.md`](../../../performance/avx512.md), AMX section.

## What is measured

A fused dense `2^k x 2^k` unitary (k = 5, 6, 7) is applied to an n-qubit complex f32 state.
Targets are the k lowest qubits of an interleaved (re, im) state, so each group of `D = 2^k`
amplitudes is one row of `X` (2D reals) and the update is the real GEMM `Y = X R^T`, with `R` the
2D x 2D real form of the unitary (rounded to f32 once; every method uses the same f32 matrix).

| method | what it does |
|---|---|
| `avx512-f32` | AVX-512 FMA GEMM on the same layout: blocks of 6 rows x 64 columns (24 zmm accumulators; 4 B loads and 6 broadcasts per 24 FMAs; 4- and 2-row tails), B packed in 64-column panels; the hot loop has no spills (checked with objdump). It replaced an 8 x 32 block that was 10-40 % slower |
| `amx-bf16x3` | AMX `TDPBF16PS`: every f32 of both operands split error-free into hi + mid + lo bf16 (round to nearest even; 24 = 8 + 8 + 8 significand bits), the six products with i + j <= 2 accumulated in f32 tiles; 2 x 2 C tiles (32 rows x 32 columns), 24 TDPBF16PS and 16 tile loads per K chunk of 32, each followed by 16 NOPs (back-to-back AMX instructions issue at about half rate here, see `micro`); the state's f32 -> bf16 conversion (AVX512-BF16 `vcvtne2ps2bf16`) is inside the timed region, double-buffered |
| `amx-bf16x2` | hi + mid only, products hi*hi, mid*hi, hi*mid (12 TDPBF16PS, 10 loads per chunk): diagnostic, not f32 accuracy |
| `amx-bf16x1` | one bf16 product, no split: the upper bound on AMX speed (about 3 digits) |
| `(amx3 convert only)` / `(amx3 tiles only)` | the two halves of `amx-bf16x3`, to attribute its time |
| `scalar-f32` | plain f32 complex mat-vec (accuracy table only) |

Unitaries: `brick` = product of k layers of Haar 4x4 unitaries on local pairs (q, q+1), alternating
offsets (what a fusion pass produces from a random-circuit brickwork); `haar` = Haar `2^k x 2^k`
(Gram-Schmidt on complex Gaussians, positive R diagonal). Accuracy is against an f64 complex
mat-vec from the same f32 input: `random` = normalised Gaussian state on 16 qubits; `basis` =
|0...0> on k + 5 qubits (all the norm in one group: the largest amplitudes, the hardest case for a
max-abs bound); 1 and 20 successive applications.

## Reproduce

```sh
rustc --edition 2021 -C opt-level=3 -C target-cpu=native examples/amx_dense.rs -o amx_dense
./amx_dense acc                      # accuracy table (no lock needed)
./amx_dense micro                    # TDPBF16PS / tileloadd / FMA throughput
./amx_dense speed <n> <threads> <reps> [k ...]
research/data/avx512/amx/run.sh ./amx_dense research/data/avx512/amx   # everything, under the bench lock
```

`run.sh` pins 1 thread to vCPU 4 and 8 threads to one vCPU per physical core (0,2,...,14), takes
the shared bench lock for the timed block and records `uptime` / `free -g` before and after. It
times **one method per process** (`only=<method>`), alternating the methods, two processes per
method and min of 5 in-process repetitions each: running AMX lowers the core clock for a while,
so an interleaved AVX-512 measurement right after it was 20-25 % slow
(`speed-20261005T1206.md` used that interleaved protocol, with the unpadded kernel).
`python3 summarize.py speed-<tag>.md` prints the per-cell minima and ratios.

## Results in brief (details and caveats: notebook)

Accuracy (`accuracy.md`, k = 5..7, both unitary kinds, worst case over cells, max |dAmp| vs f64):
after 20 applications to |0..0>, scalar f32 3.3e-7, AVX-512 f32 3.1e-7, **AMX bf16x3 1.5e-7**,
bf16x2 7.8e-6, bf16x1 3.9e-3. bf16x3 is as accurate as f32 FMA (about half its error) and meets
the engine's 1e-5 by two orders of magnitude; bf16x2 and bf16x1 do not have f32 accuracy.

Speed, AMX bf16x3 / AVX-512 f32 (one method per process, min of 2 processes x 5 reps):

| run | load (1-min) | 1 thread, k = 5 / 6 / 7 | 8 threads, k = 5 / 6 / 7 |
|---|---|---|---|
| `speed-20261005T1342.md` | 6.1 before, 3.8 after | 0.90-1.12 / 1.54-1.73 / 1.86-1.95 | 0.84-0.93 / 1.45-1.57 / 1.38-1.77 |
| `speed-20261005T1350.md` | 14.9 before, 10.3 after | 0.82-0.85 / 1.20-1.25 / 1.17-1.25 | 0.59-0.96 / 1.14-1.72 / 0.76-1.63 |
| `speed-20261005T1236.md` | 27.5 / 20.3 (loaded) | 0.79-0.84 / 0.94-1.18 / 1.17-1.29 | (n = 20 cells corrupted by load) |

The AMX rate depends on the machine state: `micro` gives 5.5 ns per TDPBF16PS back to back (18
cycles at 3.3 GHz, near the documented 16) and overlapping tile loads in the quiet run, but 11.9-15.7
ns back to back (8.1-8.3 ns with 16 NOPs between) with tile loads serialised behind the products in
the loaded runs; the AVX-512 kernel loses less. Likely cause (not proven): busy SMT siblings
(the vCPUs 2k+1 next to our pinned vCPUs 2k run other users' jobs).

## Files

- `accuracy.md`: the accuracy table (`amx_dense acc`).
- `speed-<timestamp>.md`: `micro` and `speed` output with the load before and after each locked block.
- `speed-20261005T1206.md`: first run, unpadded kernel, methods interleaved within one process.
- `pad-ab.md`: NOP-padding A/B of the kernel (exploratory, load 22).
- `pad-quiet.md`: padded vs unpadded kernel, one process per run (load 11): padded 1.20x (k = 5), 1.32-1.37x (k = 6), 1.36-1.42x (k = 7).
- `amx-issue-spacing.md`, `micro_spacing.rs`: TDPBF16PS rate back to back vs with NOPs / adds between.
- `run.sh`, `summarize.py`: the driver and the table builder.
