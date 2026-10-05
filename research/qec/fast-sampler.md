# Fast detector sampling: Poisson hits into precomputed detector tables

Branch `exp/fast-sampler`. Code: `src/stabilizer/fast_sampler.rs`, `examples/stim_compare.rs`
(`sample-fast`, `bench-fast`, `dem-support-fast`, `probe-bitsliced`). Tests: `tests/fast_sampler.rs`
plus a unit test in the module. Data and scripts: `research/data/fast-sampler/`.

**Question.** On *Stim's own* `surface_code:rotated_memory_z` circuit, the SymPhase detector sampler
was at parity with natively built AVX2 Stim (0.89–1.01×, `research/qec/qec-r4.md` §1.5). 72–82% of our time
went into drawing noise variables at about 45 ns per fault. Can we get a clear single-thread lead,
with the same output distribution?

**Answer.** Yes. On x86 (VPS, AVX2-native Stim 1.16 built from source) **qsim-lab is 8.3–18× faster
than native Stim on identical circuits, and 8.3–17× faster than the better of pip and native Stim** (d = 3–15, rounds = d, p = 0.1% and 0.3%, same
`.stim` file, same ptb64 output, single thread). Two caveats:
- Our compile step is slower: 55–80 ms against 3.5 ms for Stim's whole 64-shot process at d = 15.
  End to end (process start to finish, 128k shots at d = 15) the lead is therefore 2.5–3.7×.
- On the Mac the comparison flatters us: Stim has no NEON backend.

The distribution is unchanged:
- every hit-table entry is an error signature of Stim's DEM, exactly (16/16 cells);
- 0 rejections in 155,531 two-sample tests at 10⁶ shots;
- the perturbed-circuit control is rejected at |z| = 17 and 61.

---

## 1. Why the old sampler was slow, and what replaced it

SymPhase compiles the circuit once into `detectors = A · v` over independent noise groups `v`:
- a 1-qubit depolarizing channel is 2 variables (x, z);
- a 2-qubit depolarizing channel is 4 variables;
- a flip is 1 variable.

The old loop:
- drew every fault by geometric skipping (ChaCha12 `f64`, `ln`, a division);
- drew the Pauli with a second `random_range`;
- OR-ed the fault into a variable buffer;
- then evaluated `A v`.

That costs about 45 ns per fault, while the evaluation itself is cheap.

**FastSampler** draws the same distribution differently. It takes four steps.

1. **Hit model (exact).** A group has `m` non-identity patterns (`m` = 1 flip, 3 depol1, 15 depol2),
   each with probability `p/m`. Treat it as a Poisson number of *hits*, each XOR-ing in a uniformly
   random non-identity pattern. The patterns with identity form `Z₂^b` (`m + 1 = 2^b`), so the net
   pattern is a random walk on `Z₂^b`. Every non-identity pattern ends up equally likely, because
   GL(b,2) acts transitively on non-zero vectors. A Fourier sum gives
   `P(net = I) = 2^-b (1 + m e^{-λ(m+1)/m})`, so the rate
   **λ = −(m/(m+1)) ln(1 − (m+1)p/m)** reproduces the channel exactly. Cells (group × shot) are
   independent because a Poisson process has independent counts on disjoint cells.
   - The test `hit_model_reproduces_every_group_distribution_exactly` sums the series over the hit
     count to 200 terms. That is independent of the closed form. It matches every outcome
     probability of `VarDist::outcomes()` to 1e-14 for p from 10⁻⁶ to 0.25.
   - Groups with p > 0.25, and coins, take a dense per-shot path.
2. **One stream per (kind, p) class, Pauli folded into the location.** All groups with the same
   `(kind, p)` form a class, however they are interleaved in the circuit. For a batch of S shots the
   class's *slots* are `(group, pattern, shot)` triples, `G·m·S` of them, each with hit rate `λ/m`.
   One uniform integer picks the location and the Pauli together: no `ln`, no separate Pauli draw.
   This is the "pool all equal-p locations into one stream" and "depolarizing as one draw" idea.
3. **Precomputed hit table.** For every `(group, pattern)` the rows it flips are precomputed: the
   XOR of the pattern's variable columns, padded to the class's maximum width K with a scratch "sink"
   row. Stim's circuit needs K ≤ 4, and the table uses `u16` rows when the row count fits. A hit is
   then `out[shot_word·(rows+1) + table[t·K + i]] ^= bit` for `i < K`, branch-free. No variable
   buffer is cleared or scanned, and there is no CSC indirection. At d = 15 the table is 1.7 MB.
4. **Blocked generation.** The slots are cut into blocks of `2^k` (chosen so that a block averages
   8 hits). Each block takes:
   - an independent `Poisson(μ_b)` count, from a precomputed inverse-CDF table with a 256-entry guide
     table, so one random word per block;
   - its hits as `block_start + (random >> (64 − k))`, which is exact because the block size is a
     power of two.

   Table accesses therefore walk forward through memory instead of jumping across 1.7 MB. The batch
   is 16 × 64 = 1024 shots: larger batches amortise the table walk, and 16–32 words was the measured
   optimum (§3.3).

The RNG is **wyrand** (one 64×64→128 multiply per word, 64-bit state; period 2⁶⁴ ≫ the ~10¹¹ words of
the largest run here). Xoshiro256++ (`rand::rngs::SmallRng`) is supported too: it passes the same
equivalence tests and is 0.90–0.98× as fast (§3.2).

## 2. Correctness

All on the VPS. Raw output is in `research/data/fast-sampler/*.jsonl`, scripts are alongside.

**Exact checks.**
- *Hit algebra:* the series test above.
- *Poisson inverse-CDF table:* the pmf is reproduced to 1e-15. The guide table gives the same answer
  as a plain scan on 200k random and boundary inputs (`poisson_table_matches_pmf_and_scan`).
- *Support (`support_fast.py` → `support_fast.jsonl`):* the distinct non-empty row sets of the hit
  tables equal the error-target sets of **Stim's DEM**. This holds for both directions (A: qsim-lab's
  circuit exported to Stim; B: Stim's generated circuit parsed by us), d = 3, 7, 11, 15,
  p = 0.1% and 0.3%: **16/16 equal** (up to 66,087 signatures at d = 15).

**Statistical checks in `cargo test`** (deterministic seeds):
- Exact outcome distributions of 60 small random noisy circuits pass a χ² test. The circuits mix
  rare and dense groups, coins, gate/measurement/reset noise and batch widths 1–16, and cycle through
  all four code paths and both RNGs. The test uses Wilson–Hilferty z < 4.5 (Bonferroni); the worst
  z was 3.3.
- A circuit with one channel at +25% is rejected at z > 10.
- Surface-code d = 5, p = 0.5%, 2²⁰ shots: every detector marginal and every pair rate of all four
  code paths against the old sampler passes (10,804 tests, max |z| 4.35 against z* 5.40).
- The Poisson sampler (moments and pmf χ²) and the Lemire uniform pass as well.

**Stim equivalence, 10⁶ shots per side** (`equivalence_fast.py`). This reuses
`research/data/qec-r4/stim_equivalence.py` unchanged, with our side switched to `sample-fast`:
- T0: DEM support;
- T1: per-detector and observable rates;
- T2: every DEM-correlated detector pair;
- T3: mean and variance of events per shot.

Each cell is tested at 1% family-wise error.

| direction | d | p | RNG | detectors | DEM support equal | max abs z marg | max abs z pairs (#) | z events mean, var | obs rate ours / Stim | z* | rejections | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: ours→Stim | 3 | 0.3% | wyrand | 16 | yes (55) | 2.36 | 2.85 (39) | -1.08, -0.44 | 0.0408 / 0.0409 | 3.76 | 0/58 | PASS |
| A: ours→Stim | 7 | 0.3% | wyrand | 192 | yes (883) | 2.66 | 3.22 (819) | -1.20, +1.38 | 0.1868 / 0.1870 | 4.42 | 0/1014 | PASS |
| A: ours→Stim | 11 | 0.3% | wyrand | 720 | yes (3631) | 2.93 | 3.46 (3487) | -1.01, +0.22 | 0.3425 / 0.3434 | 4.72 | 0/4210 | PASS |
| A: ours→Stim | 15 | 0.3% | wyrand | 1792 | yes (9451) | 3.50 | 4.00 (9195) | +0.74, +0.89 | 0.4414 / 0.4417 | 4.91 | 0/10990 | PASS |
| B: Stim→ours | 3 | 0.3% | wyrand | 24 | yes (219) | 1.79 | 2.72 (94) | +0.73, +0.34 | 0.0653 / 0.0650 | 3.94 | 0/121 | PASS |
| B: Stim→ours | 7 | 0.3% | wyrand | 336 | yes (5471) | 2.86 | 4.30 (2638) | +0.37, +0.79 | 0.2538 / 0.2538 | 4.65 | 0/2977 | PASS |
| B: Stim→ours | 11 | 0.3% | wyrand | 1320 | yes (24483) | 3.45 | 4.32 (12158) | -0.50, +0.91 | 0.4100 / 0.4101 | 4.95 | 0/13481 | PASS |
| B: Stim→ours | 15 | 0.3% | wyrand | 3360 | yes (66087) | 4.74 | 4.67 (33262) | -0.89, +0.37 | 0.4788 / 0.4788 | 5.14 | 0/36625 | PASS |
| A: ours→Stim | 3 | 0.1% | wyrand | 16 | yes (55) | 1.55 | 2.50 (39) | +0.18, -0.54 | 0.0141 / 0.0139 | 3.76 | 0/58 | PASS |
| A: ours→Stim | 7 | 0.1% | wyrand | 192 | yes (883) | 3.11 | 3.08 (819) | -1.39, -1.94 | 0.0717 / 0.0728 | 4.42 | 0/1014 | PASS |
| A: ours→Stim | 11 | 0.1% | wyrand | 720 | yes (3631) | 2.88 | 3.48 (3487) | -0.19, +1.24 | 0.1599 / 0.1603 | 4.72 | 0/4210 | PASS |
| A: ours→Stim | 15 | 0.1% | wyrand | 1792 | yes (9451) | 3.78 | 3.91 (9195) | +0.41, +1.44 | 0.2565 / 0.2568 | 4.91 | 0/10990 | PASS |
| B: Stim→ours | 3 | 0.1% | wyrand | 24 | yes (219) | 2.01 | 2.59 (94) | -0.49, -0.28 | 0.0226 / 0.0227 | 3.94 | 0/121 | PASS |
| B: Stim→ours | 7 | 0.1% | wyrand | 336 | yes (5471) | 2.85 | 3.15 (2638) | -0.20, -0.16 | 0.1049 / 0.1055 | 4.65 | 0/2977 | PASS |
| B: Stim→ours | 11 | 0.1% | wyrand | 1320 | yes (24483) | 3.63 | 3.86 (12158) | -0.97, +0.07 | 0.2167 / 0.2175 | 4.95 | 0/13481 | PASS |
| B: Stim→ours | 15 | 0.1% | wyrand | 3360 | yes (66087) | 3.78 | 4.39 (33262) | +1.61, -1.20 | 0.3243 / 0.3246 | 5.14 | 0/36625 | PASS |
| B: Stim→ours | 3 | 0.3% | Xoshiro256++ | 24 | yes (219) | 1.99 | 2.91 (94) | +0.16, +1.55 | 0.0650 / 0.0650 | 3.94 | 0/121 | PASS |
| B: Stim→ours | 7 | 0.3% | Xoshiro256++ | 336 | yes (5471) | 2.93 | 3.62 (2638) | +1.35, +0.11 | 0.2544 / 0.2538 | 4.65 | 0/2977 | PASS |
| B: Stim→ours | 11 | 0.3% | Xoshiro256++ | 1320 | yes (24483) | 3.35 | 4.37 (12158) | -1.35, +0.23 | 0.4104 / 0.4101 | 4.95 | 0/13481 | PASS |

**0 rejections in 155,531 tests.**

**Negative control** (`negative_control_fast.py`: Stim's d = 7 circuit, our side sampling a
deliberately wrong copy with `sample-fast`):

| perturbation | max abs z | rejections |
|---|---|---|
| every `DEPOLARIZE2(0.003)` → `0.0033` (+10% on one channel) | 17.4 | 1449 |
| reset `X_ERROR`s removed | 61.1 | 1693 |

The test has power at the 10% level, so the new path is not "close"; it is the same distribution.
The new path is not bit-identical to the old one, since it consumes randomness differently, so a
bit-identity assertion does not apply. The old sparse path's bit-identity test with the old dense
path still passes.

## 3. Speed

### 3.1 x86 (VPS, AMD EPYC-Rome, AVX2), Stim's circuit

`timing.py` / `run_vps_timing.sh`. The circuit is `stim.Circuit.generated("surface_code:rotated_memory_z", d, rounds = d)`
with all four noise knobs = p, written once to a `.stim` file that every contender reads. All runs are
single-threaded, write ptb64 with observables appended to `/dev/null`, are interleaved (the order
alternates each rep) and report the min of 3. Mshot/s:

- **Stim pip write:** `compile_detector_sampler().sample_write(..., "ptb64")`. pip's wheel is the
  SSE2 build.
- **Stim native:** Stim 1.16.0 built from source with `-DSIMD_WIDTH=256` (AVX2; 5,816 ymm
  instructions in the binary), `stim detect --out_format ptb64 --append_observables`. The wall time
  **minus the wall time of the same command at 64 shots** (start-up, parse and compile removed), which
  favours Stim.
- **ours:** `stim_compare bench-fast` internal timer: FastSampler sampling plus ptb64 packing and
  writing, with compile excluded (as for Stim pip).
- **end-to-end:** process wall time of `stim detect` against `stim_compare sample-fast`, both
  including parse and compile.

| d | p | shots | Stim pip write | Stim native (AVX2) | qsim-lab FastSampler | ours / native Stim | end-to-end ours / Stim | load |
|---|---|---|---|---|---|---|---|---|
| 3 | 0.1% | 4,000,000 | 41.29 | 38.49 | 696.0 | **18.1×** | 10.9× | 0.8 |
| 7 | 0.1% | 1,000,000 | 4.43 | 4.98 | 55.13 | **11.1×** | 8.4× | 1.1 |
| 11 | 0.1% | 256,000 | 1.13 | 1.27 | 11.77 | **9.3×** | 4.8× | 1.6 |
| 15 | 0.1% | 128,000 | 0.46 | 0.48 | 4.41 | **9.2×** | 2.5× | 1.8 |
| 3 | 0.3% | 4,000,000 | 25.87 | 26.54 | 317.5 | **12.0×** | 10.4× | 2.1 |
| 7 | 0.3% | 1,000,000 | 2.25 | 2.64 | 24.49 | **9.3×** | 7.9× | 2.5 |
| 11 | 0.3% | 256,000 | 0.57 | 0.65 | 6.08 | **9.4×** | 6.0× | 2.6 |
| 15 | 0.3% | 128,000 | 0.23 | 0.27 | 2.24 | **8.3×** | 3.7× | 3.0 |

(`timing_vps_epyc_run1.jsonl`. The 1-minute load is recorded per row and was ≤ 3.0 throughout. In
every cell the better of the two Stims is the native one, except d = 3, p = 0.1%, where pip is 7%
faster: against pip the ratio there is 16.9×.)

**Stim's DEM sampler is not faster.** We checked `circuit.detector_error_model().compile_sampler()`,
the same "sample error mechanisms, XOR into detectors" representation, with DEM extraction excluded.
pip, VPS, load 3.9: 1.06 Mshot/s at d = 7, p = 0.3%; 0.09 / 0.11 Mshot/s at d = 15, p = 0.3% / 0.1%.
That is slower than Stim's circuit sampler, because it does work per error mechanism per batch. These
x86 DEM numbers are a one-off check. `timing.py` now times the DEM sampler in every cell, but a second
x86 grid was not run: VPS load stayed at 6–11 for the rest of the session. The Mac grid (§3.4) has it
in every cell, at 0.06–0.09 Mshot/s for d = 15.

**Where the end-to-end lead goes.** Our compile at d = 15 is 69–80 ms in run 1. That is the
symbolic SymPhase frame (about 42 ms), the parity rows and the hit table. A flatter table build since
then brings it to about 55 ms, with the FastSampler build at 12 ms. Stim's entire 64-shot process
(start-up, parse, compile, sample) takes 3.5 ms. At 128k shots our sampling takes only 29–57 ms, so
compile dominates. For long runs (≥ 10⁶ shots, the usual case for LER estimation) the end-to-end
ratio converges to the sampling ratio.

For scale: Stim's own `circuit.detector_error_model()`, the comparable precomputation, takes 61–63 ms
at d = 15 on the same machine.

### 3.2 Where the speed comes from (ablation, same runs)

Mshot/s. Each column adds one change to the one before. "old" is the SymPhase path before this
branch; "hits, columns" is the Poisson-hit model (steps 1–2) evaluated through the CSC columns.

| d | p | old dense, StdRng | old sparse, StdRng | old sparse, Xoshiro | hits, columns, Xoshiro | + hit table | + blocked | + u16 table | + wyrand | total |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.1% | 42.80 | 48.89 | 54.18 | 241.02 | 587.46 | 643.71 | 657.57 | 696.02 | 16.3× |
| 7 | 0.1% | 3.71 | 5.25 | 5.81 | 18.51 | 37.03 | 48.74 | 51.07 | 55.13 | 14.9× |
| 11 | 0.1% | 0.99 | 1.51 | 1.53 | 3.99 | 7.58 | 10.67 | 10.96 | 11.77 | 11.9× |
| 15 | 0.1% | 0.35 | 0.60 | 0.62 | 1.21 | 2.40 | 3.51 | 3.99 | 4.41 | 12.6× |
| 3 | 0.3% | 25.77 | 25.86 | 28.93 | 89.91 | 277.10 | 311.58 | 305.81 | 317.46 | 12.3× |
| 7 | 0.3% | 2.25 | 2.38 | 2.71 | 6.97 | 14.78 | 22.44 | 22.89 | 24.49 | 10.9× |
| 11 | 0.3% | 0.59 | 0.65 | 0.73 | 1.47 | 3.09 | 5.62 | 5.98 | 6.08 | 10.2× |
| 15 | 0.3% | 0.22 | 0.25 | 0.28 | 0.46 | 0.93 | 2.03 | 2.20 | 2.24 | 10.3× |

Individual contributions at d = 11–15 (ratios of adjacent columns):

| technique | gain |
|---|---|
| Poisson hits: one uniform word picks location and Pauli, no `ln`, no variable buffer | **1.6–2.6×** over the old sparse path with the same RNG |
| precomputed (group, Pauli) → detector hit table, branch-free | **2.0×** |
| blocked generation (per-block Poisson count from an inverse-CDF table, forward-walking table access) | **1.4–2.2×**, more at higher p |
| `u16` table rows | 1.03–1.14× |
| wyrand vs Xoshiro256++ | 1.02–1.10× |
| ChaCha12 → Xoshiro256++ on the old path | 1.0–1.12× |

The relative order is the same at d = 3 and 7. The gains multiply to 10–16× over the old dense path.

### 3.3 Ideas that did not win (measured)

- **Pooled geometric gaps** (`SymPhaseSampler::pool_equal_dists`). Sort the groups so all
  equal-distribution groups form one geometric-skip stream: at d = 15 this cuts the stream starts per
  64 shots from 61 to 3. Old sparse path + Xoshiro, d = 15, load ~7: 0.204 → 0.199 s at p = 0.1%,
  0.460 → 0.440 s at p = 0.3%, a **2–4% gain**. The cost was never the run starts but the per-fault
  `ln` plus Pauli draw plus scatter, which the hit model removes. The FastSampler does pool by class
  (step 2), so the idea survives in that form.
- **Bit-sliced Bernoulli** (`stim_compare probe-bitsliced`): one exact Bernoulli(p) word per group per
  64 shots, by lazy bit-plane comparison of random words against the binary expansion of p (exact,
  unlike a k/2^m truncation).
  - It needs **7.3 random words per group-word** at both p = 0.1% and 0.3%: the expansion is not
    short, and log₂64 ≈ 6 planes are needed before all 64 lanes are decided.
  - At d = 15 that alone (no Pauli choice, no evaluation, no output) takes 1.74 s per 128k shots,
    **about 30–60× slower than the entire FastSampler** (0.029 s / 0.057 s at p = 0.1% / 0.3%).
  - At p ≤ 1% the expected faults per group-word are 0.064–0.19. Any method that touches every
    group-word loses to one that touches only the faults. This is also why Stim's DEM sampler is
    slow here.
- **Batch width.** At d = 15, p = 0.3% (load 3–4), blocked path, wyrand, 200k shots: 0.131 / 0.110 /
  0.104 / 0.111 s at W = 8 / 16 / 32 / 64 words. The unblocked path *loses* with large W (cache) and
  is best at W = 4. We use W = 16.
- **Pooling across kinds.** Folding flips, depol1 and depol2 of equal p into one stream would need a
  weighted (alias) pick. There are only 3 classes per batch of 1024 shots, so the per-class overhead
  is negligible and this was not pursued.

### 3.4 Apple M1 Pro

`mac_timing.sh`. It holds the swarm bench lock per (d, p) cell, waits up to 120 s for load < 3 and
records the load. Only pip Stim is used: **Stim has no NEON backend, so on arm64 both the wheel and a
native build use 64-bit words** (see `qec-r4.md` §1.5, where a native `-mcpu=native` build measured
the same as pip). The Mac ratios therefore overstate what the algorithm buys on its own.

`timing_mac_m1.jsonl`. The load is the 1-minute load at the end of each cell: 2.9–6.7, because other agents'
jobs were running. The bench lock guarantees no other *timing* run overlapped, but not an idle
machine. Mshot/s:

| d | p | Stim pip write | Stim DEM sampler pip | qsim-lab FastSampler | ours / pip Stim | ours / best Stim | end-to-end ratio | load |
|---|---|---|---|---|---|---|---|---|
| 3 | 0.1% | 26.05 | 20.05 | 1035.20 | **39.7×** | 39.7× | – | 6.7 |
| 7 | 0.1% | 2.28 | 0.95 | 62.27 | **27.3×** | 27.3× | – | 5.7 |
| 11 | 0.1% | 0.61 | 0.23 | 16.46 | **26.9×** | 26.9× | – | 3.2 |
| 15 | 0.1% | 0.24 | 0.09 | 6.52 | **27.0×** | 27.0× | – | 3.0 |
| 3 | 0.3% | 13.97 | 12.77 | 485.55 | **34.7×** | 34.7× | – | 2.9 |
| 7 | 0.3% | 1.13 | 0.69 | 32.09 | **28.4×** | 28.4× | – | 4.1 |
| 11 | 0.3% | 0.29 | 0.16 | 8.13 | **28.0×** | 28.0× | – | 3.7 |
| 15 | 0.3% | 0.11 | 0.06 | 3.18 | **27.7×** | 27.7× | – | 5.3 |

- Against pip Stim: **27–40×**. Stim's DEM sampler is again slower than its circuit sampler.
- This is 2–3× the x86 ratio, consistent with Stim using 64-bit words here instead of 256-bit
  AVX2. It is not a fair measure of the algorithm.
- Our own Mac throughput is 1.1–1.5× the VPS's.

Mac ablation (same runs):

| d | p | old dense, StdRng | old sparse, StdRng | old sparse, Xoshiro | hits, columns, Xoshiro | + hit table | + blocked | + u16 table | + wyrand | total |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.1% | 55.52 | 63.92 | 84.35 | 303.49 | 841.22 | 993.79 | 1023.80 | 1035.20 | 18.6× |
| 7 | 0.1% | 4.47 | 6.82 | 8.79 | 24.13 | 58.47 | 64.49 | 63.23 | 62.27 | 13.9× |
| 11 | 0.1% | 1.32 | 1.96 | 2.59 | 6.02 | 13.25 | 15.69 | 15.96 | 16.46 | 12.5× |
| 15 | 0.1% | 0.54 | 0.82 | 1.09 | 2.37 | 4.75 | 6.04 | 6.35 | 6.52 | 12.0× |
| 3 | 0.3% | 33.46 | 31.39 | 41.81 | 111.81 | 393.20 | 480.02 | 477.95 | 485.55 | 14.5× |
| 7 | 0.3% | 2.83 | 3.06 | 4.27 | 9.27 | 27.37 | 31.41 | 32.20 | 32.09 | 11.3× |
| 11 | 0.3% | 0.77 | 0.82 | 1.14 | 2.32 | 6.47 | 8.06 | 8.08 | 8.13 | 10.6× |
| 15 | 0.3% | 0.31 | 0.33 | 0.47 | 0.91 | 2.24 | 3.22 | 3.14 | 3.18 | 10.3× |

The order of the contributions is the same as on x86. On the M1 the hit table gains more (2.0–3.5×)
and the blocked generation less (1.1–1.4×), plausibly because of the M1's larger caches and lower
memory latency (not investigated).

## 4. Limits and caveats

- The lead is per shot on low-noise circuits; it scales with faults per shot, not gates. The hit
  model is exact for any p, but groups with p > 0.25 and coins go through a dense per-shot path that
  was not optimised (not exercised by QEC detector sampling, where every coin is pruned).
- Entries wider than 8 rows (circuits where a single fault flips many outputs, e.g. raw measurement
  sampling of long circuits) fall back to the column path ("hits, columns": still 1.6–2.6× faster
  than the old path).
- Compile is 55 ms at d = 15 against 3.5 ms for Stim. It is reported, not hidden: at 128k shots the
  end-to-end lead is 2.5–3.7× rather than 8–9×. A faster compile would replace the dense symbolic
  frame (2n lines of #variables bits) with sparse or chunked columns; that was not attempted here.
- wyrand has a 64-bit state (period 2⁶⁴). Xoshiro256++ is one argument away (`sample-fast … xo`) at
  0.90–0.98× of the speed, and both pass the equivalence tests.
- Single-threaded throughout. Both samplers parallelise trivially over shots.
- VPS numbers were taken at 1-min load 0.8–3.0 on a shared 4-vCPU machine. The pip Stim runs in the
  same process as the interleaving harness, while native Stim and ours are separate processes.

## 5. Prior art and scope (added by the independent audit, `research/qec/fast-sampler-audit.md`)

- **Not a new algorithm class.** Sampling detection events by drawing sparse faults and XOR-ing
  their precomputed detector sets in O(npd + 1) is described in the Stim paper (Gidney 2021,
  arXiv:2103.02202, §5.6). Gidney left it unimplemented because he expected the constant factors to
  lose at p ≈ 0.1%. The hit model's per-Pauli flip probability `½(1 − (1 − (m+1)p/m)^{2/(m+1)})` is
  exactly Stim's DEM decomposition of DEPOLARIZE1/2 into independent mechanisms (`error_decomp.cc`).
  What is new is the constant-factor engineering:
  - one pooled Poisson stream per (kind, p), with location and Pauli taken from one draw;
  - blocked forward-walking generation;
  - the padded branch-free hit table.

  Together these make the sparse method win by about 9–27× per shot.
- **Scope.** The Stim front-end accepts: H, S, S_DAG, Paulis, CX, CZ, SWAP, R/RX, M/MX/MR/MRX
  with one global flip probability, X/Y/Z_ERROR, DEPOLARIZE1/2, DETECTOR, OBSERVABLE_INCLUDE,
  REPEAT. Everything else is rejected with an error and is never approximated: PAULI_CHANNEL_*,
  E/ELSE_CORRELATED_ERROR, HERALDED_*, MPP, Y-basis operations, feedback and other gates. Stim's
  colour code works after `stim.Circuit.decomposed()`.
- **Fixed by the audit.** Circuits with a detector or observable of noiseless parity 1 (e.g. the
  decomposed colour code at d ≥ 5) used to panic in `stim_compare`. Outputs are now reported
  relative to the noiseless reference, as Stim does (`SymPhaseSampler::relative_to_reference`).
