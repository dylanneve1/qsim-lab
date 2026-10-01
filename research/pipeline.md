# Pipeline: one entry point, blocked executor by default (lab notebook)

Branch `exp/pipeline` (from main 7112b5a). Author: Claudius.
Machine: shared 4-vCPU AMD EPYC-Rome VPS, rustc 1.93.1, other agents compiling/benchmarking at the
same time (`bench.sh` load at start of every timed run: 5.7-6.5 on 4 cores), so absolute times and
even ratios move by up to ~2x between runs (compare `bench1.txt` and `bench_final.txt`). Treat the
numbers below as orders of magnitude, not as precise speedups.

## What was built

1. **Blocked executor by default for state vectors** (`src/circuit.rs`, `src/statevector.rs`).
   `Simulator` gains `apply_gates(&[Gate])` (default: one `apply` per gate). `StateVector`
   overrides it: 2+ gates go through `apply_gates_blocked(&BlockConfig::default())`
   (`split_phases` stays `false`). `Circuit::run` / `run_noisy` collect *maximal runs of
   `Op::Gate`* and flush them as one batch before every measurement, reset, classically
   conditioned op and noise op, and at the end. With gate noise (`p_1q`/`p_2q > 0`) each gate is
   applied singly because the noise draws from the RNG after every gate. The RNG consumption
   order is unchanged, so outcome records are identical to the old path for the same seed.
   `plan::prepare_statevector` (all dense component paths) also applies its gates as one batch.
   The old attempt on `exp/sv-wip` (689c799) predates `Reset`/`ClassicControlled`/noise ops and was
   used as a reference only.
2. **`qsim_lab::pipeline::simulate(circuit, &Request, &Budget)`** (`src/pipeline.rs`).
   `Request` = `Samples{shots, seed}` | `Amplitudes(Vec<u128>)` | `Expectation(Vec<usize>)`
   (Z-product). It runs `src/compile` (peephole, SWAP elimination, state propagation, light cone,
   components, classical suffix) and dispatches per component. `Budget{mem_bytes}` lowers the
   dense-register cap (the crate-wide 1 GiB cap still applies). Returns the output and
   `(qubits, gates, engine)` per component. A time budget is not implemented.
   New engine in the compile layer: `Backend::Adaptive` + `AdaptiveRule` (opt-in via
   `PlanOptions::adaptive`, off by default so the existing compile API is unchanged;
   `pipeline::plan_options()` turns it on) and `adaptive::active_dimension(circuit)` (polynomial
   time computation of the compressed-register size `d`).
3. **CLI** (`src/main.rs`, `pipeline::choose_shor_path`): `qsim run shor` picks the gate-level
   3n-qubit dense path only if it fits the 1 GiB cap; otherwise it switches itself to
   `--semiclassical --sparse` and prints a note on stderr. A semiclassical-dense request whose
   register does not fit also becomes sparse. Explicit flags are never overridden when they fit.
4. **Exactness tests** (`tests/pipeline.rs`, 10 tests, proptest), see below.
5. **Benchmark** `examples/pipeline_bench.rs`, raw output in `research/data/pipeline/`.

## Rule-based planner (seed for the learned planner)

Per component, in this order (`compile::plan::choose_backend`, `pipeline.rs` constants):

| rule | engine |
|---|---|
| no gates | idle (stays `|0>`) |
| all gates Clifford | stabilizer tableau (any size) |
| terminal sampling, `needed <= 12`, `T <= 22`, `2^needed * 2^T * ceil(n/64) < 2^n / 4` | Pauli-path marginals |
| unitary, `n >= 12`, active dim `d <= 26` and `d + 1 <= n` | **compressed state** (`adaptive`) |
| otherwise | dense f64 state vector, blocked executor, refused above `Budget::mem_bytes` |

* Expectation (`expectation_z_product`): Clifford or (`T <= 24` and `2^T * ceil(n/64) < 2^n/4`) ->
  Pauli paths; else adaptive under the same `n`/`d` rule; else dense.
* Amplitudes never use the compressed state: it drops global phases (`T = e^{i pi/8} Rz(pi/4)`
  etc.), the pipeline guarantees amplitudes with the phase.
* Mid-circuit-measurement circuits: components are still split (classical control joins reader
  and measured qubits); each component runs shot by shot with `Circuit::run` (blocked between
  non-unitary ops) or a tableau; Clifford prefix is prepared once and cloned per shot.

Thresholds come from `research/data/pipeline/thresholds{1,2}.txt` (random Clifford layers with `t`
T gates, 1000 shots, min of 3, adaptive forced on vs off for the same compile plan):

* Adaptive ties the dense vector at n = 10 (`d` up to 10; ratio 0.7-1.08, noise level), is 1.3-1.7x
  at n = 12, 2-4x at n = 14, 2-11x at n = 16, 16-36x at n = 18 and 50-100x at n = 20 (`t` 4..16).
  It was never slower beyond noise once `d <= n - 1`. That fixed `ADAPTIVE_MIN_QUBITS = 12`,
  `ADAPTIVE_MARGIN = 1`. (The first guess 14/3 was too conservative.)
* Expectation of `<Z0 Z1 Z2>`: adaptive ties the existing Pauli-path/dense choice when `t <= 12`
  (0.6-1.7x, noise) and wins by 11-1700x when `t` is 20-30 (where the Pauli path blows up and the
  dense vector is large).

Known weakness of the rules: they are static. The Pauli-path/adaptive crossover for sampling
(`needed` small) is decided by the formula above, not measured; no engine is *measured* before
being chosen. That is what the learned cost model is for.

## Exactness evidence (all run, all green)

`cargo test` (full suite, dev profile) plus `cargo fmt --check` and
`cargo clippy --all-targets -- -D warnings`. New: `tests/pipeline.rs` (+1 unit test in
`pipeline.rs`):

| test | what is compared | cases |
|---|---|---|
| `simulate_amplitudes_match_reference` | `simulate(Amplitudes)` vs an independent naive reference (`audit_common::RefSv` + textbook matrices for Sx/Sxdg/U/ISwap/I) and vs `apply_circuit`, **all 2^n amplitudes, max |d| < 1e-12, global phase included**; circuits over *every* gate type, n 1..8 | 200 |
| `simulate_amplitudes_structured` | same on 3-block circuits (components, state propagation, Clifford prefix) | 200 |
| `simulate_amplitudes_large` | same vs gate-by-gate vector, n 15..17 | 12 |
| `simulate_plan_distribution_is_exact` | the pipeline's sampling plan's exact outcome distribution vs the branching reference `exact_outcome_distribution` (<1e-10), circuits with **every op type**: measure, reset, classical control, X/Y/Z flips, Depolarize1q/2q, terminal measurements | 200 |
| `simulate_expectation_matches_reference` | `<Z..Z>` vs reference (<1e-10) | 200 |
| `run_blocked_matches_gatewise` | `Circuit::run` (blocked) vs the old gate-by-gate run with the same seed: **identical outcome records and final amplitudes < 1e-12**, every op type, n 1..10 | 200 |
| `run_blocked_matches_gatewise_large` | same at n 15..18 (several blocked stages) | 12 |
| `adaptive_is_chosen_and_exact` | n 14..16 Clifford+4T: engine reported is `Adaptive`; sampled marginals within 5 sigma (60k shots) and `<Z>` products within 1e-10 of the dense vector | 6 trials |
| `budget_is_enforced`, `amplitudes_refuse_non_unitary` | budget error / NotSupported | - |
| `pipeline::tests::shor_path_follows_the_memory_cap` | CLI path choice | - |

Not covered: the *sampling* path through `simulate` is checked statistically (adaptive test) and by
its exact plan distribution, not by a goodness-of-fit test per engine over all circuit families.
f32 is not used anywhere in the pipeline (f64 only).

A pre-existing flake: `tests/dag.rs::prop_light_cone_exact` can fail with a deviation of
1.95e-12 against its 1e-12 tolerance (seed 5042061061179466422, n=3, len=25). I reproduced the
identical failure on unmodified main 7112b5a with that fixed seed, so it is not caused by this
branch (rounding after dividing by a small branch probability); it needs a looser tolerance or a
relative one. Not changed here.

## CLI

```
$ qsim run shor --modulus 16777207          # = 4093 x 4099, was a panic before (per the task brief)
note: the dense register for N=16777207 would exceed the 1024 MiB memory cap; running semiclassical sparse (exact; ...)
a=11554613  qubits=25  ... order=Some(127038)  factor=Some(4099)  peak_amplitudes=254076
16777207 = 4093 x 4099
time 14.591 s  peak RSS 179.1 MiB
$ qsim run shor --modulus 15    # unchanged: gate-level dense path, 12 qubits, 15 = 3 x 5
```
(I did not re-run the old binary to see the panic; the "was a panic" is from the task brief.)

## End-to-end benchmark (`examples/pipeline_bench.rs`, 1000 shots, min of 3, shared VM)

Terminal circuits: `gatewise` = apply every gate (pre-change `Circuit::run` behaviour) then
`sample`; `run` = `Circuit::run` (blocked) then `sample`; `simulate` = `Request::Samples`.
Mid-circuit: 100 shots, time of 20 shots scaled x5.
Final run (`research/data/pipeline/bench_final.txt`, load 5.7):

| family | n | gatewise s | run s | simulate s | engine(s) |
|---|---|---|---|---|---|
| random dense (rx/ry/rz/h + cnot/cz, depth 20) | 20 | 0.975 | 0.250 | 0.187 | SV |
| random dense (depth 12) | 22 | 1.93 | 0.488 | 0.544 | SV |
| QFT after Ry layer | 22 | 0.512 | 0.126 | 0.107 | SV |
| two independent 12-qubit dense blocks | 24 | 5.67 | 1.93 | 0.0027 | SV x2 (components) |
| GHZ | 24 | 0.238 | 0.134 | 0.0017 | tableau |
| Clifford + 4 T | 22 | 1.32 | 0.322 | 0.0010 | adaptive |
| Clifford + 8 T | 22 | 2.19 | 0.602 | 0.0015 | adaptive |
| Clifford + 14 T | 22 | 3.44 | 1.01 | 0.0026 | adaptive |
| mid-circuit measure/reset/classical control, 6 rounds | 16 | 3.94 | 1.48 | 1.48 | SV (shot by shot) |

Reading it honestly:
* The blocked-by-default `Circuit::run` is the generic win: 2.7-4x over gate-by-gate on dense
  circuits and mid-circuit circuits on this box. (An earlier run, `bench1.txt`, measured 1.4x at
  n = 20 and 3.7x at n = 22: the load was equally high, so the n = 20 figure is uncertain. Block
  size is tuned for n > 14.)
* `simulate` on a *dense* circuit is the same engine as `run`: 0.9-1.2x (noise). The pipeline only
  adds value when the circuit has structure: independent components (1000x+ here, because 2x12
  qubits replace one 24-qubit vector), Clifford (tableau), and few-T Clifford+T (adaptive,
  1000x+ at n = 22). These large factors compare against a dense vector the structure never needs;
  they are the same effect that the compile layer and adaptive notebooks already reported, now
  reachable from one call.
* Mid-circuit circuits: no gain from the pipeline over `run`; it just does the same shot loop.
  Re-preparing nothing per shot is the next step (snapshot at the first non-unitary op).

## Recommendations / next steps

* Merge: the `Circuit::run` change (small, exact, tested against the old semantics with identical
  RNG streams), `pipeline::simulate`, and the CLI fallback. Keep `PlanOptions::adaptive` opt-in in
  the compile layer.
* The planner should be fitted on measured timings (`research/data/pipeline/*.txt` is a first
  sample): features n, d, T count, needed; candidate set incl. Pauli path vs adaptive for sampling
  with small `needed` (currently a formula), HSF/MPS (not wired in here).
* `amplitudes` through adaptive would need exact global-phase tracking in `compile_state`
  (T/Phase/Rz decompositions); until then it is excluded.
* Time budgets are not implemented.
