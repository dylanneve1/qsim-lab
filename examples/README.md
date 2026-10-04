# Examples

Run any of them with `cargo run --release --example <name> [-- args]`. Each
file's header comment documents its arguments. Examples are grouped by
purpose; research drivers and benchmarks belong to the notebook named in the
last column (see [../research/README.md](../research/README.md)).

## Tutorials — start here

| Example | What it shows |
|---|---|
| [bell](bell.rs) | A Bell pair on three backends, sampled |
| [ghz](ghz.rs) | GHZ states at each backend's natural size: 20 qubits (state vector), 10 000 (tableau), 1 000 (MPS) |
| [qft](qft.rs) | QFT of a basis state; QFT followed by its inverse is the identity |
| [grover](grover.rs) | Grover search for one marked item |
| [bernstein_vazirani](bernstein_vazirani.rs) | One-query hidden-string recovery; Clifford, so the tableau runs it at large sizes |
| [clifford_t](clifford_t.rs) | Why one T gate matters: the tableau refuses it, Pauli-path cost doubles per T |
| [shor](shor.rs) | Shor's algorithm on the state vector: factors 15 and 21 |
| [repetition](repetition.rs) | Bit-flip repetition code: logical vs physical error rate |
| [surface_threshold](surface_threshold.rs) | Surface-code threshold sweep, one CSV line per (d, p) — [qec.md](../research/qec/qec.md) |

## Benchmarks

Timing harnesses. Run them on an idle machine (or under the bench lock),
interleaved A/B, and report machine and load ([CONTRIBUTING.md](../CONTRIBUTING.md#timing-discipline)).

| Example | Measures | Notebook |
|---|---|---|
| [sv_speed](sv_speed.rs) | State-vector executors (min of `reps` runs) | [sv.md](../research/performance/sv.md) |
| [kernel_micro](kernel_micro.rs) | Single-thread in-cache 2×2 kernels, AoS vs SoA | [sv.md](../research/performance/sv.md) |
| [metal_bench](metal_bench.rs) | Metal GPU vs CPU blocked executor, f32 (needs `--features metal`, macOS) | [metal.md](../research/performance/metal.md) |
| [ooc_bench](ooc_bench.rs) | Out-of-core state vector, one CSV line per configuration | [ooc.md](../research/performance/ooc.md) |
| [ooc_plan](ooc_plan.rs) | Out-of-core schedulers: passes over the file (plan only, no I/O) | [ooc.md](../research/performance/ooc.md) |
| [pipeline_bench](pipeline_bench.rs) | `pipeline::simulate` vs plain simulation, end to end | [pipeline.md](../research/performance/pipeline.md) |
| [stab_bench](stab_bench.rs) | Stabilizer tableau speed | [stab.md](../research/performance/stab.md) |
| [symphase_bench](symphase_bench.rs) | SymPhase sampler vs shot-by-shot tableau on the surface-code memory | [stab.md](../research/performance/stab.md), [RESULTS.md](../RESULTS.md) |
| [stim_compare](stim_compare.rs) | Identical-circuit comparison with Stim, both directions | [qec-r4.md](../research/qec/qec-r4.md), [fast-sampler.md](../research/qec/fast-sampler.md) |
| [stim_export](stim_export.rs) | Exports the surface-code circuit to .stim; per-detector rates and timing vs Stim | [qec-r4.md](../research/qec/qec-r4.md) |
| [compile_bench](compile_bench.rs) | Compiler passes vs always simulating the original circuit | [compiler.md](../research/compiler/compiler.md) |
| [dag_bench](dag_bench.rs) | DAG IR construction cost and gate counts after each peephole | [dag.md](../research/compiler/dag.md) |
| [phasepoly_bench](phasepoly_bench.rs) | T-count: DAG peephole vs phase folding | [phasepoly.md](../research/compiler/phasepoly.md) |
| [phasepoly_e2e](phasepoly_e2e.rs) | End-to-end wall time with phase folding off vs on | [phasepoly.md](../research/compiler/phasepoly.md) |
| [repeat_bench](repeat_bench.rs) | The repeat pass (`compile::repeat`) | [repeat.md](../research/compiler/repeat.md) |

## Research drivers

Campaign and data-generation binaries; their output lives in `research/data/<study>/`.

| Example | Produces | Notebook |
|---|---|---|
| [ge_shor](ge_shor.rs) | Gidney–Ekerå windowing / Ekerå–Håstad / coset runs | [ge-shor.md](../research/shor/ge-shor.md) |
| [mbu_counts](mbu_counts.rs) | Whole-run gate/Toffoli/measurement counts, measurement-based oracles | [mbu-shor.md](../research/shor/mbu-shor.md) |
| [superopt_counts](superopt_counts.rs) | Oracle gate/Toffoli counts per superoptimisation | [superopt.md](../research/shor/superopt.md) |
| [shor_noise](shor_noise.rs) | Noisy gate-level Shor trajectories at scale | [shor-noise.md](../research/shor/shor-noise.md) |
| [shor_noise_validate](shor_noise_validate.rs) | Statistical validation of the noisy trajectory sampler | [shor-noise.md](../research/shor/shor-noise.md) |
| [shor_precision](shor_precision.rs) | f32 vs f64 amplitudes in the sliced Shor state | [shor.md](../research/shor/shor.md) |
| [shor_support](shor_support.rs) | Reachable support of the semiclassical Shor circuit | [shor.md](../research/shor/shor.md) |
| [shor_seed_orders](shor_seed_orders.rs) | Base and order drawn by each `--seed` | [mbu-shor.md](../research/shor/mbu-shor.md) |
| [theory_coset](theory_coset.rs) | Data for the coset-error theory | [theory-coset.md](../research/theory/theory-coset.md) |
| [schedule_search](schedule_search.rs) | Surface-code CNOT schedule search | [schedules.md](../research/qec/schedules.md) |
| [color_search](color_search.rs) | Colour-code schedule experiments | [qec-r4.md](../research/qec/qec-r4.md), [colour-global.md](../research/qec/colour-global.md) |
| [color_ler](color_ler.rs) | Colour-code memory logical error rate (SymPhase + BP+OSD) | [colour-flags.md](../research/qec/colour-flags.md) |
| [dem_distance](dem_distance.rs) | Exact minimum-weight logical of a DEM on stdin | [colour-global.md](../research/qec/colour-global.md) |
| [simulability](simulability.rs) | Simulability phase-diagram campaign | [simulability.md](../research/simulability/simulability.md) |
| [magic_atlas](magic_atlas.rs) | Magic atlas CLI | [magic-atlas.md](../research/simulability/magic-atlas.md) |
| [magic_transition](magic_transition.rs) | Monitored Clifford+T transition campaign | [magic-transition.md](../research/simulability/magic-transition.md) |
| [transition_theory](transition_theory.rs) | d-only polynomial campaign for the transition theory | [transition-theory.md](../research/theory/transition-theory.md) |
| [planner_v2](planner_v2.rs) | Planner v2 features, plans and end-to-end runs | [planner-v2.md](../research/simulability/planner-v2.md) |

## Audit helpers

Dump intermediate data so independent (Python) checks can verify a claim.

| Example | Notebook |
|---|---|
| [audit_dump_shor_rounds](audit_dump_shor_rounds.rs) | [audit.md](../research/process/audit.md) §16 |
| [audit_dump_states](audit_dump_states.rs) | [audit.md](../research/process/audit.md) §16 |
| [audit_shor_r4](audit_shor_r4.rs) | [shor-r4-audit.md](../research/shor/shor-r4-audit.md) |
