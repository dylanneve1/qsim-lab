# Archived branches

Branches that are no longer on GitHub. Their code is kept in the git bundle
`qsim-lab-stale-branches-2026-10-04.bundle` (heads `refs/heads/archive/<branch>`;
copies on the storage box, on Dylan's Mac in `~/qsim-archive`, and as local
`archive/*` branches in the VPS clone `/tmp/qsim-lab` and the Mac clone
`~/qsim-lab`). Restore one with
`git fetch qsim-lab-stale-branches-2026-10-04.bundle refs/heads/archive/exp/zx:exp/zx`.

The 48 other remote branches deleted on 4 Oct 2026 were fully merged into
main (every commit reachable from main), so they need no archive.

Integration pass of 4 Oct 2026 (integration agent `qsim-integrator`): every
branch below was checked hunk by hunk against main by content (each added
line of the branch diff looked up in main's version of the file), then
merged, ported or dropped.

| branch | head | what it was | fate | why | surviving work on main |
|---|---|---|---|---|---|
| `exp/repeat` | 30bab74 | repeated-block detection and exact fast paths (Clifford power, diagonal fold, 2^k unitary power, plan reuse) | superseded | its audited rebase `exp/repeat-r4` (6e1f9d0) was merged in round 4; all 3159 added lines are on main except 9 that main has in evolved form (extra arguments added later: `planner_debug`, `reuse_max_qubits`) | `src/compile/repeat/`, `tests/repeat.rs`, `tests/audit_repeat.rs`, `research/repeat.md`; audit: `research/audit.md` §15 |
| `exp/phasepoly` | a98e012 | phase-folding pass (`compile::phasefold`) + PyZX comparison data | superseded | its rebase `exp/phasepoly-r4` (a22570f) was merged; all 61 695 added lines are on main verbatim. PR #11 (from this branch) closed | `src/compile/phasefold.rs`, `tests/audit_phasefold.rs`, `research/phasepoly.md`; audit §15 |
| `exp/compiler-pre-rebase` | 7b79fd7 | first compile module + WIP state-propagation pass and probes (pre-rebase copy of `exp/compiler`) | superseded | main's `src/compile/` is a strict superset: every function is present (`op_qubits` moved from `peephole.rs` to `compile/mod.rs`); `stateprop.rs` on main additionally handles resets, classical control, noise ops and iSWAP; `examples/compile_probe.rs` is on main. The 95 "missing" lines are the old API (`compile_sampling(c, opts)` etc.) | `src/compile/{analysis,peephole,plan,stabsv,stateprop}.rs`, `examples/compile_probe.rs`, `tests/compile.rs`; audit §7 |
| `exp/zx` | ed45cc6 | ZX phase-fusion pass and graph-like ZX draft | **dropped (broken)** | 281 of 300 random Clifford+T circuits come out with a different unitary; decomposes each CCX into 7 T and fuses none; 139 commits behind with the first-week `Op` enum. No code merged | the verdict and the full-unitary harness description: `research/audit.md` §15 "exp/zx — verdict DROP"; the idea (PyZX `teleport_reduce`) is listed there as the next step after phase folding |
| `exp/l1-tiling` | 2cebf9c | nested L1 tiling inside blocked-executor blocks, NEON FMA kernels, Mac sweeps | superseded | reviewed rebase `exp/l1-tiling-r4` | see next row |
| `exp/l1-tiling-r4` | 6153e88 | reviewed rebase of `exp/l1-tiling` + audit differential test + M1 Pro A/B | **merged behind a flag** (4 Oct) | negative result: 1.04-1.06x on brickwork over the best untiled block, 0.96-1.01x on QFT on the M1; +7-11% brickwork, -3 to -15% QFT on the VPS. Kept on main as `BlockConfig::l1_tile_bytes` (default 0 = off) so the code and the negative result live together; the FMA part had already landed via `exp/neon-fma-r4` | `src/blocked.rs` (`TilePlan`, `tile_segments`, `apply_tile_run`, `tile_stats`), `tests/l1_tiling.rs` (incl. `tiled_matches_audit_reference`), `examples/l1_bench.rs`, `examples/l1_micro.rs`, `research/data/l1/`, write-up `research/mac-m1.md` |
| `exp/sv-monomial` | 7be7697 | prior-art analysis + WIP dense k-qubit fusion (`src/dense_fusion.rs`) | **ported and merged** (`BlockConfig::dense_fusion`; default k=2 on aarch64, off on x86_64) | unreviewed WIP at agent timeout; reviewed, restricted to cached qubits, given a cost rule (fusing brickwork's `U1 U1 CNOT` is a 0.66-0.98x loss, permutation chains 0.5x), differential-tested on x86_64 and aarch64, measured: 1.5-2.15x on circuits of generic 2-qubit unitaries on the M1 Pro, bit-identical elsewhere; `research/dense-fusion.md` | `src/dense_fusion.rs`, `src/dense_kernels.rs`, `BlockConfig::dense_fusion`, `tests/dense_fusion.rs`, `research/dense-fusion.md`, `research/sv-monomial.md` |
| `wip/fusion-avx2` | 327b346 | `exp/sv-monomial` + lane-exchange dense kernels toward AVX2+FMA dispatch ("may not build") | **ported and merged** (with the row above) | the kernels (`dense_kernels.rs`) were self-contained and are used as is, now monomorphised for the existing FMA/AVX2 dispatch; the other parts of this branch (a `tile_u1` lane-exchange 1q kernel, `KOp::Dense` in the IR, `sv_micro`) were not ported: the 1q kernel was superseded by main's `exp/simd` kernels, and fusion runs per stage without changing `KOp` | as above |
| `exp/sv-wip` | 689c799 | Grover workload in executor IR; wiring the blocked executor into `Circuit::run` | **partly merged** | the `Circuit::run` wiring was already on main (`Simulator::apply_gates`, batches of gates go through the blocked executor); the Grover workload was ported | `algorithms::{grover_state, grover_kops}`, `sv_speed grover`, `tests/blocked.rs::grover_matches_gate_by_gate_and_closed_form` |
