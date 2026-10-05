# Code discovery: exhaustive search of weight-6 two-block (BB / GB / coprime-BB) codes, n ≤ 300

Branch `exp/code-discovery`. Code: `src/qec/bicycle.rs` (codes, k, exact distance), `src/qec/bb_search.rs`
(enumeration up to equivalence), `src/qec/bb_circuit.rs` (depth-7 syndrome circuits, schedule validity),
`examples/bb_codes.rs` (CLI: `params`, `search`, `schedules`, `schedsearch`, `cdist`, `ler`).
Tests: `tests/bicycle_codes.rs` + unit tests in the three modules. Data: `research/data/code-discovery/`.
All runs on the Mac (M1 Pro), ≤ 2 workers, no GPU.

__RESULTS__
