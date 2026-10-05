Dense-fusion A/B on the Apple M1 Pro (4 Oct 2026), `examples/l1_bench.rs`,
one locked chunk per workload under `/tmp/qsim-mac-bench.lock`, configs
round-robin per repetition, min of 3 (5 at n = 20). The `## ...` lines carry
the UTC time, thread count and the load average at start and end of each
chunk (11-24 on 8 cores: the Mac was shared).

- `fusion_ab.*`  — first run, branch 7d83d46 (no cost rule): off / k=2 / k=3. Stopped after 3 chunks to add the cost rule.
- `fusion_ab2.*` — c5de967: `off`, `k2raw`/`k3raw` (`dmin=2`: every group of >= 2 gates), `k2`/`k3` (rule v1: >= 2^k gates of any kind).
- `fusion_ab3.*` — 7329bd3, final rule (>= 2^k uncontrolled 1-qubit gates with a dense 2x2): `off`, `k2`, `off2` (duplicate of `off`: in-chunk noise floor), `k3`.
- `bench*.sh` — the drivers.
