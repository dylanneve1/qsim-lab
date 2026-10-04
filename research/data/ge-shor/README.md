# research/data/ge-shor

* `counts_all.txt` — `ge_shor counts` for 544 configurations (VPS).
* `cost_model.py`, `cost_model.out` — GE19 cost model reconstruction vs ours.
* `coset_exact.log` — `ge_shor coset` exact whole-distribution TV / success (Mac, 2 threads).
* `coset_paths.log` — `ge_shor cosetmc` path sampling with exact likelihoods (VPS).
* `eh_mc.log` — `ge_shor ehmc` Monte-Carlo success rates (VPS; two batches with different seeds, combined in the note).
* Timings (Mac, bench lock): `bench_31_campaign1.log`, `bench_31_campaign2.log` (31-bit Shor A/B),
  `bench_24.log` (24-bit A/B), `bench_24eh_28_31eh.log` (24-bit EH, 28-bit A/B, 31-bit EH),
  driver scripts `bench.sh`, `drive*.sh`.
  Caveat: in `bench_24.log` the `eh` / `eh-odd` configurations ran an older build (A-register first;
  `eh-odd` not yet implemented, so it ran Shor); they are superseded by `bench_24eh_28_31eh.log`.
  The last 31-bit `eh` (random base) run in `bench_24eh_28_31eh.log` was killed (needs ≈ 16 GB).
* `early_dev_runs.log` — timings of earlier engine versions during development (not used in the note).
