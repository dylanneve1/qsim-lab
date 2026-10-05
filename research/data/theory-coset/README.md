# research/data/theory-coset

Driver: `examples/theory_coset.rs` (model shared with `tests/theory/theory_coset.rs` via
`tests/theory_coset_model/mod.rs`). All runs on the VPS, `nice -n 15`, single thread, < 200 MB.
`B=target/release/examples/theory_coset`.

* `tv_21.log`, `tv_small.log` — `$B tv N a 2 2 cmax`: exact TV, Theorem-A bounds (square reference
  at offset 0, best square offset, majority reference), unfaithful fraction, per-window ge-shor
  "deviant" weight, first-unfaithful-lookup histogram, temporarily wrapped accumulators.
* `scan_windows.log` — `$B scan N a we wm c tmax`: TV and bounds vs number of windows (t), and the
  last five lines vs w_m (K) at 6 windows.
* `zero_we2.log` (N ≤ 63, c ≤ 3), `zero_we1.log` (N ≤ 45, c ≤ 2) — `$B zero Nmax cmax we wm`:
  TV for every base; summary line counts (r power of two, TV = 0 at all c).
* `pow2_we2.log`, `pow2_we1.log` — `$B pow2 5 129 3 we 2`: power-of-two-order bases only, TV and
  off-diagonal κ mass via Theorem B (asserts Φ_E = Φ_{E mod r}).
* `mc31_eh_a{3,5}_c{8,10,11,12,13,14,16}.log` (400 exponents × 1000 branches), `mc31_c8.log`,
  `mc31_eh_a{3,5}_c12_big.log` (2000 × 2000), `mc31_shor_a3_c12.log` —
  `$B mc 1537596787 eh|shor a 2 3 c E branches seed`: Theorem-A bound on the exact 31-bit schedule by
  Monte Carlo over branch walks; per-window table (square offset chosen on an independent pilot,
  δ̄·2^c with standard error, δ_rms·2^c, unfaithful·2^c, mean and variance of the coset-index
  displacement of faithful branches, ge-shor deviant weight).
* `fit_growth.py` — fits δ̄·2^c(k) (linear vs √k) and the per-window index variance from the
  31-bit logs (`python3 fit_growth.py`).
* `counts31.txt` — `examples/ge_shor counts 1537596787 2 we wm all eh coset c` for c = 8…16 and
  six (w_e, w_m), plus the exact EH baselines.
