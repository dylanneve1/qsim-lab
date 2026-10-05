# Pre-registration for exp/shor-xl (written 5 Oct 2026, 11:45 IST, before any run on these instances)

The commit that adds this file is its timestamp. Everything below is fixed before the runs.

## A. N_W = 549 755 813 701 (Willsch et al. 2023, arXiv:2308.05047)

* **Circuit**: Ekerå–Håstad (EH) with an odd-order base, gate level (X / CNOT / Toffoli plus
  measurement-based uncomputation), `examples/ge_shor.rs`:
  `ge_shor run 549755813701 <seed> <w_e> <w_m> lookups eh-odd f32 [runs]`.
  Two circuits, both run:
  * config **A** = `(w_e, w_m) = (2, 3)`: the configuration of the repo's 31-bit EH run (166 qubits);
  * config **B** = `(1, 4)`: one exponent qubit per window (165 qubits), half the branch memory.
* **Base rule** (uses nothing but N and the seed): `h` = first draw of
  `StdRng::seed_from_u64(seed).random_range(2..N−1)`, `g = h^(2^39) mod N` (39 = bits of N − 1).
  The measurement outcomes are drawn from the same RNG stream after the base draw.
  Seeds are tried in the order 1, 2, 3, … until a run factors N; every run is reported.
  Seed 1: h = 535 596 708 274, g = 345 241 646 758 (`nw_bases.txt`).
* **Memory safety (the only use of the published factors before the run; it selects nothing):**
  every seed's g has odd order dividing the odd part of λ(N_W) = lcm(712 320, 771 780) =
  2^7·3·5·7·19·53·677, i.e. dividing 71 582 595, so the window holds at most
  2^{w_e}·71 582 595 branches for *every* seed. Predicted peak RSS ≈ 20 B per branch + 0.2 GB:
  A ≈ 5.9 GB, B ≈ 3.1 GB. Seed 1: ord(g) = 71 582 595 (`orders_seed1.txt`, last line).
* **Shor's order finding on the same g** (`shor-odd`, 78 exponent bits, config B) followed by
  Miller's reduction from the multiple 2^39·ord(g) of ord(h) (`shor::ge::factor_from_power_order`):
  run with seed 1.
* **Plain Shor with the random base h itself**: not run if the predicted peak exceeds the budget
  (seed 1: ord(h) = λ = 9 162 572 160, ν₂ = 7; peak ≥ ord(h) branches in the last window).

## B. Generic N: frontier and beyond 39 bits

* **Instances**: `gen_generic.py` = the repo's generator (`research/data/shor_r4/gen_instances.py`,
  `random.seed(1)`, first balanced semiprime per bit size) with the loop extended to 63 bits
  (22–32-bit N unchanged). List: `generic_instances.txt`.
* **Base**: seed 1 only (base rule as in A), EH with the odd-order base, config B, f32.
* **Rule**: an instance is run iff its predicted peak RSS (2·ord(g)·20 B + 0.2 GB, config B) is
  ≤ 10 GB. ord(g) is computed from the factors before the run (`orders.py`, output
  `orders_seed1.txt`) as a feasibility filter only; the simulation never reads it. The instances
  that do not fit are reported with the memory they would need. Note: these seed-1 orders for
  22–63 bits were computed (and seed 2 for 30/33/36/39 bits looked at) before this rule was
  written down; no other seed was examined for N above 39 bits.
  Under this rule the instances above 39 bits that fit are exactly one: 43 bits,
  N = 4 911 456 443 897, whose seed-1 base has ord(g) = 115 574 445 = λ_odd/83 (a property of
  this base, not of N: a base with the full odd order would need ≈ 83× the memory).
* **Retries**: if a run's classical post-processing fails, the same base is re-run with the
  measurement stream continuing (`runs` argument), up to 3 runs; all runs reported.
* **Memory rules** (MACHINE.md): runs predicted above 4 GB hold `/dev/shm/qsim/bench.lock`,
  check `free -g` first (≥ 6 GB must stay available to others) and run under
  `prlimit --as` with a 12 GB cap plus `QSIM_GE_MAX_GB`.
