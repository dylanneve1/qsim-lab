# research/data/shor-xl

Data and scripts of [research/shor/shor-xl.md](../../shor/shor-xl.md). Binaries:
`cargo build --release --example ge_shor --bin qsim` (paths below assume `$G` = the `ge_shor`
example, `$Q` = `qsim`). Every timed run held `/dev/shm/qsim/bench.lock` (`flock`), and went through
`timed.sh` (load / free memory before and after, `/usr/bin/time`, 13 GB address-space cap).

* `PREREG.md` — the instance and base rules, committed before the runs.
* `gen_generic.py` — the repo's seeded generic generator (round 4), loop extended to 63 bits;
  `generic_instances.txt` is its output (`bits N p q λ λ_odd ν₂(λ)`).
* `orders.py` — `ord(h)`, `ord(g)` of the base rule's seed-1 bases, from the factors (write-up and
  RAM estimates only); `orders_seed1.txt` (22–63 bits and N_W). `nw_bases.txt`: the base rule's
  `h`, `g`, `y` for N_W, seeds 1–3 (`ge_shor info`, no factors used).
* `timed.sh`, `ab.sh` (interleaved AVX2 / AVX-512 A/B), `frontier.sh` (generic-N runs),
  `oracle_check.py` (independent full-size check of dumped window blocks), `tables.py` (the
  notebook's tables from these logs).
* N_W runs: `nw_B_seed1.log`, `nw_A_seed1.log`, `nw_shorodd_B_seed1.log` (first build),
  `nw_B_seed1_v2.log` (final build); `oracle_check_nw.out`.
* Generic N: `frontier_22_33.log`, `g43_B_seed1.log`, `g43_B_seed1_oomkilled.log` (first attempt,
  killed by the kernel when another job exhausted the machine's memory).
* AVX-512: `slicebench.log` (kernel, one thread), `ab_28.log`, `ab_31_qsim.log`, `ab_31_eh.log`,
  `ab_nw.log` (whole runs), `lanes_31_eh.log` (lane counts).
* `tests_small_n.log` — output of the small-N distribution test (`odd_order_bases_match_textbook`).

Reproduce (each line is one run; the N_W and 43-bit runs need ≈ 2.3 / 3.7 GB):

```sh
$G run 549755813701 1 1 4 lookups eh-odd f32 3      # N_W, config B
$G run 549755813701 1 2 3 lookups eh-odd f32 3      # N_W, config A
$G run 549755813701 1 1 4 lookups shor-odd f32 3    # Shor on g + Miller
$G run 4911456443897 1 1 4 lookups eh-odd f32 3     # 43-bit generator N
./frontier.sh $G 22 23 24 25 26 27 28 29 30 31 32 33
for w in 0 19 20 40 59; do python3 oracle_check.py $G 549755813701 1 1 4 lookups eh-odd $w; done
$G slicebench 549755813701 1 1 4 lookups eh-odd 40 20
QSIM_NO_AVX512=1 $Q run shor --modulus 1537596787 --semiclassical --sliced --window 4 --oracle windowed-mbu-lookup --f32 --seed 2 --tries 1
python3 orders.py $G 1 22 63; python3 tables.py
```
