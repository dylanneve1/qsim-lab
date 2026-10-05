# Distributed state vector across two machines (Mac + VPS)

Branch `exp/distributed-sv`. Code: `src/engines/dist.rs` (transport trait,
planner, executor), driver `examples/dist_sv.rs`, tests `tests/dist.rs`. Data
and scripts: `research/data/distributed/`.

## TL;DR

* **It works, and it is exact.** A 2-node MPI-style state vector (ranks of
  `2^L` amplitudes, `G = n - L` global qubits, global-qubit swaps streamed
  over a byte link) runs between Dylan's MacBook Pro and the VPS. Over the
  real WAN path it agrees with the single-node executor to <= 3.4e-17 (f64)
  and <= 1.2e-8 (f32), for every gate type and all ownership layouts tried.
* **Largest exact simulation across both machines: 29 qubits, f32 (4 GiB),
  7/8 of the state on the Mac and 1/8 (512 MiB) on the VPS**, QFT checked
  amplitude-by-amplitude against the analytic result on both nodes
  (max error 6.6e-11 on amplitudes of modulus 4.3e-5, i.e. 1.5e-6 relative). It does *not* extend the frontier: the Mac alone
  holds 29 f32 qubits in RAM, and with the VPS limited to ~2.5 GB the pair
  cannot reach 30 qubits (8 GiB) without the Mac holding >= 5.5 GiB of it.
* **It is bandwidth-bound by a factor of ~100.** The only path is an SSH
  session (the VPS firewall drops other ports and sshd forbids port
  forwarding; the Mac is behind NAT and Tailscale is stopped on it): RTT
  56 ms, Mac->VPS 6.1 MiB/s, VPS->Mac 26 MiB/s. Every swap moves half of the
  VPS's share each way, so the Mac uplink sets the pace. QFT-28 f32: **67.7 s
  distributed vs 0.83 s in RAM on the Mac and 4.5 s out-of-core on the Mac's
  SSD** (15x worse than OOC, 82x worse than RAM). The same code over loopback
  TCP takes 1.73 s, so >97% of the WAN time is the link.
* **Verdict: not worth pursuing on this hardware.** Seen from the Mac, the
  VPS is a remote swap device about 700x slower than its own SSD. The design
  only pays when the link is in the multi-GB/s class (Thunderbolt/25-100 GbE)
  *and* the remote node brings a comparable share of RAM. The transport trait,
  the role-aware planner and the tests are worth keeping (small, exact,
  reusable for a LAN/cluster setting).

## How the machines talk

| path | status |
|---|---|
| Tailscale | VPS is on the tailnet; the Mac's Tailscale is **stopped** (not started: it is Dylan's laptop network config) |
| direct TCP Mac -> VPS:47101 | **dropped** by the VPS firewall (INPUT policy DROP); VPS -> Mac impossible (NAT, 192.168.0.250) |
| SSH local port forward (`ssh -L`) | **refused**: `/etc/ssh/sshd_config.d/99-lockdown.conf` has `AllowTcpForwarding no` (not changed) |
| **SSH session stdin/stdout** | **works**: the Mac's key is authorised for `dylan@77.42.21.5` (`Host claudius` in the Mac's ssh config) |

So the transport is behind a trait, `Link` (split into independent send and
receive halves for full-duplex streaming), with three implementations:
`TcpLink` (TCP, used over loopback), `PipeLink` (any `Read`/`Write` pair: the
Mac spawns `ssh claudius 'systemd-run --user --scope -p MemoryMax=… dist_sv
… --stdio 1'` and talks over the child's stdin/stdout; the VPS side uses its
own fds 0/1) and `ChanLink` (in-process, for tests). Nothing was installed or
reconfigured on either machine.

Link benchmark (`dist_sv link`, 64 MiB per direction, 3 runs, data in
`link.log`):

| path | RTT median | node1->node0 | node0->node1 | full duplex, each way |
|---|---|---|---|---|
| Mac <-> VPS, SSH session (node 0 = Mac) | 56.0 ms | VPS->Mac 26.4-27.2 MiB/s | Mac->VPS 6.1 MiB/s | 6.0-6.1 MiB/s |
| Mac loopback TCP, 1 GiB | 0.061 ms | 6070 MiB/s | 6230 MiB/s | 2190 MiB/s |
| Mac SSD (OOC bench, read+write, page cache warm) | - | - | - | 4.1-6.6 GB/s effective |

The Mac's home uplink (~50 Mbit/s) is the bottleneck; SSH encryption is not
(the VPS sends at 26 MiB/s through the same channel). An earlier plain
`dd | ssh cat` gave 5.7 up / 21.5 down MiB/s.

## Model

* `2^n` amplitudes in `2^G` ranks of `2^L` (`L = n - G`). Physical qubits
  `0..L` index inside a rank, `L..n` pick the rank. Each rank has an owner
  (node 0 or 1); the table is arbitrary, so the symmetric MPI split (`G = 1`,
  `01`) and an asymmetric one (`G = 3`, `00000001`: Mac 7/8, VPS 1/8) are the
  same code. This is the chunk/global-qubit model of `ooc.md` with "chunk on
  disk" replaced by "rank on another machine".
* **Local runs.** A gate runs without communication when all of its
  *non-diagonal* qubits are local. Global qubits may appear in diagonal or
  control roles: in rank `r` they have the fixed value `bit(r, p - L)`, so the
  gate is specialised per rank (`CPhase(g, l)` -> `Phase(l)` or nothing,
  `Cnot(g, t)` -> `X(t)` or nothing, `Ccx` with global controls -> `Cnot`/`X`
  /nothing, `Rz/Phase/T/S/Z(g)` -> a scalar on the whole rank, `Cz`/`CPhase` on
  two globals -> a scalar). Specialised gates go through the cache-blocked
  kernels (`BlockedChunkExecutor`), one executor per distinct specialised
  gate list.
* **Global-qubit swap** `(l, g)`: for each rank pair `(r, r|1<<g)`, the `l=1`
  half of `r` is exchanged with the `l=0` half of the partner. Same-node pairs
  swap in memory (`swap_with_slice`, parallel); cross-node pairs stream both
  halves at once in 4 MiB messages, sender and receiver on separate threads,
  block `j` overwritten only after it was sent (one staging block per
  direction, no half-rank buffer). A 16-byte header per exchange checks both
  sides are at the same step; a handshake compares an FNV fingerprint of plan
  + ownership + element size.
* **Planner** (greedy, DAG-driven, like `schedule_ooc`): drain every ready gate
  whose non-diagonal qubits are local; when stuck, swap in the global qubits
  of the ready gate that unlocks most others, evicting local qubits with
  Belady's rule on the next *non-diagonal* use. Two free optimisations:
  - the **initial layout is free** (the start is a basis state, any qubit
    relabelling of it is the same basis state at a permuted index), so the
    qubits whose first non-diagonal use is latest start global;
  - **`Swap` gates touching a global qubit are folded** into the
    logical->physical map (`Rename` steps, no data moves).
  Order restoration at the end is optional (`--restore 1`); without it the
  final layout is kept and `gather`/`logical_index` account for it.

Swap counts (`dist_sv plan`, n = 30; `G` global qubits, restore off / on):

| workload | G=1 | G=2 | G=3 | G=4 |
|---|---|---|---|---|
| QFT (with final bit-reversal swaps) | 1 / 2 | 2 / 4 | 3 / 6 | 4 / 8 |
| brickwork 4 layers (`brick`) | 1 / 2 | 2 / 4 | 3 / 6 | 4 / 8 |
| brickwork 16 layers (`brick16`) | 1 / 2 | 2 / 4 | 3 / 6 | 4 / 8 |
| brickwork 60 layers (`brickd`, depth 2n) | 3 / 4 | 6 / 8 | 9 / 12 | 13 / 17 |

QFT needs one swap per global qubit because all its controlled phases are
diagonal (a role-unaware scheduler, `schedule_ooc` in `ooc.md`, pays 31 swap
passes for QFT-28 with 6 high qubits). Without the free initial layout QFT
G=3 needs 4 swaps instead of 3 (9 vs 6 with restore). Brickwork shallower than
`n` needs only `G` swaps: the light cone of the global qubits lets every other
qubit finish its layers first.

**Communication volume.** With one cross pair per swap (the `00000001`
layout), a swap moves `S / 2^(G+1)` bytes each way (`S` = state size). For
QFT that is `G * S / 2^(G+1)` per direction in total: `S/4` at G=1, `3S/16` at
G=3. The asymmetric layout halves the per-swap volume relative to the
symmetric `G=1` split *and* matches the VPS's small memory budget.

## Correctness

* `tests/dist.rs` (in-process `ChanLink` and loopback `TcpLink`, all green on
  the Mac; release and debug/`debug_assert` builds):
  - every gate type (incl. `I`, `U`, `Sx`, `Sxdg`, `ISwap`, `ISwapdg`, `Swap`,
    `CPhase`, `Ccx`) in random circuits, n = 6..12, `G` = 1..4, ownership
    tables `one rank on node 1`, symmetric, `node 1 owns rank 0`, parity,
    random, all-on-node-0, all 8 planner option combinations, f64 <= 1e-12 and
    f32 <= 1e-5 against `StateVector::apply_circuit`;
  - every gate type with every assignment of its qubits to {local, local,
    global, global} (identity layout so the globals really are global): 400+
    cases, <= 1e-12;
  - QFT on random basis states vs the analytic `2^{-n/2} e^{2πixk/2^n}`,
    plus a check that QFT needs exactly one swap per global qubit;
  - a proptest against the independent audit reference (`audit_common::RefSv`,
    edge-biased gates and angles), random n, G, layout, options, <= 1e-10;
  - planner invariants (no global qubit in a non-diagonal role inside a local
    run) and handshake rejection of mismatched plans / node ids.
* **Across the real link** (`wan_verify.log`): QFT, brick, brickd at n = 20/21,
  f64 and f32, layouts `01`, `00000001`, `01101001`, `10000000`, restore on and
  off, gathered on the Mac and compared with the single-node blocked executor:
  max error **3.4e-17 (f64)**, **1.2e-8 (f32)**; QFT-24 f64 on a basis state
  vs analytic: 8.9e-19. (The VPS runs the AVX2 kernels and the Mac the NEON
  ones, so f32 agreement is to rounding, not bitwise.)

## Timings

f32, `G = 3`, layout `00000001` (Mac 7/8, VPS 1/8), restore off, 3 swaps for
every workload below. Mac M1 Pro (load 3-3.7 from other agents; every timing
under `/tmp/qsim-mac-bench.lock`), VPS 4 vCPU EPYC (load ~1, 2 rayon threads,
`MemoryMax` cgroup). Min of 3 (Mac-only columns, interleaved within each locked chunk, order rotated per round) or of 2 (WAN).
Raw lines: `mac_baselines.log`, `wan_bench.log`.

| workload | n | state | Mac in RAM | Mac OOC (SSD, c=20 k=4) | distributed, loopback (2 procs on the Mac) | distributed, Mac <-> VPS | WAN exchange time | bytes each way |
|---|---|---|---|---|---|---|---|---|
| qft | 24 | 128 MiB | - | - | - | 4.61 s | 4.20 s | 24 MiB |
| brick | 24 | 128 MiB | - | - | - | 4.47 s | 4.02 s | 24 MiB |
| qft | 26 | 512 MiB | 0.197 s | 1.257 s | 0.423 s | 16.76 s | 16.20 s | 96 MiB |
| brick | 26 | 512 MiB | 0.387 s | 0.747 s | 0.732 s | 16.71 s | 15.93 s | 96 MiB |
| qft | 28 | 2 GiB | 0.828 s | 4.534 s | 1.727 s | **67.47 s** | 65.80 s | 384 MiB |
| brick | 28 | 2 GiB | 1.647 s | 2.696 s | 2.910 s | **67.44 s** | 65.13 s | 384 MiB |
| qft (analytic check) | 29 | 4 GiB | 1.742 s | (file > 4 GB budget) | - | **129.30 s** (1 run) | 126.75 s | 768 MiB |
| brick | 29 | 4 GiB | - | - | - | 131.60 s (1 run) | 127.23 s | 768 MiB |

* WAN time is the link: exchange time / bytes = 5.7-6.1 MiB/s each way, i.e.
  the measured Mac uplink. Compute on the Mac (7 ranks, 6 threads) is
  1.3-2.0 s at n=28 and 2.1-4.0 s at n=29; the VPS's 1/8 share takes about as
  long on 2 threads. Ratios: QFT-28 is 81x the in-RAM time and 15x OOC;
  QFT-29 is 74x the in-RAM time.
* Loopback isolates the protocol: 384 MiB each way in 0.25 s (1.5 GiB/s, the
  full-duplex loopback rate), and compute is 1.4-2.6 s vs 0.83-1.65 s in RAM:
  node 0 runs 4 rayon threads instead of 8 (both processes share the Mac) and
  the plan has 4 local runs instead of one pass. So even on a free link the
  two-node version is ~2x the single-node time at equal total cores.
* OOC numbers ride the page cache (2 GiB file, 16 GB Mac), as in `ooc.md`.

## Is it a net win vs out-of-core on the Mac's SSD?

No, and the gap is structural, not an implementation detail.

Cost model (checked against the table): `T_dist ≈ T_compute + swaps *
(S / 2^(G+1)) / B_link`, `T_ooc ≈ passes * 2S / B_ssd + T_compute`.

* Here `B_link` = 6.1 MiB/s and `B_ssd` ≈ 4-6 GB/s: ~700x. QFT-28 moves
  384 MiB per direction over the link (66 s) where OOC moves 28 GiB through
  the SSD (7 passes x 2 GiB x 2) in 4.5 s.
* Break-even link bandwidth for the *same* memory split: `B_link* ≈ B_ssd *
  swaps / (2^(G+2) * passes)`. QFT-30, G=3 (3 swaps) vs OOC's 9 windowed passes:
  `B_link* ≈ 4.3 GB/s * 3 / (32 * 9) ≈ 45 MB/s`, ~7x the measured uplink. But
  that comparison flatters the distributed version: it assumes the Mac holds
  7/8 of the state in RAM. If it can, the right baseline keeps the remaining
  1/8 on the local SSD (each swap = S/16 of disk I/O at GB/s, well under a
  second), and the VPS loses by its full ~700x bandwidth ratio.
* The only regime where two nodes win is a symmetric split with a fast
  link: two 16 GB Macs over Thunderbolt/10-40 GbE (1-4 GB/s) holding 30-31
  f32 qubits where neither fits it alone. QFT-30 G=1 would move S/4 = 2 GiB
  each way per swap (1 swap): ~0.5-2 s at those rates, comparable to the
  in-RAM compute (~3.5 s extrapolated from QFT-28). Over this WAN the same
  swap takes ~340 s.
* What would help on this link, none of it enough: fewer global qubits
  touched (already 1 swap per global qubit for QFT/shallow brickwork), a
  layout where the VPS holds less (volume `G * S / 2^(G+1)` falls with G, but
  so does its contribution), compression (QFT/brickwork amplitudes are dense
  random-phase floats; nothing to gain losslessly).

## Unfinished / caveats

* Two nodes only (owner table is 0/1); N nodes would need one `Link` per peer
  and a pairwise exchange schedule. Nothing in the planner limits it.
* No overlap of a swap with the next local run (the WAN time is >97%
  exchange; overlap would hide at most the compute share).
* Unitary circuits only (no measurement/reset), like `ooc`.
* The OOC comparison point at n=29 is missing: the 4 GiB file exceeds the
  swarm's 4 GB OOC budget on the Mac. QFT-29 in RAM on the Mac: 1.742 s
  (min of 3, under the lock).
* n = 29 WAN runs are single runs (each ~130 s of a 150 s lock slot); n <= 28
  are min of 2. The WAN numbers are reproducible to ~5-10% (see the run
  lists in `wan_bench.log`), set by the uplink.
* Timing caveats: Mac shared with other agents (load 3-3.7) and Logic Pro;
  VPS shared; WAN numbers depend on Dylan's home uplink at the time.
