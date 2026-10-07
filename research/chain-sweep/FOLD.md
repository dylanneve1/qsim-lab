# Folding several worldlines into one memory pass (chain sweep, nq70 D = 70)

Question: can the chain sweep apply several qubits' worldlines per pass over
the 2^35 bond register, so that a quantized register is rounded far fewer
than R = 71 times, without growing the stored register?

**Answer: no.** With the stored register fixed at 2^35, at least one pass per
qubit is needed (R ≥ 70; the existing 71-pass schedule is optimal up to 1).
Folding k worldlines per pass is possible, but every pass boundary then holds
k worldlines in mid-flight, and that costs ⌊k/2⌋ extra stored bits
(2^⌊k/2⌋ × the register). R = 12–24 needs k = 4–7 (×4 to ×8). This is not a
scheduling limit. It is the geometry of the brickwork network.

Code (Python, exact f64): `fold-plan/`. `chain.py` ports `compile` from
`src/engines/chain_sweep.rs`. It reproduces the state vector to 1e-15
(`t_port.py`), and at n = D = 70 it gives the same plan: width 35 and 10907
ops. The other files: `cuts.py` (cut widths), `fold.py` (multi-head builder,
pass assignment, commutation check, out-of-core block executor) and
`t_table.py` (schedules and validation).

## 1. Op structure of one worldline

Slots are time-ordered: slot t holds the bond at time position t
(`t_struct.py`). Each worldline is a **staircase** of 2-bit terms on
neighbouring slots (t, t+1), plus one non-diagonal 1-bit op on every slot. It
works in place: the slot of the consumed left bond t becomes the right bond.
The head (the qubit wire) is *tied* to the bond it last created, so it costs
no register bit.

The engine alternates the direction: even qubits run forwards (slot 0→34),
odd qubits backwards. This is forced. In brickwork, qubit i's left and right
bonds sit in layers of opposite parity. Only the direction in which each step
**consumes before it creates** keeps the cut at 35. In the other direction
every intermediate cut has 36 bits (`cuts.py`: one qubit in progress, all 69
positions: 35 in its preferred direction, 36 in the other; checked at
q = 30 and q = 31).

## 2. Why R ≥ 70 at 35 stored bits

- Between passes the store holds a cut of the network.
- `cuts.py` enumerates cuts with two neighbouring worldlines both in
  progress. That is every pair of positions and all four direction choices,
  at q = 30 and q = 31. The minimum is **36**. Three in progress (on a grid
  of positions): also at least 36.
- Worldlines in progress always form a contiguous run, since a worldline
  can only finish once its left neighbour has finished. So a 35-bit cut has
  at most one worldline in progress, in its preferred direction.
- Every worldline touches all 35 slots non-diagonally. A pass with a buffer
  of w < 35 local bits therefore cannot run a whole worldline.
- So from one 35-bit cut to the next, a pass can finish at most the one
  worldline in progress and start the next one. That gives at least one
  pass per qubit: R ≥ 70.
- The alternating directions make this a zigzag: each pass finishes qubit j
  at one time-end and starts qubit j+1 at the same end. That gives
  n + 1 = 71 passes, which is what `lowprec --count` already reports.

The head state the brief asked about is real. Carrying the heads to
completion inside one pass would need all 35 slots local. Stopping the heads
in mid-flight stores their untied wires.

## 3. Folding with growth: the group zigzag

- Split the qubits into groups of k. All heads of a group run in the same
  direction, and groups alternate up, down, up, …. A pass finishes group c
  at one time-end and starts group c+1 there, with the heads staggered by
  one layer.
- Passes: ⌈n/k⌉ + 1.
- Stored at every boundary: 35 + (number of heads running against their
  preferred direction) = 35 + ⌊k/2⌋ (k odd: the group's two end heads run in
  their preferred direction).
- Transient within a pass: none beyond the pass's local bits. The buffer is
  2^(local bits), and "max local" below already counts the head bits.

D = 70, n = 70 (`table70.out`). "local" is the largest number of bits a
single pass touches non-diagonally, for the schedule picked at each w. Each
buffer b = 26–29 bits (f32: 0.5–4 GiB) admits the schedule unless marked "—".

| k | passes R | stored bits | store vs 2^35 | max local @ w = 26 / 27 / 28 / 29 |
|---|---|---|---|---|
| 1 | 71 | 35 | ×1 | 22 / 22 / 22 / 22 |
| 2 | 36 | 36 | ×2 | 23 / 23 / 23 / 23 |
| 3 | 25 | 36 | ×2 | 24 / 24 / 24 / 24 |
| 4 | 19 | 37 | ×4 | 26 / 26 / 26 / 26 |
| 5 | 15 | 37 | ×4 | 26 / 27 / 27 / 27 |
| 6 | 13 | 38 | ×8 | 26 / 27 / 28 / 29 |
| 7 | 11 | 38 | ×8 | 26 / 27 / 28 / 29 |
| 8 | 10 | 39 | ×16 | — / 27 / 28 / 29 |
| 9 | 9 | 39 | ×16 | — / — / 28 / 29 |
| 11 | — | | | no fit in the searched split range |

"—" means none of the searched schedules fits that buffer: stagger 1 layer,
split points D/2−6 … D/2+5+2k. The pass counts do not depend on the buffer;
only feasibility does.

## 4. Validation (exact, block-executed)

`fold.py:block_execute` emulates the out-of-core run.

- The state is a tensor over the bits stored between passes.
- Each pass loops over every assignment of its global bits and applies its
  ops to the 2^local sub-block only. A non-diagonal op on a global bit is an
  assertion failure.
- Finished bits are checked to be |0> and dropped.
- Op reordering across passes is checked to be commutation-legal.

The reference is a dense f64 state vector. Amplitude errors are over the RMS
amplitude 2^-n/2.

| n | D (tail) | W | k | w | passes | stored | err / rms |
|---|---|---|---|---|---|---|---|
| 16 | 32 | 16 | 1 | 11 | 17 | 16 | 2.3e-15 |
| 16 | 32 | 16 | 2 | 11 | 9 | 17 | 3.3e-15 |
| 16 | 32 | 16 | 3 | 11 | 7 | 17 | 4.1e-15 |
| 16 | 32 | 16 | 4 | 12 | 5 | 18 | 2.8e-15 |
| 16 | 32 | 16 | 5 | 13 | 5 | 18 | 4.1e-15 |
| 20 | 24 | 12 | 1 | 9 | 21 | 12 | 2.9e-15 |
| 20 | 24 | 12 | 2 | 9 | 11 | 13 | 4.8e-15 |
| 20 | 24 | 12 | 3 | 9 | 8 | 13 | 4.7e-15 |

## Bottom line

- At a 2^35 store, R_total = 71 (at least 70). Folding cannot lower it.
- The cheapest folds are k = 3: R = 25 at ×2 store, and k = 5: R = 15 at ×4
  store.
- Under a "transient ≤ 1.5× register" budget only k = 1 is possible: any
  extra stored bit doubles the store.
- With F ≈ exp(−r·R) this does not help the 16 GB target. Halving the bits
  per component to pay for ×2 storage raises r far more than R = 71 → 25
  saves (int4 r ≈ 0.009 vs int3 r ≈ 0.055 in LOWPREC.md).
