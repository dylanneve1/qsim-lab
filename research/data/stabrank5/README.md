# stabrank5: χ(|T⟩^{⊗5}) = 6 by Galois-pair enumeration and one-qubit lifts

Write-up: [research/theory/stabrank5.md](../../theory/stabrank5.md).

`stabrank5.rs` is a standalone program (std only). It includes `../stabrank-lower/stabrank.rs` as a
module, so build it from this directory:

    rustc -O -C target-cpu=native stabrank5.rs -o stabrank5

| command | what it does |
|---|---|
| `bcheck n [step]` | `bottoms()` (Pauli-orbit completions, Lemma 2) equals the `is_stabilizer`-based `completions()` of stabrank.rs |
| `galois K n k` | Lemma 1: ψ^⊥_n lies in the span of every optimal k-term decomposition (old search) |
| `gsearch K n k` | all minimal k-term decompositions (k = 4, 5) by the Galois-pair search; writes `min_K{n}_k{k}_reps.txt` and `.u16` (orbit representatives); `SIMPLE=1` selects the simpler k = 5 implementation |
| `pipeline K n k` | k = χ(n−1) + 1: optimal and minimal lists at n − 1, then all lifts (Algorithm 1); for n ≤ 4 expands and compares with the stabrank-lower lists |
| `liftfile K n k file [deg\|type1\|all]` | Algorithm 1 from a stored list of minimal representatives (`.txt` or `.u16`); `BRA=s` picks the restriction bra (index into the six 1-qubit states: 0 = \|0⟩, 1 = \|1⟩, 4 = \|+i⟩) |
| `symsplit K n k file` | Algorithm 2: degenerate lifts for every bra orbit + the all-type-I case by `glue_from` |
| `gluesym K n k file` | the all-type-I case alone |
| `olddeg K n k` | the degenerate part by the older brute-force `degenerate_search` (cross-check) |
| `suborbits K n k` | orbit counts of the k-term list under subgroups of the symmetry group |
| `orbitsizes K n` | orbit sizes and the predicted work profile of the k = 5 search |

Environment: `THREADS` (default 4), `PROGRESS=1`, `BRA` (default 0), `SIMPLE=1`.

Files:

* `m5_H4_reps.u16`: the certificate, i.e. the 14,181 canonical orbit representatives of the
  2,662,464 minimal 5-term decompositions of H^{⊗4}, as little-endian u16 5-tuples of indices into
  `enum_states(4)` (141,810 bytes; sha256
  `53e1134e2078596f2a44df788a17ad04c122015aaa073315782eb7728d972234`).
* `*.out`: the logs the tables of the write-up are taken from.
* `qpg_plateau_check.py`: the falsification check of §3.3 on the 6-term decomposition of T^{⊗6} of
  Qassim–Pashayan–Gosset (numpy).
