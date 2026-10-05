# stabrank-lower: exact small-n stabilizer-rank computations

`stabrank.rs` is a standalone program (std only):

    rustc -O -C target-cpu=native stabrank.rs -o stabrank

| command | what it does |
|---|---|
| `count` | number of n-qubit stabilizer states, n ≤ 4 |
| `search K n k [prod]` | all independent k-sets of stabilizer states (or product stabilizer states, `prod`) whose span contains K^{⊗n}, with K = H or F (or W / D<w> for W/Dicke states); writes `dec_K{n}_k{k}.txt` |
| `brute K n k` | naive enumeration of all k-subsets (k ≤ 4), used to cross-check `search` |
| `chain K n0 k nmax` | exhaustive search at n0 (asserts χ = k there), then plateau-gluing up to nmax |
| `deg K n` | the degenerate-restriction half of the search one step past a plateau (k = χ(n−1)+1) |
| `gluemin K n k` | the non-degenerate half: glue all minimal k-term decompositions of K^{⊗(n−1)} |
| `degcheck K n s`, `gluecheck K n k` | validate `deg` / `glue` against a direct search at n = 3 |
| `pblock K b` | the kill probability p_b for the block-local restricted model |
| `anneal K n k steps restarts` | simulated annealing for upper bounds (weak; not used for any claim) |

Environment: `THREADS` (default 1), `GROUP_CAP` (largest symmetry group stored explicitly, default
1000), `PROGRESS=1`.

Outputs (`*.out`) are the logs that the table in `research/stabrank-lower.md` is taken from.
Decomposition files list state indices into `enum_states(n)` order, plus coefficients.
