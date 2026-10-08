# Notes on Majorana/Fermion Propagation

1. **Formalism**: We represent observables as sums of normal-ordered fermionic monomials in the occupation basis $c^\dagger_{x_1} \dots c_{y_k}$. Total particle number per chain is conserved, so we only track monomials with an equal number of creation and annihilation operators per chain.
2. **Operations**:
   - Single-site phases: $c^\dagger \to e^{i\phi} c^\dagger$.
   - Hopping ($V_{sp}$): $c^\dagger_x \to V_{xx} c^\dagger_x + V_{xy} c^\dagger_y$. Implemented as sparse index substitutions on the normal-ordered bitmasks, generating up to 4 branches per hop.
   - Interaction: $V = \exp(i g n_a n_b)$. Using exact combinatorial rules, each monomial splits into at most 2 terms: one unchanged, and one with an added $n_a n_b$ factor (and coefficient multiplied by $e^{\pm i g} - 1$).
3. **Optimizations**:
   - Bitmask representation: 60 sites fit exactly in a 64-bit integer.
   - Pauli principle / anticommutation signs evaluated in $O(1)$ via `__builtin_popcountll`.
   - Normal ordering signs during interactions only depend on the bitmasks of the same chain.
   - Core propagation written in a C extension (`heis_c.c`) using a fast linear-probing hash table, achieving $~100\times$ speedup over Python dicts.
4. **Validation**:
   - Setting $g=0$ locally but retaining circuit $a_0, a_1$ reproduces exactly the `gauss.py` free results (e.g. `n_f=0.4963` at step 20).
   - Simulating the full interactive circuit gives exact results matching small-system/early-time statevectors (e.g. `stag_SCV(step=2) = -40.855`, which aligns with the physical RZZ compilation).

