# A simulability transition in monitored Clifford+T circuits

Branch `exp/magic-transition` (from main e7e102d). Author: qsim-magic-transition agent, 4 Oct 2026.
Code: `src/monitored/{mod.rs,circuit.rs,ent.rs}` (new engine), `examples/magic_transition.rs` (campaign
driver), `tests/magic_transition.rs` (4 tests). Data: `research/data/magic-transition/` (`raw.csv` one row
per trajectory, `aggregate.csv`, `fss.json`, PNGs, `analyze.py`, `jobs*.txt`, `run*.sh`). All runs on the
M1 Pro (statistical campaign, ≤ 4 single-threaded workers, paused while the bench lock was held).

RESULTS_PLACEHOLDER

## 1. The engine: exact monitored Clifford+T with a shrinking register

The state is kept in the rotation-frame form of `src/adaptive.rs`,

    |ψ⟩ = C (|φ⟩_A ⊗ |0⟩_rest),   C Clifford (n qubits),   |φ⟩ dense on d = |A| "active" coordinates,

but `C` is now a Schrödinger (CHP-style) tableau with rows `D_j = C X_j C†`, `S_j = C Z_j C†` and signed
phases, so that both **left** multiplication by physical Cliffords (column updates, one 16-entry table
lookup per row for a uniformly random two-qubit Clifford) and **right** multiplication by virtual Cliffords
(row products) are cheap. For an operation on physical qubit `a` the engine reads the virtual image
`Q = C† Z_a C` off column `a` (x bit j ⇔ `S_j` anticommutes with `Z_a`, z bit j ⇔ `D_j` does):

| operation | `Q` has an x bit on an inactive coordinate `v` | otherwise (`Q` acts on the register only) |
|---|---|---|
| `T_a` | absorb `CNOT(v,·)`, `CZ(v,·)`, `S_v` (all controlled by `v`, which holds \|0⟩, so the state is unchanged) into `C` until `Q = ±X_v`; `v` joins the register in `cos π/8 \|0⟩ ∓ i sin π/8 \|1⟩`: **d → d+1** | `exp(−iπ/8 · ε Q_A)` on `\|φ⟩` (sign ε from a row product); d unchanged |
| `M_a` (Z measurement) | same isolation; outcome ±1 with probability exactly 1/2; `C → C H_v (X_v)`; d unchanged | Born probability `(1 ± ε⟨φ\|Q_A\|φ⟩)/2`, project `\|φ⟩`, rotate `Q_A → Z_u` with register Cliffords (applied to `\|φ⟩` and absorbed into `C`), drop the now-factorised coordinate `u`: **d → d−1** (or nothing if `Q_A = 1`: outcome deterministic) |

This is what Clifft does at measurements; here it is a ~600-line module with no dependence on the earlier
engines. Cost: `O(n)` per two-qubit Clifford (tableau column update), `O(n · n/64)` per T/measurement
(row products), `O(2^d)` per dense update. A d-only mode (`Mode::DimensionOnly`) skips the amplitudes.

**Cut entropies, exactly** (`ent.rs`). With `G = ⟨S_j : j inactive⟩`, the Paulis on a region `R` that commute
with `G` form a group `H_R ⊇ G_R = G ∩ Paulis(R)` (`g` generators) whose quotient maps onto a logical Pauli
group on the register with `k` generators, symplectic rank `2a` and centre `b = k − 2a`. Up to a Clifford on
`R`, `ρ_R = |0⟩⟨0|^{⊗g} ⊗ σ ⊗ (I/2)^{⊗(|R|−g−a−b)}`, so

    S_α(ρ_R) = |R| − g − a − b + S_α(σ),    0 ≤ S_α(σ) ≤ a + b,

where `σ` is the state of `|φ⟩` on the `a` logical pairs with the `b` central logicals dephased. The frame
part is GF(2) elimination (`O(n³/64)`), `S_2(σ)` is computed from `|φ⟩` after a register Clifford that puts
the logical group in standard form. In d-only mode the two bounds are reported.

### 1.1 A structural fact: d is the entropy of a dephased monitored Clifford circuit

The *unsigned* tableau never depends on the amplitudes or on the measurement outcomes (outcomes only flip
row signs), so **d(t) is a property of the circuit, identical for every Born trajectory**
(`dimension_is_outcome_independent` checks it for 20 circuits, d-only vs exact vs two outcome records).
Moreover `n − d = log2 |G|`, and the table above is, operation by operation, the update of the stabilizer
group of a *mixed* stabilizer state ρ ∝ Π_{g∈G}(1+g)/2:

* Clifford: `G → U G U†`;
* `Z_a` measurement: anticommuting → standard replacement (|G| unchanged); commuting, `±Z_a ∉ G` →
  `G → ⟨G, ±Z_a⟩` (purification, d−1); `±Z_a ∈ G` → unchanged;
* `T_a`: `G → {g ∈ G : [g, Z_a] = 0}` — exactly the full Z-dephasing channel `ρ → (ρ + Z_a ρ Z_a)/2`.

So **d(t) = S(ρ_t)**, the von Neumann entropy (bits) of the *same* monitored Clifford circuit with every
`T` replaced by a dephasing channel. Two consequences:

1. d is computable in polynomial time at any `n` (we run n = 1024 in d-only mode), while `2^d` is the exact
   amplitude cost. The "simulability transition" of this engine is therefore a **purification transition of a
   monitored Clifford circuit with dephasing noise injected at the T sites** (Gullans–Huse 2020 with a
   noise source; cf. Weinstein–Bao–Altman 2022, Li–Vijay–Fisher 2023 on noise in monitored circuits).
2. A rigorous chain of magic bounds per trajectory: the stabilizer nullity of ψ equals that of φ
   (`ν(ψ) = ν(φ) ≤ d`), and for the stabilizer 2-Rényi entropy
   `M_2 = −log2(Σ_P ⟨P⟩⁴ / 2^n) ≤ ν` (the 2^{n−ν} stabilizers alone contribute `2^{n−ν}` to the sum). Hence

       M_2(ψ_t) ≤ ν(ψ_t) ≤ d(t)   for every trajectory and time.

   Any measurement rate at which `d/n → 0` is one at which the magic density `M_2/n → 0`.

## 2. Validation (exactness)

`tests/magic_transition.rs` (dev profile with all `debug_assert`s on, and release):

* `clifford_group_has_11520_elements`: the BFS over {H, S, CNOT} words gives the whole two-qubit Clifford
  group mod phase (uniform sampling as in Li–Chen–Fisher / Bejan et al.).
* `exact_against_state_vector_with_midcircuit_measurements`: 60 random monitored circuits, n = 3…10,
  depth 3n, `p_m ∈ {0.05, 0.15, 0.3, 0.5}`, `p_T ∈ {0.1, 0.25, 0.5}`, open and periodic. The engine samples
  each outcome from the Born rule; the dense state vector is collapsed onto the same outcome and its
  probability is compared (|Δp| < 1e-9 at every one of the mid-circuit measurements); every 3 layers the full
  state (expanded from the gate logs) has fidelity 1 ± 1e-9 with the SV; the exact Rényi-2 entropy of the
  half chain **and of random qubit subsets** matches the SV to 1e-7, equals that of the complement, and lies
  in the frame bounds; for d ≤ 8 the stabilizer nullity and M_2 of the register equal those of the full SV
  state (`magic_atlas::state_magic`).
* `dimension_is_outcome_independent` (above) and `dephasing_picture_matches` (full-region frame algebra:
  g = n − d, a = d).
* `examples/magic_transition validate`: VALIDATE_PLACEHOLDER

## 3. Model and observables

Brickwork of uniformly random two-qubit Cliffords on a ring of n qubits (even/odd layers alternate); after
every layer, each qubit gets `T` with probability `p_T` and then a Z measurement with probability `p_m`
(Bejan–McLauchlan–Béri's "uncorrelated monitoring"; Fux et al. use the same ingredients). Initial state
|0ⁿ⟩, depth 4n, steady-state `d̄` = average of d over the last n layers. Families:

| family | `p_T` | n | use |
|---|---|---|---|
| E0 | 0 (pure Clifford) | 16–256 | entanglement-transition reference: tripartite mutual information `I_3` of quarters (exact) |
| E1 | `1/n` (η = 1 T per layer, `qD = O(1)` of Bejan et al.) | 16–512 | dilute magic: d/n, P(d ≥ 0.05 n), FSS |
| E3 | `2/n` (η = 2, Fux et al.'s main case) | 32–1024 | direct test of their `p_c^magic ≈ 0.22` |
| E2 | 0.01, 0.05, 0.2 (constant density) | 16–512 | dense magic |
| E4, E5 | `η/n`, exact amplitudes | 32–2048 | ν, M_2 (d ≤ 12), exact S_2; largest exact sizes |

Samples per point: 400 (n ≤ 32) down to 60 (n = 512) and 16 (n = 1024); see `aggregate.csv`.
FSS: weighted polynomial master curve (degree 4), minimise χ²/dof over `(p_c, ν[, β/ν])` for
`Φ n^{β/ν} = F((p − p_c) n^{1/ν})`; error bars = bootstrap over trajectories (60 resamples).
