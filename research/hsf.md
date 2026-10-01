# Hybrid Schrödinger--Feynman (HSF)

## Hypothesis

Cutting a circuit into two balanced blocks stores two vectors of size
`2^(n/2)` instead of one vector of size `2^n`; the tradeoff is one path per
operator-Schmidt term of each cross-cut gate.

## Change

`HybridSchrodingerFeynman` (`src/hsf.rs`) represents the wavefunction as a
sum of products of two `StateVectorF64`s. Cross-cut two-qubit gates use the
exact four-term matrix-unit expansion, so arbitrary supported two-qubit gates
remain exact (CNOT/CZ/CPhase generally have lower rank, but the generic path
bound is four). `amplitude` avoids full output allocation; `simulate` rebuilds
full output only under the existing 1 GiB cap. `automatic` performs a balanced
single-swap local search to reduce crossing gates.

## Verification

`cargo fmt` and `CARGO_BUILD_JOBS=2 cargo test` were run. The implementation
uses the existing state-vector gate kernels and therefore keeps f64 arithmetic.
The full cross-check and benchmark matrix is pending integration by the parent
agent; timings must be taken through `bench.sh` on the shared VM.

## Limitations

The backend currently accepts circuits without measurements and supports
cross-cut two-qubit gates for which `Gate::matrix_2q` exists. Toffoli crossing
the cut should be decomposed into Clifford+T gates by a caller before use.
