//! The theorem checks of `tests/theory_shor.rs` (support law T1 on whole
//! gate-level measurement trees, the work-counter identity, ...) re-run on
//! the measurement-based oracle `Oracle::WindowedMbu(4)` (exp/mbu-shor).
//! The T3 noise-window checks need the noisy trajectory engine, which only
//! runs reversible oracles; they return early here. Run:
//! `cargo test --release --test theory_shor_mbu`.
#[path = "theory_shor.rs"]
mod theory_shor_mbu;
