//! Every theorem check of `tests/theory_shor.rs` (support law T1, work
//! counter identity, noise windows T3, ...) re-run on the superoptimised
//! windowed oracle `Oracle::WindowedOpt(4)` (exp/superopt): the file is
//! included as a module and its `theory_oracle()` switches on the module
//! path. Run: `cargo test --release --test theory_shor_opt`.
#[allow(clippy::map_identity, clippy::needless_range_loop)]
#[path = "theory_shor.rs"]
mod theory_shor;
