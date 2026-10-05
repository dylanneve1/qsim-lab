# Contributing to qsim-lab

qsim-lab is a research codebase: every change is either an engine/library
change (which must stay exact and be differentially tested) or an experiment
(which must be written up, numbers and caveats included). This file is the
house style for both.

## Repository layout

| Path | What lives there |
|---|---|
| `src/` | the `qsim_lab` library and the `qsim` CLI (`src/main.rs`) |
| `tests/` | integration tests by area (`core/ engines/ compiler/ simulability/ shor/ qec/ theory/ audit/`), each listed as a `[[test]]` in `Cargo.toml`; shared helpers in `tests/common/` and `tests/audit_common/` (the independent reference state vector every engine is checked against) |
| `examples/` | tutorials, benchmarks and research drivers — see [examples/README.md](examples/README.md) |
| `research/` | lab notebooks by topic, plus `research/data/<study>/` — see [research/README.md](research/README.md) |
| `tools/` | repo tooling: link checker, relinker, data packer, audit adapters, superopt scripts |
| `RESULTS.md` | headline results, each pointing at its notebook |

## Build and test

The toolchain is pinned in `rust-toolchain.toml` so local clippy matches CI.

```sh
cargo build --release
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --release                  # full suite; some audit tests take minutes
cargo build --release --examples      # CI builds every example
python3 tools/check_links.py --anchors   # every relative link in every .md resolves
```

CI (`.github/workflows/ci.yml`) runs fmt, clippy `-D warnings`, the test
suite and the examples build on every push to `main` and every PR. All five
must be green before a branch is merged.

Optional features: `--features metal` builds the Apple-GPU backend
(`metal_sv`, macOS only; a no-op elsewhere).

On shared machines keep builds polite: `nice -n 15 cargo ... -j 2`, one
`CARGO_TARGET_DIR` per worktree, `RAYON_NUM_THREADS<=2` for tests, and delete
target directories you are done with.

## Code conventions

* **Where code goes** (see the README's Architecture section): simulators in
  `src/engines/`, file formats in `src/io/`, gate-level Shor in `src/shor/`,
  QEC in `src/qec/`, timing harnesses in `src/bench/`. Use the canonical paths
  (`qsim_lab::engines::statevector`, `qsim_lab::shor::ge`, `qsim_lab::io::qasm`);
  the old flat paths are hidden compatibility re-exports only.
* **Errors**: a fallible public function returns its subsystem's error type
  (`SimError`, `DagError`, `DemError`, `StimError`, ...), which implements
  `std::error::Error` and converts into the crate-wide `qsim_lab::Error` (use
  `qsim_lab::Result` when mixing subsystems). Don't add new `Result<_, String>`
  APIs.
* **Docs**: every new public item gets a doc comment; module files start with a
  `//!` summary that says what the module is for and links its notebook.
* Names: `snake_case` modules named after what they contain, no `_v2`-style
  suffixes in `src/` (keep experiments on branches or behind options).

## Correctness discipline

* **Every engine change is differentially tested** against the reference state
  vector in `tests/audit_common` (random and adversarial circuits, edge sizes,
  every configuration knob). An engine is exact or it is wrong: tolerances are
  floating-point round-off, never "statistically close".
* Statistical claims (sampling, logical error rates) state the number of
  shots, the confidence interval and the test used.
* A bug fix comes with the test that would have caught it.
* New integration test: put it in `tests/<area>/<name>.rs`, add a `[[test]]`
  entry to `Cargo.toml`, and pull helpers in with
  `#[path = "../common/mod.rs"] mod common;`. Give it a weight in
  `tools/ci_shard.py` if it runs for more than a few seconds.
* Independent audits are recorded in [research/process/audit.md](research/process/audit.md);
  a claim that an audit overturned is corrected in place and listed under
  "Corrections we have published" in the README.

## Timing discipline

* Interleaved A/B runs, minimum of at least 3 repetitions, same binary flags.
* Always report the machine, the core count used and the load average.
* No timing claims from a loaded machine (load above the core count): re-time
  on an idle one.
* Compare against the strongest baseline available (natively built Stim,
  qsim, Aer, ...), configured as their authors recommend.

## Research notebooks

One notebook per study, in the topic folder that fits
(`research/{shor,qec,simulability,performance,compiler,theory,process}/`),
added to the index in [research/README.md](research/README.md).

A notebook contains, in order:

1. **Title and provenance** — branch, base commit, date, machine(s).
2. **Headline** — the result in two or three sentences, with numbers.
3. **Method** — what was run, exactly enough to reproduce it (commands,
   seeds, parameters).
4. **Results** — tables with units and uncertainties; figures as PNG in
   `research/data/<study>/`.
5. **Caveats and negative results** — what did not work and why; a negative
   result written up is a valid result.
6. **Reproduction** — the scripts in `research/data/<study>/` and how to run them.

Data conventions:

* Raw data and the scripts that produced it go in `research/data/<study>/`
  (flat, named after the study, never moved when the notebook is re-filed).
  Scripts should take their inputs as arguments or paths relative to their
  own directory.
* Raw text files over ~200 kB are committed xz-compressed:
  `tools/datafiles.py pack <file>` (records the sha256 in
  `research/data/COMPRESSED.tsv`); readers run `tools/datafiles.py unpack`.
* Never commit build output, virtualenvs, `__pycache__/`, or macOS `._*`
  files (all in `.gitignore`).
* Link to other notebooks and data with relative links, and run
  `tools/check_links.py` before pushing. To move or rename documents, use
  `tools/relink.py --mv OLD NEW`, which rewrites every reference repo-wide.

## Git and merge rules

* Commit author is **Claudius** (`Claudius <claudiusthebot@gmail.com>`,
  already the repo config). No personal names in commits, notebooks or code;
  no `Co-Authored-By` trailers.
* Work happens on `exp/<topic>` branches (one git worktree per branch). Only
  the maintainer merges to `main`; nobody pushes to `main` directly.
* Rebase or merge `origin/main` into your branch before asking for a merge,
  and re-run the full checklist above on the result.
* Commit messages: `<area>: <what changed>`, with the measured effect when
  there is one (e.g. `sv: fuse diagonal runs; QFT-28 1.31x on M1 Pro`).
* Keep merges reviewable: one topic per branch, no drive-by reformatting of
  files you are not otherwise changing.
