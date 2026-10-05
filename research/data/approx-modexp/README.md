# Data and scripts for `research/shor/approx-modexp.md`

The paper's code release (Gidney 2025, CC-BY-4.0, doi:10.5281/zenodo.15347487,
`code.zip` sha256 `e627abdeb91e880ec8500a3015ab59eb09c3171e1c8f8d9c5eab96728064c94d`)
is **not** committed; download and unzip it, and point `GIDNEY_SRC` at its
`src/` directory for the Python checks (numpy, sympy; the release's own
`requirements.txt`).

| file | what |
|---|---|
| `run_experiments.sh` | reproduces everything (`xcheck verify dist sweep gate model moon`) |
| `gidney_env.py` | locates the release, keeps its process pools small, builds the paper's `ExecutionConfig` for a given prime set |
| `xcheck_tables.py` | Rust precomputation vs the paper's table code, entry by entry; the paper's own prime verifier and search |
| `qbackend.py` | genuinely quantum backend for the paper's `scatter_script`; runs the paper's `approx_modexp` on the full superposition and compares with the Rust simulator |
| `paper_model.py` | the paper's success-rate model (`main1`), evaluated exactly |
| `model_vs_paper_csv.py` | that exact evaluation against the paper's released Monte-Carlo data |
| `paper_trajectories.py` | the paper's own trajectory verifier on one of our instances (for context) |
| `eq28_check.py` | dense numpy check of the paper's Eq. 28 bound (uniform shift) |
| `plot_results.py` | figures from `out/` |
| `out/` | raw outputs of the runs (text and CSV) |
| `xcheck/` | intermediate files of the cross-checks (dumped tables, outcome logs, distributions) |
