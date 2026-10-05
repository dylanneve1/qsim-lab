# Research data

Raw data, logs and the scripts that produced them, one folder per study. Each
folder belongs to a notebook (see [../README.md](../README.md)):

| Data | Notebook |
|---|---|
| [`adaptive/`](adaptive/) | [adaptive.md](../simulability/adaptive.md) |
| [`alphaqubit-lite/`](alphaqubit-lite/) | [alphaqubit-lite.md](../qec/alphaqubit-lite.md) |
| [`audit/`](audit/) | [audit.md](../process/audit.md) |
| [`colour-flags/`](colour-flags/) | [colour-flags.md](../qec/colour-flags.md) |
| [`code-discovery/`](code-discovery/) | [code-discovery.md](../qec/code-discovery.md) |
| [`colour-global/`](colour-global/) | [colour-global.md](../qec/colour-global.md) |
| [`compiler/`](compiler/) | [compiler.md](../compiler/compiler.md) |
| [`dag/`](dag/) | [dag.md](../compiler/dag.md) |
| [`fast-sampler/`](fast-sampler/) | [fast-sampler.md](../qec/fast-sampler.md) |
| [`fast-sampler-audit/`](fast-sampler-audit/) | [fast-sampler-audit.md](../qec/fast-sampler-audit.md) |
| [`dense-fusion/`](dense-fusion/) | [dense-fusion.md](../performance/dense-fusion.md) |
| [`ft-shor/`](ft-shor/) | [ft-shor.md](../shor/ft-shor.md) |
| [`ge-shor/`](ge-shor/) | [ge-shor.md](../shor/ge-shor.md) |
| [`graph-compiler/`](graph-compiler/) | [graph-compiler.md](../compiler/graph-compiler.md) |
| [`hsf/`](hsf/) | [hsf.md](../performance/hsf.md) |
| [`mac-m1/`](mac-m1/) | [mac-m1.md](../performance/mac-m1.md) |
| [`l1/`](l1/) | [mac-m1.md](../performance/mac-m1.md) |
| [`lowmagic-chem/`](lowmagic-chem/) | [lowmagic-chem.md](../simulability/lowmagic-chem.md) |
| [`magic-atlas/`](magic-atlas/) | [magic-atlas.md](../simulability/magic-atlas.md) |
| [`magic-transition/`](magic-transition/) | [magic-transition.md](../simulability/magic-transition.md) |
| [`mbu-shor/`](mbu-shor/) | [mbu-shor.md](../shor/mbu-shor.md) |
| [`metal/`](metal/) | [metal.md](../performance/metal.md) |
| [`neural-decoder/`](neural-decoder/) | [neural-decoder.md](../qec/neural-decoder.md) |
| [`noise-oracles/`](noise-oracles/) | [noise-oracles.md](../shor/noise-oracles.md) |
| [`ooc/`](ooc/) | [ooc.md](../performance/ooc.md) |
| [`pauli/`](pauli/) | [pauli.md](../performance/pauli.md) |
| [`phasepoly/`](phasepoly/) | [phasepoly.md](../compiler/phasepoly.md) |
| [`pipeline/`](pipeline/) | [pipeline.md](../performance/pipeline.md) |
| [`planner/`](planner/) | [planner.md](../simulability/planner.md) |
| [`planner-v2/`](planner-v2/) | [planner-v2.md](../simulability/planner-v2.md) |
| [`qec/`](qec/) | [qec.md](../qec/qec.md) |
| [`qec-r4/`](qec-r4/) | [qec-r4.md](../qec/qec-r4.md) |
| [`r4-audit2/`](r4-audit2/) | [audit.md](../process/audit.md) |
| [`repeat/`](repeat/) | [repeat.md](../compiler/repeat.md) |
| [`schedules/`](schedules/) | [schedules.md](../qec/schedules.md) |
| [`shor/`](shor/) | [shor.md](../shor/shor.md) |
| [`shor-noise/`](shor-noise/) | [shor-noise.md](../shor/shor-noise.md) |
| [`shor_r4/`](shor_r4/) | [shor.md](../shor/shor.md) |
| [`shor_r4_audit/`](shor_r4_audit/) | [shor-r4-audit.md](../shor/shor-r4-audit.md) |
| [`simd/`](simd/) | [sv-monomial.md](../performance/sv-monomial.md) |
| [`simulability/`](simulability/) | [simulability.md](../simulability/simulability.md) |
| [`spoof-utility/`](spoof-utility/) | [spoof-utility.md](../simulability/spoof-utility.md) |
| [`stab/`](stab/) | [stab.md](../performance/stab.md) |
| [`stabrank-lower/`](stabrank-lower/) | [stabrank-lower.md](../theory/stabrank-lower.md) |
| [`superopt/`](superopt/) | [superopt.md](../shor/superopt.md) |
| [`theory-colour/`](theory-colour/) | [theory-colour.md](../theory/theory-colour.md) |
| [`theory-coset/`](theory-coset/) | [theory-coset.md](../theory/theory-coset.md) |
| [`theory-shor/`](theory-shor/) | [theory-shor.md](../theory/theory-shor.md) |
| [`theory-rank/`](theory-rank/) | [theory-rank.md](../theory/theory-rank.md) |
| [`transition-theory/`](transition-theory/) | [transition-theory.md](../theory/transition-theory.md) |

## Compressed raw files

Raw files over ~200 kB are stored xz-compressed (`<name>.xz`). The manifest
[COMPRESSED.tsv](COMPRESSED.tsv) records the original path, its sha256 and
size, and the sha256 of the `.xz`. The analysis scripts read the original
paths, so unpack once before re-running them (from the repo root):

```
tools/datafiles.py unpack     # restores every original, verified against the manifest
tools/datafiles.py verify     # integrity check only
```

Unpacked originals are listed in [.gitignore](.gitignore), so `git status`
stays clean. Rust tests that read data directly (`tests/planner.rs` reads
`simulability/raw/*.csv`) use files that are kept uncompressed. To compress
new large output: `tools/datafiles.py pack path/to/file.csv` (or
`tools/datafiles.py pack` to sweep every tracked text file over 200 kB).
