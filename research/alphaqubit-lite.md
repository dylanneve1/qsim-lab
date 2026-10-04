# AlphaQubit-lite: an open, laptop-scale reimplementation of AlphaQubit on public Google data

Branch `exp/alphaqubit-lite`. Code: `research/data/alphaqubit-lite/` (`aq_data.py` loaders + metrics,
`aq_model.py` model, `aq_gen.py` FastSampler pretraining data, `aq_train.py` pretrain / fine-tune /
eval, `baselines_real.py` decoders on real data, `remote_zip.py` partial Zenodo downloads,
`test_aq.py` tests). Data is downloaded, never committed (see §6).

**Question.** AlphaQubit (Bausch et al., *Nature* 635, 834 (2024)) is the strongest published decoder
on Google's Sycamore surface-code data, but neither its code nor its weights are public. How much of
its accuracy can an open reimplementation recover on a laptop (M1 Pro GPU, ≤ 2 GB MLX memory,
45-min runs), using our own FastSampler for pretraining and only public data?

**Answer.** _(filled in §5 when the GPU runs finish)_

---

## 1. Data (all public, CC-BY-4.0)

| dataset | Zenodo | what we use | size on disk |
|---|---|---|---|
| Sycamore 2022 "Suppressing quantum errors by scaling a surface code logical qubit" | [6804040](https://zenodo.org/records/6804040) | all 130 surface-code experiments: d = 3 (4 areas), d = 5, X and Z, R = 1, 3, …, 25 rounds, 50,000 shots each; pij DEMs fitted per half (even/odd shots); XEB noisy circuits; per-shot predictions of PyMatching, correlated matching, belief matching and the tensor-network decoder | 315 MB zip, 360 MB unpacked (repetition code skipped) |
| Willow 2024 "Quantum error correction below the surface code threshold" | [13273331](https://zenodo.org/records/13273331) | `google_105Q_surface_code_d3_d5_d7.zip` (5.7 GB) read **remotely by HTTP range requests**: only d = 3 (9 patches) and d = 5 (4 patches), X and Z, R ∈ {1, 10, 13, 30, 50}: detection events, observables, circuits, SI1000 noisy circuit, shipped predictions (correlated matching and Harmony with SI1000 and RL-optimised priors, Libra with RL prior) and their DEMs | 112 MB fetched, 384 MB unpacked |

arXiv:2609.04557 (sim-to-real decoder benchmark) uses exactly this Willow archive (its Table VII,
rounds 2–30), so our Willow numbers are directly comparable to theirs.

**One canonical layout.** Detectors are mapped to (round, stabilizer) by their coordinates
(Sycamore: `(r, c, t)`; Willow: several `(r, c, t)` triples, the last being the measure qubit).
Stabilizers sit on a 45°-rotated lattice; with `u = r + c, v = r − c` they occupy a
(d+1)×(d+1) grid. Each experiment is mapped by one of the 8 symmetries of the square so that the
on-basis stabilizers and the logical observable coincide. Result (tested): **all 120 Sycamore
experiments with R ≥ 3 and all 104 Willow experiments of one distance share one canonical layout**
(per distance), whatever the area, basis or device. One model per distance therefore serves every
area and basis (a learned per-(area, basis) context embedding lets it specialise), and a model can be
moved between the two devices.

## 2. Architecture: paper vs lite

Transcribed from the Supplementary Information pseudocode (Alg. 1 AttentionWithBias, 2
MHAttentionWithBias, 3 GatedDenseBlock, 4 ScatteringResidualConvBlock, 5 RNNCore), Methods ("Input
representation", "Syndrome transformer", "Attention bias", "Readout network", "Auxiliary tasks") and
Extended Data Figs 4 (architecture) and 8 (hyperparameters).

| component | paper (Sycamore models) | lite (this work) |
|---|---|---|
| inputs per stabilizer and round | measurement + event (soft/leakage when available; Sycamore: binary; measurements = cumulative XOR of events when pretraining on a DEM) | same: event + cumulative-XOR measurement (both devices' public data are binary) |
| StabilizerEmbedder | sum of linear projections of each input + learned index embedding → 2-layer ResNet; separate final-round projections for on-basis computed stabilizers, one learned embedding for undefined off-basis ones | same, plus a learned context (area, basis) embedding |
| state update | X ← (X + S_n)/√2 | same |
| syndrome transformer layers per round | 3 | 3 |
| dims per stabilizer | 320 | **96** |
| heads × key size | 4 × 32 | 4 × **24** |
| gated dense block widening | 5 | **4** |
| dilated 3×3 convs per layer, channels, dilations | 3 × 160; d=3: 1,1,1; d=5: 1,1,2 | 3 × **48**; same dilations |
| attention bias | 48-dim embedding of (coords i, coords j, signed offset, Manhattan distance, same-type bit) → 8-layer ResNet; + 7 event-indicator features per round; projected per head and layer | **24**-dim, **2**-layer ResNet (we add an i = j bit); 7 indicator features; per head and layer |
| readout | scatter → 2×2 conv to data qubits → project → mean-pool data-qubit lines → + round embedding → 16-layer ResNet (64 dims) → logit | same structure, **4** layers × **32** dims; one logit per line parallel to the observable, line 0 is the measured observable |
| auxiliary loss | next-stabilizer prediction, weight 0.02 | same |
| parameters | ≈ 5.4 M (all distances, scaling model); Sycamore model larger | ≈ 0.80 M (d = 3) |
| optimiser | Lamb, wd 1e-5, b2 0.95, batch 256 → 1024, piecewise-constant LR (2.45–3.46e-4) after 10k warm-up | AdamW (no Lamb in MLX), b2 0.95, wd 1e-5, linear warm-up + cosine, grad-norm clip 1 |
| pretraining | ≤ 2 × 10⁹ samples of the pij DEM of the training half, R uniform in {1, 3, …, 25}, noise curriculum f = 0.5 → 1 | FastSampler samples of the same pij DEMs (DEM → equivalent circuit, §3), R uniform in {3, …, 25}; size limited by the 45-min runs (§4) |
| fine-tuning | training half: 19,880 shots per experiment train, 5,120 dev; wd 0.08 towards the pretrained weights; ≤ 30k steps (~120 epochs) | same split; decoupled wd towards the pretrained weights |
| model selection | EMA parameters (1e-4); lowest dev LER fitted over R = 3…25 | EMA (1e-3, shorter runs); lowest dev LER, same fit |
| ensembling | 20 seeds (−0.03 / −0.08 ×10⁻² LER at d = 3 / 5) | none by default |

Implementation notes. MLX has no Lamb; we use AdamW. The recurrence is a Python loop over rounds with
`mx.checkpoint` per round (activation memory O(R · B · S · D) instead of O(R · layers · …)), which keeps
training under the 1.5 GB MLX cap. All index buffers are private (`_`-prefixed) so MLX does not try
to differentiate through gathers. Scatter/gather to the grid are pure gathers with a learned padding
vector (the paper's `P`).

## 3. Pretraining data from our FastSampler

The paper pretrains on samples of the pij DEM fitted to the training half. A DEM is a list of
independent mechanisms; `aq_data.dem_to_circuit` writes an equivalent Stim circuit that qsim-lab's
FastSampler can run (`R a; X_ERROR(p) a; CX a q_1 a q_2 …` per mechanism, one detector/observable qubit
each, one reused ancilla). Tests (`test_aq.py`):
- the converted circuit's own DEM reproduces all 799 mechanisms of a d = 3, R = 9 pij DEM exactly
  (probabilities to 1e-9);
- Stim's circuit sampler and FastSampler (`nd_tool stream`, wyrand) on the converted circuit agree
  with Stim's DEM sampler: per-detector marginals, observable rate and 50 random pair correlations,
  max |z| = 2.5 (Stim) and 2.9 (FastSampler) at 4 × 10⁵ shots.

Generating 102,400 shots for each of the 96 d = 3 and 24 d = 5 sources took about 20 s in total on the
Mac (one core).

## 4. Training cost

_(filled in from the runs: samples/s on the M1 Pro GPU, wall time, MLX peak memory)_

## 5. Results on real data

### 5.1 Sycamore 2022 (paper protocol)

Per (area, basis, fold) the per-round LER ε is fitted over R = 3, 5, …, 25 (log-fidelity fit, R = 1
excluded); mean over the 16 (d = 3) or 4 (d = 5) datasets; errors are the bootstrap (499 resamples)
per dataset combined in quadrature. Both folds unless stated.

| decoder | d = 3 LER (%) | d = 5 LER (%) | source |
|---|---|---|---|
| PyMatching (shipped predictions) | 4.002 ± 0.015 | 4.435 ± 0.034 | Zenodo |
| PyMatching 2.4, pij DEM (ours) | 3.885 ± 0.014 | 4.421 ± 0.035 | this work |
| correlated matching (shipped) | 3.495 ± 0.012 | 3.653 ± 0.024 | Zenodo |
| PyMatching 2.4 correlated, pij DEM (ours) | 3.439 ± 0.011 | 3.547 ± 0.023 | this work |
| belief matching (shipped) | 3.118 ± 0.010 | 3.109 ± 0.019 | Zenodo |
| tensor network (shipped) | 3.063 ± 0.009 | 2.983 ± 0.017 | Zenodo |
| tensor network, as quoted by the AlphaQubit paper | 3.028 ± 0.023 | 2.915 ± 0.016 | paper |
| AlphaQubit (20-model ensemble), paper | 2.901 ± 0.023 | 2.748 ± 0.015 | paper |

Our pipeline puts the shipped tensor-network predictions 1.2 % (d = 3) and 2.3 % (d = 5) above the
values the paper quotes for the same decoder; no fit variant we tried (R ≥ 3, ≥ 5, ≥ 7, F₀ = 1,
fixed R = 25) reproduces both exactly, so the paper's TN numbers were probably computed on a slightly
different pipeline. We therefore compare everything **inside one pipeline** and quote the paper's
AlphaQubit result as its ratio to TN: 0.958 (d = 3) and 0.943 (d = 5).

_(Tesseract, BP+OSD and AlphaQubit-lite rows: §5 update)_

### 5.2 Willow 2024

_(filled in)_

## 6. Reproducing

```
# Sycamore (315 MB)
curl -L -o syc.zip https://zenodo.org/api/records/6804040/files/google_qec3v5_experiment_data.zip/content
unzip syc.zip 'surface_code*' -d syc
# Willow d3/d5 subset (112 MB of the 5.7 GB archive, by HTTP range requests)
python remote_zip.py https://zenodo.org/api/records/13273331/files/google_105Q_surface_code_d3_d5_d7.zip/content \
   get 'd[35]_at_[^/]+/[XZ]/r(01|10|13|30|50)/(detection_events|obs_flips_actual|circuit_ideal|circuit_noisy_si1000|metadata|decoding_results/.*(obs_flips_predicted|error_model))' willow
python test_aq.py syc <nd_tool>
python baselines_real.py syc out/bl            # shipped + PyMatching (+ tess:15)
python aq_gen.py syc data/d3 --d 3             # FastSampler pretraining samples + packed real shots
python aq_train.py data/d3 syc runs/d3pre --mode pretrain --d 3 ...
python aq_train.py data/d3 syc runs/d3ft  --mode finetune --init runs/d3pre --wd-anchor 0.08 ...
python aq_train.py data/d3 syc runs/d3ft  --mode eval --init runs/d3ft
```
