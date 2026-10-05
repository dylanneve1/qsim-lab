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
gradient checkpointing per round (`nn.utils.checkpoint`, activation memory O(R · B · S · D) instead of
O(R · layers · …)) and the whole training step is `mx.compile`d (one trace per round count).

**A bug worth recording.** The first version wrapped the round step in a bare `mx.checkpoint(self._step)`.
MLX's `mx.checkpoint` only differentiates with respect to the function's explicit inputs: gradients
with respect to the module parameters the closure captures are silently dropped. The loss and
forward pass were exactly right, but the bulk-round embedding got **zero** gradient and the transformer
layers learned only from the final round. Three 42-minute GPU runs (≈ 4 M samples) trained like this
and stalled at the level of a trivial decoder (dev LER 6–11 % against 11.5 % for "always predict no
flip"). A finite-gradient comparison (checkpointed vs plain vs compiled autodiff) exposed it; the fix
is `nn.utils.checkpoint(self, self._step)`, which passes the parameters explicitly. `test_aq.py` now
asserts that the three gradients agree for every tensor. After the fix the training loss at 150 k
samples is 0.26 instead of 0.62. All index buffers are private (`_`-prefixed) so MLX does not try
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

## 4. Training runs and cost

All on the M1 Pro GPU (MLX 0.32.3), one run at a time, each capped at 42 min wall clock *including*
pauses (other agents' bench-lock timing runs pause us; the Mac's load average was 11–38 throughout,
from other agents' CPU jobs). MLX memory: `set_memory_limit` 1.2 GB + 256 MB cache, hard stop if the
MLX peak passes 1.75 GB; the run never exceeded 1.65 GB (wired memory stayed 2.9–3.4 GB, free +
inactive ≥ 5.3 GB). Lite model: D = 64, key 16, conv 32, 0.37 M parameters.

| run | data | samples | GPU time | samples/s | MLX peak | dev LER at end (best) |
|---|---|---|---|---|---|---|
| d3 pretrain (correct gradients) | FastSampler pij-DEM samples, noise curriculum f = 0.5→1 (t_c = 0.4 M, s_c = 2), rounds curriculum R_max 7→25 over 0.8 M, batch 256, lr 5e-4 cosine | 1.25 M | 42 min | 1,070 (short R) → 520 (R ≤ 25) | 1.64 GB | 3.55 % |
| d3 fine-tune | real training half (even shots, 19,880 / experiment, 1.9 M shots), batch 256, lr 2e-4, decoupled wd 0.02 towards the pretrained weights | 0.96 M (0.5 epoch) | 42 min | ~380 | 1.64 GB | 3.35 % |
| d3 held-out evaluation | odd shots, 2.4 M | – | 17 min | 2,330 | | 3.512 % (test) |
| d5 pretrain | as d3, warm start from the d3 model (180/185 tensors), batch 80, t_c = 0.12 M | 0.32 M | 42 min | 220–330 | 1.60 GB | 5.68 % |
| d5 fine-tune | real training half | _(running)_ | | | | |

Total for the reported d = 3 model: 2.2 M training samples, 84 GPU-minutes. AlphaQubit's Sycamore
models: up to 2 × 10⁹ pretraining samples + ~120 fine-tuning epochs per model, ×20 ensemble members,
a ~3–15× larger model, on TPUs. We are ~10³ below the paper in samples and ~10⁴–10⁵ below it in
FLOPs.

**What went wrong first (and is worth knowing).** (i) The gradient-checkpointing bug of §2 cost three
runs. (ii) Without a curriculum, even correct gradients learn slowly on Sycamore-strength noise
(detection density 15 %); the paper's noise curriculum plus a short-experiments-first rounds
curriculum gave a dev LER of 4.6 % after 0.5 M samples. (iii) An EMA with decay constant 1e-3 still
carries 37 % of the initial weights after 1,000 steps; we warm the EMA up (rate max(1e-3, 1/(1 + it/10)))
and select on the better of raw and EMA weights.

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

**Held-out comparison, d = 3 (odd-index shots, 8 datasets × 12 round counts × 25,000 shots = 2.4 M
shots; every decoder on the same shots; 95 % CIs from a paired bootstrap over the joint per-shot
fail pattern; ratio = LER / tensor-network LER on the same bootstrap draws).**

| decoder | LER (fit R = 3…25) | 95 % CI | ratio vs TN | ε at R = 3 | ε at R = 25 |
|---|---|---|---|---|---|
| PyMatching (shipped) | 4.008 % | [3.972, 4.050] | 1.310 [1.296, 1.325] | 3.18 % | 3.89 % |
| PyMatching 2.4, pij DEM (ours) | 3.882 % | [3.843, 3.923] | 1.269 [1.254, 1.282] | 3.09 % | 3.74 % |
| correlated matching (shipped) | 3.492 % | [3.464, 3.525] | 1.141 [1.129, 1.154] | 2.95 % | 3.43 % |
| **AlphaQubit-lite (this work)** | **3.512 %** | **[3.482, 3.542]** | **1.148 [1.137, 1.158]** | **2.64 %** | **3.38 %** |
| PyMatching 2.4 correlated, pij DEM (ours) | 3.434 % | [3.409, 3.464] | 1.122 [1.112, 1.133] | 2.82 % | 3.33 % |
| BP+OSD-CS order 10, pij DEM (ours, nd_tool) | 3.358 % | [3.331, 3.390] | 1.097 [1.089, 1.106] | 2.74 % | 3.25 % |
| belief matching (shipped) | 3.122 % | [3.098, 3.149] | 1.020 [1.012, 1.029] | 2.71 % | 3.06 % |
| tensor network (shipped) | 3.060 % | [3.038, 3.087] | 1.000 | 2.61 % | 2.98 % |
| AlphaQubit, paper (both folds, 20-model ensemble) | 2.901 ± 0.023 % | | 0.958 (vs paper's TN) | | |

(ε at fixed R is ½(1 − (1 − 2E)^{1/R}) averaged over the 8 datasets.)

Reading the table:
- AlphaQubit-lite beats PyMatching by 10 % and ties Google's shipped correlated matching; it is
  15 % above the tensor network, where the paper's AlphaQubit is 4 % below it. We recover about
  half of the PyMatching → AlphaQubit gap (PM 1.269 → lite 1.148 → paper 0.958 in TN units).
- The model is strong on short experiments and weaker on long ones: at R = 3 its per-round error
  (2.64 %) is within 1.5 % of the tensor network's and below belief matching's, but at R = 25 it is 14 %
  above TN. The fitted intercepts F₀ are 1.06–1.15 (the paper requires F₀ ≈ 1 for a good fit), i.e.
  the fidelity decays faster than exponentially-in-R would predict from the short runs. This is the
  signature of an under-trained recurrent state: our curriculum spent most early samples on short
  experiments, and we stopped after 2.2 M samples (paper: ≤ 2 × 10⁹ + 120 fine-tuning epochs).
- Tesseract (beam 15) on this data: §5.3 (running, 5,000 shots per experiment).

### 5.2 Willow 2024 (105-qubit processor, d = 3: 9 patches, d = 5: 4 patches)

Test fold: odd-index shots (25,000 per configuration). Per-round LER ε = ½(1 − (1 − 2E)^{1/r}) per
configuration, averaged over patches and bases; "mean 10–30" averages r ∈ {10, 13, 30}, which is
exactly the "rounds 2–30" window of arXiv:2609.04557 Table VII (the archive has no other counts
there; r = 1 is excluded). "fit" is the paper-style log-fidelity fit over r ∈ {10, 13, 30, 50}.

| decoder (prior) | d = 3 mean 10–30 | d = 3 fit | d = 5 mean 10–30 | d = 5 fit | source |
|---|---|---|---|---|---|
| PyMatching 2.4 (SI1000 prior) | 0.994 % | 1.005 % | 0.622 % | 0.638 % | this work |
| MWPM, arXiv:2609.04557 Table VII | 0.995 % | – | 0.622 % | – | paper |
| PyMatching 2.4 correlated (SI1000) | 0.868 % | 0.864 % | 0.428 % | 0.427 % | this work |
| correlated matching (SI1000), shipped | 0.819 % | 0.815 % | 0.420 % | 0.420 % | Zenodo |
| correlated matching (RL prior), shipped | 0.739 % | 0.745 % | 0.387 % | 0.388 % | Zenodo |
| Harmony, 101-ensemble (SI1000), shipped | 0.759 % | 0.757 % | 0.370 % | 0.370 % | Zenodo |
| Harmony (RL prior), shipped | 0.714 % | 0.714 % | 0.349 % | 0.352 % | Zenodo |
| Tesseract (SI1000), arXiv:2609.04557 | 0.738 % | – | 0.352 % | – | paper |
| BeliefMatching (SI1000), arXiv:2609.04557 | 0.815 % | – | 0.421 % | – | paper |
| BP+OSD (SI1000), arXiv:2609.04557 | 0.936 % | – | 0.659 % | – | paper |

Our uncorrelated-matching numbers reproduce arXiv:2609.04557 to the third digit (0.994 vs 0.995 %,
0.622 vs 0.622 %), so the two pipelines agree. The Willow paper's own neural decoder (an AlphaQubit
descendant, ref. [27] there) is not in the archive; it reports ε₇ = 0.143 % and Λ = 2.14 on this
device.

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
