# AlphaQubit-lite: an open, laptop-scale reimplementation of AlphaQubit on public Google data

Branch `exp/alphaqubit-lite`. Code: `research/data/alphaqubit-lite/` (`aq_data.py` loaders + metrics,
`aq_model.py` model, `aq_gen.py` FastSampler pretraining data, `aq_train.py` pretrain / fine-tune /
eval, `baselines_real.py` decoders on real data, `remote_zip.py` partial Zenodo downloads,
`test_aq.py` tests). Data is downloaded, never committed (see §7). Result files (per-run logs, held-out
evaluations, paired comparisons) are in `research/data/alphaqubit-lite/results/`.

**Question.** AlphaQubit (Bausch et al., *Nature* 635, 834 (2024)) is the strongest published decoder
on Google's Sycamore surface-code data, but neither its code nor its weights are public. How much of
its accuracy can an open reimplementation recover on a laptop (M1 Pro GPU, ≤ 2 GB MLX memory,
45-min runs), using our own FastSampler for pretraining and only public data?

**Answer.** On held-out real data, with 2.2 M training samples and 84 GPU-minutes, AlphaQubit-lite
recovers roughly half of AlphaQubit's d = 3 gain over PyMatching but not the rest:

| real data (held out) | AlphaQubit-lite | PyMatching | correlated matching | Tesseract | best published |
|---|---|---|---|---|---|
| Sycamore d = 3, LER (fit R = 3…25) | **3.51 %** | 3.88 % | 3.43 % (ours) / 3.49 % (Google) | 3.11 % | tensor net 3.06 %; AlphaQubit 2.90 % (paper) |
| Sycamore d = 5 | **5.55 %** | 4.39 % | 3.53 % / 3.61 % | 3.31 % (1–2 k shots/exp.) | tensor net 2.98 %; AlphaQubit 2.75 % (paper) |
| Willow d = 3, mean ε r = 10/13/30 | **0.760 %** (0.807 % zero-shot) | 0.993 % | 0.865 % (ours) / 0.739 % (Google, RL prior) | 0.738 % (arXiv:2609.04557) | Harmony-RL 0.714 % |

- d = 3 Sycamore: 10 % better than PyMatching, level with Google's correlated matching, 15 % above
  the tensor network (paper: 4 % *below* it). Near-optimal on short experiments (R = 3: within 1.5 %
  of the tensor network), weaker on long ones.
- d = 5 Sycamore: under-trained (0.67 M samples at ~200 samples/s); worse than PyMatching.
- Willow d = 3: a Sycamore-trained model transfers zero-shot (beats PyMatching by 19 %); after one
  fine-tuning run it matches a 101-member matching ensemble (Harmony-SI1000) and Google's correlated
  matching with the RL-optimised prior at the trained round counts.
- The binding constraint is compute: ~10³ fewer samples than the paper (and a ≥ 15× smaller model);
  §6 lists what limits further gains.

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
| dims per stabilizer | 320 | **64** |
| heads × key size | 4 × 32 | 4 × **16** |
| gated dense block widening | 5 | **4** |
| dilated 3×3 convs per layer, channels, dilations | 3 × 160; d=3: 1,1,1; d=5: 1,1,2 | 3 × **32**; same dilations |
| attention bias | 48-dim embedding of (coords i, coords j, signed offset, Manhattan distance, same-type bit) → 8-layer ResNet; + 7 event-indicator features per round; projected per head and layer | **24**-dim, **2**-layer ResNet (we add an i = j bit); 7 indicator features; per head and layer |
| readout | scatter → 2×2 conv to data qubits → project → mean-pool data-qubit lines → + round embedding → 16-layer ResNet (64 dims) → logit | same structure, **4** layers × **32** dims; one logit per line parallel to the observable, line 0 is the measured observable |
| auxiliary loss | next-stabilizer prediction, weight 0.02 | same |
| parameters | ≈ 5.4 M (all distances, scaling model); Sycamore model larger | **0.37 M** (d = 3 and d = 5; the code default D = 96 gives 0.80 M but trained at ~2/3 the speed) |
| optimiser | Lamb, wd 1e-5, b2 0.95, batch 256 → 1024, piecewise-constant LR (2.45–3.46e-4) after 10k warm-up | AdamW (no Lamb in MLX), b2 0.95, wd 1e-5, linear warm-up + cosine, grad-norm clip 1 |
| pretraining | ≤ 2 × 10⁹ samples of the pij DEM of the training half, R uniform in {1, 3, …, 25}, noise curriculum f = 0.5 → 1 | FastSampler samples of the same pij DEMs (DEM → equivalent circuit, §3), same noise curriculum (eq. 6–7, f ∈ {0.5, …, 1.0}) plus a rounds curriculum (R_max 7 → 25), R ∈ {3, …, 25}; 1.25 M samples (§4) |
| fine-tuning | training half: 19,880 shots per experiment train, 5,120 dev; wd 0.08 towards the pretrained weights; ≤ 30k steps (~120 epochs) | same split; decoupled wd 0.02 towards the pretrained weights; 0.5 epoch (§4) |
| model selection | EMA parameters (1e-4); lowest dev LER fitted over R = 3…25 | EMA (1e-3, warmed up) or raw weights, whichever has the lower dev LER (same fit, first 512 dev shots per experiment) |
| ensembling | 20 seeds (−0.03 / −0.08 ×10⁻² LER at d = 3 / 5) | none by default |

Implementation notes. MLX has no Lamb; we use AdamW. The recurrence is a Python loop over rounds with
gradient checkpointing per round (`nn.utils.checkpoint`, activation memory O(R · B · S · D) instead of
O(R · layers · …)) and the whole training step is `mx.compile`d (one trace per round count).

**A bug worth recording.** The first version wrapped the round step in a bare `mx.checkpoint(self._step)`.
MLX's `mx.checkpoint` only differentiates with respect to the function's explicit inputs: gradients
with respect to the module parameters the closure captures are silently dropped. The loss and
forward pass were exactly right, but the bulk-round embedding got **zero** gradient and the transformer
layers learned only from the final round. Three GPU runs (≈ 3.3 M samples) trained like this
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
| d5 fine-tune, lr 3e-4 | real training half, batch 80 | 0.12 M (stopped) | 20 min | ~170 | 1.60 GB | 12–16 % (wrecked; discarded) |
| d5 fine-tune, lr 5e-5 | real training half, batch 80 | 0.35 M | 42 min | ~170 | 1.60 GB | 5.51 % (best at 0.1 M) |
| d5 held-out evaluation | odd shots, 0.6 M | – | 12 min | 850 | | 5.554 % (test) |
| Willow d3 fine-tune | from the Sycamore d3 model; Willow even shots, r ∈ {10, 13} (0.72 M), batch 256, lr 2e-4 | 1.02 M | 42 min | ~680 | 1.24 GB | (dev fit too noisy, §5.3) |
| Willow d3 evaluations (zero-shot, fine-tuned) | odd shots, 10 k per configuration, r ∈ {10, 13, 30, 50} | – | 12 + 9 min | | | §5.3 |

Total for the reported d = 3 model: 2.2 M training samples, 84 GPU-minutes. AlphaQubit's Sycamore
models: up to 2 × 10⁹ pretraining samples + ~120 fine-tuning epochs per model, ×20 ensemble members,
a model ≥ 15× larger (the 256-dim scaling model alone has 5.4 M parameters; the Sycamore model is 320-dim), on TPUs. We are ~10³ below the paper in samples and ~10⁴–10⁵ below it in
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
- Tesseract (beam 15, 16 detector orders, pij DEM) on the first 5,000 held-out shots of every
  experiment (paired subset of 0.8 M shots): **3.112 % [3.054, 3.176], ratio to TN 1.010 [0.997,
  1.026]** — statistically tied with the tensor network and with belief matching (1.009). On the same
  subset AlphaQubit-lite is 3.532 % (1.147), BP+OSD 3.366 % (1.093), correlated PyMatching 3.429 %
  (1.113), PyMatching 3.878 % (1.259).

**Held-out comparison, d = 5 (odd shots, 2 bases × 12 round counts × 25,000 = 0.6 M shots).**

| decoder | LER (fit R = 3…25) | 95 % CI | ratio vs TN | ε at R = 3 | ε at R = 25 |
|---|---|---|---|---|---|
| **AlphaQubit-lite (this work)** | **5.554 %** | **[5.403, 5.716]** | **1.865 [1.813, 1.926]** | **2.64 %** | **5.17 %** |
| PyMatching (shipped) | 4.454 % | [4.365, 4.557] | 1.495 [1.459, 1.535] | 2.74 % | 4.29 % |
| PyMatching 2.4, pij DEM (ours) | 4.388 % | [4.303, 4.496] | 1.473 [1.442, 1.514] | 2.67 % | 4.16 % |
| correlated matching (shipped) | 3.614 % | [3.552, 3.679] | 1.213 [1.192, 1.239] | 2.42 % | 3.44 % |
| PyMatching 2.4 correlated (ours) | 3.529 % | [3.467, 3.588] | 1.185 [1.162, 1.207] | 2.30 % | 3.38 % |
| belief matching (shipped) | 3.086 % | [3.039, 3.131] | 1.036 [1.019, 1.052] | 2.07 % | 2.97 % |
| tensor network (shipped) | 2.978 % | [2.936, 3.024] | 1.000 | 1.92 % | 2.88 % |
| AlphaQubit, paper | 2.748 ± 0.015 % | | 0.943 (vs paper's TN) | | |

BP+OSD-CS (order 10, ours) on the first 5,000 shots per experiment (paired 120 k-shot subset):
3.803 % [3.654, 3.982], ratio 1.249 — worse than correlated matching at d = 5 although better at
d = 3. Tesseract at d = 5: §5.2.

At d = 5 AlphaQubit-lite **loses to PyMatching** (5.55 % vs 4.39 %). It only had 0.67 M d = 5
samples (0.32 M pretraining warm-started from the d = 3 model, 0.35 M fine-tuning, of which the best
checkpoint saw 0.1 M) at 180–330 samples/s; it matches PyMatching on R = 3 experiments and falls
behind as R grows. Two observations from the runs: (i) fine-tuning at the d = 3 learning rate
(3e-4, batch 80) wrecked the pretrained model within 0.1 M samples (dev 5.7 % → 12–16 %); lr 5e-5
was stable but gained little (5.68 → 5.51 %); (ii) the released d = 5 pij DEMs produce 9 % more
detection events than the real device (d = 3: 2–6 %), and Sycamore d = 5 sits at threshold
(Λ₃/₅ ≈ 1.04), so DEM-simulated d = 5 data is substantially harder than the real data (simulated-dev
LER 13 % vs real-dev 5.7 % for the same model).

### 5.2 Tesseract at d = 5 (Sycamore)

Tesseract (beam 15, pij DEM) is slow on d = 5 × 25 rounds on the shared VPS (up to 19 min per 1,000
shots), so it ran on the first 1,000–2,000 held-out shots per experiment (paired 29 k-shot subset):
**3.305 % [3.061, 3.639], ratio to TN 1.047 [0.973, 1.139]** — statistically tied with the tensor
network and belief matching (1.012) and clearly better than correlated matching (1.175), BP+OSD
(1.230) and PyMatching (1.455). AlphaQubit-lite on the same subset: 5.47 % (1.73).

### 5.3 Willow 2024 (105-qubit processor, d = 3: 9 patches, d = 5: 4 patches)

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
0.622 vs 0.622 %), so the two pipelines agree.

**AlphaQubit-lite on Willow d = 3** (held-out odd shots, first 10,000 per configuration, 18
patch × basis datasets × r ∈ {10, 13, 30, 50} = 720 k shots, every decoder on the same shots; 95 % CIs
and ratios from the paired bootstrap; reference = Harmony with the RL-optimised prior, the best
decoder in the archive).

| decoder | mean ε, r = 10/13/30 | ratio vs Harmony-RL | ε r = 10 | ε r = 13 | ε r = 30 | ε r = 50 |
|---|---|---|---|---|---|---|
| PyMatching 2.4 (SI1000) | 0.993 % [0.985, 1.000] | 1.390 | 0.910 | 0.950 | 1.118 | 0.979 |
| PyMatching 2.4 correlated (SI1000) | 0.865 % [0.857, 0.872] | 1.211 | 0.798 | 0.834 | 0.962 | 0.854 |
| correlated matching (SI1000), shipped | 0.817 % [0.809, 0.824] | 1.144 | 0.755 | 0.787 | 0.908 | 0.802 |
| **AlphaQubit-lite, zero-shot (Sycamore-trained only)** | **0.807 % [0.800, 0.814]** | **1.131** | 0.725 | 0.761 | 0.936 | 0.948 |
| **AlphaQubit-lite, fine-tuned on Willow** | **0.760 % [0.754, 0.766]** | **1.065 [1.057, 1.072]** | **0.679** | **0.710** | 0.891 | 0.865 |
| Harmony (SI1000), shipped | 0.757 % [0.751, 0.764] | 1.061 | 0.703 | 0.732 | 0.838 | 0.745 |
| correlated matching (RL prior), shipped | 0.739 % [0.733, 0.745] | 1.036 | 0.678 | 0.707 | 0.834 | 0.728 |
| Harmony (RL prior), shipped | 0.714 % [0.708, 0.720] | 1.000 | 0.658 | 0.684 | 0.799 | 0.701 |
| Tesseract (SI1000), arXiv:2609.04557 (all shots) | 0.738 % | | | | | |
| BeliefMatching (SI1000), arXiv:2609.04557 | 0.815 % | | | | | |

- **Zero-shot transfer works.** The d = 3 model trained only on Sycamore 2022 (DEM samples + real
  Sycamore shots: a different chip with ≈ 4× higher logical error per round and a different circuit variant), with
  10 of the 18 Willow context embeddings untrained, already beats PyMatching by 19 % and edges out
  Google's correlated matching with the SI1000 prior. The canonical layout (§1) is what makes this
  possible.
- **Fine-tuning** (one 42-min run, 1.0 M samples ≈ 1.4 epochs of the Willow training half,
  r ∈ {10, 13} only, lr 2e-4) brings it level with Harmony-SI1000 (a 101-member matching ensemble) and
  equal (within 0.5 % relative) to correlated matching with the RL-optimised prior at the trained round counts
  (r = 10: 0.679 vs 0.678 %; r = 13: 0.710 vs 0.707 %). It generalises less well to r = 30 and 50,
  which it never saw (it is 7–23 % above the RL-prior decoders there).
- The paper-style log-fidelity *fit* over r = 10…50 is unkind to the network (0.922 % vs Harmony-RL
  0.723 %): on Willow every decoder is worse at r = 30 than at r = 50 (the configurations were taken
  separately), and the network's long-r degradation dominates a 4-point fit. We therefore quote the
  fixed-round numbers (the convention of the dataset release and of arXiv:2609.04557).
- Model selection on the Willow dev split used a two-point fit (r = 10, 13), which is too noisy to rank
  checkpoints; we report the final weights. A fixed-round dev metric would be the right choice.
- Willow d = 5 was not trained (time). The Willow paper's own neural decoder (an AlphaQubit
descendant, ref. [27] there) is not in the archive; it reports ε₇ = 0.143 % and Λ = 2.14 on this
device.

## 6. What limits further gains

1. **Samples.** Every learning curve we have is still falling when the 42-min cap hits (d = 3 dev LER
   4.6 → 4.0 → 3.7 → 3.55 % across the last pretraining evaluations, fine-tuning 3.55 → 3.35 %). The
   paper used ≤ 2 × 10⁹ pretraining samples per model and ~120 fine-tuning epochs; we used 1.25 M and
   0.5 epochs. On the M1 Pro under a 1.2 GB MLX cap we get 500–1,000 samples/s at d = 3 and 200–330 at
   d = 5, so a paper-scale run is ~10⁶ GPU-seconds — weeks, not minutes. At d = 5 this is decisive.
2. **Long experiments.** The model is near-optimal at R = 3 and degrades with R (fit intercepts
   F₀ = 1.06–1.15). The rounds curriculum front-loads short experiments; with the sample budget fixed,
   more R = 25 data (and the paper's per-round auxiliary labels, unavailable for DEM samples) is the
   obvious lever.
3. **Model size.** D = 64 vs the paper's 320, a 4-layer instead of 16-layer readout, a 2-layer
   attention-bias ResNet instead of 8. Under the 2 GB memory rule a larger model also means a smaller
   batch; we have not measured the trade-off.
4. **Pretraining prior.** The released pij DEMs over-predict the real detection density (d = 3:
   +2–6 %, d = 5: +9 %), which at d = 5 (Λ ≈ 1.04) makes simulated data much harder than the real data.
   The paper's XEB-DEM pretraining and fine-tuning on many more real epochs mitigate this.
5. **Ensembling.** The paper's numbers are 20-model ensembles (−0.03 / −0.08 percentage points at
   d = 3 / 5); ours are single models.
6. **Throughput engineering.** The recurrence is a Python loop of small kernels; `mx.compile` gave
   only ~10 % here. A fused per-round kernel, bf16, or a larger effective batch via gradient
   accumulation are untried.

## 7. Reproducing

```
# Sycamore (315 MB)
curl -L -o syc.zip https://zenodo.org/api/records/6804040/files/google_qec3v5_experiment_data.zip/content
unzip syc.zip 'surface_code*' -d syc
# Willow d3/d5 subset (112 MB of the 5.7 GB archive, by HTTP range requests)
python remote_zip.py https://zenodo.org/api/records/13273331/files/google_105Q_surface_code_d3_d5_d7.zip/content \
   get 'd[35]_at_[^/]+/[XZ]/r(01|10|13|30|50)/(detection_events|obs_flips_actual|circuit_ideal|circuit_noisy_si1000|metadata|decoding_results/.*(obs_flips_predicted|error_model))' willow
python test_aq.py syc <nd_tool>                 # layout, gradient and sampler-equivalence tests
python baselines_real.py syc out/bl --folds odd --decoders shipped:pymatching,shipped:correlated_matching,shipped:belief_matching,shipped:tensor_network_contraction,pm,pmcorr
python baselines_real.py syc out/bl --folds odd --decoders tess:15 --tess-shots 5000     # and bposd:10 (needs nd_tool)
python baselines_real.py willow out/blw --folds odd --rounds 10,13,30,50 --decoders shipped:...,pm,pmcorr
# pretraining data (FastSampler, pij DEMs of the even half) + packed real shots
python aq_gen.py syc data/d3 --d 3 --shots 102400
python aq_gen.py syc data/d3 --d 3 --shots 51200 --scales 0.5,0.6,0.7,0.8,0.9 --no-real
# d = 3 (MLX GPU; AQ_MEM_GB=1.2)
python aq_train.py data/d3 syc runs/d3pre --mode pretrain --d 3 --D 64 --conv 32 --key 16 \
    --scales 0.5,0.6,0.7,0.8,0.9,1.0 --curr-tc 400000 --curr-sc 2 --rmax0 7 --rcurr 800000 \
    --steps 6400 --warmup 400 --lr 5e-4 --batch 256 --eval-every 1000 --dev-max 512 --max-minutes 42
python aq_train.py data/d3 syc runs/d3ft --mode finetune --d 3 --init runs/d3pre --wd-anchor 0.02 \
    --steps 5000 --warmup 200 --lr 2e-4 --batch 256 --eval-every 1250 --dev-max 512 --max-minutes 42
python aq_train.py data/d3 syc runs/d3ft_test --mode eval --d 3 --init runs/d3ft --eval-bs 1024
# d = 5: --init-partial runs/d3ft, batch 80, --curr-tc 120000 --rcurr 250000; fine-tune lr 5e-5
# Willow d = 3: aq_gen.py willow data/w3 --d 3 --shots 0 --rounds 10,13,30,50, then
python aq_train.py data/w3 willow runs/w3ft --mode finetune --d 3 --init runs/d3ft --rounds 10,13 \
    --eval-rounds 10,13 --steps 4000 --lr 2e-4 --batch 256 --eval-at-start --max-minutes 42
python aq_train.py data/w3 willow runs/w3ft_test --mode eval --d 3 --init runs/w3ft --weights last.safetensors \
    --rounds 10,13 --eval-rounds 10,13,30,50 --test-max 10000
python compare.py syc 3 odd shipped_tensor_network_contraction <decoders>,nn out/bl runs/d3ft_test
python compare.py willow 3 odd shipped_harmony_decoder_with_rl_optimized_prior <decoders>,nn out/blw runs/w3ft_test --mean-rounds 10,13,30
```
