# GIAnT paper — summary

Xie, Friedrich, Wirsching, Shibu, Seyedolmohadesin, Ouellette, Wang, Svoboda, Charles*, Podgorski*.
*GIAnT: a Glutamate Imaging Analysis Toolbox.* bioRxiv `10.64898/2026.08.07.743580`, posted 2026-08-13.
JHU BME + Allen Institute, Neural Dynamics. Preprint, not peer reviewed. CC-BY 4.0.

## Claim

- First validated automated pipeline for in vivo two-photon **glutamate** imaging.
- Two parts: **StripRegistration** (motion) + **SILo** (source extraction).
- Beats Suite2p and CaImAn on motion, detection, demixing. Validated against histology.

## Why cellular pipelines fail here

| Constraint | Consequence |
|---|---|
| Glutamate decay ~20 ms | must image >100 Hz |
| Speed via narrow strips (resonant scanner, fixed line rate) | few anatomical features to register on |
| Brain motion several um vs ~12 um strip | motion is **not** << FOV; objects leave the window |
| Synapses <1 fL, 1-2 per um | sources at the diffraction limit, optically overlapping |
| Sparse firing (0.1-10 Hz) | correlation summary images drown sparse sources |

## StripRegistration

- Cross-correlation, rigid, sequential, frame by frame.
- Optional temporal downsampling: average every `2^ds_time` frames.
- Init: NoRMCorre on first 1000 frames, seeded with the mean of a high-mutual-correlation cluster.
- Template = that mean, NaN-padded to `(X+2m) x (Y+2m)` — **larger than the FOV**.
- Per frame: max correlation vs the template crop at the previous shift, constrained to `+-clipShift`.
- Solved by matrix-multiply DFT (Guizar-Sicairos 2008) → **quarter-pixel**.
- If normalized shift magnitude > ~0.6: re-solve with NaN-aware **normalized** xcorr over +-50 px; quadratic sub-pixel peak. Shifts beyond `maxShift` rejected.
- Template updated online; a pixel contributes only after **>=100 observations** (censors undersampled pixels).
- Motion upsampled back to full rate by **PCHIP** spline.
- Output canvas `X + 2 max|sx|`; unobserved pixels NaN.
- **RecNegErr** per frame (10 s chunks, vs median of nearby frames) → out-of-plane motion proxy (Fig. S2).

## SILo

Detection + localization on downsampled frames; traces at full rate.

**1. Censoring / normalization**

- Drop frames where high-passed RecNegErr `phi > motionThresh`, plus neighbors within 25 ms above half-threshold; gaps <=50 ms also dropped.
- Keep pixels NaN in fewer than `nanThresh` of frames.
- Baseline `R0` = moving median (`baselineWindow_Glu_s`) of a moving-mean-smoothed trace (`denoiseWindow_s`).
- Noise: `var = alpha*R0 + sigma0^2`, fit on the first 500 valid frames; `sigma0^2` = 10th percentile of variances x `VIF`. Z-score.

**2. Event detection**

- Matched filter = spatial **DoG** (`sigma_px`, `5*sigma_px`) x temporal **exponential** (`tau_s`, time-reversed).
- DoG negative surround kills flat bright regions (the shaft).
- Candidate events = local maxima in x, y **and** t (6-neighbour).
- **Activity image** = sum of squared filtered values at those maxima, then spatially high-passed.
- Localization error is isotropic → peaks tighter than the PSF, tighter than NNCorr (Fig. 3a-b).

**3. Source localization**

- Fit the activity image as a sum of Gaussians (amplitude, center, width).
- Seeds: local maxima with a neighbor above `b + peakth*sigma_b` (`b`, `sigma_b` = median / robust SD of the image).
- Constraints: `a >= 0`, `sigma^2 <= 25`, center within 1.5 px of its seed.
- Iteratively add the largest residual peak, refit, stop when it falls below `b + peakth*sigma_b`.
- Drop peaks with low amplitude relative to spread.

**4. Trace extraction (full rate)**

- Convert to photons: `Z / photonScale` (estimated from variance/mean if not given).
- Constrained NMF: `Z ~ S theta + B`.

| Term | Constraint |
|---|---|
| Spatial `S` | `Ss * G_sigma` — point source convolved with the **known PSF**, `Ss >= 0` |
| Temporal `theta` | `e * exp(-t/tau)`, `e >= 0`, L1 penalty `lambda` |
| Background `B` | per-pixel, time-varying, `>= minBaseline` |
| Loss | `sum (Z - S theta - B)^2 / (S theta + B + 1)` — Poisson-weighted, not L2 |

- Coordinate descent, `nmfIter` iterations; then **debias** refit on the recovered support with `phi*lambda`.
- Outputs denoised `theta` **and** least-squares `theta_ls = (Z-B)' S (S'S)^-1`.

## Named parameters

| StripRegistration | SILo |
|---|---|
| `ds_time`, `maxShift`, `clipShift` | `motionThresh`, `nanThresh`, `VIF` |
| | `denoiseWindow_s`, `baselineWindow_Glu_s` |
| | `sigma_px`, `tau_s`, `peakth` |
| | `photonScale`, `lambda`, `phi`, `minBaseline`, `nmfIter` |

## Simulations

Built from real volumetric stacks; statistics matched to real recordings (Fig. S1).

| | Value |
|---|---|
| FOV / pixel | 125 x 45 px, 250 nm |
| Frame time | 2.3 ms (435 Hz) |
| Brightness (99th pct px, photons) | 0.06, 0.2, **0.6**, 2.0, 6.0 |
| Sources | 10, 20, **30**, 40, 50 |
| Motion sigma (um) | 0, 0.5, **1**, 1.5, 2 |
| Source shape | Gaussian sigma 0.33 um x local anatomy; >=0.5 voxel apart |
| Spiking | Bernoulli ~ Poisson; dF/F lognormal (mu 0, sigma 0.25); decay 27 ms |
| Motion | smooth 3D vector, X/Y/Z correlations from in vivo |
| Noise | Poisson shot + lognormal detector + Gaussian electronic |
| Other | photobleaching tau 1200 s |

Bold = the **default** condition, used to optimize parameters for all methods.

## Benchmarking rules

- All source-extraction methods fed the **same** StripRegistration output.
- Cropped for competitors: pixels observed >=35% (CaImAn), >=94% (Suite2p); gaps linearly interpolated.
- All three Bayesian-optimized on the default sim, then frozen.
- Suite2p used `sparsery` (beat `sourcery` throughout).
- Rigid stages only — non-rigid and rolling shutter negligible at this FOV/rate.

## Results

**Motion** — RMSE vs ground truth, plus mCM and CMM in vivo (Hattori & Komiyama 2022).

- Lower RMSE than Suite2p and CaImAn across motion sigma 0-2 um and brightness down to 10x dimmer.
- Largest margin on frames with >6 um ground-truth motion.
- Better mCM and CMM in vivo; both competitors show large-motion artifacts (Fig. 2d).

**Source extraction** — aggregated over all conditions, mean +- s.e.m.

| | SILo | CaImAn | Suite2p |
|---|---|---|---|
| F1 | **0.759 +- 0.021** | 0.694 +- 0.018 (p<0.001) | 0.637 +- 0.028 (p<0.001) |
| Demixing purity | **0.745 +- 0.010** | 0.717 +- 0.014 (p=0.734) | 0.735 +- 0.008 (p<0.001) |

- Paired Wilcoxon signed-rank. Purity vs CaImAn is **not** significant.
- Main claim is **stability** across density and brightness, not the aggregate margin.
- F1 matching: linear sum assignment on spatial cosine similarity, >=0.5 required.

**In vivo tuning** — 1158 sources, 31 recordings, drifting gratings.

- Median OSI 0.412 (IQR 0.246-0.579).
- 417 sources with OSI > 0.5; preferred orientations span all angles.

**Histology validation** — same-day in vivo → perfusion → LICONN expansion microscopy, Bassoon / PSD-95 / iGluSnFR, lightsheet. 4 mice, 5 branches, blind annotation with control centroids.

| | Value |
|---|---|
| PPV (one-to-one matched) | **0.73 +- 0.04** |
| Near matches among the rest (<0.75 um) | 45% → splitting |
| Unmatched | 23 / 149 (15.4%) |
| Unmatched, peak SNR < 3 | 21 (18.3%) |
| Unmatched, peak SNR > 3 | 2 (5.9%) |

## Setup

| | |
|---|---|
| Indicator | iGluSnFR4f; AAV1-hSyn-FLEX-iGluSnFR4f + AAV9-CamKII-Cre, VISp 300 um |
| Scope | Bergamo II (Thorlabs), 12 kHz resonant, ScanImage, Olympus 25x 1.0 NA |
| Laser / detection | 1030 nm InSight X3; GaAsP PMT, 525/50 |
| Imaging | ~440 Hz, 50 lines x 128 px, 10x zoom; L2/3 visual cortex, awake |
| Stimulus | drifting gratings, 8 directions, 0.03 cyc/deg, 2 Hz, 2 s on / 1 s gray |
| Stacks | 21 planes, 0.5 um spacing |

## Stated limits & future work

- Complex synaptic structures (large or multiple marker patches) — source count ambiguous even in histology.
- Anatomy is deliberately unused; the authors note it could be integrated as complementary information.
- Sequential template growth is "in principle compatible with online" processing; decentralized methods (DREDge, Varol) suggested as an alternative.
- Generalizes to non-image modalities (SLAP2 band scanning) — only needs an approximate PSF.
- Validation datasets released as a benchmark for the field, mirroring NAOMi for calcium.

## Availability

- MATLAB: `github.com/AllenNeuralDynamics/GIAnT-MATLAB`.
- Python implementation "forthcoming" — this repo (`subcell`) is that lineage.
- Data: aind-open-data S3 bucket; **the link is a literal `(MISSING)` placeholder in the preprint**.
