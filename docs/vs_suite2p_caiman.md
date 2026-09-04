# GIAnT / subcell vs. Suite2p & CaImAn

Xie et al., bioRxiv `10.64898/2026.08.07.743580` (2026-08-13). `subcell` = Python port of GIAnT-MATLAB (Bergamo lineage).

## Regime

| | Suite2p / CaImAn | GIAnT / subcell |
|---|---|---|
| Target | somata, shafts, axons | single synapses |
| Size | 10-20 px | ~3-5 px, PSF sigma 1.33 px |
| Count / FOV | 10^2-10^4 | 10-50 |
| Rate | 1-30 Hz | 435 Hz |
| FOV | 512^2 | strip, 50 x 128 px @ 250 nm |
| Indicator | Ca2+, 100 ms-1 s | glutamate, tau 27 ms |
| Motion | << FOV | comparable to FOV |
| Overlap | separable | unresolvable, must demix |

Everything below follows from the last three rows.

## Motion correction

| | Suite2p | CaImAn (NoRMCorre) | StripRegistration |
|---|---|---|---|
| Template | FOV-sized, fixed | FOV-sized, fixed | canvas padded by `maxShift`, NaN outside |
| Update | batch | batch / per-patch | online, pixel used after >=100 obs |
| Max shift | < FOV | < FOV | **> FOV** |
| Sub-pixel | integer (rigid stage) | yes | DFT to 1/4 px |
| Big-motion fallback | - | - | NaN-aware normalized xcorr, +-50 px |
| Output canvas | crops in | crops in | **pads out**, NaN unobserved |
| Non-rigid | yes | yes | no (negligible here) |
| Z-motion | - | - | `RecNegErr` frame censoring |

- Init: NoRMCorre on first 1000 frames, seeded from a high-correlation frame cluster.
- Paper: lower RMSE across motion 0-2 um and 10x dimmer; better mCM + CMM in vivo.

## Detection

Competitors seed from correlation images — resolution-limited, sparse sources drown out. SILo localizes events instead.

- Baseline: moving median; noise from `sigma^2 = a*mu + s0^2`; z-score.
- Matched filter: spatial DoG (sigma, 5 sigma) x causal exponential (tau).
- Events = local maxima in x, y **and** t.
- Activity image = sum of squared peaks, high-passed. Peaks tighter than the PSF.
- Centers: paper = Gaussian-mixture fit; **subcell = iterative NMS + parabolic refine**.

| | Suite2p `sparsery` | CaImAn CNMF | SILo |
|---|---|---|---|
| Seed | sparse decomposition | correlation/PNR | event-density image |
| Uses timing to split | no | indirect | **yes** |
| Spatial | free mask | free, sparse | point source x known PSF |
| Overlap | `max_overlap` **discards** | free footprints | expected, identifiable |

## Traces

| | Suite2p | CaImAn | SILo |
|---|---|---|---|
| Model | `F - 0.7 Fneu` | `Y ~ AC + bf` | `Z ~ S theta + B` |
| Loss | subtraction | L2 | `(Z-Stheta-B)^2 / (Stheta+B+1)` (Poisson) |
| Temporal | OASIS, post hoc | AR(p) | `theta = e * exp(-t/tau)`, `e>=0`, L1, debias refit |
| Background | scalar neuropil | low-rank | per-pixel, time-varying |
| Out | `F`,`Fneu`,`spks` | `C`,`S`,`YrA` | denoised + least-squares trace |

- Neuropil annulus is wrong here: it contains the shaft and neighbors, i.e. signal.

## Missing data

- GIAnT is NaN-aware end to end.
- Benchmarks needed cropping: pixels observed >=35% (CaImAn), >=94% (Suite2p), gaps interpolated.

## Numbers

Same registered input, all three Bayesian-optimized on the default sim.

| | SILo | CaImAn | Suite2p |
|---|---|---|---|
| F1 | **0.759 +- .021** | 0.694 +- .018 *** | 0.637 +- .028 *** |
| Demix purity | **0.745 +- .010** | 0.717 +- .014 n.s. | 0.735 +- .008 *** |

- Margins modest; the claim is **stability** across density (10-50) and brightness (0.06-6 photons).
- Histology (expansion microscopy, 5 dendrites, 4 mice): PPV 0.73 +- 0.04. 45% of misses are splits (<0.75 um); 23/149 unmatched, mostly low SNR.

## Not in scope

- No non-rigid, no volumetric, no cell classifier, no curation GUI, no online, no 10^3-source FOVs.

## subcell deltas

- **Cross-trial align** + valid-trial cut at r 0.90; sources from the averaged activity image. Not in paper or competitors.
- PyTorch L-BFGS replaces `fmincon`; connected-component subproblems; CUDA optional.
- Zarr store (downsampled + full res); ScanImage reader with flyback removal.
- Peak localization differs from paper (NMS + parabolic, not Gaussian fit).
- Registration validated vs MATLAB; **extraction not formally validated**.
- Defaults: `sigma_px 1.33`, `tau_s 0.03`, `ds_time 3`, `maxshift 50`, `clip_shift 10`, `nan_thresh 0.33`, `motion_thresh 2.5`, `dXY 3`.
- Head-to-head: `subcell_vs_suite2p.ipynb`, `figures/s2p_*.png`. Traces agree (median r > 0.8) like-for-like; detection is where they part.
