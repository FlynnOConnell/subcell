"""Test each stashed localize.py change in isolation to find which one
causes the subproblem count to explode from ~6 to ~48.

Loads registered data, runs localize_sources with variants, and reports
peak counts and subproblem counts for each.

Usage:
    python examples/test_localize_changes.py
"""

from __future__ import annotations

import logging
import math
import time
import warnings

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.ndimage import label as ndimage_label

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_localize")

# ── Data path ──────────────────────────────────────────────────────────
import sys

_scan = sys.argv[1] if len(sys.argv) > 1 else "scan_00001_20240924_110500"
from _paths import get_dir_scan

data_dir = get_dir_scan(_scan)
zarr_path = data_dir / "registered.zarr"

# ── Load data once ─────────────────────────────────────────────────────
from subcell.config import ExtractionConfig
from subcell.io.zarr_store import ExperimentStore

store = ExperimentStore(zarr_path)
adata = store.load_alignment_data(1)
reg_ds = store.load_registered_ds(1)

n_ch = adata.num_channels
n_ds_frames = reg_ds.shape[2] // n_ch
movie_4d = reg_ds.reshape(
    reg_ds.shape[0], reg_ds.shape[1], n_ds_frames, n_ch
).transpose(0, 1, 3, 2)

ext_config = ExtractionConfig(
    microscope="bergamo",
    sigma_px=1.33,
    nmf_iter=2,
    dXY=3,
    denoise_window_s=0.2,
    baseline_window_glu_s=4.0,
    tau_s=0.03,
    max_synapse_density=0.01,
    motion_thresh=2.5,
    nan_thresh=0.33,
    activity_channel=2,
)

act_ch = ext_config.activity_channel - 1
movie_act = movie_4d[:, :, act_ch, :].copy()
align_hz = adata.align_hz

logger.info("Movie shape: %s, align_hz=%.1f", movie_act.shape, align_hz)


# ── Helper: count subproblems from peaks ───────────────────────────────
def count_subproblems(peaks_row, peaks_col, h, w, sigma_px):
    """Simulate extract_trial's connected component decomposition."""
    if len(peaks_row) == 0:
        return 0, []
    sigma_radius = int(np.ceil(1.5 * sigma_px + 1))
    zones = np.zeros((h, w), dtype=bool)
    for r, c in zip(peaks_row, peaks_col):
        ri, ci = int(round(r)), int(round(c))
        if 0 <= ri < h and 0 <= ci < w:
            zones[ri, ci] = True
    y, x = np.mgrid[-sigma_radius : sigma_radius + 1, -sigma_radius : sigma_radius + 1]
    disk = (x**2 + y**2) <= sigma_radius**2
    zones = binary_dilation(zones, structure=disk)
    labeled, n_problems = ndimage_label(zones)

    # Count sources per subproblem
    sources_per = []
    for p in range(1, n_problems + 1):
        mask = labeled == p
        count = sum(
            1
            for r, c in zip(peaks_row, peaks_col)
            if 0 <= int(round(r)) < h
            and 0 <= int(round(c)) < w
            and mask[int(round(r)), int(round(c))]
        )
        sources_per.append(count)
    return n_problems, sorted(sources_per, reverse=True)


# ── Localization variants ──────────────────────────────────────────────
from subcell.filters.morphology import (
    apply_density_threshold,
    local_maxima_3x3,
    spatiotemporal_nms,
)
from subcell.filters.spatial import difference_of_gaussians, nanmedfilt2
from subcell.filters.temporal import (
    exponential_matched_filter,
    moving_mad_noise,
    moving_mean,
    moving_median_baseline,
)


def run_localize(
    movie,
    config,
    align_hz,
    *,
    do_interp=True,
    do_denoise=True,
    do_median_fill=True,
    do_nms_normalize=False,
    label="",
):
    """Run localization with toggleable changes.

    do_interp:      Fill NaN gaps via temporal interpolation (current=True, stash=False)
    do_denoise:     True: overwrite IMf with smoothed (current behavior)
                    "smooth_baseline": separate IMfden copy, baseline from smoothed,
                        subtract from raw, MAD from smoothed residual × denoiseWindow
                        (matches current MATLAB)
                    "smooth_baseline_sqrt": same but MAD × sqrt(denoiseWindow)
                    "smooth_baseline_noscale": same but no MAD scaling
                    False: no smoothing at all (stash smooth_baseline=False)
    do_median_fill: Fill NaN in activity image with median (current=True, stash=False)
    do_nms_normalize: Pass nan_mask to NMS for obs normalization (current=False, stash=True)
    """
    tau = config.tau_s * align_hz
    sigma = config.sigma_px
    baseline_window = int(math.ceil(config.baseline_window_glu_s * align_hz))
    denoise_window = int(math.ceil(config.denoise_window_s * align_hz))
    n_time = movie.shape[2]
    nan_mask = np.isnan(movie)

    nan_frac_spatial = np.mean(nan_mask, axis=2)
    valid = nan_frac_spatial < config.nan_thresh

    IMf = movie.copy()
    invalid_3d = np.broadcast_to(~valid[:, :, np.newaxis], IMf.shape)
    IMf[invalid_3d] = np.nan
    nan_mask = np.isnan(IMf)

    # Change 1: temporal interpolation
    if do_interp:
        shape = IMf.shape
        flat = IMf.reshape(-1, shape[2])
        nan_frac_px = np.mean(np.isnan(flat), axis=1)
        incomplete = (nan_frac_px > 0) & (nan_frac_px < 1)
        if np.any(incomplete):
            smoothed = moving_mean(flat[incomplete], window=baseline_window, axis=1)
            nans_flat = np.isnan(flat[incomplete])
            flat[incomplete] = np.where(nans_flat, smoothed, flat[incomplete])
        IMf = flat.reshape(shape)

    # Change 2: denoise before baseline
    if do_denoise in (
        "smooth_baseline",
        "smooth_baseline_sqrt",
        "smooth_baseline_noscale",
    ):
        # MATLAB-match: baseline from smoothed copy, subtract from raw,
        # MAD from smoothed residual scaled by denoiseWindow
        IMfden = moving_mean(IMf, window=denoise_window, axis=2)
        IMb = moving_median_baseline(IMfden, window=baseline_window, axis=2)
        IMf = IMf - IMb  # subtract from RAW
        std_IM = moving_mad_noise(IMfden - IMb, window=baseline_window, axis=2)
        if do_denoise == "smooth_baseline":
            std_IM = std_IM * denoise_window
        elif do_denoise == "smooth_baseline_sqrt":
            std_IM = std_IM * np.sqrt(denoise_window)
        # else: no scaling
        std_IM = np.maximum(std_IM, 1e-10)
        del IMfden, IMb
    else:
        if do_denoise:
            IMf = moving_mean(IMf, window=denoise_window, axis=2)
        IMb = moving_median_baseline(IMf, window=baseline_window, axis=2)
        IMf = IMf - IMb
        del IMb
        std_IM = moving_mad_noise(IMf, window=baseline_window, axis=2)
        std_IM = np.maximum(std_IM, 1e-10)

    IMf = IMf / std_IM
    del std_IM

    IMf = exponential_matched_filter(IMf, tau)
    IMf[nan_mask] = np.nan
    IMf = difference_of_gaussians(IMf, sigma, nan_mask=nan_mask)

    # Change 3: NMS normalization
    nms_mask = nan_mask if do_nms_normalize else None
    activity_image, _ = spatiotemporal_nms(IMf, tau, nan_mask=nms_mask)
    del IMf

    activity_image[~valid] = np.nan

    # Change 4: median fill
    if do_median_fill:
        nan_vals = np.isnan(activity_image)
        med_val = np.nanmedian(activity_image) if np.any(~nan_vals) else 0.0
        activity_image[nan_vals] = med_val

    med_filt = nanmedfilt2(activity_image, 5)
    activity_image = activity_image - med_filt
    activity_image[~valid] = np.nan

    peak_mask = local_maxima_3x3(activity_image)
    rows, cols, vals = apply_density_threshold(
        activity_image,
        peak_mask,
        config.max_synapse_density,
        n_time_points=n_time,
        align_hz=align_hz,
        valid_mask=valid,
    )

    return rows, cols, vals


# ── Run all variants ───────────────────────────────────────────────────
h, w = movie_act.shape[:2]

variants = [
    # (label, kwargs)
    (
        "CURRENT (baseline)",
        {
            "do_interp": True,
            "do_denoise": True,
            "do_median_fill": True,
            "do_nms_normalize": False,
        },
    ),
    (
        "STASH (all changes)",
        {
            "do_interp": False,
            "do_denoise": False,
            "do_median_fill": False,
            "do_nms_normalize": False,
        },
    ),
    (
        "Change 1 only: remove interpolation",
        {
            "do_interp": False,
            "do_denoise": True,
            "do_median_fill": True,
            "do_nms_normalize": False,
        },
    ),
    (
        "Change 2 only: remove denoise",
        {
            "do_interp": True,
            "do_denoise": False,
            "do_median_fill": True,
            "do_nms_normalize": False,
        },
    ),
    (
        "Change 3 only: add NMS normalization",
        {
            "do_interp": True,
            "do_denoise": True,
            "do_median_fill": True,
            "do_nms_normalize": True,
        },
    ),
    (
        "Change 4 only: remove median fill",
        {
            "do_interp": True,
            "do_denoise": True,
            "do_median_fill": False,
            "do_nms_normalize": False,
        },
    ),
    (
        "Changes 1+2: remove interp + denoise",
        {
            "do_interp": False,
            "do_denoise": False,
            "do_median_fill": True,
            "do_nms_normalize": False,
        },
    ),
    (
        "Changes 1+2+4: remove interp + denoise + median fill (full stash minus NMS)",
        {
            "do_interp": False,
            "do_denoise": False,
            "do_median_fill": False,
            "do_nms_normalize": False,
        },
    ),
    (
        "MATLAB-MATCH: interp + smooth_baseline + NMS norm + no median fill",
        {
            "do_interp": True,
            "do_denoise": "smooth_baseline",
            "do_median_fill": False,
            "do_nms_normalize": True,
        },
    ),
    (
        "MATLAB-MATCH + no interp (NaN-aware DoG)",
        {
            "do_interp": False,
            "do_denoise": "smooth_baseline",
            "do_median_fill": False,
            "do_nms_normalize": True,
        },
    ),
    (
        "smooth_baseline + NMS norm + interp + median fill",
        {
            "do_interp": True,
            "do_denoise": "smooth_baseline",
            "do_median_fill": True,
            "do_nms_normalize": True,
        },
    ),
    (
        "smooth_baseline only (keep interp, median fill, no NMS norm)",
        {
            "do_interp": True,
            "do_denoise": "smooth_baseline",
            "do_median_fill": True,
            "do_nms_normalize": False,
        },
    ),
    (
        "NMS norm only (keep interp, denoise, median fill)",
        {
            "do_interp": True,
            "do_denoise": True,
            "do_median_fill": True,
            "do_nms_normalize": True,
        },
    ),
    (
        "PROPOSED: smooth_baseline + interp + median fill (no NMS norm)",
        {
            "do_interp": True,
            "do_denoise": "smooth_baseline",
            "do_median_fill": True,
            "do_nms_normalize": False,
        },
    ),
    (
        "smooth_baseline_sqrt + NMS norm + interp + median fill",
        {
            "do_interp": True,
            "do_denoise": "smooth_baseline_sqrt",
            "do_median_fill": True,
            "do_nms_normalize": True,
        },
    ),
    (
        "smooth_baseline_noscale + NMS norm + interp + median fill",
        {
            "do_interp": True,
            "do_denoise": "smooth_baseline_noscale",
            "do_median_fill": True,
            "do_nms_normalize": True,
        },
    ),
]

print("\n" + "=" * 80)
print("LOCALIZE.PY CHANGE ISOLATION TEST")
print("=" * 80)
print(f"{'Variant':<58s} {'Peaks':>6s} {'Subprobs':>9s} {'Max/sub':>8s}")
print("-" * 80)

for label, kwargs in variants:
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rows, cols, vals = run_localize(
            movie_act, ext_config, align_hz, label=label, **kwargs
        )
    elapsed = time.perf_counter() - t0

    n_peaks = len(rows)
    n_sub, sources_per = count_subproblems(rows, cols, h, w, ext_config.sigma_px)
    max_per = sources_per[0] if sources_per else 0

    print(f"  {label:<56s} {n_peaks:>6d} {n_sub:>9d} {max_per:>8d}  ({elapsed:.1f}s)")
    if sources_per:
        print(f"    sources/subproblem: {sources_per}")

print("=" * 80)
