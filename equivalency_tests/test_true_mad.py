"""Test true per-window MAD using numpy sliding_window_view, verify it fixes correlation."""

import math
import time
import warnings

import numpy as np
import scipy.io as sio
from _paths import get_dir_scan

from subcell.filters.morphology import (
    apply_density_threshold,
    local_maxima_3x3,
    spatiotemporal_nms,
)
from subcell.filters.spatial import difference_of_gaussians, nanmedfilt2
from subcell.filters.temporal import (
    exponential_matched_filter,
    moving_mean,
    moving_median_baseline,
)
from subcell.io.zarr_store import ExperimentStore

data_dir = get_dir_scan()

MAD_TO_STD = 0.6741891400433162


def true_movmad_chunked(data_3d, window, axis=2):
    """Compute per-window MAD matching MATLAB's movmad exactly.

    For each window position, computes median(|window - median(window)|).
    Processes pixel-by-pixel with numpy operations.
    """
    import pandas as pd

    if axis != 2:
        data_3d = np.moveaxis(data_3d, axis, 2)

    _, _, n_time = data_3d.shape
    half_w = window // 2
    filter_w = 2 * half_w + 1
    result = np.full_like(data_3d, np.nan)

    flat = data_3d.reshape(-1, n_time)
    out = result.reshape(-1, n_time)
    n_pixels = flat.shape[0]

    t0 = time.time()
    # Use pandas rolling.apply with the true MAD function
    for i in range(n_pixels):
        ts = flat[i]
        if np.all(np.isnan(ts)):
            continue
        s = pd.Series(ts)
        mad = s.rolling(filter_w, center=True, min_periods=1).apply(
            lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True
        )
        out[i] = mad.values

        if i > 0 and i % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (n_pixels - i)
            print(
                f"  {i}/{n_pixels} pixels, {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"
            )

    if axis != 2:
        result = np.moveaxis(result, 2, axis)
    return result


# Load data
store = ExperimentStore(data_dir / "registered.zarr")
adata = store.load_alignment_data(1)
reg_ds = store.load_registered_ds(1)
n_ch = adata.num_channels
n_ds_frames = reg_ds.shape[2] // n_ch
movie_4d = reg_ds.reshape(
    reg_ds.shape[0], reg_ds.shape[1], n_ds_frames, n_ch
).transpose(0, 1, 3, 2)
IMf = movie_4d[:, :, 1, :].copy()

align_hz = adata.align_hz
tau = 0.03 * align_hz
sigma = 1.33
baseline_window = int(math.ceil(4.0 * align_hz))
denoise_window = int(math.ceil(0.2 * align_hz))
n_time = IMf.shape[2]

nan_frac = np.mean(np.isnan(IMf), axis=2)
valid = nan_frac < 0.33
IMf[np.broadcast_to(~valid[:, :, np.newaxis], IMf.shape)] = np.nan
nan_mask_orig = np.isnan(IMf)

print("Computing smoothing and baseline...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    IMfden = moving_mean(IMf, window=denoise_window, axis=2)
    IMb = moving_median_baseline(IMfden, window=baseline_window, axis=2)

IMf_hp = IMf - IMb
residual = IMfden - IMb

print("Computing true per-window MAD (this will take a while)...")
t0 = time.time()
std_true = (
    true_movmad_chunked(residual, window=baseline_window) / MAD_TO_STD * denoise_window
)
print(f"Done in {time.time() - t0:.0f}s")

del IMfden, IMb, residual

# Z-score with true MAD
with np.errstate(divide="ignore", invalid="ignore"):
    IMf_z = IMf_hp / std_true
non_finite = ~np.isfinite(IMf_z)
IMf_z[non_finite] = np.nan
nan_mask = nan_mask_orig | non_finite
del IMf_hp, std_true, non_finite

# Matched filter
IMf_mf = exponential_matched_filter(IMf_z, tau)
IMf_mf[nan_mask] = np.nan
del IMf_z

# DoG
IMf_dog = difference_of_gaussians(IMf_mf, sigma, nan_mask=nan_mask)
del IMf_mf

# NMS
activity_image, _ = spatiotemporal_nms(IMf_dog, tau, nan_mask=nan_mask)

# Post-process
activity_image[~valid] = np.nan
med_filt = nanmedfilt2(activity_image, 5)
activity_image = activity_image - med_filt
activity_image[~valid] = np.nan

# Compare with MATLAB
mat6 = sio.loadmat(str(data_dir / "matlab_step6_activity.mat"))
mat_raw = mat6["step6_activity_raw"]
mat_final = mat6["step6_activity_final"]

print("\n=== Results with true per-window MAD ===")
vb = ~np.isnan(activity_image) & ~np.isnan(mat_final)
corr = np.corrcoef(activity_image[vb], mat_final[vb])[0, 1]
print(f"Activity (final) correlation: {corr:.4f}")

from scipy.stats import spearmanr

rho, _ = spearmanr(activity_image[vb], mat_final[vb])
print(f"Spearman rank correlation: {rho:.4f}")

print(
    f"Python range: [{np.nanmin(activity_image):.6f}, {np.nanmax(activity_image):.6f}]"
)
print(f"MATLAB range: [{np.nanmin(mat_final):.6f}, {np.nanmax(mat_final):.6f}]")

# Peak detection
peak_mask = local_maxima_3x3(activity_image)
rows, cols, vals = apply_density_threshold(
    activity_image,
    peak_mask,
    max_density=0.01,
    n_time_points=n_time,
    align_hz=align_hz,
    valid_mask=valid,
)

mat_act = sio.loadmat(str(data_dir / "matlab_activityImage.mat"))
matlab_peaks_r = mat_act["peaks"]["row"][0, 0].flatten()
matlab_peaks_c = mat_act["peaks"]["col"][0, 0].flatten()

print(f"\nPython peaks: {len(rows)}, MATLAB peaks: {len(matlab_peaks_r)}")

# Match peaks
if len(rows) > 0 and len(matlab_peaks_r) > 0:
    from scipy.spatial.distance import cdist

    py_coords = np.column_stack([rows.astype(float), cols.astype(float)])
    mat_coords = np.column_stack([matlab_peaks_r, matlab_peaks_c])
    dists = cdist(py_coords, mat_coords)

    for thr in [1.0, 2.0, 3.0, 5.0]:
        py_m = np.sum(np.min(dists, axis=1) <= thr)
        mat_m = np.sum(np.min(dists, axis=0) <= thr)
        print(
            f"  Within {thr:.0f}px: {py_m}/{len(rows)} Python, {mat_m}/{len(matlab_peaks_r)} MATLAB"
        )
