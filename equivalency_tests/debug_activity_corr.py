"""Investigate why activity image correlation is low despite good upstream match."""

import math, warnings
import numpy as np
import scipy.io as sio
from scipy.ndimage import maximum_filter

from spine_extraction.io.zarr_store import ExperimentStore
from spine_extraction.filters.temporal import (
    exponential_matched_filter, moving_median_baseline,
    moving_mean, moving_mad_noise,
)
from spine_extraction.filters.spatial import difference_of_gaussians, nanmedfilt2
from spine_extraction.filters.morphology import spatiotemporal_nms

from _paths import get_dir_scan

data_dir = get_dir_scan()

# Full pipeline to DoG (with Inf fix)
store = ExperimentStore(data_dir / "registered.zarr")
adata = store.load_alignment_data(1)
reg_ds = store.load_registered_ds(1)
n_ch = adata.num_channels
n_ds_frames = reg_ds.shape[2] // n_ch
movie_4d = reg_ds.reshape(reg_ds.shape[0], reg_ds.shape[1], n_ds_frames, n_ch).transpose(0, 1, 3, 2)
IMf = movie_4d[:, :, 1, :].copy()

align_hz = adata.align_hz
tau = 0.03 * align_hz
sigma = 1.33
baseline_window = int(math.ceil(4.0 * align_hz))
denoise_window = int(math.ceil(0.2 * align_hz))

nan_frac = np.mean(np.isnan(IMf), axis=2)
valid = nan_frac < 0.33
IMf[np.broadcast_to(~valid[:, :, np.newaxis], IMf.shape)] = np.nan
nan_mask_orig = np.isnan(IMf)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    IMfden = moving_mean(IMf, window=denoise_window, axis=2)
    IMb = moving_median_baseline(IMfden, window=baseline_window, axis=2)

IMf_hp = IMf - IMb
std_IM = moving_mad_noise(IMfden - IMb, window=baseline_window, axis=2) * denoise_window
del IMfden, IMb

with np.errstate(divide="ignore", invalid="ignore"):
    IMf_z = IMf_hp / std_IM
non_finite = ~np.isfinite(IMf_z)
IMf_z[non_finite] = np.nan
nan_mask = nan_mask_orig | non_finite
del IMf_hp, std_IM, non_finite

IMf_mf = exponential_matched_filter(IMf_z, tau)
IMf_mf[nan_mask] = np.nan
del IMf_z

IMf_dog = difference_of_gaussians(IMf_mf, sigma, nan_mask=nan_mask)
del IMf_mf

# Load MATLAB activity
mat6 = sio.loadmat(str(data_dir / "matlab_step6_activity.mat"))
mat_raw = mat6['step6_activity_raw']

# Run NMS
py_raw, _ = spatiotemporal_nms(IMf_dog, tau, nan_mask=nan_mask)

print("=== Activity Image Analysis ===")
print(f"Python raw range: [{np.nanmin(py_raw):.6f}, {np.nanmax(py_raw):.6f}]")
print(f"MATLAB raw range: [{np.nanmin(mat_raw):.6f}, {np.nanmax(mat_raw):.6f}]")

# Compare interior vs border
interior = nan_frac == 0
border = (nan_frac > 0) & valid

vb_int = interior & ~np.isnan(py_raw) & ~np.isnan(mat_raw)
vb_brd = border & ~np.isnan(py_raw) & ~np.isnan(mat_raw)

if np.sum(vb_int) > 2:
    c = np.corrcoef(py_raw[vb_int], mat_raw[vb_int])[0, 1]
    nrmse = np.sqrt(np.mean((py_raw[vb_int] - mat_raw[vb_int])**2)) / np.std(mat_raw[vb_int])
    print(f"\nInterior pixels ({np.sum(vb_int)}): corr={c:.4f}, NRMSE={nrmse:.4f}")
    ratio = py_raw[vb_int] / np.maximum(mat_raw[vb_int], 1e-15)
    finite_r = ratio[np.isfinite(ratio) & (mat_raw[vb_int] > 1e-10)]
    if len(finite_r) > 0:
        print(f"  Ratio: mean={np.mean(finite_r):.4f}, median={np.median(finite_r):.4f}, std={np.std(finite_r):.4f}")

if np.sum(vb_brd) > 2:
    c = np.corrcoef(py_raw[vb_brd], mat_raw[vb_brd])[0, 1]
    print(f"Border pixels ({np.sum(vb_brd)}): corr={c:.4f}")

# Check scatter of activity values
vb = ~np.isnan(py_raw) & ~np.isnan(mat_raw)
print(f"\nAll valid pixels: {np.sum(vb)}")

# Percentile comparison
for p in [50, 75, 90, 95, 99]:
    pp = np.percentile(py_raw[vb], p)
    mp = np.percentile(mat_raw[vb], p)
    print(f"  p{p}: Python={pp:.6f}, MATLAB={mp:.6f}, ratio={pp/mp:.4f}" if mp != 0 else f"  p{p}: Python={pp:.6f}, MATLAB={mp:.6f}")

# How many peaks are found at each frame?
rows, cols, n_frames = IMf_dog.shape
skip_end = int(np.ceil(1.5 * tau))
end_frame = n_frames - skip_end

print(f"\n=== Per-frame peak analysis ===")
py_peaks_per_frame = []
for fr in range(1, min(end_frame, 100)):  # first 100 frames
    cur = IMf_dog[:, :, fr]
    prev = IMf_dog[:, :, fr - 1]
    nxt = IMf_dog[:, :, fr + 1]

    spatial_max = maximum_filter(cur, size=3)
    is_spatial = (cur == spatial_max)
    nan_cur = np.isnan(cur)
    if np.any(nan_cur):
        nan_neighbor = maximum_filter(nan_cur.astype(np.float32), size=3) > 0
        is_spatial[nan_neighbor] = False
    is_peak = is_spatial & (cur > prev) & (cur >= nxt)
    py_peaks_per_frame.append(np.sum(is_peak))

print(f"Peaks per frame (first 100): mean={np.mean(py_peaks_per_frame):.1f}, "
      f"min={np.min(py_peaks_per_frame)}, max={np.max(py_peaks_per_frame)}")

# Post-process and compare
py_final = py_raw.copy()
py_final[~valid] = np.nan
med_filt = nanmedfilt2(py_final, 5)
py_final = py_final - med_filt
py_final[~valid] = np.nan

mat_final = mat6['step6_activity_final']

vb_f = ~np.isnan(py_final) & ~np.isnan(mat_final)
c = np.corrcoef(py_final[vb_f], mat_final[vb_f])[0, 1]
print(f"\n=== Post-processed activity ===")
print(f"Correlation: {c:.4f}")

# Interior only
vb_int_f = interior & ~np.isnan(py_final) & ~np.isnan(mat_final)
if np.sum(vb_int_f) > 2:
    c_int = np.corrcoef(py_final[vb_int_f], mat_final[vb_int_f])[0, 1]
    print(f"Interior only: corr={c_int:.4f}")

# Check if the low correlation is driven by different spatial patterns or just scale
# Try rank correlation
from scipy.stats import spearmanr
rho, _ = spearmanr(py_final[vb_f], mat_final[vb_f])
print(f"Spearman rank correlation: {rho:.4f}")

# Check if zeroing out border fixes things
py_interior_only = py_final.copy()
py_interior_only[border] = np.nan
mat_interior_only = mat_final.copy()
mat_interior_only[border] = np.nan

vb_io = ~np.isnan(py_interior_only) & ~np.isnan(mat_interior_only)
if np.sum(vb_io) > 2:
    c_io = np.corrcoef(py_interior_only[vb_io], mat_interior_only[vb_io])[0, 1]
    print(f"Interior-only (no border): corr={c_io:.4f}")
