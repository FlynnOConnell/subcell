"""Compare Python vs MATLAB activity images and source selection."""

import warnings
import numpy as np
import scipy.io as sio

from spine_extraction.io.zarr_store import ExperimentStore
from spine_extraction.config import ExtractionConfig
from spine_extraction.extraction.localize import localize_sources
from spine_extraction.extraction.source_selection import select_sources
from spine_extraction.filters.morphology import iterative_nms_peaks

from _paths import get_dir_scan

data_dir = get_dir_scan()

# Load MATLAB activity image
mat = sio.loadmat(str(data_dir / "matlab_activityImage.mat"))
matlab_act = mat['activityImage']
matlab_peaks_r = mat['peaks']['row'][0, 0].flatten()
matlab_peaks_c = mat['peaks']['col'][0, 0].flatten()

# Run Python localization
store = ExperimentStore(data_dir / "registered.zarr")
adata = store.load_alignment_data(1)
reg_ds = store.load_registered_ds(1)
n_ch = adata.num_channels
n_ds_frames = reg_ds.shape[2] // n_ch
movie_4d = reg_ds.reshape(reg_ds.shape[0], reg_ds.shape[1], n_ds_frames, n_ch).transpose(0, 1, 3, 2)
movie_act = movie_4d[:, :, 1, :]

ext_config = ExtractionConfig(
    microscope="bergamo",
    sigma_px=1.33,
    dXY=3,
    denoise_window_s=0.2,
    baseline_window_glu_s=4.0,
    tau_s=0.03,
    max_synapse_density=0.01,
    nan_thresh=0.33,
    activity_channel=2,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    py_act, py_peaks = localize_sources(movie_act, ext_config, adata.align_hz)

print(f"\n=== Activity Image Comparison ===")
print(f"Python shape: {py_act.shape}, MATLAB shape: {matlab_act.shape}")

# Compare raw activity images (before median subtraction in select_sources)
valid_both = ~np.isnan(py_act) & ~np.isnan(matlab_act)
print(f"Valid pixels: {np.sum(valid_both)}")
print(f"Python  - min: {np.nanmin(py_act):.4f}, max: {np.nanmax(py_act):.4f}, median: {np.nanmedian(py_act):.4f}")
print(f"MATLAB  - min: {np.nanmin(matlab_act):.4f}, max: {np.nanmax(matlab_act):.4f}, median: {np.nanmedian(matlab_act):.4f}")

# Correlation
py_vals = py_act[valid_both]
mat_vals = matlab_act[valid_both]
corr = np.corrcoef(py_vals, mat_vals)[0, 1]
print(f"Correlation: {corr:.4f}")
print(f"Mean abs diff: {np.mean(np.abs(py_vals - mat_vals)):.4f}")
print(f"Max abs diff: {np.max(np.abs(py_vals - mat_vals)):.4f}")

# NaN pattern
py_nan = np.isnan(py_act)
mat_nan = np.isnan(matlab_act)
print(f"\nPython NaN pixels: {np.sum(py_nan)}, MATLAB NaN pixels: {np.sum(mat_nan)}")
print(f"NaN mismatch: {np.sum(py_nan != mat_nan)} pixels")
# Where Python has a value but MATLAB has NaN (or vice versa)
py_val_mat_nan = ~py_nan & mat_nan
mat_val_py_nan = py_nan & ~mat_nan
print(f"  Python valid, MATLAB NaN: {np.sum(py_val_mat_nan)}")
print(f"  MATLAB valid, Python NaN: {np.sum(mat_val_py_nan)}")

# Now check what select_sources produces
print(f"\n=== Source Selection Comparison ===")
print(f"\n--- Python activity image fed to select_sources ---")
from spine_extraction.filters.spatial import nanmedfilt2

kernel_size = 2 * int(np.ceil(1.5 * ext_config.dXY)) + 1
py_act_sel = py_act.copy()
# The activity image from localize already has median subtracted + NaN at invalid
# But select_sources subtracts another median
med = nanmedfilt2(py_act_sel, kernel_size)
py_act_sel = py_act_sel - med

py_peak_mask = iterative_nms_peaks(py_act_sel)
py_peak_vals = py_act_sel[py_peak_mask]
print(f"Candidate peaks (iterative NMS): {len(py_peak_vals)}")
print(f"  >0: {np.sum(py_peak_vals > 0)}")
sorted_p = np.sort(py_peak_vals)[::-1]

total_pix = np.sum(~np.isnan(py_act_sel))
max_sources = max(1, int(np.ceil(total_pix * ext_config.max_synapse_density)))
thresh_idx = min(len(sorted_p) - 1, max_sources - 1)
threshold = 2.0 * sorted_p[thresh_idx]
print(f"Total valid pix: {total_pix}, max_sources: {max_sources}, thresh_idx: {thresh_idx}")
print(f"Threshold value: {threshold:.6f}")
print(f"sorted_p[thresh_idx] = {sorted_p[thresh_idx]:.6f}")
n_above = np.sum(py_peak_vals > threshold)
print(f"Peaks above threshold: {n_above}")

print(f"\n--- MATLAB activity image fed to same select_sources logic ---")
mat_act_sel = matlab_act.copy()
med_m = nanmedfilt2(mat_act_sel, kernel_size)
mat_act_sel = mat_act_sel - med_m

mat_peak_mask = iterative_nms_peaks(mat_act_sel)
mat_peak_vals = mat_act_sel[mat_peak_mask]
print(f"Candidate peaks (iterative NMS): {len(mat_peak_vals)}")
print(f"  >0: {np.sum(mat_peak_vals > 0)}")
sorted_m = np.sort(mat_peak_vals)[::-1]

total_pix_m = np.sum(~np.isnan(mat_act_sel))
max_sources_m = max(1, int(np.ceil(total_pix_m * ext_config.max_synapse_density)))
thresh_idx_m = min(len(sorted_m) - 1, max_sources_m - 1)
threshold_m = 2.0 * sorted_m[thresh_idx_m]
print(f"Total valid pix: {total_pix_m}, max_sources: {max_sources_m}, thresh_idx: {thresh_idx_m}")
print(f"Threshold value: {threshold_m:.6f}")
print(f"sorted_p[thresh_idx] = {sorted_m[thresh_idx_m]:.6f}")
n_above_m = np.sum(mat_peak_vals > threshold_m)
print(f"Peaks above threshold: {n_above_m}")

# Check border region specifically
print(f"\n=== Border Analysis ===")
valid_mask = ~np.isnan(py_act)
# Find border pixels (valid but adjacent to NaN)
from scipy.ndimage import maximum_filter
nan_neighbor = maximum_filter(np.isnan(py_act).astype(float), size=3) > 0
border = valid_mask & nan_neighbor
print(f"Border pixels (valid, adjacent to NaN): {np.sum(border)}")
print(f"Python activity at border - mean: {np.nanmean(py_act[border]):.4f}, max: {np.nanmax(py_act[border]):.4f}")
print(f"MATLAB activity at border - mean: {np.nanmean(matlab_act[border & ~np.isnan(matlab_act)]):.4f}, max: {np.nanmax(matlab_act[border & ~np.isnan(matlab_act)]):.4f}")

# Interior pixels (>2 pixels from NaN)
nan_neighbor2 = maximum_filter(np.isnan(py_act).astype(float), size=5) > 0
interior = valid_mask & ~nan_neighbor2
print(f"Interior pixels (>2px from NaN): {np.sum(interior)}")
print(f"Python activity at interior - mean: {np.nanmean(py_act[interior]):.4f}, max: {np.nanmax(py_act[interior]):.4f}")
print(f"MATLAB activity at interior - mean: {np.nanmean(matlab_act[interior]):.4f}, max: {np.nanmax(matlab_act[interior]):.4f}")
