"""Compare Python vs MATLAB MAD in detail, including a true per-window MAD."""

import math, warnings
import numpy as np
import scipy.io as sio
import pandas as pd

from spine_extraction.io.zarr_store import ExperimentStore
from spine_extraction.filters.temporal import (
    moving_median_baseline, moving_mean,
)

from _paths import get_dir_scan

data_dir = get_dir_scan()

store = ExperimentStore(data_dir / "registered.zarr")
adata = store.load_alignment_data(1)
reg_ds = store.load_registered_ds(1)
n_ch = adata.num_channels
n_ds_frames = reg_ds.shape[2] // n_ch
movie_4d = reg_ds.reshape(reg_ds.shape[0], reg_ds.shape[1], n_ds_frames, n_ch).transpose(0, 1, 3, 2)
IMf = movie_4d[:, :, 1, :].copy()

align_hz = adata.align_hz
baseline_window = int(math.ceil(4.0 * align_hz))
denoise_window = int(math.ceil(0.2 * align_hz))

nan_frac = np.mean(np.isnan(IMf), axis=2)
valid = nan_frac < 0.33
IMf[np.broadcast_to(~valid[:, :, np.newaxis], IMf.shape)] = np.nan

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    IMfden = moving_mean(IMf, window=denoise_window, axis=2)
    IMb = moving_median_baseline(IMfden, window=baseline_window, axis=2)

residual = IMfden - IMb

# Load MATLAB raw movmad
mat = sio.loadmat(str(data_dir / "matlab_raw_movmad.mat"))
mat_movmad = mat['raw_movmad_frame']  # frame 1000, raw movmad (before /0.674 * denoiseWindow)

# Python two-pass MAD (current implementation)
from spine_extraction.filters.temporal import moving_mad_noise
MAD_TO_STD = 0.6741891400433162
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    py_mad_scaled = moving_mad_noise(residual, window=baseline_window, axis=2)
# Get raw MAD at frame 999 (0-indexed) before scaling
# moving_mad_noise returns MAD / MAD_TO_STD, so multiply back to get raw MAD
py_mad_raw = py_mad_scaled[:, :, 999] * MAD_TO_STD

# Python true per-window MAD (matching MATLAB exactly)
# For a single frame, compute MAD using the actual window
half_w = baseline_window // 2
frame_idx = 999  # 0-indexed
n_time = residual.shape[2]

# Pick test pixels: interior pixel (34,74) and a few others
test_pixels = [(34, 74), (13, 38), (35, 75)]

for r, c in test_pixels:
    ts = residual[r, c, :]
    # Get the window around frame_idx
    start = max(0, frame_idx - half_w)
    end = min(n_time, frame_idx + half_w + 1)
    window = ts[start:end]
    valid_window = window[~np.isnan(window)]

    if len(valid_window) > 0:
        # True per-window MAD (MATLAB style)
        med_w = np.median(valid_window)
        true_mad = np.median(np.abs(valid_window - med_w))
    else:
        true_mad = np.nan

    mat_val = mat_movmad[r, c] if r < mat_movmad.shape[0] and c < mat_movmad.shape[1] else np.nan

    print(f"Pixel ({r},{c}):")
    print(f"  MATLAB raw movmad: {mat_val:.6f}")
    print(f"  Python two-pass MAD: {py_mad_raw[r,c]:.6f}")
    print(f"  Python true per-window MAD: {true_mad:.6f}")
    print(f"  Ratio py_twopass/matlab: {py_mad_raw[r,c]/mat_val:.4f}")
    print(f"  Ratio true/matlab: {true_mad/mat_val:.4f}")
    print()

# Now compute true per-window MAD for ALL pixels at frame 999
print("=== Full-frame comparison ===")
true_mad_frame = np.full((residual.shape[0], residual.shape[1]), np.nan)
for r in range(residual.shape[0]):
    for c in range(residual.shape[1]):
        ts = residual[r, c, :]
        start = max(0, frame_idx - half_w)
        end = min(n_time, frame_idx + half_w + 1)
        window = ts[start:end]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) > 0:
            med_w = np.median(valid_window)
            true_mad_frame[r, c] = np.median(np.abs(valid_window - med_w))

# Compare true MAD vs MATLAB
vb = ~np.isnan(true_mad_frame) & ~np.isnan(mat_movmad)
if np.sum(vb) > 0:
    corr_true = np.corrcoef(true_mad_frame[vb], mat_movmad[vb])[0, 1]
    ratio_true = true_mad_frame[vb] / mat_movmad[vb]
    print(f"True per-window MAD vs MATLAB: corr={corr_true:.6f}")
    print(f"  Ratio: mean={np.mean(ratio_true):.4f}, std={np.std(ratio_true):.4f}")

# Compare two-pass MAD vs MATLAB
vb2 = ~np.isnan(py_mad_raw) & ~np.isnan(mat_movmad)
if np.sum(vb2) > 0:
    corr_tp = np.corrcoef(py_mad_raw[vb2], mat_movmad[vb2])[0, 1]
    ratio_tp = py_mad_raw[vb2] / mat_movmad[vb2]
    print(f"\nTwo-pass MAD vs MATLAB: corr={corr_tp:.6f}")
    print(f"  Ratio: mean={np.mean(ratio_tp):.4f}, std={np.std(ratio_tp):.4f}")

# Compare true MAD vs two-pass MAD
vb3 = ~np.isnan(true_mad_frame) & ~np.isnan(py_mad_raw)
if np.sum(vb3) > 0:
    corr_tt = np.corrcoef(true_mad_frame[vb3], py_mad_raw[vb3])[0, 1]
    ratio_tt = py_mad_raw[vb3] / true_mad_frame[vb3]
    print(f"\nTwo-pass MAD vs True MAD: corr={corr_tt:.6f}")
    print(f"  Ratio: mean={np.mean(ratio_tt):.4f}, std={np.std(ratio_tt):.4f}")
