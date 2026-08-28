"""Debug the problematic pixel (13,38) through each processing step."""

import math, warnings
import numpy as np

from spine_extraction.io.zarr_store import ExperimentStore
from spine_extraction.filters.temporal import (
    moving_median_baseline, moving_mean, moving_mad_noise,
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
nan_mask = np.isnan(IMf)

r, c = 13, 38  # problematic pixel

print(f"=== Pixel ({r},{c}) Debug ===")
raw = IMf[r, c, :]
nan_ts = np.isnan(raw)
print(f"Raw: NaN={np.sum(nan_ts)}/{len(raw)}, nan_frac={nan_frac[r,c]:.4f}")
print(f"Raw valid range: [{np.nanmin(raw):.4f}, {np.nanmax(raw):.4f}]")

# Step 1: Moving mean
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    IMfden = moving_mean(IMf, window=denoise_window, axis=2)

smooth = IMfden[r, c, :]
print(f"\nSmoothed: NaN={np.sum(np.isnan(smooth))}, range=[{np.nanmin(smooth):.4f}, {np.nanmax(smooth):.4f}]")

# Step 2: Baseline
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    IMb = moving_median_baseline(IMfden, window=baseline_window, axis=2)

baseline = IMb[r, c, :]
hp = raw - baseline
print(f"Baseline: NaN={np.sum(np.isnan(baseline))}, range=[{np.nanmin(baseline):.4f}, {np.nanmax(baseline):.4f}]")
print(f"Highpass: NaN={np.sum(np.isnan(hp))}, range=[{np.nanmin(hp):.4f}, {np.nanmax(hp):.4f}]")

# Step 3: MAD noise
residual = smooth - baseline
print(f"\nResidual (smooth-baseline): NaN={np.sum(np.isnan(residual))}, range=[{np.nanmin(residual):.4f}, {np.nanmax(residual):.4f}]")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    std_full = moving_mad_noise(IMfden - IMb, window=baseline_window, axis=2)

std_ts = std_full[r, c, :] * denoise_window
print(f"Std (before max): NaN={np.sum(np.isnan(std_ts))}, range=[{np.nanmin(std_ts):.6f}, {np.nanmax(std_ts):.6f}]")
print(f"Std == 0: {np.sum(std_ts == 0)}, Std < 1e-5: {np.sum(std_ts < 1e-5)}")

# Check where std is zero/tiny
zero_std = std_ts < 1e-5
if np.any(zero_std):
    zero_frames = np.where(zero_std)[0]
    print(f"Near-zero std at frames: {zero_frames[:20]}...")
    print(f"Highpass at those frames: {hp[zero_frames[:20]]}")
    print(f"Z-score at those frames would be: {hp[zero_frames[:20]] / np.maximum(std_ts[zero_frames[:20]], 1e-10)}")

std_ts_clipped = np.maximum(std_ts, 1e-10)
zscore = hp / std_ts_clipped
print(f"\nZ-score: NaN={np.sum(np.isnan(zscore))}, range=[{np.nanmin(zscore):.4f}, {np.nanmax(zscore):.4f}]")
print(f"|Z| > 10: {np.sum(np.abs(zscore) > 10)}, |Z| > 100: {np.sum(np.abs(zscore) > 100)}")
print(f"|Z| > 1000: {np.sum(np.abs(zscore) > 1000)}, |Z| > 1e6: {np.sum(np.abs(zscore) > 1e6)}")

# Also check a few other border pixels
print("\n=== Checking all pixels for extreme z-scores ===")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    std_all = moving_mad_noise(IMfden - IMb, window=baseline_window, axis=2) * denoise_window

std_all_clipped = np.maximum(std_all, 1e-10)
hp_all = IMf - IMb
zscore_all = hp_all / std_all_clipped

extreme_mask = np.abs(zscore_all) > 1000
n_extreme = np.sum(extreme_mask)
print(f"|Z| > 1000: {n_extreme} values out of {zscore_all.size} ({100*n_extreme/zscore_all.size:.4f}%)")

extreme_mask_1e6 = np.abs(zscore_all) > 1e6
n_extreme_1e6 = np.sum(extreme_mask_1e6)
print(f"|Z| > 1e6: {n_extreme_1e6} values")

# Which pixels have extreme z-scores?
extreme_per_pixel = np.sum(np.abs(zscore_all) > 1000, axis=2)
bad_pixels = extreme_per_pixel > 0
print(f"Pixels with any |Z|>1000: {np.sum(bad_pixels)}")
if np.sum(bad_pixels) > 0:
    bad_r, bad_c = np.where(bad_pixels)
    print(f"First 10 bad pixels (row,col,count):")
    for i in range(min(10, len(bad_r))):
        print(f"  ({bad_r[i]},{bad_c[i]}): {extreme_per_pixel[bad_r[i],bad_c[i]]} frames, "
              f"nan_frac={nan_frac[bad_r[i],bad_c[i]]:.3f}")

    # Check if bad pixels are all border pixels
    bad_nan_fracs = nan_frac[bad_pixels]
    print(f"\nBad pixel nan_frac: min={np.min(bad_nan_fracs):.3f}, max={np.max(bad_nan_fracs):.3f}")
    print(f"Bad pixels with nan_frac>0: {np.sum(bad_nan_fracs > 0)}/{np.sum(bad_pixels)}")

# What does MATLAB's std look like at these pixels?
import scipy.io as sio
mat3 = sio.loadmat(str(data_dir / "matlab_step3_noise.mat"))
mat_std = mat3['step3_frame_std']  # frame 1000
py_std_frame = std_all[:, :, 999]
print(f"\nAt frame 1000, pixel (13,38):")
print(f"  Python std: {py_std_frame[13, 38]:.8f}")
print(f"  MATLAB std: {mat_std[13, 38]:.8f}")

# Check a few bad pixels at frame 1000
for i in range(min(5, len(bad_r))):
    rr, cc = bad_r[i], bad_c[i]
    if rr < mat_std.shape[0] and cc < mat_std.shape[1]:
        print(f"  Pixel ({rr},{cc}): Python std={py_std_frame[rr,cc]:.8f}, "
              f"MATLAB std={mat_std[rr,cc]:.8f}")
