"""Debug spatiotemporal NMS to find source of enormous values."""

import math, warnings
import numpy as np
import scipy.io as sio
from scipy.ndimage import maximum_filter

from spine_extraction.io.zarr_store import ExperimentStore
from spine_extraction.filters.temporal import (
    exponential_matched_filter, moving_median_baseline,
    moving_mean, moving_mad_noise,
)
from spine_extraction.filters.spatial import difference_of_gaussians

from _paths import get_dir_scan

data_dir = get_dir_scan()

# Reproduce pipeline up to DoG
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
nan_mask = np.isnan(IMf)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    IMfden = moving_mean(IMf, window=denoise_window, axis=2)
    IMb = moving_median_baseline(IMfden, window=baseline_window, axis=2)

IMf_hp = IMf - IMb
std_IM = moving_mad_noise(IMfden - IMb, window=baseline_window, axis=2) * denoise_window
std_IM = np.maximum(std_IM, 1e-10)
IMf_z = IMf_hp / std_IM
del IMfden, IMb, IMf_hp, std_IM

IMf_mf = exponential_matched_filter(IMf_z, tau)
IMf_mf[nan_mask] = np.nan
del IMf_z

IMf_dog = difference_of_gaussians(IMf_mf, sigma, nan_mask=nan_mask)
del IMf_mf

# ---- Debug the DoG output ----
print("=== DoG output statistics ===")
print(f"Shape: {IMf_dog.shape}")
print(f"NaN count: {np.sum(np.isnan(IMf_dog))}")
print(f"Inf count: {np.sum(np.isinf(IMf_dog))}")
valid_vals = IMf_dog[~np.isnan(IMf_dog)]
print(f"Valid values: {len(valid_vals)}")
print(f"Range: [{np.min(valid_vals):.4f}, {np.max(valid_vals):.4f}]")
print(f"Percentiles: 1%={np.percentile(valid_vals,1):.4f}, 99%={np.percentile(valid_vals,99):.4f}")
print(f"|val| > 10: {np.sum(np.abs(valid_vals) > 10)}")
print(f"|val| > 100: {np.sum(np.abs(valid_vals) > 100)}")
print(f"|val| > 1000: {np.sum(np.abs(valid_vals) > 1000)}")

# ---- Manual NMS to find the problem ----
rows, cols, n_frames = IMf_dog.shape
skip_end = int(np.ceil(1.5 * tau))
end_frame = n_frames - skip_end

print(f"\n=== Manual NMS debug ===")
print(f"Processing frames 1..{end_frame-1} (skip last {skip_end})")

activity_image = np.zeros((rows, cols), dtype=np.float64)
total_peaks = 0
max_peak_val = 0
max_peak_sq = 0

for fr in range(1, end_frame):
    cur = IMf_dog[:, :, fr]
    prev = IMf_dog[:, :, fr - 1]
    nxt = IMf_dog[:, :, fr + 1]

    # Spatial local max (matching MATLAB ordfilt2)
    spatial_max = maximum_filter(cur, size=3)
    is_spatial = (cur == spatial_max)

    # Suppress near NaN (MATLAB ordfilt2 propagates NaN)
    nan_cur = np.isnan(cur)
    if np.any(nan_cur):
        nan_neighbor = maximum_filter(nan_cur.astype(np.float32), size=3) > 0
        is_spatial[nan_neighbor] = False

    # Temporal check
    is_peak = is_spatial & (cur > prev) & (cur >= nxt)

    n_peaks = np.sum(is_peak)
    if n_peaks > 0:
        peak_vals = cur[is_peak]
        peak_sq = peak_vals ** 2
        activity_image[is_peak] += peak_sq
        total_peaks += n_peaks

        frame_max = np.max(np.abs(peak_vals))
        frame_max_sq = np.max(peak_sq)
        if frame_max_sq > max_peak_sq:
            max_peak_sq = frame_max_sq
            max_peak_val = frame_max
            max_peak_frame = fr

    if fr % 5000 == 0:
        print(f"  Frame {fr}: {n_peaks} peaks, activity max so far={np.max(activity_image):.4f}")

print(f"\nTotal peaks found: {total_peaks}")
print(f"Max peak value: {max_peak_val:.4f} at frame {max_peak_frame}")
print(f"Max peak^2: {max_peak_sq:.4f}")
print(f"Activity image (unnormalized) range: [{np.min(activity_image):.4f}, {np.max(activity_image):.4f}]")

# Normalize
n_valid = np.sum(~nan_mask[:, :, 1:end_frame], axis=2).astype(np.float64)
activity_image_norm = activity_image / (300.0 + n_valid)
print(f"Activity image (normalized) range: [{np.min(activity_image_norm):.4f}, {np.max(activity_image_norm):.4f}]")

# Compare with vectorized version
from spine_extraction.filters.morphology import spatiotemporal_nms
activity_vec, _ = spatiotemporal_nms(IMf_dog, tau, nan_mask=nan_mask)
print(f"\nVectorized activity range: [{np.min(activity_vec):.4f}, {np.max(activity_vec):.4f}]")

# Check if they match
vb = ~np.isnan(activity_image_norm) & ~np.isnan(activity_vec)
if np.sum(vb) > 0:
    corr = np.corrcoef(activity_image_norm[vb], activity_vec[vb])[0, 1]
    max_diff = np.max(np.abs(activity_image_norm[vb] - activity_vec[vb]))
    print(f"Manual vs vectorized: corr={corr:.6f}, max_diff={max_diff:.6e}")

# Compare with MATLAB
mat6 = sio.loadmat(str(data_dir / "matlab_step6_activity.mat"))
mat_raw = mat6['step6_activity_raw']
print(f"\nMATLAB activity_raw range: [{np.nanmin(mat_raw):.6f}, {np.nanmax(mat_raw):.6f}]")
print(f"Python activity_norm range: [{np.nanmin(activity_image_norm):.6f}, {np.nanmax(activity_image_norm):.6f}]")

vb = ~np.isnan(activity_image_norm) & ~np.isnan(mat_raw)
if np.sum(vb) > 0:
    ratio = activity_image_norm[vb] / np.maximum(mat_raw[vb], 1e-20)
    finite_ratio = ratio[np.isfinite(ratio) & (mat_raw[vb] > 1e-10)]
    if len(finite_ratio) > 0:
        print(f"Ratio py/mat (where mat>1e-10): mean={np.mean(finite_ratio):.4f}, "
              f"median={np.median(finite_ratio):.4f}, std={np.std(finite_ratio):.4f}")

# Find the pixel with the largest difference
diff = np.abs(activity_image_norm - mat_raw)
diff[np.isnan(diff)] = 0
max_r, max_c = np.unravel_index(np.argmax(diff), diff.shape)
print(f"\nLargest difference at pixel ({max_r},{max_c}):")
print(f"  Python: {activity_image_norm[max_r, max_c]:.6e}")
print(f"  MATLAB: {mat_raw[max_r, max_c]:.6e}")
print(f"  NaN frac: {nan_frac[max_r, max_c]:.3f}")
print(f"  Valid: {valid[max_r, max_c]}")

# Check that pixel's DoG time series
dog_ts = IMf_dog[max_r, max_c, :]
nan_ts = np.isnan(dog_ts)
print(f"  DoG time series: NaN={np.sum(nan_ts)}, range=[{np.nanmin(dog_ts):.4f}, {np.nanmax(dog_ts):.4f}]")
