"""Debug MAD noise difference between Python and MATLAB."""

import math
import warnings

import numpy as np
import scipy.io as sio
from _paths import get_dir_scan

from subcell.filters.temporal import (
    moving_mad_noise,
    moving_mean,
    moving_median_baseline,
)
from subcell.io.zarr_store import ExperimentStore

data_dir = get_dir_scan()

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
baseline_window = int(math.ceil(4.0 * align_hz))
denoise_window = int(math.ceil(0.2 * align_hz))

# Prepare data (no interpolation)
nan_frac = np.mean(np.isnan(IMf), axis=2)
valid = nan_frac < 0.33
IMf[np.broadcast_to(~valid[:, :, np.newaxis], IMf.shape)] = np.nan
nan_mask = np.isnan(IMf)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    IMfden = moving_mean(IMf, window=denoise_window, axis=2)
    IMb = moving_median_baseline(IMfden, window=baseline_window, axis=2)

residual = IMfden - IMb

# Our MAD noise
MAD_TO_STD = 0.6741891400433162
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    std_py = moving_mad_noise(residual, window=baseline_window, axis=2) * denoise_window

# Load MATLAB std_IM
mat3 = sio.loadmat(str(data_dir / "matlab_step3_noise.mat"))
mat_std_frame = mat3["step3_frame_std"]

# Compare at frame 1000
py_std_frame = std_py[:, :, 999]
vb = ~np.isnan(py_std_frame) & ~np.isnan(mat_std_frame)
ratio = py_std_frame[vb] / mat_std_frame[vb]
print(
    f"std_IM ratio (Python/MATLAB): mean={np.mean(ratio):.4f}, median={np.median(ratio):.4f}"
)

# Check: maybe the issue is MATLAB movmad vs our implementation
# MATLAB movmad computes: mean(|x - mean(x)|) over window
# Our moving_mad_noise does:
#   1. local_mean = moving_mean(data, window)
#   2. abs_dev = |data - local_mean|
#   3. result = moving_mean(abs_dev, window) / MAD_TO_STD

# But MATLAB's movmad has a DIFFERENT definition than what we implement:
# MATLAB movmad: "Moving mean absolute deviation"
# y(t) = mean(|x(t-w:t+w) - mean(x(t-w:t+w))|)
# The key: MATLAB computes the LOCAL mean inside the window, not a separate
# smoothed version. Our two-pass approach uses a GLOBAL moving mean (each pass
# has its own window centering), which is different from computing the mean
# within the same window as the deviation.

# Let me try the correct single-window approach
print("\n=== Testing single-window MAD (matching MATLAB movmad exactly) ===")

# For a single interior pixel, compute MAD by hand
ir, ic = 34, 74  # interior pixel
ts = residual[ir, ic, :]
valid_ts = ~np.isnan(ts)

# MATLAB movmad with window=215, centered
half_w = baseline_window // 2  # 107
filter_w = 2 * half_w + 1  # 215

# Compute for frame 999
t = 999
lo = max(0, t - half_w)
hi = min(len(ts), t + half_w + 1)
window_vals = ts[lo:hi]
window_valid = window_vals[~np.isnan(window_vals)]
if len(window_valid) > 0:
    window_mean = np.mean(window_valid)
    matlab_mad = np.mean(np.abs(window_valid - window_mean))
    our_mad_raw = moving_mad_noise(
        residual[ir : ir + 1, ic : ic + 1, :], window=baseline_window, axis=2
    )
    our_val = our_mad_raw[0, 0, 999] * denoise_window
    mat_val = mat_std_frame[ir, ic]
    print(f"  Frame 999, pixel ({ir},{ic}):")
    print(f"    Manual MAD (single window): {matlab_mad:.6f}")
    print(
        f"    Manual MAD / MAD_TO_STD * denoiseWindow: {matlab_mad / MAD_TO_STD * denoise_window:.6f}"
    )
    print(f"    Our moving_mad_noise * denoise: {our_val:.6f}")
    print(f"    MATLAB stdIM: {mat_val:.6f}")
    print(f"    Ratio (ours/MATLAB): {our_val / mat_val:.4f}")
    print(
        f"    Ratio (manual/MATLAB): {matlab_mad / MAD_TO_STD * denoise_window / mat_val:.4f}"
    )

# The two-pass moving_mean approach computes a DIFFERENT thing:
# Pass 1: local_mean[t] = mean(x[t-w:t+w])  -- this uses uniform_filter1d
# Pass 2: mad[t] = mean(|x[t-w:t+w] - local_mean[t-w:t+w]|)  -- deviation from the SMOOTHED mean
#
# But MATLAB movmad computes:
# For each window [t-w:t+w]:
#   1. Compute window_mean = mean(x[t-w:t+w])
#   2. Compute mad = mean(|x[t-w:t+w] - window_mean|)
# The difference is that in step 2, MATLAB subtracts the SAME window_mean
# from all values, while our Pass 2 subtracts local_mean which varies
# across the window.

# Let's verify: compute our two-pass result manually
local_mean_ts = moving_mean(
    residual[ir : ir + 1, ic : ic + 1, :], window=baseline_window, axis=2
)[0, 0, :]
abs_dev_ts = np.abs(
    ts - local_mean_ts
)  # deviation from smoothed mean (varies per sample)
mad_twopass = moving_mean(abs_dev_ts.reshape(1, 1, -1), window=baseline_window, axis=2)[
    0, 0, 999
]
print(f"\n  Two-pass MAD value: {mad_twopass:.6f}")
print(
    f"  Two-pass MAD / MAD_TO_STD * denoise: {mad_twopass / MAD_TO_STD * denoise_window:.6f}"
)

# Now compute single-pass MAD (matching MATLAB)
print(f"\n  Window size for movmad: {filter_w}")
print(f"  Number of valid values in window: {len(window_valid)}")
