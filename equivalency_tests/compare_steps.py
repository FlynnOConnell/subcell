"""Compare Python vs MATLAB at each processing step."""

import math
import warnings

import numpy as np
import scipy.io as sio
from _paths import get_dir_scan

from subcell.filters.morphology import spatiotemporal_nms
from subcell.filters.spatial import difference_of_gaussians
from subcell.filters.temporal import (
    exponential_matched_filter,
    moving_mad_noise,
    moving_mean,
    moving_median_baseline,
)
from subcell.io.zarr_store import ExperimentStore

data_dir = get_dir_scan()

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
nan_thresh = 0.33

print(
    f"tau={tau:.2f} frames, sigma={sigma}, baseline_window={baseline_window}, denoise_window={denoise_window}"
)

# Step 0: Initial NaN handling (NO interpolation, matching MATLAB)
nan_mask_orig = np.isnan(IMf)
nan_frac_spatial = np.mean(nan_mask_orig, axis=2)
valid = nan_frac_spatial < nan_thresh

IMf[np.broadcast_to(~valid[:, :, np.newaxis], IMf.shape)] = np.nan
nan_mask = np.isnan(IMf)

# Load MATLAB step data
mat1 = sio.loadmat(str(data_dir / "matlab_step1_smooth.mat"))
br, bc = int(mat1["br"][0, 0]) - 1, int(mat1["bc"][0, 0]) - 1  # 0-indexed
ir, ic = int(mat1["ir"][0, 0]) - 1, int(mat1["ic"][0, 0]) - 1
frame_idx = 999  # 0-indexed (MATLAB used 1000)

print(f"Border pixel: ({br}, {bc}), Interior pixel: ({ir}, {ic})")


def compare(label, py, mat):
    if py.ndim == 1 and mat.ndim == 1:
        valid_both = ~np.isnan(py) & ~np.isnan(mat.flatten())
        if np.sum(valid_both) == 0:
            print(f"  {label}: no valid overlap")
            return
        corr = np.corrcoef(py[valid_both], mat.flatten()[valid_both])[0, 1]
        mae = np.mean(np.abs(py[valid_both] - mat.flatten()[valid_both]))
        rng = np.max(np.abs(mat.flatten()[valid_both]))
        rel = mae / rng if rng > 0 else 0
        print(f"  {label}: corr={corr:.6f}, MAE={mae:.6e}, rel_err={rel:.4f}")
    elif py.ndim == 2 and mat.ndim == 2:
        valid_both = ~np.isnan(py) & ~np.isnan(mat)
        if np.sum(valid_both) == 0:
            print(f"  {label}: no valid overlap")
            return
        corr = np.corrcoef(py[valid_both], mat[valid_both])[0, 1]
        mae = np.mean(np.abs(py[valid_both] - mat[valid_both]))
        print(f"  {label}: corr={corr:.6f}, MAE={mae:.6e}")


# ============ Step 1: Smoothing ============
print("\n=== Step 1: Moving mean (smoothdata movmean) ===")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    IMfden = moving_mean(IMf, window=denoise_window, axis=2)

py_border_raw = IMf[br, bc, :]
py_border_smooth = IMfden[br, bc, :]
py_interior_raw = IMf[ir, ic, :]
py_interior_smooth = IMfden[ir, ic, :]
py_frame_smooth = IMfden[:, :, frame_idx]

compare("border_raw", py_border_raw, mat1["step1_border_raw"])
compare("border_smooth", py_border_smooth, mat1["step1_border_smooth"])
compare("interior_raw", py_interior_raw, mat1["step1_interior_raw"])
compare("interior_smooth", py_interior_smooth, mat1["step1_interior_smooth"])
compare("frame_smooth", py_frame_smooth, mat1["step1_frame_smooth"])

# ============ Step 2: Baseline ============
print("\n=== Step 2: Moving median baseline ===")
mat2 = sio.loadmat(str(data_dir / "matlab_step2_baseline.mat"))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    IMb = moving_median_baseline(IMfden, window=baseline_window, axis=2)
IMf_hp = IMf - IMb

py_border_bl = IMb[br, bc, :]
py_border_hp = IMf_hp[br, bc, :]
py_interior_bl = IMb[ir, ic, :]
py_interior_hp = IMf_hp[ir, ic, :]

compare("border_baseline", py_border_bl, mat2["step2_border_baseline"])
compare("border_highpass", py_border_hp, mat2["step2_border_hp"])
compare("interior_baseline", py_interior_bl, mat2["step2_interior_baseline"])
compare("interior_highpass", py_interior_hp, mat2["step2_interior_hp"])

# ============ Step 3: MAD noise + Z-score ============
print("\n=== Step 3: MAD noise estimation ===")
mat3 = sio.loadmat(str(data_dir / "matlab_step3_noise.mat"))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    std_IM = moving_mad_noise(IMfden - IMb, window=baseline_window, axis=2)
std_IM = std_IM * denoise_window
std_IM = np.maximum(std_IM, 1e-10)
IMf_z = IMf_hp / std_IM

compare("border_std", std_IM[br, bc, :], mat3["step3_border_std"])
compare("interior_std", std_IM[ir, ic, :], mat3["step3_interior_std"])
compare("border_zscore", IMf_z[br, bc, :], mat3["step3_border_z"])
compare("interior_zscore", IMf_z[ir, ic, :], mat3["step3_interior_z"])

del IMfden, IMb, IMf_hp, std_IM

# ============ Step 4: Matched filter ============
print("\n=== Step 4: Exponential matched filter ===")
mat4 = sio.loadmat(str(data_dir / "matlab_step4_matchedfilter.mat"))

IMf_mf = exponential_matched_filter(IMf_z, tau)
IMf_mf[nan_mask] = np.nan

compare("border_mf", IMf_mf[br, bc, :], mat4["step4_border_mf"])
compare("interior_mf", IMf_mf[ir, ic, :], mat4["step4_interior_mf"])
compare("frame_mf", IMf_mf[:, :, frame_idx], mat4["step4_frame"])

del IMf_z

# ============ Step 5: DoG ============
print("\n=== Step 5: Difference of Gaussians ===")
mat5 = sio.loadmat(str(data_dir / "matlab_step5_dog.mat"))

IMf_dog = difference_of_gaussians(IMf_mf, sigma, nan_mask=nan_mask)

compare("border_dog", IMf_dog[br, bc, :], mat5["step5_border_dog"])
compare("interior_dog", IMf_dog[ir, ic, :], mat5["step5_interior_dog"])
compare("frame_dog", IMf_dog[:, :, frame_idx], mat5["step5_frame"])

del IMf_mf

# ============ Step 6: NMS + activity image ============
print("\n=== Step 6: Spatiotemporal NMS + activity image ===")
mat6 = sio.loadmat(str(data_dir / "matlab_step6_activity.mat"))

activity_image, _ = spatiotemporal_nms(IMf_dog, tau, nan_mask=nan_mask)

compare("activity_raw", activity_image, mat6["step6_activity_raw"])

# Post-process
activity_image[~valid] = np.nan
nan_vals = np.isnan(activity_image)
med_val = np.nanmedian(activity_image) if np.any(~nan_vals) else 0.0
activity_image[nan_vals] = med_val

from subcell.filters.spatial import nanmedfilt2

med_filt = nanmedfilt2(activity_image, 5)
activity_image = activity_image - med_filt
activity_image[~valid] = np.nan

compare("activity_final", activity_image, mat6["step6_activity_final"])
