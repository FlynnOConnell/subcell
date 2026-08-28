"""Compare Python vs MATLAB at each step — multi-frame comparison."""

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

# Load movie
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

# Step 0: NaN handling (no interpolation)
nan_frac = np.mean(np.isnan(IMf), axis=2)
valid = nan_frac < 0.33
IMf[np.broadcast_to(~valid[:, :, np.newaxis], IMf.shape)] = np.nan
nan_mask = np.isnan(IMf)

# Frame indices matching MATLAB: [100 500 1000 5000 10000] (0-indexed: -1)
fidxs = [99, 499, 999, 4999, 9999]
fidx_labels = [100, 500, 1000, 5000, 10000]


def corr_frames(py_frames, mat_frames, step_name):
    """Compare multiple frames."""
    for i, fl in enumerate(fidx_labels):
        py = py_frames[:, :, i] if py_frames.ndim == 3 else py_frames
        ma = mat_frames[:, :, i] if mat_frames.ndim == 3 else mat_frames
        vb = ~np.isnan(py) & ~np.isnan(ma)
        if np.sum(vb) == 0:
            print(f"  frame {fl}: no valid overlap")
            continue
        c = np.corrcoef(py[vb], ma[vb])[0, 1]
        mae = np.mean(np.abs(py[vb] - ma[vb]))
        scale = np.std(ma[vb])
        nrmse = np.sqrt(np.mean((py[vb] - ma[vb]) ** 2)) / scale if scale > 0 else 0
        print(
            f"  {step_name} frame {fl}: corr={c:.6f}, NRMSE={nrmse:.4f}, MAE={mae:.2e}"
        )


# ============ Step 1: Smoothing ============
print("=== Step 1: Moving mean ===")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    IMfden = moving_mean(IMf, window=denoise_window, axis=2)

mat1b = sio.loadmat(str(data_dir / "matlab_step1b_frames.mat"))
corr_frames(IMfden[:, :, fidxs], mat1b["step1_frames_smooth"], "smooth")

# ============ Step 2: Baseline ============
print("\n=== Step 2: Moving median baseline ===")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    IMb = moving_median_baseline(IMfden, window=baseline_window, axis=2)
IMf_hp = IMf - IMb

mat2 = sio.loadmat(str(data_dir / "matlab_step2_baseline.mat"))
corr_frames(IMb[:, :, fidxs], mat2["step2_frames_baseline"], "baseline")
corr_frames(IMf_hp[:, :, fidxs], mat2["step2_frames_hp"], "highpass")

# ============ Step 3: MAD noise + Z-score ============
print("\n=== Step 3: MAD noise + Z-score ===")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    std_IM = moving_mad_noise(IMfden - IMb, window=baseline_window, axis=2)
std_IM = std_IM * denoise_window
with np.errstate(divide="ignore", invalid="ignore"):
    IMf_z = IMf_hp / std_IM
non_finite_z = ~np.isfinite(IMf_z)
IMf_z[non_finite_z] = np.nan
nan_mask = nan_mask | non_finite_z

mat3 = sio.loadmat(str(data_dir / "matlab_step3_noise.mat"))
corr_frames(IMf_z[:, :, fidxs], mat3["step3_frames_z"], "zscore")

# Check std_IM at single frame
py_std_frame = std_IM[:, :, 999]
mat_std_frame = mat3["step3_frame_std"]
vb = ~np.isnan(py_std_frame) & ~np.isnan(mat_std_frame)
c = np.corrcoef(py_std_frame[vb], mat_std_frame[vb])[0, 1]
print(f"  std_IM frame 1000: corr={c:.6f}")

# Check ratio of std_IM
ratio = py_std_frame[vb] / mat_std_frame[vb]
print(f"  std_IM ratio: mean={np.mean(ratio):.4f}, std={np.std(ratio):.4f}")

del IMfden, IMb, IMf_hp, std_IM

# ============ Step 4: Matched filter ============
print("\n=== Step 4: Exponential matched filter ===")
IMf_mf = exponential_matched_filter(IMf_z, tau)
IMf_mf[nan_mask] = np.nan

mat4 = sio.loadmat(str(data_dir / "matlab_step4_matchedfilter.mat"))
corr_frames(IMf_mf[:, :, fidxs], mat4["step4_frames"], "mf")
del IMf_z

# ============ Step 5: DoG ============
print("\n=== Step 5: Difference of Gaussians ===")
IMf_dog = difference_of_gaussians(IMf_mf, sigma, nan_mask=nan_mask)

mat5 = sio.loadmat(str(data_dir / "matlab_step5_dog.mat"))
corr_frames(IMf_dog[:, :, fidxs], mat5["step5_frames"], "dog")
del IMf_mf

# ============ Step 6: NMS + activity image ============
print("\n=== Step 6: Activity image ===")
mat6 = sio.loadmat(str(data_dir / "matlab_step6_activity.mat"))

activity_image, _ = spatiotemporal_nms(IMf_dog, tau, nan_mask=nan_mask)

vb = ~np.isnan(activity_image) & ~np.isnan(mat6["step6_activity_raw"])
c = np.corrcoef(activity_image[vb], mat6["step6_activity_raw"][vb])[0, 1]
print(f"  activity_raw: corr={c:.6f}")

# Post-process
activity_image[~valid] = np.nan
from subcell.filters.spatial import nanmedfilt2

med_filt = nanmedfilt2(activity_image, 5)
activity_image = activity_image - med_filt
activity_image[~valid] = np.nan

vb = ~np.isnan(activity_image) & ~np.isnan(mat6["step6_activity_final"])
c = np.corrcoef(activity_image[vb], mat6["step6_activity_final"][vb])[0, 1]
print(f"  activity_final: corr={c:.6f}")
