"""Full step-by-step comparison of Python vs MATLAB source localization.

Loads MATLAB intermediate results and runs each Python step, comparing:
1. Moving mean smoothing
2. Moving median baseline
3. MAD noise estimation + Z-score
4. Exponential matched filter
5. Difference of Gaussians
6. Spatiotemporal NMS + activity image
7. Peak detection + source counts
"""

import math
import warnings

import numpy as np
import scipy.io as sio
from _paths import get_dir_scan

from subcell.config import ExtractionConfig
from subcell.extraction.localize import localize_sources
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
from subcell.io.zarr_store import ExperimentStore

data_dir = get_dir_scan()

print("=" * 70)
print("PYTHON vs MATLAB SOURCE LOCALIZATION - STEP-BY-STEP COMPARISON")
print("=" * 70)

# ---- Load Python movie data ----
store = ExperimentStore(data_dir / "registered.zarr")
adata = store.load_alignment_data(1)
reg_ds = store.load_registered_ds(1)
n_ch = adata.num_channels
n_ds_frames = reg_ds.shape[2] // n_ch
movie_4d = reg_ds.reshape(
    reg_ds.shape[0], reg_ds.shape[1], n_ds_frames, n_ch
).transpose(0, 1, 3, 2)
IMf = movie_4d[:, :, 1, :].copy()  # activity channel (ch2, 0-indexed=1)

align_hz = adata.align_hz
tau = 0.03 * align_hz
sigma = 1.33
baseline_window = int(math.ceil(4.0 * align_hz))
denoise_window = int(math.ceil(0.2 * align_hz))
nan_thresh = 0.33

print(
    f"\nParameters: tau={tau:.2f}fr, sigma={sigma}, baseline_window={baseline_window}, denoise_window={denoise_window}"
)
print(f"Movie shape: {IMf.shape}, align_hz={align_hz:.1f}")

# ---- Step 0: NaN handling ----
nan_frac = np.mean(np.isnan(IMf), axis=2)
valid = nan_frac < nan_thresh
IMf[np.broadcast_to(~valid[:, :, np.newaxis], IMf.shape)] = np.nan
nan_mask = np.isnan(IMf)
n_time = IMf.shape[2]

print(f"Valid pixels: {np.sum(valid)} / {valid.size}")

# Load MATLAB reference pixel coords
mat1 = sio.loadmat(str(data_dir / "matlab_step1_smooth.mat"))
br, bc = int(mat1["br"][0, 0]) - 1, int(mat1["bc"][0, 0]) - 1  # 0-indexed
ir, ic = int(mat1["ir"][0, 0]) - 1, int(mat1["ic"][0, 0]) - 1
print(f"Border pixel: ({br},{bc}), Interior pixel: ({ir},{ic})")

fidxs = [99, 499, 999, 4999, min(9999, n_time - 1)]
fidx_labels = [100, 500, 1000, 5000, min(10000, n_time)]


def compare_traces(label, py, mat):
    """Compare 1D traces."""
    mat = mat.flatten()
    vb = ~np.isnan(py) & ~np.isnan(mat)
    if np.sum(vb) < 2:
        print(f"    {label}: insufficient valid overlap ({np.sum(vb)} pts)")
        return
    corr = np.corrcoef(py[vb], mat[vb])[0, 1]
    mae = np.mean(np.abs(py[vb] - mat[vb]))
    rng = np.ptp(mat[vb])
    rel = mae / rng if rng > 0 else float("inf")
    print(f"    {label}: corr={corr:.6f}, MAE={mae:.3e}, rel_err={rel:.5f}")


def compare_frames(label, py_stack, mat_stack):
    """Compare 2D frames stacked along axis 2."""
    n = min(
        py_stack.shape[2] if py_stack.ndim == 3 else 1,
        mat_stack.shape[2] if mat_stack.ndim == 3 else 1,
    )
    for i in range(n):
        py = py_stack[:, :, i] if py_stack.ndim == 3 else py_stack
        ma = mat_stack[:, :, i] if mat_stack.ndim == 3 else mat_stack
        vb = ~np.isnan(py) & ~np.isnan(ma)
        if np.sum(vb) < 2:
            print(f"    {label}[{fidx_labels[i]}]: no overlap")
            continue
        corr = np.corrcoef(py[vb], ma[vb])[0, 1]
        nrmse = np.sqrt(np.mean((py[vb] - ma[vb]) ** 2)) / np.std(ma[vb])
        print(f"    {label}[fr{fidx_labels[i]}]: corr={corr:.6f}, NRMSE={nrmse:.5f}")


def compare_2d(label, py, mat):
    """Compare 2D images."""
    vb = ~np.isnan(py) & ~np.isnan(mat)
    if np.sum(vb) < 2:
        print(f"    {label}: no overlap")
        return
    corr = np.corrcoef(py[vb], mat[vb])[0, 1]
    mae = np.mean(np.abs(py[vb] - mat[vb]))
    rng = np.ptp(mat[vb])
    rel = mae / rng if rng > 0 else float("inf")
    print(f"    {label}: corr={corr:.6f}, MAE={mae:.3e}, rel_err={rel:.5f}")


# ========== STEP 1: Moving mean smoothing ==========
print("\n--- STEP 1: Moving mean (smoothdata movmean) ---")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    IMfden = moving_mean(IMf, window=denoise_window, axis=2)

compare_traces("border_raw", IMf[br, bc, :], mat1["step1_border_raw"])
compare_traces("border_smooth", IMfden[br, bc, :], mat1["step1_border_smooth"])
compare_traces("interior_raw", IMf[ir, ic, :], mat1["step1_interior_raw"])
compare_traces("interior_smooth", IMfden[ir, ic, :], mat1["step1_interior_smooth"])
compare_2d("frame1000_smooth", IMfden[:, :, 999], mat1["step1_frame_smooth"])

mat1b = sio.loadmat(str(data_dir / "matlab_step1b_frames.mat"))
compare_frames("smooth", IMfden[:, :, fidxs], mat1b["step1_frames_smooth"])

# ========== STEP 2: Moving median baseline ==========
print("\n--- STEP 2: Moving median baseline ---")
mat2 = sio.loadmat(str(data_dir / "matlab_step2_baseline.mat"))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    IMb = moving_median_baseline(IMfden, window=baseline_window, axis=2)
IMf_hp = IMf - IMb

compare_traces("border_baseline", IMb[br, bc, :], mat2["step2_border_baseline"])
compare_traces("border_hp", IMf_hp[br, bc, :], mat2["step2_border_hp"])
compare_traces("interior_baseline", IMb[ir, ic, :], mat2["step2_interior_baseline"])
compare_traces("interior_hp", IMf_hp[ir, ic, :], mat2["step2_interior_hp"])
compare_frames("baseline", IMb[:, :, fidxs], mat2["step2_frames_baseline"])
compare_frames("highpass", IMf_hp[:, :, fidxs], mat2["step2_frames_hp"])

# ========== STEP 3: MAD noise + Z-score ==========
print("\n--- STEP 3: MAD noise estimation + Z-score ---")
mat3 = sio.loadmat(str(data_dir / "matlab_step3_noise.mat"))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    std_IM = moving_mad_noise(IMfden - IMb, window=baseline_window, axis=2)
std_IM = std_IM * denoise_window
with np.errstate(divide="ignore", invalid="ignore"):
    IMf_z = IMf_hp / std_IM
# Convert Inf/NaN z-scores to NaN (matching MATLAB: Inf→NaN in DoG)
non_finite_z = ~np.isfinite(IMf_z)
IMf_z[non_finite_z] = np.nan
nan_mask = nan_mask | non_finite_z

compare_traces("border_std", std_IM[br, bc, :], mat3["step3_border_std"])
compare_traces("interior_std", std_IM[ir, ic, :], mat3["step3_interior_std"])
compare_traces("border_zscore", IMf_z[br, bc, :], mat3["step3_border_z"])
compare_traces("interior_zscore", IMf_z[ir, ic, :], mat3["step3_interior_z"])
compare_frames("zscore", IMf_z[:, :, fidxs], mat3["step3_frames_z"])

# Check std ratio
py_std_f = std_IM[:, :, 999]
mat_std_f = mat3["step3_frame_std"]
vb = ~np.isnan(py_std_f) & ~np.isnan(mat_std_f)
if np.sum(vb) > 0:
    ratio = py_std_f[vb] / mat_std_f[vb]
    print(
        f"    std_IM ratio (py/mat): mean={np.mean(ratio):.4f}, std={np.std(ratio):.4f}, "
        f"min={np.min(ratio):.4f}, max={np.max(ratio):.4f}"
    )

del IMfden, IMb, IMf_hp, std_IM

# ========== STEP 4: Exponential matched filter ==========
print("\n--- STEP 4: Exponential matched filter ---")
mat4 = sio.loadmat(str(data_dir / "matlab_step4_matchedfilter.mat"))

IMf_mf = exponential_matched_filter(IMf_z, tau)
IMf_mf[nan_mask] = np.nan

compare_traces("border_mf", IMf_mf[br, bc, :], mat4["step4_border_mf"])
compare_traces("interior_mf", IMf_mf[ir, ic, :], mat4["step4_interior_mf"])
compare_2d("frame1000_mf", IMf_mf[:, :, 999], mat4["step4_frame"])
compare_frames("mf", IMf_mf[:, :, fidxs], mat4["step4_frames"])

del IMf_z

# ========== STEP 5: Difference of Gaussians ==========
print("\n--- STEP 5: Difference of Gaussians ---")
mat5 = sio.loadmat(str(data_dir / "matlab_step5_dog.mat"))

IMf_dog = difference_of_gaussians(IMf_mf, sigma, nan_mask=nan_mask)

compare_traces("border_dog", IMf_dog[br, bc, :], mat5["step5_border_dog"])
compare_traces("interior_dog", IMf_dog[ir, ic, :], mat5["step5_interior_dog"])
compare_2d("frame1000_dog", IMf_dog[:, :, 999], mat5["step5_frame"])
compare_frames("dog", IMf_dog[:, :, fidxs], mat5["step5_frames"])

del IMf_mf

# ========== STEP 6: Spatiotemporal NMS + Activity Image ==========
print("\n--- STEP 6: Spatiotemporal NMS + Activity Image ---")
mat6 = sio.loadmat(str(data_dir / "matlab_step6_activity.mat"))

activity_image, _ = spatiotemporal_nms(IMf_dog, tau, nan_mask=nan_mask)
compare_2d("activity_raw", activity_image, mat6["step6_activity_raw"])

# Post-process (matching MATLAB localizeSources_vIM.m lines 119-124)
activity_image[~valid] = np.nan
med_filt = nanmedfilt2(activity_image, 5)
activity_image = activity_image - med_filt
activity_image[~valid] = np.nan

compare_2d("activity_final", activity_image, mat6["step6_activity_final"])

# ========== STEP 7: Peak detection comparison ==========
print("\n--- STEP 7: Peak detection & source locations ---")

# Python peaks from activity_image
peak_mask_py = local_maxima_3x3(activity_image)
rows_py, cols_py, vals_py = apply_density_threshold(
    activity_image,
    peak_mask_py,
    max_density=0.01,
    n_time_points=n_time,
    align_hz=align_hz,
    valid_mask=valid,
)

# MATLAB peaks from saved activity image
mat_act = sio.loadmat(str(data_dir / "matlab_activityImage.mat"))
matlab_act_img = mat_act["activityImage"]
matlab_peaks_r = mat_act["peaks"]["row"][0, 0].flatten()
matlab_peaks_c = mat_act["peaks"]["col"][0, 0].flatten()

print(f"    Python peaks:  {len(rows_py)}")
print(f"    MATLAB peaks:  {len(matlab_peaks_r)}")
print(
    f"    Python activity range: [{np.nanmin(activity_image):.4f}, {np.nanmax(activity_image):.4f}]"
)
print(
    f"    MATLAB activity range: [{np.nanmin(matlab_act_img):.4f}, {np.nanmax(matlab_act_img):.4f}]"
)

# Also run the full localize_sources function for end-to-end comparison
print("\n--- END-TO-END: Full localize_sources ---")
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

movie_act = movie_4d[:, :, 1, :].copy()
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    py_act_full, py_peaks_full = localize_sources(movie_act, ext_config, align_hz)

print(f"    localize_sources peaks: {len(py_peaks_full.row)}")
compare_2d("e2e_activity", py_act_full, mat6["step6_activity_final"])

# ========== STEP 8: Source location matching ==========
print("\n--- STEP 8: Source location matching ---")
# Match Python peaks to closest MATLAB peaks
if len(py_peaks_full.row) > 0 and len(matlab_peaks_r) > 0:
    # MATLAB uses 1-indexed rows/cols; the peaks struct stores them as-is
    # Check if MATLAB coords need adjustment
    py_coords = np.column_stack([py_peaks_full.row, py_peaks_full.col])
    mat_coords = np.column_stack([matlab_peaks_r, matlab_peaks_c])

    from scipy.spatial.distance import cdist

    dists = cdist(py_coords, mat_coords)

    # For each Python peak, find closest MATLAB peak
    min_dists_py = np.min(dists, axis=1)
    # For each MATLAB peak, find closest Python peak
    min_dists_mat = np.min(dists, axis=0)

    thresholds = [1.0, 2.0, 3.0, 5.0]
    for thr in thresholds:
        py_matched = np.sum(min_dists_py <= thr)
        mat_matched = np.sum(min_dists_mat <= thr)
        print(
            f"    Within {thr:.0f}px: {py_matched}/{len(py_coords)} Python matched, "
            f"{mat_matched}/{len(mat_coords)} MATLAB matched"
        )

    print(f"    Mean closest distance (Python->MATLAB): {np.mean(min_dists_py):.2f}px")
    print(f"    Mean closest distance (MATLAB->Python): {np.mean(min_dists_mat):.2f}px")

    # Show unmatched peaks
    unmatched_py = min_dists_py > 3.0
    unmatched_mat = min_dists_mat > 3.0
    if np.any(unmatched_py):
        print(
            f"    Python-only peaks (>3px from any MATLAB peak): {np.sum(unmatched_py)}"
        )
        um_rows = py_peaks_full.row[unmatched_py]
        um_cols = py_peaks_full.col[unmatched_py]
        for r, c in zip(um_rows[:5], um_cols[:5]):
            print(f"      ({r:.1f}, {c:.1f})")
    if np.any(unmatched_mat):
        print(
            f"    MATLAB-only peaks (>3px from any Python peak): {np.sum(unmatched_mat)}"
        )
        um_rows = matlab_peaks_r[unmatched_mat]
        um_cols = matlab_peaks_c[unmatched_mat]
        for r, c in zip(um_rows[:5], um_cols[:5]):
            print(f"      ({r:.1f}, {c:.1f})")

# ========== Mean image alignment check ==========
print("\n--- MEAN IMAGE CHECK ---")
py_mean = np.nanmean(movie_4d[:, :, 1, :], axis=2)
print(f"    Python mean image shape: {py_mean.shape}")
print(
    f"    Python mean image range: [{np.nanmin(py_mean):.1f}, {np.nanmax(py_mean):.1f}]"
)

# Check if MATLAB mean image is available
if "meanIM" in mat_act:
    mat_mean = mat_act["meanIM"]
    if mat_mean.ndim == 3:
        mat_mean = mat_mean[:, :, 1]  # ch2
    compare_2d("mean_image_ch2", py_mean, mat_mean)

print("\n" + "=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)
