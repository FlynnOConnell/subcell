"""Timed step-by-step comparison of Python vs MATLAB source localization.

Loads MATLAB intermediate results and runs each Python step with timing,
comparing numerical equivalency at each stage.

Test data: Mouse 750098, session 2024-09-24, scan test_scan_00001_20240924_110500
MATLAB intermediates exported by: equivalency_tests/matlab_step_by_step.m
"""

import math
import time
import warnings
import numpy as np
import scipy.io as sio

from spine_extraction.io.zarr_store import ExperimentStore
from spine_extraction.config import ExtractionConfig
from spine_extraction.extraction.localize import localize_sources
from spine_extraction.filters.temporal import (
    exponential_matched_filter, moving_median_baseline,
    moving_mean, moving_mad_noise,
)
from spine_extraction.filters.spatial import difference_of_gaussians, nanmedfilt2
from spine_extraction.filters.morphology import spatiotemporal_nms, local_maxima_3x3, apply_density_threshold

from _paths import get_dir_scan

data_dir = get_dir_scan()

timings = {}


def timed(name):
    """Context manager for timing a code block."""
    class Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.t0
            timings[name] = elapsed
            print(f"    [TIME] {name}: {elapsed:.3f}s")
    return Timer()


def compare_traces(label, py, mat):
    mat = mat.flatten()
    vb = ~np.isnan(py) & ~np.isnan(mat)
    if np.sum(vb) < 2:
        print(f"    {label}: insufficient valid overlap ({np.sum(vb)} pts)")
        return
    corr = np.corrcoef(py[vb], mat[vb])[0, 1]
    mae = np.mean(np.abs(py[vb] - mat[vb]))
    rng = np.ptp(mat[vb])
    rel = mae / rng if rng > 0 else float('inf')
    print(f"    {label}: corr={corr:.6f}, MAE={mae:.3e}, rel_err={rel:.5f}")


def compare_frames(label, py_stack, mat_stack, fidx_labels):
    n = min(py_stack.shape[2] if py_stack.ndim == 3 else 1,
            mat_stack.shape[2] if mat_stack.ndim == 3 else 1)
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
    vb = ~np.isnan(py) & ~np.isnan(mat)
    if np.sum(vb) < 2:
        print(f"    {label}: no overlap")
        return
    corr = np.corrcoef(py[vb], mat[vb])[0, 1]
    mae = np.mean(np.abs(py[vb] - mat[vb]))
    rng = np.ptp(mat[vb])
    rel = mae / rng if rng > 0 else float('inf')
    print(f"    {label}: corr={corr:.6f}, MAE={mae:.3e}, rel_err={rel:.5f}")


print("=" * 70)
print("PYTHON vs MATLAB SOURCE LOCALIZATION - TIMED COMPARISON")
print("=" * 70)

# ---- Load Python movie data ----
print("\n--- Loading data ---")
with timed("data_load"):
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
nan_thresh = 0.33

print(f"\nDataset: Mouse 750098, 2024-09-24, test_scan_00001")
print(f"Parameters: tau={tau:.2f}fr, sigma={sigma}, baseline_window={baseline_window}, denoise_window={denoise_window}")
print(f"Movie shape: {IMf.shape} (rows, cols, time), align_hz={align_hz:.1f}")

# ---- Step 0: NaN handling ----
nan_frac = np.mean(np.isnan(IMf), axis=2)
valid = nan_frac < nan_thresh
IMf[np.broadcast_to(~valid[:, :, np.newaxis], IMf.shape)] = np.nan
nan_mask = np.isnan(IMf)
n_time = IMf.shape[2]

print(f"Valid pixels: {np.sum(valid)} / {valid.size} ({100*np.sum(valid)/valid.size:.1f}%)")

# Load MATLAB reference pixel coords
mat1 = sio.loadmat(str(data_dir / "matlab_step1_smooth.mat"))
br, bc = int(mat1['br'][0, 0]) - 1, int(mat1['bc'][0, 0]) - 1
ir, ic = int(mat1['ir'][0, 0]) - 1, int(mat1['ic'][0, 0]) - 1

fidxs = [99, 499, 999, 4999, min(9999, n_time - 1)]
fidx_labels = [100, 500, 1000, 5000, min(10000, n_time)]

# ========== STEP 1: Moving mean smoothing ==========
print("\n--- STEP 1: Moving mean (smoothdata movmean) ---")
with timed("step1_moving_mean"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        IMfden = moving_mean(IMf, window=denoise_window, axis=2)

compare_traces("border_smooth", IMfden[br, bc, :], mat1['step1_border_smooth'])
compare_traces("interior_smooth", IMfden[ir, ic, :], mat1['step1_interior_smooth'])

mat1b = sio.loadmat(str(data_dir / "matlab_step1b_frames.mat"))
compare_frames("smooth", IMfden[:, :, fidxs], mat1b['step1_frames_smooth'], fidx_labels)

# ========== STEP 2: Moving median baseline ==========
print("\n--- STEP 2: Moving median baseline (smoothdata movmedian) ---")
mat2 = sio.loadmat(str(data_dir / "matlab_step2_baseline.mat"))

with timed("step2_moving_median"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        IMb = moving_median_baseline(IMfden, window=baseline_window, axis=2)

IMf_hp = IMf - IMb

compare_traces("border_baseline", IMb[br, bc, :], mat2['step2_border_baseline'])
compare_traces("interior_baseline", IMb[ir, ic, :], mat2['step2_interior_baseline'])
compare_frames("baseline", IMb[:, :, fidxs], mat2['step2_frames_baseline'], fidx_labels)
compare_frames("highpass", IMf_hp[:, :, fidxs], mat2['step2_frames_hp'], fidx_labels)

# ========== STEP 3: MAD noise + Z-score ==========
print("\n--- STEP 3: MAD noise estimation (movmad) ---")
mat3 = sio.loadmat(str(data_dir / "matlab_step3_noise.mat"))

with timed("step3_mad_noise"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        std_IM = moving_mad_noise(IMfden - IMb, window=baseline_window, axis=2)
    std_IM = std_IM * denoise_window

compare_traces("border_std", std_IM[br, bc, :], mat3['step3_border_std'])
compare_traces("interior_std", std_IM[ir, ic, :], mat3['step3_interior_std'])

# Check std ratio
py_std_f = std_IM[:, :, 999]
mat_std_f = mat3['step3_frame_std']
vb = ~np.isnan(py_std_f) & ~np.isnan(mat_std_f)
if np.sum(vb) > 0:
    ratio = py_std_f[vb] / mat_std_f[vb]
    print(f"    std_IM ratio (py/mat): mean={np.mean(ratio):.6f}, std={np.std(ratio):.6f}")

print("\n--- STEP 3b: Z-score normalization ---")
with timed("step3b_zscore"):
    with np.errstate(divide="ignore", invalid="ignore"):
        IMf_z = IMf_hp / std_IM
    non_finite_z = ~np.isfinite(IMf_z)
    IMf_z[non_finite_z] = np.nan
    nan_mask = nan_mask | non_finite_z

compare_traces("border_zscore", IMf_z[br, bc, :], mat3['step3_border_z'])
compare_traces("interior_zscore", IMf_z[ir, ic, :], mat3['step3_interior_z'])
compare_frames("zscore", IMf_z[:, :, fidxs], mat3['step3_frames_z'], fidx_labels)

del IMfden, IMb, IMf_hp, std_IM

# ========== STEP 4: Exponential matched filter ==========
print("\n--- STEP 4: Exponential matched filter ---")
mat4 = sio.loadmat(str(data_dir / "matlab_step4_matchedfilter.mat"))

with timed("step4_matched_filter"):
    IMf_mf = exponential_matched_filter(IMf_z, tau)
    IMf_mf[nan_mask] = np.nan

compare_traces("border_mf", IMf_mf[br, bc, :], mat4['step4_border_mf'])
compare_traces("interior_mf", IMf_mf[ir, ic, :], mat4['step4_interior_mf'])
compare_frames("mf", IMf_mf[:, :, fidxs], mat4['step4_frames'], fidx_labels)

del IMf_z

# ========== STEP 5: Difference of Gaussians ==========
print("\n--- STEP 5: Difference of Gaussians (imgaussfilt) ---")
mat5 = sio.loadmat(str(data_dir / "matlab_step5_dog.mat"))

with timed("step5_dog"):
    IMf_dog = difference_of_gaussians(IMf_mf, sigma, nan_mask=nan_mask)

compare_traces("border_dog", IMf_dog[br, bc, :], mat5['step5_border_dog'])
compare_traces("interior_dog", IMf_dog[ir, ic, :], mat5['step5_interior_dog'])
compare_frames("dog", IMf_dog[:, :, fidxs], mat5['step5_frames'], fidx_labels)

del IMf_mf

# ========== STEP 6: Spatiotemporal NMS + Activity Image ==========
print("\n--- STEP 6: Spatiotemporal NMS + Activity Image ---")
mat6 = sio.loadmat(str(data_dir / "matlab_step6_activity.mat"))

with timed("step6_nms"):
    activity_image, _ = spatiotemporal_nms(IMf_dog, tau, nan_mask=nan_mask)

compare_2d("activity_raw", activity_image, mat6['step6_activity_raw'])

with timed("step6_postprocess"):
    activity_image[~valid] = np.nan
    med_filt = nanmedfilt2(activity_image, 5)
    activity_image = activity_image - med_filt
    activity_image[~valid] = np.nan

compare_2d("activity_final", activity_image, mat6['step6_activity_final'])

# ========== STEP 7: Peak detection ==========
print("\n--- STEP 7: Peak detection & density thresholding ---")
with timed("step7_peaks"):
    peak_mask_py = local_maxima_3x3(activity_image)
    rows_py, cols_py, vals_py = apply_density_threshold(
        activity_image, peak_mask_py,
        max_density=0.01,
        n_time_points=n_time,
        align_hz=align_hz,
        valid_mask=valid,
    )

mat_act = sio.loadmat(str(data_dir / "matlab_activityImage.mat"))
matlab_act_img = mat_act['activityImage']
matlab_peaks_r = mat_act['peaks']['row'][0, 0].flatten()
matlab_peaks_c = mat_act['peaks']['col'][0, 0].flatten()

print(f"    Python peaks:  {len(rows_py)}")
print(f"    MATLAB peaks:  {len(matlab_peaks_r)}")
print(f"    Python activity range: [{np.nanmin(activity_image):.6f}, {np.nanmax(activity_image):.6f}]")
print(f"    MATLAB activity range: [{np.nanmin(matlab_act_img):.6f}, {np.nanmax(matlab_act_img):.6f}]")

# ========== END-TO-END ==========
print("\n--- END-TO-END: Full localize_sources ---")
ext_config = ExtractionConfig(
    microscope="bergamo",
    sigma_px=1.33, dXY=3,
    denoise_window_s=0.2, baseline_window_glu_s=4.0,
    tau_s=0.03, max_synapse_density=0.01,
    nan_thresh=0.33, activity_channel=2,
)

movie_act = movie_4d[:, :, 1, :].copy()
with timed("end_to_end_localize"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        py_act_full, py_peaks_full = localize_sources(movie_act, ext_config, align_hz)

print(f"    localize_sources peaks: {len(py_peaks_full.row)}")
compare_2d("e2e_activity", py_act_full, mat6['step6_activity_final'])

# ========== Source location matching ==========
print("\n--- Source location matching ---")
if len(py_peaks_full.row) > 0 and len(matlab_peaks_r) > 0:
    py_coords = np.column_stack([py_peaks_full.row, py_peaks_full.col])
    mat_coords = np.column_stack([matlab_peaks_r, matlab_peaks_c])

    from scipy.spatial.distance import cdist
    dists = cdist(py_coords, mat_coords)
    min_dists_py = np.min(dists, axis=1)
    min_dists_mat = np.min(dists, axis=0)

    thresholds = [1.0, 2.0, 3.0, 5.0]
    for thr in thresholds:
        py_matched = np.sum(min_dists_py <= thr)
        mat_matched = np.sum(min_dists_mat <= thr)
        print(f"    Within {thr:.0f}px: {py_matched}/{len(py_coords)} Python, "
              f"{mat_matched}/{len(mat_coords)} MATLAB")

    print(f"    Mean closest distance (Py->Mat): {np.mean(min_dists_py):.2f}px")
    print(f"    Mean closest distance (Mat->Py): {np.mean(min_dists_mat):.2f}px")

# ========== Timing Summary ==========
print("\n" + "=" * 70)
print("TIMING SUMMARY")
print("=" * 70)
total = 0
for name, elapsed in timings.items():
    if name != "end_to_end_localize" and name != "data_load":
        total += elapsed
    print(f"  {name:30s} {elapsed:8.3f}s")
print(f"  {'TOTAL (steps 1-7)':30s} {total:8.3f}s")

print("\n" + "=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)
