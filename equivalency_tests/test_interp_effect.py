"""Test effect of interpolation on source count."""

import warnings
import numpy as np
from scipy.ndimage import binary_dilation, label as ndimage_label
import scipy.io as sio

from spine_extraction.io.zarr_store import ExperimentStore
from spine_extraction.config import ExtractionConfig
from spine_extraction.extraction.source_selection import select_sources

from _paths import get_dir_scan

data_dir = get_dir_scan()

store = ExperimentStore(data_dir / "registered.zarr")
adata = store.load_alignment_data(1)
reg_ds = store.load_registered_ds(1)
n_ch = adata.num_channels
n_ds_frames = reg_ds.shape[2] // n_ch
movie_4d = reg_ds.reshape(reg_ds.shape[0], reg_ds.shape[1], n_ds_frames, n_ch).transpose(0, 1, 3, 2)
movie_act = movie_4d[:, :, 1, :]

ext_config = ExtractionConfig(
    microscope="bergamo", sigma_px=1.33, dXY=3, denoise_window_s=0.2,
    baseline_window_glu_s=4.0, tau_s=0.03, max_synapse_density=0.01,
    nan_thresh=0.33, activity_channel=2,
)

valid_pix = np.mean(~np.isnan(np.nanmean(movie_4d, axis=3)), axis=2) > (1 - ext_config.nan_thresh)

def run_and_report(label, act_img):
    act_stack = act_img[:, :, np.newaxis]
    sources = select_sources(act_stack, np.array([0]), ext_config, valid_pixel_mask=valid_pix)

    h, w = act_img.shape
    sigma_radius = int(np.ceil(1.5 * ext_config.sigma_px + 1))
    zones = np.zeros((h, w), dtype=bool)
    for i in range(sources.n_sources):
        r, c = int(round(sources.rows[i])), int(round(sources.cols[i]))
        if 0 <= r < h and 0 <= c < w:
            zones[r, c] = True
    y, x = np.mgrid[-sigma_radius:sigma_radius+1, -sigma_radius:sigma_radius+1]
    disk = (x**2 + y**2) <= sigma_radius**2
    zones = binary_dilation(zones, structure=disk)
    labeled, n_problems = ndimage_label(zones)
    sizes = []
    for i in range(1, n_problems + 1):
        count = sum(1 for j in range(sources.n_sources)
                    if labeled[int(round(sources.rows[j])), int(round(sources.cols[j]))] == i)
        sizes.append(count)
    sizes.sort(reverse=True)
    print(f"  {label}: {sources.n_sources} sources, {n_problems} subproblems, max={max(sizes) if sizes else 0}, sizes={sizes}")

# 1. Current Python (with interpolation)
from spine_extraction.extraction.localize import localize_sources
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    py_act, py_peaks = localize_sources(movie_act, ext_config, adata.align_hz)
print(f"Python with interp: {len(py_peaks.row)} localized")
run_and_report("with_interp", py_act)

# 2. Python without interpolation (matching MATLAB)
# Monkey-patch to skip interpolation
import spine_extraction.extraction.localize as loc_mod
_orig = loc_mod.localize_sources

import math
from spine_extraction.filters.temporal import exponential_matched_filter, moving_median_baseline, moving_mean, moving_mad_noise
from spine_extraction.filters.spatial import difference_of_gaussians, nanmedfilt2
from spine_extraction.filters.morphology import spatiotemporal_nms, local_maxima_3x3, apply_density_threshold
from spine_extraction.extraction.localize import PeakSet

def localize_no_interp(movie, config, align_hz, variance_image=None):
    tau = config.tau_s * align_hz
    sigma = config.sigma_px
    baseline_window = int(math.ceil(config.baseline_window_glu_s * align_hz))
    denoise_window = int(math.ceil(config.denoise_window_s * align_hz))
    n_time = movie.shape[2]
    nan_mask = np.isnan(movie)
    nan_frac_spatial = np.mean(nan_mask, axis=2)
    valid = nan_frac_spatial < config.nan_thresh

    IMf = movie.copy()
    invalid_3d = np.broadcast_to(~valid[:, :, np.newaxis], IMf.shape)
    IMf[invalid_3d] = np.nan
    nan_mask = np.isnan(IMf)

    # NO interpolation - skip it entirely like MATLAB

    IMfden = moving_mean(IMf, window=denoise_window, axis=2)
    IMb = moving_median_baseline(IMfden, window=baseline_window, axis=2)
    IMf = IMf - IMb
    std_IM = moving_mad_noise(IMfden - IMb, window=baseline_window, axis=2)
    std_IM = std_IM * denoise_window
    std_IM = np.maximum(std_IM, 1e-10)
    del IMfden, IMb
    IMf = IMf / std_IM
    del std_IM
    IMf = exponential_matched_filter(IMf, tau)
    IMf[nan_mask] = np.nan
    IMf = difference_of_gaussians(IMf, sigma, nan_mask=nan_mask)
    activity_image, _ = spatiotemporal_nms(IMf, tau, nan_mask=nan_mask)
    del IMf
    activity_image[~valid] = np.nan
    nan_vals = np.isnan(activity_image)
    med_val = np.nanmedian(activity_image) if np.any(~nan_vals) else 0.0
    activity_image[nan_vals] = med_val
    med_filt = nanmedfilt2(activity_image, 5)
    activity_image = activity_image - med_filt
    activity_image[~valid] = np.nan
    peak_mask = local_maxima_3x3(activity_image)
    rows, cols, vals = apply_density_threshold(
        activity_image, peak_mask, config.max_synapse_density,
        n_time_points=n_time, align_hz=align_hz, valid_mask=valid,
    )
    peaks = PeakSet(row=rows.astype(float), col=cols.astype(float), val=vals, peak_image=activity_image)
    return activity_image, peaks

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    py_act_ni, py_peaks_ni = localize_no_interp(movie_act, ext_config, adata.align_hz)
print(f"Python no interp: {len(py_peaks_ni.row)} localized")
run_and_report("no_interp", py_act_ni)

# 3. MATLAB activity image
mat = sio.loadmat(str(data_dir / "matlab_activityImage.mat"))
matlab_act = mat['activityImage']
print(f"MATLAB: {len(mat['peaks']['row'][0,0].flatten())} localized")
run_and_report("matlab_act", matlab_act)

# Compare activity images
valid_both = ~np.isnan(py_act_ni) & ~np.isnan(matlab_act)
corr = np.corrcoef(py_act_ni[valid_both], matlab_act[valid_both])[0, 1]
print(f"\nCorrelation (no_interp vs MATLAB): {corr:.4f}")
corr2 = np.corrcoef(py_act[valid_both], matlab_act[valid_both])[0, 1]
print(f"Correlation (with_interp vs MATLAB): {corr2:.4f}")
