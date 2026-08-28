"""Quick test: compare Python localize_sources with MATLAB results."""

import matplotlib
import numpy as np
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
from _paths import get_dir_scan

from subcell.config import ExtractionConfig
from subcell.extraction.localize import localize_sources
from subcell.io.zarr_store import ExperimentStore

data_dir = get_dir_scan("scan_00001_20240924_110500")
zarr_path = data_dir / "registered.zarr"
mat_path = data_dir / "matlab_localization_result.mat"

# Load MATLAB results
mat = scipy.io.loadmat(str(mat_path), squeeze_me=True)
mat_actIM = mat["actIM"]
mat_peaks_row = np.atleast_1d(mat["peaks"]["row"].item())
mat_peaks_col = np.atleast_1d(mat["peaks"]["col"].item())
mat_meanIM = mat["meanIM"]  # (159, 75, 2) in MATLAB
print(f"MATLAB: {mat_actIM.shape} activity image, {len(mat_peaks_row)} peaks")
print(f"MATLAB meanIM shape: {mat_meanIM.shape}")

# Load Python data
store = ExperimentStore(zarr_path)
adata = store.load_alignment_data(1)
reg_ds = store.load_registered_ds(1)
n_ch = adata.num_channels
n_ds_frames = reg_ds.shape[2] // n_ch
movie_4d = reg_ds.reshape(
    reg_ds.shape[0], reg_ds.shape[1], n_ds_frames, n_ch
).transpose(0, 1, 3, 2)  # (H, W, C, T)

config = ExtractionConfig(
    microscope="bergamo",
    sigma_px=1.33,
    nmf_iter=2,
    dXY=3,
    denoise_window_s=0.2,
    baseline_window_glu_s=4.0,
    tau_s=0.03,
    max_synapse_density=0.01,
    motion_thresh=2.5,
    nan_thresh=0.33,
    activity_channel=2,
)

act_ch = config.activity_channel - 1
movie_act = movie_4d[:, :, act_ch, :]
py_mean_ch2 = np.nanmean(movie_act, axis=2)
print(f"Python movie: {movie_act.shape}, align_hz={adata.align_hz:.1f}")

# Run localization
import time

t0 = time.time()
act_img, peaks = localize_sources(movie_act, config, adata.align_hz)
elapsed = time.time() - t0
print(f"Python: {act_img.shape} activity image, {len(peaks.row)} peaks, {elapsed:.1f}s")
print(f"  MATLAB peaks: {len(mat_peaks_row)}")
print(f"  Python peaks: {len(peaks.row)}")

# Create comparison figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Python: peaks on mean image
ax = axes[0]
vmin, vmax = np.nanpercentile(py_mean_ch2, [1, 99.5])
ax.imshow(py_mean_ch2, cmap="gray", vmin=vmin, vmax=vmax)
if len(peaks.row) > 0:
    ax.scatter(
        peaks.col, peaks.row, s=40, facecolors="none", edgecolors="r", linewidths=1.5
    )
ax.set_title(f"Python Ch2 mean + {len(peaks.row)} peaks")
ax.axis("off")

# MATLAB: peaks on mean image (ch2)
ax = axes[1]
mat_ch2 = mat_meanIM[:, :, 1]  # channel 2 (0-indexed in loaded array)
vmin_m, vmax_m = np.nanpercentile(mat_ch2[mat_ch2 > 0], [1, 99.5])
ax.imshow(mat_ch2, cmap="gray", vmin=vmin_m, vmax=vmax_m)
ax.scatter(
    mat_peaks_col,
    mat_peaks_row,
    s=40,
    facecolors="none",
    edgecolors="r",
    linewidths=1.5,
)
ax.set_title(f"MATLAB Ch2 mean + {len(mat_peaks_row)} peaks")
ax.axis("off")

fig.suptitle("Source Localization: Python vs MATLAB", fontweight="bold")
plt.tight_layout()
out_path = data_dir / "python_vs_matlab_peaks.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

# Also save activity image comparison
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

ax = axes2[0]
vmax_p = np.nanpercentile(act_img, 99.5) if np.any(~np.isnan(act_img)) else 1
ax.imshow(act_img, cmap="hot", vmin=0, vmax=vmax_p)
if len(peaks.row) > 0:
    ax.scatter(peaks.col, peaks.row, s=20, c="cyan", marker="+", linewidths=0.8)
ax.set_title(f"Python activity image ({len(peaks.row)} peaks)")
ax.axis("off")

ax = axes2[1]
mat_act_clean = np.where(np.isfinite(mat_actIM), mat_actIM, np.nan)
vmax_m = (
    np.nanpercentile(mat_act_clean, 99.5) if np.any(np.isfinite(mat_act_clean)) else 1
)
ax.imshow(mat_act_clean, cmap="hot", vmin=0, vmax=vmax_m)
ax.scatter(mat_peaks_col, mat_peaks_row, s=20, c="cyan", marker="+", linewidths=0.8)
ax.set_title(f"MATLAB activity image ({len(mat_peaks_row)} peaks)")
ax.axis("off")

fig2.suptitle("Activity Images: Python vs MATLAB", fontweight="bold")
plt.tight_layout()
out_path2 = data_dir / "python_vs_matlab_activity.png"
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path2}")
