"""Export registered data from zarr to .mat for MATLAB comparison."""

import numpy as np
import scipy.io as sio
from _paths import get_dir_scan

from subcell.io.zarr_store import ExperimentStore

zarr_path = get_dir_scan() / "registered.zarr"

store = ExperimentStore(zarr_path)
adata = store.load_alignment_data(1)
reg_ds = store.load_registered_ds(1)

n_ch = adata.num_channels
n_ds_frames = reg_ds.shape[2] // n_ch
movie_4d = reg_ds.reshape(
    reg_ds.shape[0], reg_ds.shape[1], n_ds_frames, n_ch
).transpose(0, 1, 3, 2)  # (H, W, C, T)

# Activity channel (channel 2, 0-indexed = 1)
act_ch = 1
movie_act = movie_4d[:, :, act_ch, :].copy()

print(f"Movie shape: {movie_act.shape}")
print(f"align_hz: {adata.align_hz}")
print(f"frame_time: {adata.frame_time}")
print(f"num_channels: {adata.num_channels}")
print(f"NaN fraction: {np.mean(np.isnan(movie_act)):.3f}")

out_path = zarr_path.parent / "registered_ds_act.mat"
sio.savemat(
    str(out_path),
    {
        "movie_act": movie_act.astype(np.float32),
        "align_hz": float(adata.align_hz),
        "frame_time": float(adata.frame_time),
        "num_channels": int(adata.num_channels),
    },
    do_compression=True,
)
print(f"Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
