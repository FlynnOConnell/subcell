"""Bilinear interpolation for frame alignment and motion correction.

Uses PyTorch grid_sample for GPU-accelerated interpolation, mapping
MATLAB's interp2(..., 'linear', nan) semantics.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import map_coordinates


def apply_shift_numpy(
    frame: np.ndarray,
    row_shift: float,
    col_shift: float,
    view_r: np.ndarray | None = None,
    view_c: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply sub-pixel shift via bilinear interpolation using NumPy/SciPy.

    Fallback for when PyTorch is not needed. Ports MATLAB:
        interp2(1:cols, 1:rows, frame, viewC + col_shift, viewR + row_shift, 'linear', nan)

    Parameters
    ----------
    frame : np.ndarray
        2D array (rows, cols).
    row_shift : float
        Row shift in pixels.
    col_shift : float
        Col shift in pixels.
    view_r : np.ndarray, optional
        Row coordinates grid (default: 0..rows-1).
    view_c : np.ndarray, optional
        Col coordinates grid (default: 0..cols-1).

    Returns
    -------
    np.ndarray
        Shifted image with NaN for out-of-bounds.
    """

    h, w = frame.shape[:2]

    if view_r is None or view_c is None:
        view_r, view_c = np.mgrid[0:h, 0:w].astype(np.float64)

    coords_r = view_r + row_shift
    coords_c = view_c + col_shift

    oob = (coords_r < 0) | (coords_r > h - 1) | (coords_c < 0) | (coords_c > w - 1)

    frame_clean = np.where(np.isnan(frame), 0.0, frame)
    result = map_coordinates(
        frame_clean, [coords_r, coords_c], order=1, mode="constant", cval=0.0
    )
    result[oob] = np.nan
    return result


def apply_shifts_batch(
    frames: np.ndarray,
    row_shifts: np.ndarray,
    col_shifts: np.ndarray,
    out_h: int,
    out_w: int,
    view_r_start: float,
    view_c_start: float,
    device: torch.device | None = None,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Apply per-frame shifts to a stack of frames using batched grid_sample.

    Uses torch grid_sample on both GPU and CPU — the batched C++ kernel is
    ~3x faster than a per-frame scipy map_coordinates loop even on CPU.

    Parameters
    ----------
    frames : np.ndarray
        Input frames, shape (H, W, N).
    row_shifts : np.ndarray
        Per-frame row shifts, shape (N,).
    col_shifts : np.ndarray
        Per-frame col shifts, shape (N,).
    out_h : int
        Output height.
    out_w : int
        Output width.
    view_r_start : float
        Starting row coordinate of the output grid (typically -maxshift).
    view_c_start : float
        Starting col coordinate of the output grid (typically -maxshift).
    device : torch.device, optional
        Torch device (default: CPU). GPU gives ~5x over CPU.
    batch_size : int
        Frames per batch (reduce if GPU OOM).

    Returns
    -------
    np.ndarray
        Shifted frames, shape (out_h, out_w, N), with NaN for out-of-bounds.
    """
    h, w = frames.shape[:2]
    n_frames = frames.shape[2]
    result = np.zeros((out_h, out_w, n_frames), dtype=np.float32)

    if device is None:
        device = torch.device("cpu")

    row_coords = torch.arange(out_h, dtype=torch.float32, device=device)
    col_coords = torch.arange(out_w, dtype=torch.float32, device=device)
    base_r = row_coords + view_r_start
    base_c = col_coords + view_c_start
    base_r_norm = 2.0 * base_r / (h - 1) - 1.0
    base_c_norm = 2.0 * base_c / (w - 1) - 1.0
    grid_r_base, grid_c_base = torch.meshgrid(base_r_norm, base_c_norm, indexing="ij")

    is_gpu = device.type != "cpu"

    for start in range(0, n_frames, batch_size):
        end = min(start + batch_size, n_frames)

        frames_t = torch.from_numpy(
            np.ascontiguousarray(frames[:, :, start:end].transpose(2, 0, 1))
        ).unsqueeze(1)
        if is_gpu:
            frames_t = frames_t.to(device)

        dr = torch.from_numpy(row_shifts[start:end].astype(np.float32))
        dc = torch.from_numpy(col_shifts[start:end].astype(np.float32))
        if is_gpu:
            dr = dr.to(device)
            dc = dc.to(device)
        dr_norm = 2.0 * dr / (h - 1)
        dc_norm = 2.0 * dc / (w - 1)

        grid_r = grid_r_base.unsqueeze(0) + dr_norm.view(-1, 1, 1)
        grid_c = grid_c_base.unsqueeze(0) + dc_norm.view(-1, 1, 1)
        grid = torch.stack([grid_c, grid_r], dim=-1)

        out = F.grid_sample(
            frames_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )  # (N, 1, out_h, out_w)

        oob = (
            (grid[..., 0] < -1)
            | (grid[..., 0] > 1)
            | (grid[..., 1] < -1)
            | (grid[..., 1] > 1)
        )
        out[:, 0][oob] = float("nan")

        if is_gpu:
            result[:, :, start:end] = out.squeeze(1).permute(1, 2, 0).cpu().numpy()
        else:
            result[:, :, start:end] = out.squeeze(1).permute(1, 2, 0).numpy()

    return result
