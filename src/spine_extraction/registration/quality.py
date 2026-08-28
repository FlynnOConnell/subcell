"""Alignment quality metrics.

Ports the alignment quality computation from stripRegBergamo.m lines 276-291.
Computes normalized reconstruction error per downsampled frame in 10-second chunks.
"""

from __future__ import annotations

import warnings

import numpy as np


def compute_alignment_quality(
    aligned_ds: np.ndarray,
    align_hz: float,
    running_template: np.ndarray | None = None,
    chunk_duration_s: float = 10.0,
) -> np.ndarray:
    """Compute per-frame alignment quality as normalized reconstruction error.

    Ports stripRegBergamo.m lines 276-291:
    For each 10-second chunk:
    1. Compute median template from frames in chunk
    2. Apply gamma correction (sqrt)
    3. Compute normalized negative reconstruction error

    Args:
        aligned_ds: Spatially downsampled aligned frames, shape (H, W, T).
        align_hz: Alignment sampling rate in Hz.
        running_template: Final running template from registration loop. Used to
            compute the gamma-correction offset (MATLAB line 277). If None, falls
            back to computing the offset from aligned_ds.
        chunk_duration_s: Duration of each chunk in seconds.

    Returns:
        rec_neg_err: Reconstruction error per frame, shape (T,).
    """
    n_frames = aligned_ds.shape[2]
    n_chunks = max(1, int(np.ceil(n_frames / (align_hz * chunk_duration_s))))
    chunk_edges = np.round(np.linspace(0, n_frames, n_chunks + 1)).astype(int)

    # Global offset for gamma correction
    # MATLAB: offset = max(0, prctile(template(:), 25))  — uses the running template
    if running_template is not None:
        valid_vals = running_template[~np.isnan(running_template)]
    else:
        valid_vals = aligned_ds[~np.isnan(aligned_ds)]
    offset = max(0.0, float(np.percentile(valid_vals, 25))) if len(valid_vals) > 0 else 0.0

    rec_neg_err = np.full(n_frames, np.nan)

    for chunk_ix in range(n_chunks):
        t_start = chunk_edges[chunk_ix]
        t_end = chunk_edges[chunk_ix + 1]
        if t_start >= t_end:
            continue

        t_ixs = slice(t_start, t_end)
        chunk_data = aligned_ds[:, :, t_ixs]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            # Compute template as median of chunk
            template = np.nanmedian(chunk_data, axis=2)

            # Mask pixels with too many NaNs
            nan_frac = np.mean(np.isnan(chunk_data), axis=2)
            template[nan_frac > 0.2] = np.nan

            # Gamma correction
            template_gamma = np.sqrt(np.maximum(0, template) + offset)

            # Expand template for broadcasting
            template_3d = template[:, :, np.newaxis] * np.ones((1, 1, t_end - t_start))
            template_3d[np.isnan(chunk_data)] = np.nan

            # Compute normalized negative reconstruction error
            # recNegErr = sqrt(mean(max(0, (T-A)/T_gamma)^2) / mean((T/T_gamma)^2))
            tg = template_gamma[:, :, np.newaxis]
            diff = np.maximum(0, (template_3d - chunk_data) / tg)
            numerator = np.nanmean(diff**2, axis=(0, 1))
            denom = np.nanmean((template_3d / tg) ** 2, axis=(0, 1))
            denom = np.maximum(denom, 1e-10)  # Avoid division by zero

        rec_neg_err[t_start:t_end] = np.sqrt(numerator / denom)

    return rec_neg_err
