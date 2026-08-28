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
    """
    Compute per-frame alignment quality as normalized reconstruction error.

    Ports stripRegBergamo.m lines 276-291:
    For each 10-second chunk:
    1. Compute median template from frames in chunk
    2. Apply gamma correction (sqrt)
    3. Compute normalized negative reconstruction error

    Parameters
    ----------
    aligned_ds : np.ndarray
        Spatially downsampled aligned frames, shape (H, W, T).
    align_hz : float
        Alignment sampling rate in Hz.
    running_template : np.ndarray, optional
        Final running template from registration loop. Used to
        compute the gamma-correction offset (MATLAB line 277). If None, falls
        back to computing the offset from aligned_ds.
    chunk_duration_s : float
        Duration of each chunk in seconds.

    Returns
    -------
    np.ndarray
        rec_neg_err
        Reconstruction error per frame, shape (T,).
    """
    n_frames = aligned_ds.shape[2]
    n_chunks = max(1, int(np.ceil(n_frames / (align_hz * chunk_duration_s))))
    chunk_edges = np.round(np.linspace(0, n_frames, n_chunks + 1)).astype(int)

    if running_template is not None:
        valid_vals = running_template[~np.isnan(running_template)]
    else:
        valid_vals = aligned_ds[~np.isnan(aligned_ds)]
    offset = (
        max(0.0, float(np.percentile(valid_vals, 25))) if len(valid_vals) > 0 else 0.0
    )

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

            template = np.nanmedian(chunk_data, axis=2)

            nan_frac = np.mean(np.isnan(chunk_data), axis=2)
            template[nan_frac > 0.2] = np.nan

            template_gamma = np.sqrt(np.maximum(0, template) + offset)

            template_3d = template[:, :, np.newaxis] * np.ones((1, 1, t_end - t_start))
            template_3d[np.isnan(chunk_data)] = np.nan

            tg = template_gamma[:, :, np.newaxis]
            diff = np.maximum(0, (template_3d - chunk_data) / tg)
            numerator = np.nanmean(diff**2, axis=(0, 1))
            denom = np.nanmean((template_3d / tg) ** 2, axis=(0, 1))
            denom = np.maximum(denom, 1e-10)

        rec_neg_err[t_start:t_end] = np.sqrt(numerator / denom)

    return rec_neg_err
