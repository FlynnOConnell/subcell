"""Temporal filtering, porting the time-domain steps of localizeSources_vIM.m."""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor

import bottleneck as bn
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger(__name__)

_N_WORKERS = min(os.cpu_count() or 1, 8)
_MAD_TO_STD = 0.6741891400433162


def _centered_window(window: int) -> tuple[int, int]:
    """Half-width and odd full width of a centered window."""
    half_w = window // 2
    return half_w, 2 * half_w + 1


def _rolling_median(flat: np.ndarray, filter_w: int, dtype: np.dtype) -> np.ndarray:
    """Centered rolling median over the last axis, NaN excluded."""
    df = pd.DataFrame(flat.T)
    rolled = df.rolling(filter_w, center=True, min_periods=1).median()
    return rolled.values.T.astype(dtype)


def _rolling_median_chunked(
    flat: np.ndarray, filter_w: int, dtype: np.dtype
) -> np.ndarray:
    """
    Rolling median over row chunks in parallel.

    pandas rolling is Cython-based and releases the GIL, so the threads do run
    concurrently.

    Parameters
    ----------
    flat : np.ndarray
        2D array, (n_series, n_time).
    """
    n_rows = flat.shape[0]
    if n_rows == 0:
        return flat.copy()

    n_chunks = min(_N_WORKERS, max(1, n_rows // 100))
    if n_chunks <= 1:
        return _rolling_median(flat, filter_w, dtype)

    chunk_size = (n_rows + n_chunks - 1) // n_chunks
    with ThreadPoolExecutor(max_workers=n_chunks) as pool:
        futures = [
            pool.submit(
                _rolling_median, flat[start : start + chunk_size], filter_w, dtype
            )
            for start in range(0, n_rows, chunk_size)
        ]
        return np.concatenate([f.result() for f in futures], axis=0)


def _movmad_pixel_batch(
    flat_batch: np.ndarray, filter_w: int, half_w: int
) -> np.ndarray:
    """
    Per-window MAD for a batch of time series, matching MATLAB movmad.

    Each window yields ``median(|window - median(window)|)``. Windows are
    NaN-padded at both ends so edge windows shrink as MATLAB's do.

    Parameters
    ----------
    flat_batch : np.ndarray
        2D array, (n_series, n_time).

    Returns
    -------
    np.ndarray
        MAD per sample, same shape as ``flat_batch``, NaN for all-NaN series.
    """
    n_px, n_time = flat_batch.shape
    out = np.full_like(flat_batch, np.nan)

    padded = np.empty(n_time + 2 * half_w, dtype=flat_batch.dtype)
    abs_dev = np.empty((n_time, filter_w), dtype=flat_batch.dtype)

    for i in range(n_px):
        ts = flat_batch[i]
        has_nan = bool(np.any(np.isnan(ts)))
        if has_nan and np.all(np.isnan(ts)):
            continue

        padded[:] = np.nan
        padded[half_w : half_w + n_time] = ts
        windows = sliding_window_view(padded, filter_w)

        if has_nan:
            with np.errstate(all="ignore"):
                med = bn.nanmedian(windows, axis=1)
                np.abs(windows - med[:, np.newaxis], out=abs_dev)
                out[i] = bn.nanmedian(abs_dev, axis=1)
            continue

        interior_s, interior_e = half_w, n_time - half_w
        if interior_e > interior_s:
            int_win = windows[interior_s:interior_e]
            int_dev = abs_dev[interior_s:interior_e]
            med_int = bn.median(int_win, axis=1)
            np.abs(int_win - med_int[:, np.newaxis], out=int_dev)
            out[i, interior_s:interior_e] = bn.median(int_dev, axis=1)

        with np.errstate(all="ignore"):
            for region in (slice(0, interior_s), slice(interior_e, n_time)):
                edge_win = windows[region]
                if edge_win.shape[0] == 0:
                    continue
                edge_dev = abs_dev[region]
                med_e = bn.nanmedian(edge_win, axis=1)
                np.abs(edge_win - med_e[:, np.newaxis], out=edge_dev)
                out[i, region] = bn.nanmedian(edge_dev, axis=1)

    return out


def exponential_matched_filter(movie: np.ndarray, tau_frames: float) -> np.ndarray:
    """
    Backward causal exponential matched filter.

    Runs backward through time so a transient peaks at its onset. NaN samples
    inherit the running memory rather than propagating.

    Parameters
    ----------
    movie : np.ndarray
        3D array, (rows, cols, time). Modified in place.
    tau_frames : float
        Decay time constant in frames.

    Returns
    -------
    np.ndarray
        The same array, filtered.
    """
    gamma = math.exp(-1.0 / tau_frames)
    mem = np.maximum(0, gamma * movie[:, :, -1])

    for t in range(movie.shape[2] - 1, -1, -1):
        frame = movie[:, :, t].copy()
        nan_mask = np.isnan(frame)
        frame[nan_mask] = mem[nan_mask]
        movie[:, :, t] = gamma * mem + (1 - gamma) * frame
        mem = movie[:, :, t].copy()

    return movie


def moving_median_baseline(data: np.ndarray, window: int, axis: int = 2) -> np.ndarray:
    """
    Moving median baseline, porting smoothdata(..., 'movmedian', 'omitnan').

    Parameters
    ----------
    data : np.ndarray
        N-D array.
    window : int
        Window size in samples; rounded up to the next odd width.
    axis : int
        Time axis.

    Returns
    -------
    np.ndarray
        Baseline estimate, same shape as ``data``.
    """
    moved = axis != -1 and axis != data.ndim - 1
    if moved:
        data = np.moveaxis(data, axis, -1)

    shape = data.shape
    _, filter_w = _centered_window(window)
    logger.debug(
        "moving_median_baseline: %d series x %d time, window=%d",
        int(np.prod(shape[:-1])),
        shape[-1],
        filter_w,
    )

    result = _rolling_median_chunked(
        data.reshape(-1, shape[-1]), filter_w, data.dtype
    ).reshape(shape)
    return np.moveaxis(result, -1, axis) if moved else result


def moving_mean(data: np.ndarray, window: int, axis: int = 2) -> np.ndarray:
    """
    Moving mean, porting smoothdata(..., 'movmean', 'omitnan').

    ``bn.move_mean`` is left-aligned, so the series is padded on the right with
    NaN and the result shifted back; with ``min_count=1`` the padding drops out
    and edge windows shrink as MATLAB's do.

    Parameters
    ----------
    data : np.ndarray
        N-D array.
    window : int
        Window size in samples; rounded up to the next odd width.
    axis : int
        Time axis.

    Returns
    -------
    np.ndarray
        Smoothed array, same shape as ``data``.
    """
    moved = axis != -1 and axis != data.ndim - 1
    if moved:
        data = np.moveaxis(data, axis, -1)

    shape = data.shape
    flat = data.reshape(-1, shape[-1])
    n_px, n_time = flat.shape
    half_w, filter_w = _centered_window(window)
    logger.debug("moving_mean: %d series x %d time, window=%d", n_px, n_time, filter_w)

    padded = np.full((n_px, n_time + half_w), np.nan, dtype=np.float64)
    padded[:, :n_time] = flat
    smoothed = bn.move_mean(padded, window=filter_w, min_count=1, axis=1)

    result = smoothed[:, half_w : half_w + n_time].astype(data.dtype).reshape(shape)
    return np.moveaxis(result, -1, axis) if moved else result


def moving_mad_noise(data: np.ndarray, window: int, axis: int = 2) -> np.ndarray:
    """
    Robust noise estimate from the moving median absolute deviation.

    Ports ``movmad(IMf, baselineWindow, 3, 'omitmissing') / 0.674...``, which
    converts the MAD to a standard-deviation estimate.

    Parameters
    ----------
    data : np.ndarray
        N-D array.
    window : int
        Window size in samples; rounded up to the next odd width.
    axis : int
        Time axis.

    Returns
    -------
    np.ndarray
        Standard-deviation estimate, same shape as ``data``.
    """
    moved = axis != -1 and axis != data.ndim - 1
    if moved:
        data = np.moveaxis(data, axis, -1)

    shape = data.shape
    flat = data.reshape(-1, shape[-1])
    n_pixels = flat.shape[0]
    half_w, filter_w = _centered_window(window)
    logger.debug(
        "moving_mad_noise: %d series x %d time, window=%d",
        n_pixels,
        shape[-1],
        filter_w,
    )

    chunk_size = max(1, n_pixels // _N_WORKERS)
    if n_pixels <= chunk_size:
        result_flat = _movmad_pixel_batch(flat, filter_w, half_w)
    else:
        with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
            futures = [
                pool.submit(
                    _movmad_pixel_batch,
                    flat[start : start + chunk_size],
                    filter_w,
                    half_w,
                )
                for start in range(0, n_pixels, chunk_size)
            ]
            result_flat = np.concatenate([f.result() for f in futures], axis=0)

    result = (result_flat / _MAD_TO_STD).reshape(shape)
    return np.moveaxis(result, -1, axis) if moved else result
