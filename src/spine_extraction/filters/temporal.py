"""Temporal filtering operations.

Ports the temporal processing from localizeSources_vIM.m:
- Exponential matched filter (backward causal)
- Moving median baseline subtraction
- MAD-based robust noise estimation
"""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Number of parallel threads for rolling operations
_N_WORKERS = min(os.cpu_count() or 1, 8)


def _rolling_chunked(
    flat: np.ndarray, filter_w: int, dtype: np.dtype, method: str = "median"
) -> np.ndarray:
    """Apply pandas rolling statistic in parallel across row chunks.

    Splits the (n_pixels, n_time) array into chunks of rows and processes
    each in a separate thread. Pandas rolling is Cython-based and releases
    the GIL, so threads run truly in parallel.

    Args:
        flat: 2D array (n_pixels, n_time).
        filter_w: Rolling window size.
        dtype: Output dtype.
        method: "median" or "mean".
    """
    n_rows = flat.shape[0]
    if n_rows == 0:
        return flat.copy()

    n_chunks = min(_N_WORKERS, max(1, n_rows // 100))
    if n_chunks <= 1:
        df = pd.DataFrame(flat.T)
        roller = df.rolling(filter_w, center=True, min_periods=1)
        return getattr(roller, method)().values.T.astype(dtype)

    chunk_size = (n_rows + n_chunks - 1) // n_chunks

    def process_chunk(start: int) -> np.ndarray:
        end = min(start + chunk_size, n_rows)
        df = pd.DataFrame(flat[start:end].T)
        roller = df.rolling(filter_w, center=True, min_periods=1)
        return getattr(roller, method)().values.T.astype(dtype)

    with ThreadPoolExecutor(max_workers=n_chunks) as pool:
        futures = [
            pool.submit(process_chunk, i * chunk_size)
            for i in range(n_chunks)
            if i * chunk_size < n_rows
        ]
        results = [f.result() for f in futures]

    return np.concatenate(results, axis=0)


def exponential_matched_filter(
    movie: np.ndarray, tau_frames: float
) -> np.ndarray:
    """Apply backward causal exponential matched filter.

    Ports localizeSources_vIM.m lines 72-81:
        gamma = exp(-1/tau);
        mem = max(0, gamma*IMf(:,:,end));
        for t = size(IMf,3):-1:1
            IMt = IMf(:,:,t);
            nanst = isnan(IMt);
            IMt(nanst) = mem(nanst);
            IMf(:,:,t) = gamma*mem + (1-gamma)*IMt;
            mem = IMf(:,:,t);
        end

    This filter emphasizes transients with decay time constant tau, running
    backward through time so that the peak of a transient is at its onset.

    Args:
        movie: 3D array (rows, cols, time).
        tau_frames: Decay time constant in frames.

    Returns:
        Filtered movie (in-place modification, also returned).
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


def moving_median_baseline(
    data: np.ndarray, window: int, axis: int = 2
) -> np.ndarray:
    """Moving median for baseline estimation.

    Ports: smoothdata(IMf, 3, 'movmedian', baselineWindow, 'omitnan')
    MATLAB's 'omitnan' excludes NaN values from the median computation.

    Uses pandas rolling median which:
    - Employs an efficient O(log w) skiplist algorithm per element
    - Handles NaN natively via min_periods=1 (matching MATLAB 'omitnan')
    - Shrinks window at edges (matching MATLAB endpoint behavior)

    Args:
        data: N-D array.
        window: Window size in samples.
        axis: Axis along which to compute.

    Returns:
        Baseline estimate.
    """
    if axis != -1 and axis != data.ndim - 1:
        data = np.moveaxis(data, axis, -1)

    shape = data.shape
    flat = data.reshape(-1, shape[-1])

    # Match original loop's effective window: 2*(window//2)+1 (always odd)
    half_w = window // 2
    filter_w = 2 * half_w + 1

    logger.debug(
        "moving_median_baseline: %d rows x %d cols, window=%d",
        flat.shape[0], flat.shape[1], filter_w,
    )

    result = _rolling_chunked(flat, filter_w, data.dtype, method="median")

    result = result.reshape(shape)
    if axis != -1 and axis != data.ndim - 1:
        result = np.moveaxis(result, -1, axis)
    return result


def moving_mean(
    data: np.ndarray, window: int, axis: int = 2
) -> np.ndarray:
    """Moving mean with NaN handling.

    Ports: smoothdata(IMf, 3, 'movmean', denoiseWindow, 'omitnan')

    Uses bottleneck.move_mean (C implementation) with NaN-padding trick
    to produce centered windows that shrink at edges, matching MATLAB's
    smoothdata endpoint behavior.  Falls back to pandas rolling if
    bottleneck is not installed.

    Args:
        data: N-D array.
        window: Window size in samples.
        axis: Axis along which to compute.

    Returns:
        Smoothed array.
    """
    if axis != -1 and axis != data.ndim - 1:
        data = np.moveaxis(data, axis, -1)

    shape = data.shape
    flat = data.reshape(-1, shape[-1])
    n_px, n_time = flat.shape

    # Match original loop's effective window: 2*(window//2)+1 (always odd)
    half_w = window // 2
    filter_w = 2 * half_w + 1

    logger.debug(
        "moving_mean: %d pixels x %d time, window=%d",
        n_px, n_time, filter_w,
    )

    try:
        import bottleneck as bn

        # bottleneck.move_mean is left-aligned: result[t] = mean(x[t-w+1:t+1]).
        # To get centered windows, pad the right side with NaN and then
        # take result[half_w : half_w + n_time].  NaN values (including
        # padding) are excluded when min_count=1, so edges naturally shrink.
        padded = np.full((n_px, n_time + half_w), np.nan, dtype=np.float64)
        padded[:, :n_time] = flat
        result_padded = bn.move_mean(padded, window=filter_w, min_count=1, axis=1)
        result = result_padded[:, half_w:half_w + n_time].astype(data.dtype)
    except ImportError:
        result = _rolling_chunked(flat, filter_w, data.dtype, method="mean")

    result = result.reshape(shape)
    if axis != -1 and axis != data.ndim - 1:
        result = np.moveaxis(result, -1, axis)
    return result


def _movmad_pixel_batch(
    flat_batch: np.ndarray, filter_w: int, half_w: int
) -> np.ndarray:
    """Compute per-window MAD for a batch of pixel time series.

    Uses numpy sliding_window_view + bottleneck.nanmedian for speed.
    Matches MATLAB's movmad exactly: for each window, computes
    median(|window - median(window)|).
    """
    from numpy.lib.stride_tricks import sliding_window_view

    try:
        import bottleneck as bn
        _nanmedian = bn.nanmedian
        _median = bn.median
    except ImportError:
        _nanmedian = np.nanmedian
        _median = np.median

    n_px, n_time = flat_batch.shape
    out = np.full_like(flat_batch, np.nan)

    # Pre-allocate buffers (reused per pixel)
    padded = np.empty(n_time + 2 * half_w, dtype=flat_batch.dtype)
    abs_dev = np.empty((n_time, filter_w), dtype=flat_batch.dtype)

    for i in range(n_px):
        ts = flat_batch[i]
        has_nan = np.any(np.isnan(ts))
        if has_nan and np.all(np.isnan(ts)):
            continue

        padded[:] = np.nan
        padded[half_w : half_w + n_time] = ts

        windows = sliding_window_view(padded, filter_w)  # (n_time, filter_w) view

        if has_nan:
            with np.errstate(all="ignore"):
                med = _nanmedian(windows, axis=1)
                np.abs(windows - med[:, np.newaxis], out=abs_dev)
                out[i] = _nanmedian(abs_dev, axis=1)
        else:
            # NaN-free pixel: only edge windows (from padding) have NaN
            interior_s = half_w
            interior_e = n_time - half_w

            if interior_e > interior_s:
                int_win = windows[interior_s:interior_e]  # contiguous, no NaN
                med_int = _median(int_win, axis=1)
                int_dev = abs_dev[interior_s:interior_e]
                np.abs(int_win - med_int[:, np.newaxis], out=int_dev)
                out[i, interior_s:interior_e] = _median(int_dev, axis=1)

            # Edge windows (have NaN from padding)
            with np.errstate(all="ignore"):
                for region in (slice(0, interior_s), slice(interior_e, n_time)):
                    edge_win = windows[region]
                    if edge_win.shape[0] == 0:
                        continue
                    edge_dev = abs_dev[region]
                    med_e = _nanmedian(edge_win, axis=1)
                    np.abs(edge_win - med_e[:, np.newaxis], out=edge_dev)
                    out[i, region] = _nanmedian(edge_dev, axis=1)

    return out


def moving_mad_noise(
    data: np.ndarray, window: int, axis: int = 2
) -> np.ndarray:
    """Moving median-absolute-deviation noise estimation.

    Ports localizeSources_vIM.m line 65:
        stdIM = movmad(IMf, baselineWindow, 3, 'omitmissing') / 0.6741891400433162;

    MATLAB's ``movmad`` computes ``median(|x - median(x)|)`` — the
    *median* absolute deviation from the *median* — for each sliding
    window.  The result is divided by the constant 0.674… to convert
    to a standard-deviation estimate.

    Computes the true per-window MAD (matching MATLAB exactly) using
    numpy sliding_window_view for each pixel, processed in parallel
    chunks.

    Args:
        data: 3D array (rows, cols, time).
        window: Window size in frames.
        axis: Time axis.

    Returns:
        Robust standard deviation estimate, same shape as input.
    """
    MAD_TO_STD = 0.6741891400433162

    if axis != -1 and axis != data.ndim - 1:
        data = np.moveaxis(data, axis, -1)

    shape = data.shape
    flat = data.reshape(-1, shape[-1])
    n_pixels = flat.shape[0]

    half_w = window // 2
    filter_w = 2 * half_w + 1

    logger.debug(
        "moving_mad_noise: %d pixels x %d time, window=%d",
        n_pixels, flat.shape[1], filter_w,
    )

    # Process in parallel chunks
    chunk_size = max(1, n_pixels // _N_WORKERS)
    n_chunks = (n_pixels + chunk_size - 1) // chunk_size

    def process_chunk(start: int) -> np.ndarray:
        end = min(start + chunk_size, n_pixels)
        return _movmad_pixel_batch(flat[start:end], filter_w, half_w)

    if n_chunks <= 1:
        result_flat = process_chunk(0)
    else:
        with ThreadPoolExecutor(max_workers=min(n_chunks, _N_WORKERS)) as pool:
            futures = [
                pool.submit(process_chunk, i * chunk_size)
                for i in range(n_chunks)
                if i * chunk_size < n_pixels
            ]
            results = [f.result() for f in futures]
        result_flat = np.concatenate(results, axis=0)

    result = (result_flat / MAD_TO_STD).reshape(shape)
    if axis != -1 and axis != data.ndim - 1:
        result = np.moveaxis(result, -1, axis)
    return result
