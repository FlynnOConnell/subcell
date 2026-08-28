"""Baseline estimation for NMF signal extraction.

Ports splitFreq, fitB, and getSurround from extractTrial.m lines 315-381.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time as _time
import warnings

import numpy as np
import psutil
import torch
from scipy.fft import irfft2, next_fast_len, rfft2
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d, rank_filter
from scipy.ndimage import gaussian_filter as gf

logger = logging.getLogger(__name__)

_TARGET_VRAM_PER_SUBPROBLEM = 2 * 1024**3  # 2 GB


class _VRAMBudget:
    """
    Coordinate GPU memory across concurrent subproblems.

    Serializes GPU access so only ONE thread runs GPU FFT at a time.
    This avoids deadlocks from concurrent CUDA submissions via the
    Windows WDDM driver (observed with ThreadPoolExecutor + cuFFT).

    Threads that cannot fit on the GPU fall back to CPU immediately.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._in_use = False
        self._usable = 0

    def _get_usable(self) -> int:
        if self._usable == 0:
            free = torch.cuda.mem_get_info()[0]
            self._usable = int(free * 0.60)
            logger.debug(
                "VRAM budget: %.1f GB free, %.1f GB usable (60%%)",
                free / 1024**3,
                self._usable / 1024**3,
            )
        return self._usable

    def reserve(self, needed_bytes: int, timeout: float = 120.0) -> bool:
        """
        Reserve exclusive GPU access, waiting for the current user to finish.

        Returns True if the reservation succeeded, False only if the
        subproblem is too large to ever fit or the timeout expired.
        """
        deadline = _time.monotonic() + timeout
        with self._cond:
            usable = self._get_usable()
            if needed_bytes > usable:
                return False
            while self._in_use:
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "VRAM reserve timed out after %.0fs (need %.0f MB, "
                        "usable %.0f MB)",
                        timeout,
                        needed_bytes / 1024**2,
                        usable / 1024**2,
                    )
                    return False
                logger.debug(
                    "Waiting for GPU (need %.0f MB)…",
                    needed_bytes / 1024**2,
                )
                self._cond.wait(timeout=remaining)
            self._in_use = True
            return True

    def release(self, amount: int) -> None:
        with self._cond:
            self._in_use = False
            self._cond.notify_all()


_vram_budget = _VRAMBudget()


def split_freq(
    data: np.ndarray,
    denoise_window: int,
    lp_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split signal into low-pass and high-pass components.

    Ports extractTrial.m splitFreq (lines 343-370):
    1. Reshape into pages of denoise_window samples
    2. Average each page
    3. Lowess smoothing of page averages
    4. Remove upward outliers, re-smooth
    5. Interpolate back to original time base

    Parameters
    ----------
    data : np.ndarray
        Input data, shape (n_pixels, n_time).
    denoise_window : int
        Samples per page for downsampling.
    lp_factor : int
        Smoothing factor for lowess.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (low_pass, high_pass), each shape (n_pixels, n_time).
    """
    n_pixels, n_time = data.shape
    n_pages = n_time // denoise_window
    tot_samps = n_pages * denoise_window

    if n_pages < 2:
        lp = np.broadcast_to(np.nanmean(data, axis=1, keepdims=True), data.shape).copy()
        return lp, data - lp

    paged = data[:, :tot_samps].reshape(n_pixels, n_pages, denoise_window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        page_means = np.nanmean(paged, axis=2)  # (n_pixels, n_pages)

    smoothed = _lowess_smooth(page_means, lp_factor)

    residual = page_means - smoothed
    low_vals = residual <= _local_order_stat(residual, lp_factor, 0.15)
    page_means_clean = page_means.copy()
    page_means_clean[~low_vals] = np.nan

    smoothed = _lowess_smooth(page_means_clean, lp_factor)

    for _ in range(3):
        nans = np.isnan(smoothed)
        if not np.any(nans):
            break
        tmp = _lowess_smooth(smoothed, lp_factor)
        smoothed[nans] = tmp[nans]

    t_ds = (denoise_window + 1) / 2 + denoise_window * np.arange(n_pages)
    t_full = np.arange(n_time, dtype=np.float64)

    nan_rows = np.isnan(smoothed).any(axis=1)
    lp = np.empty_like(data)

    if not np.any(nan_rows):
        f = interp1d(t_ds, smoothed, kind="linear", fill_value="extrapolate", axis=1)
        lp[:] = f(t_full)
    else:
        clean = ~nan_rows
        if np.any(clean):
            f = interp1d(
                t_ds, smoothed[clean], kind="linear", fill_value="extrapolate", axis=1
            )
            lp[clean] = f(t_full)

        for px in np.where(nan_rows)[0]:
            valid = ~np.isnan(smoothed[px])
            if np.sum(valid) >= 2:
                lp[px] = np.interp(t_full, t_ds[valid], smoothed[px, valid])
            else:
                lp[px] = np.nanmean(data[px])

    hp = data - lp
    return lp, hp


def _lowess_smooth(data: np.ndarray, window: int) -> np.ndarray:
    """
    LOWESS (locally weighted scatterplot smoothing) along axis=1.

    Matches MATLAB ``smoothdata(..., 'lowess', window, 'omitmissing')``:
    tricube weight function with local linear (degree-1) regression.

    For interior positions with a full symmetric window and no NaN, the
    local linear regression intercept simplifies to a tricube-weighted
    moving average (because sum(w_j * t_j) = 0 by symmetry).  This is
    computed via a single fast C-level ``convolve1d`` call.

    Edge positions (first/last ``half_w`` columns) and any rows containing
    NaN use a per-column fallback with full local linear regression.

    Parameters
    ----------
    data : np.ndarray
        2D array (n_rows, n_cols).
    window : int
        Window size in number of points.

    Returns
    -------
    np.ndarray
        Smoothed array with same shape.
    """

    n_rows, n_cols = data.shape
    window = max(1, window)
    if n_cols <= 1 or window <= 1:
        return data.copy()

    half_w = window // 2

    t = np.arange(-half_w, half_w + 1, dtype=np.float64)
    h = half_w + 1.0
    u = np.abs(t) / h
    kernel = np.where(u < 1.0, (1.0 - u**3) ** 3, 0.0)
    kernel /= kernel.sum()

    has_nan = np.isnan(data).any()

    if not has_nan:
        result = convolve1d(
            data.astype(np.float64),
            kernel,
            axis=1,
            mode="nearest",
        )
        if half_w > 0 and n_cols > 1:
            _lowess_edges(data, result, half_w, h)
        return result.astype(data.dtype)

    nan_row_mask = np.isnan(data).any(axis=1)
    clean_rows = ~nan_row_mask

    result = np.empty((n_rows, n_cols), dtype=np.float64)

    if clean_rows.any():
        result[clean_rows] = convolve1d(
            data[clean_rows].astype(np.float64),
            kernel,
            axis=1,
            mode="nearest",
        )

    if nan_row_mask.any():
        nan_data = data[nan_row_mask].astype(np.float64)
        nan_result = np.empty_like(nan_data)
        for i in range(n_cols):
            j0 = max(0, i - half_w)
            j1 = min(n_cols, i + half_w + 1)
            t_local = np.arange(j0, j1, dtype=np.float64) - i
            u_local = np.abs(t_local) / h
            w_local = np.where(u_local < 1.0, (1.0 - u_local**3) ** 3, 0.0)
            y = nan_data[:, j0:j1]
            nm = np.isnan(y)
            w2 = np.broadcast_to(w_local[np.newaxis, :], y.shape).copy()
            w2[nm] = 0.0
            yc = np.where(nm, 0.0, y)
            Sw = w2.sum(axis=1)
            Swt = (w2 * t_local[np.newaxis, :]).sum(axis=1)
            Swt2 = (w2 * (t_local**2)[np.newaxis, :]).sum(axis=1)
            Swy = (w2 * yc).sum(axis=1)
            Swty = (w2 * t_local[np.newaxis, :] * yc).sum(axis=1)
            det = Sw * Swt2 - Swt**2
            valid = np.abs(det) > 1e-15
            nan_result[:, i] = np.where(
                valid,
                (Swt2 * Swy - Swt * Swty) / np.where(valid, det, 1.0),
                np.where(Sw > 0, Swy / np.maximum(Sw, 1e-15), np.nan),
            )
        result[nan_row_mask] = nan_result

    if half_w > 0 and n_cols > 1 and clean_rows.any():
        _lowess_edges(data, result, half_w, h, row_mask=clean_rows)

    return result.astype(data.dtype)


def _lowess_edges(
    data: np.ndarray,
    result: np.ndarray,
    half_w: int,
    bandwidth: float,
    row_mask: np.ndarray | None = None,
) -> None:
    """
    Fix edge columns with full local linear regression (in-place).

    Only modifies the first and last ``half_w`` columns of ``result``.
    """
    _, n_cols = data.shape
    if row_mask is not None:
        data = data[row_mask]
    edge_cols = list(range(min(half_w, n_cols))) + list(
        range(max(n_cols - half_w, half_w), n_cols)
    )
    for i in edge_cols:
        j0 = max(0, i - half_w)
        j1 = min(n_cols, i + half_w + 1)
        t = np.arange(j0, j1, dtype=np.float64) - i
        u = np.abs(t) / bandwidth
        w = np.where(u < 1.0, (1.0 - u**3) ** 3, 0.0)

        y = data[:, j0:j1].astype(np.float64)
        nan_mask = np.isnan(y)
        w_2d = np.broadcast_to(w[np.newaxis, :], y.shape).copy()
        w_2d[nan_mask] = 0.0
        y_clean = np.where(nan_mask, 0.0, y)

        Sw = w_2d.sum(axis=1)
        Swt = (w_2d * t[np.newaxis, :]).sum(axis=1)
        Swt2 = (w_2d * (t**2)[np.newaxis, :]).sum(axis=1)
        Swy = (w_2d * y_clean).sum(axis=1)
        Swty = (w_2d * t[np.newaxis, :] * y_clean).sum(axis=1)

        det = Sw * Swt2 - Swt**2
        valid = np.abs(det) > 1e-15
        a = np.where(
            valid,
            (Swt2 * Swy - Swt * Swty) / np.where(valid, det, 1.0),
            np.where(Sw > 0, Swy / np.maximum(Sw, 1e-15), np.nan),
        )
        if row_mask is not None:
            result[row_mask, i] = a
        else:
            result[:, i] = a


def _local_order_stat(data: np.ndarray, window: int, quantile: float) -> np.ndarray:
    """
    Local order statistic filter along axis=1.

    Ports MATLAB: lowVals = resid <= ordfilt2(resid, max(2, ceil(0.15*LPfactor)), ones([1 LPfactor]));

    Uses scipy.ndimage.rank_filter which is the Python equivalent of
    MATLAB's ordfilt2.  rank_filter uses 0-based indexing (MATLAB is 1-based).

    Parameters
    ----------
    data : np.ndarray
        2D array (n_rows, n_cols).
    window : int
        Window size along columns.
    quantile : float
        Quantile to select (e.g. 0.15 for 15th percentile).

    Returns
    -------
    np.ndarray
        Order-statistic filtered array, same shape as data.
    """
    # MATLAB: order = max(2, ceil(0.15 * LPfactor)) — 1-indexed
    order_matlab = max(2, math.ceil(quantile * window))
    order_scipy = order_matlab - 1  # 0-indexed for rank_filter

    nan_mask = np.isnan(data)
    if np.any(nan_mask):
        filled = data.copy()
        filled[nan_mask] = (
            np.nanmax(data) + 1
        )  # large value so NaN positions don't affect low order stats
    else:
        filled = data

    # MATLAB: ordfilt2(resid, order, ones([1, LPfactor]))
    result = rank_filter(
        filled, rank=order_scipy, size=(1, window), mode="constant", cval=0.0
    )

    if np.any(nan_mask):
        result[nan_mask] = np.nan
    return result


def get_surround(
    hp_data: np.ndarray,
    sel_pix: np.ndarray,
    b_filter: np.ndarray,
    n_concurrent: int = 1,
    device: object | None = None,
) -> np.ndarray:
    """
    Compute surround signals from high-pass data using baseline filters.

    Ports extractTrial.m getSurround (lines 372-381).
    Uses batch FFT to convolve all time frames at once (matching MATLAB's convn).

    Parameters
    ----------
    hp_data : np.ndarray
        High-pass data, shape (n_selected_pixels, n_time).
    sel_pix : np.ndarray
        2D boolean mask.
    b_filter : np.ndarray
        Baseline filter kernels, shape (kernel_h, kernel_w, n_filters).
    n_concurrent : int
        Number of subproblems running concurrently. Used to divide
        the memory budget and FFT worker threads fairly.
    device : object, optional
        torch.device or None. If None, auto-detects CUDA availability.

    Returns
    -------
    np.ndarray
        Surround signals, shape (n_selected_pixels, n_time, n_filters).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(device, "type") and device.type == "cuda":
        h, w = sel_pix.shape
        kh, kw = b_filter.shape[:2]
        n_time = hp_data.shape[1]
        fft_h = next_fast_len(h + kh - 1)
        fft_w = next_fast_len(w + kw - 1)
        bytes_per_frame = (
            fft_h * fft_w * 4  # hp_padded
            + fft_h * (fft_w // 2 + 1) * 8  # HP_fft
            + fft_h * (fft_w // 2 + 1) * 8  # product
            + fft_h * fft_w * 4  # conv
        )
        overhead = 300 * 1024**2  # filter FFTs, sel_pix, cuFFT workspace, allocator
        target_chunk = max(
            1000, int(_TARGET_VRAM_PER_SUBPROBLEM / max(bytes_per_frame, 1))
        )
        target_chunk = min(target_chunk, n_time)
        estimated_vram = bytes_per_frame * target_chunk + overhead

        if _vram_budget.reserve(estimated_vram):
            try:
                return _get_surround_gpu(hp_data, sel_pix, b_filter, estimated_vram)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(
                        "CUDA OOM in get_surround, falling back to CPU: %s", e
                    )
                    torch.cuda.empty_cache()
                else:
                    raise
            finally:
                _vram_budget.release(estimated_vram)
        else:
            logger.warning(
                "get_surround: %dx%d crop needs %.0f MB, exceeds GPU capacity — using CPU",
                h,
                w,
                estimated_vram / 1024**2,
            )
    return _get_surround_cpu(hp_data, sel_pix, b_filter, n_concurrent)


def _get_surround_cpu(
    hp_data: np.ndarray,
    sel_pix: np.ndarray,
    b_filter: np.ndarray,
    n_concurrent: int = 1,
) -> np.ndarray:
    """CPU path using scipy.fft with adaptive chunking."""

    h, w = sel_pix.shape
    n_time = hp_data.shape[1]
    n_filters = b_filter.shape[2]
    kh, kw = b_filter.shape[:2]

    result = np.zeros((hp_data.shape[0], n_time, n_filters), dtype=np.float32)

    hp_full = np.zeros((h, w, n_time), dtype=np.float32)
    hp_full[sel_pix] = np.asarray(hp_data, dtype=np.float32)

    fft_h = next_fast_len(h + kh - 1)
    fft_w = next_fast_len(w + kw - 1)
    bytes_per_frame = (
        fft_h * (fft_w // 2 + 1) * 8  # HP_chunk complex64
        + fft_h * (fft_w // 2 + 1) * 8  # product complex64
        + fft_h * fft_w * 4  # conv float32
    )
    try:
        avail = psutil.virtual_memory().available
    except ImportError:
        avail = 2 * 1024**3  # conservative 2 GB fallback
    mem_budget = avail * 0.75 / n_concurrent
    fft_workers = max(1, (os.cpu_count() or 4) // n_concurrent)
    chunk_size = max(1000, int(mem_budget / max(bytes_per_frame, 1)))
    chunk_size = min(chunk_size, n_time)
    n_chunks = math.ceil(n_time / chunk_size)
    logger.debug(
        "get_surround: %dx%d spatial, %d frames, avail %.1f GB, "
        "concurrent=%d → chunk_size=%d (%d chunks, %.0f MB/chunk, %d fft_workers)",
        h,
        w,
        n_time,
        avail / 1024**3,
        n_concurrent,
        chunk_size,
        n_chunks,
        bytes_per_frame * chunk_size / 1024**2,
        fft_workers,
    )
    r0 = (kh - 1) // 2
    c0 = (kw - 1) // 2

    filt_f32 = b_filter.astype(np.float32)
    F_ffts = [rfft2(filt_f32[:, :, i], s=(fft_h, fft_w)) for i in range(n_filters)]

    for t0 in range(0, n_time, chunk_size):
        t1 = min(t0 + chunk_size, n_time)
        HP_fft = rfft2(
            hp_full[:, :, t0:t1],
            s=(fft_h, fft_w),
            axes=(0, 1),
            workers=fft_workers,
        )
        for filt_ix in range(n_filters):
            conv = irfft2(
                HP_fft * F_ffts[filt_ix][:, :, np.newaxis],
                s=(fft_h, fft_w),
                axes=(0, 1),
                workers=fft_workers,
            )
            result[:, t0:t1, filt_ix] = conv[r0 : r0 + h, c0 : c0 + w, :][sel_pix]
        del HP_fft

    return result


def _get_surround_gpu(
    hp_data: np.ndarray,
    sel_pix: np.ndarray,
    b_filter: np.ndarray,
    vram_budget: int,
) -> np.ndarray:
    """
    GPU path using torch.fft for accelerated 2D FFT convolution.

    Parameters
    ----------
    vram_budget : int
        Maximum VRAM (bytes) this call may use, as reserved
        by the caller via ``_vram_budget``.
    """

    h, w = sel_pix.shape
    n_time = hp_data.shape[1]
    n_filters = b_filter.shape[2]
    kh, kw = b_filter.shape[:2]

    result = np.zeros((hp_data.shape[0], n_time, n_filters), dtype=np.float32)

    hp_full = np.zeros((h, w, n_time), dtype=np.float32)
    hp_full[sel_pix] = np.asarray(hp_data, dtype=np.float32)

    fft_h = next_fast_len(h + kh - 1)
    fft_w = next_fast_len(w + kw - 1)
    r0 = (kh - 1) // 2
    c0 = (kw - 1) // 2

    F_ffts = []
    for i in range(n_filters):
        f_padded = torch.zeros(fft_h, fft_w, device="cuda")
        f_padded[:kh, :kw] = torch.from_numpy(b_filter[:, :, i].astype(np.float32))
        F_ffts.append(torch.fft.rfft2(f_padded))

    sel_pix_t = torch.from_numpy(sel_pix).cuda()

    bytes_per_frame = (
        fft_h * fft_w * 4  # hp_padded float32
        + fft_h * (fft_w // 2 + 1) * 8  # HP_fft complex64
        + fft_h * (fft_w // 2 + 1) * 8  # product complex64
        + fft_h * fft_w * 4  # conv float32
    )
    overhead = 300 * 1024**2  # filter FFTs, sel_pix, cuFFT workspace, allocator
    chunk_size = max(1000, int((vram_budget - overhead) / max(bytes_per_frame, 1)))
    chunk_size = min(chunk_size, n_time)
    n_chunks = math.ceil(n_time / chunk_size)
    logger.debug(
        "get_surround GPU: %dx%d spatial, %d frames, budget %.0f MB, "
        "chunk_size=%d (%d chunks)",
        h,
        w,
        n_time,
        vram_budget / 1024**2,
        chunk_size,
        n_chunks,
    )

    for t0 in range(0, n_time, chunk_size):
        t1 = min(t0 + chunk_size, n_time)
        C = t1 - t0

        hp_chunk_np = np.ascontiguousarray(hp_full[:, :, t0:t1].transpose(2, 0, 1))
        hp_chunk = torch.from_numpy(hp_chunk_np).cuda()
        del hp_chunk_np

        hp_padded = torch.zeros(C, fft_h, fft_w, device="cuda")
        hp_padded[:, :h, :w] = hp_chunk
        del hp_chunk

        HP_fft = torch.fft.rfft2(hp_padded)
        del hp_padded

        for filt_ix in range(n_filters):
            product = HP_fft * F_ffts[filt_ix].unsqueeze(0)
            conv = torch.fft.irfft2(product, s=(fft_h, fft_w))
            del product

            valid = conv[:, r0 : r0 + h, c0 : c0 + w]
            del conv

            # Gather selected pixels: (C, n_sel_pix) → transpose to (n_sel_pix, C)
            result[:, t0:t1, filt_ix] = valid[:, sel_pix_t].cpu().numpy().T
            del valid

        del HP_fft

    del F_ffts, sel_pix_t
    torch.cuda.empty_cache()

    return result


def build_baseline_filters(sel_radius: int) -> np.ndarray:
    """
    Build the DoG baseline filters.

    Ports setParamsExtractTrial.m lines 12-20.

    Parameters
    ----------
    sel_radius : int
        Selection radius for sources.

    Returns
    -------
    np.ndarray
        Filter bank, shape (kernel_h, kernel_w, 3).
    """

    size = 4 * sel_radius + 1
    center = size // 2
    tmp = np.zeros((size, size))
    tmp[center, center] = 1.0

    g1 = gf(tmp, sigma=sel_radius)
    g2 = gf(tmp, sigma=sel_radius / 2)
    b_filter = np.zeros((size, size, 3))

    b_filter[:, :, 0] = np.maximum(0, g1 - g2)

    mask = np.zeros((size, size))
    mask[:, center + 1 :] = 1.0
    mask[:, center] = 0.5
    mask -= np.mean(mask)

    b_filter[:, :, 1] = b_filter[:, :, 0] * mask
    b_filter[:, :, 2] = b_filter[:, :, 0] * mask.T

    return b_filter
