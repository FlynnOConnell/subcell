"""Normalized cross-correlation for images with NaN regions.

Ports xcorr2_nans.m - performs local normalized cross-correlation that gracefully
handles NaN pixels by eroding the valid mask within the search range.

Uses FFT-based correlation to avoid the O(d_shift^2) Python loop.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve


def xcorr2_nans(
    frame: np.ndarray,
    template: np.ndarray,
    shifts_center: np.ndarray,
    d_shift: int,
) -> tuple[np.ndarray, float]:
    """
    Local normalized cross-correlation for images with NaN regions.

    Ports xcorr2_nans.m by Kaspar Podgorski 2023.

    Uses FFT convolution for the shift-dependent sums, avoiding an explicit
    Python loop over the (2*d_shift+1)^2 shift grid.

    Parameters
    ----------
    frame : np.ndarray
        Image to align (may contain NaNs).
    template : np.ndarray
        Reference template (may contain NaNs).
    shifts_center : np.ndarray
        Initial shift estimate, shape (2,) as [row, col].
    d_shift : int
        Search radius in pixels.

    Returns
    -------
    tuple[np.ndarray, float]
        Tuple of (motion_vector, correlation_coefficient).
        motion_vector shape (2,) as [row_shift, col_shift].
    """
    d_shift = int(d_shift)
    shifts_center = np.asarray(shifts_center, dtype=np.float64).ravel()
    sc_r, sc_c = int(round(shifts_center[0])), int(round(shifts_center[1]))

    se_size = 2 * d_shift + 1
    se = np.ones((se_size, se_size), dtype=bool)

    template_nan = np.isnan(template)
    template_nan_dilated = binary_dilation(template_nan, structure=se)

    shifted_valid = np.roll(~template_nan_dilated, (sc_r, sc_c), axis=(0, 1))
    f_valid = ~np.isnan(frame) & shifted_valid

    f_valid[:d_shift, :] = False
    f_valid[-d_shift:, :] = False
    f_valid[:, :d_shift] = False
    f_valid[:, -d_shift:] = False

    F = np.where(f_valid, frame.astype(np.float64), 0.0)
    ss_F = np.sqrt(np.sum(F**2))

    if ss_F == 0:
        return shifts_center.copy(), 0.0

    T_shifted = np.roll(template, (sc_r, sc_c), axis=(0, 1)).astype(np.float64)
    T_clean = np.where(np.isnan(T_shifted), 0.0, T_shifted)

    cross_corr = fftconvolve(F, T_clean[::-1, ::-1], mode="full")

    T_sq = T_clean**2
    energy = fftconvolve(f_valid.astype(np.float64), T_sq[::-1, ::-1], mode="full")

    center_r = frame.shape[0] - 1
    center_c = frame.shape[1] - 1

    r_lo = max(0, center_r - d_shift)
    r_hi = min(cross_corr.shape[0], center_r + d_shift + 1)
    c_lo = max(0, center_c - d_shift)
    c_hi = min(cross_corr.shape[1], center_c + d_shift + 1)

    cc_window = cross_corr[r_lo:r_hi, c_lo:c_hi]
    e_window = energy[r_lo:r_hi, c_lo:c_hi]

    C = np.full_like(cc_window, np.nan)
    valid_e = e_window > 0
    C[valid_e] = cc_window[valid_e] / np.sqrt(e_window[valid_e])

    if np.all(np.isnan(C)):
        return shifts_center.copy(), 0.0

    max_val = np.nanmax(C)
    max_idx = np.unravel_index(np.nanargmax(C), C.shape)
    rr, cc_idx = max_idx

    R = max_val / ss_F

    shift_r_at_peak = (r_lo - center_r) + rr
    shift_c_at_peak = (c_lo - center_c) + cc_idx

    n_r, n_c = C.shape
    d_r, d_c = 0.0, 0.0

    if (
        0 < rr < n_r - 1
        and not np.isnan(C[rr - 1, cc_idx])
        and not np.isnan(C[rr + 1, cc_idx])
    ):
        denom_r = C[rr, cc_idx] - C[rr + 1, cc_idx]
        if abs(denom_r) > 0:
            ratio_r = min(1e6, (C[rr, cc_idx] - C[rr - 1, cc_idx]) / denom_r)
            d_r = (1 - ratio_r) / (1 + ratio_r) / 2

    if (
        0 < cc_idx < n_c - 1
        and not np.isnan(C[rr, cc_idx - 1])
        and not np.isnan(C[rr, cc_idx + 1])
    ):
        denom_c = C[rr, cc_idx] - C[rr, cc_idx + 1]
        if abs(denom_c) > 0:
            ratio_c = min(1e6, (C[rr, cc_idx] - C[rr, cc_idx - 1]) / denom_c)
            d_c = (1 - ratio_c) / (1 + ratio_c) / 2

    motion = shifts_center + np.array([shift_r_at_peak - d_r, shift_c_at_peak - d_c])

    if np.any(np.isnan(motion)):
        return shifts_center.copy(), R

    return motion, R
