"""Spatial filtering, porting the image filters from localizeSources_vIM.m."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


def _matlab_truncate(sigma: float) -> float:
    """
    scipy ``truncate`` reproducing MATLAB imgaussfilt's kernel size.

    MATLAB uses radius ``ceil(2*sigma)``; scipy uses ``int(truncate*sigma + 0.5)``.
    """
    return (math.ceil(2.0 * sigma) - 0.5 + 1e-9) / sigma


def _dog(data: np.ndarray, sigma: float) -> np.ndarray:
    """Difference of Gaussians on an already NaN-free array."""
    trunc_small = _matlab_truncate(sigma)
    trunc_large = _matlab_truncate(5 * sigma)
    sigma_small = [sigma, sigma, 0] if data.ndim == 3 else sigma
    sigma_large = [5 * sigma, 5 * sigma, 0] if data.ndim == 3 else 5 * sigma

    small = gaussian_filter(
        data, sigma=sigma_small, truncate=trunc_small, mode="nearest"
    )
    large = gaussian_filter(
        small, sigma=sigma_large, truncate=trunc_large, mode="nearest"
    )
    return small - large


def difference_of_gaussians(
    data: np.ndarray, sigma: float, nan_mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Difference of Gaussians, G(sigma) - G(5*sigma).

    NaN pixels are zeroed before filtering and restored afterwards, matching
    MATLAB. Zeroing deliberately attenuates values near NaN borders, which is
    what suppresses false peaks there.

    Parameters
    ----------
    data : np.ndarray
        2D image, or 3D stack filtered frame by frame.
    sigma : float
        Small Gaussian sigma in pixels.
    nan_mask : np.ndarray, optional
        Boolean mask of NaN locations in ``data``.

    Returns
    -------
    np.ndarray
        Filtered data, NaN restored where ``nan_mask`` was set.
    """
    if nan_mask is None:
        return _dog(data, sigma)

    result = _dog(np.where(nan_mask, 0.0, data), sigma)
    result[nan_mask] = np.nan
    return result


def nanmedfilt2(image: np.ndarray, kernel_size: int | tuple[int, int]) -> np.ndarray:
    """
    2D median filter that excludes NaN from each local window.

    Ports nanmedfilt2.m.

    Parameters
    ----------
    image : np.ndarray
        2D array, may contain NaN.
    kernel_size : int or tuple of int
        Filter size, square if an int.

    Returns
    -------
    np.ndarray
        Filtered image, NaN where a window held no valid samples.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if not np.any(np.isnan(image)):
        return median_filter(image, size=kernel_size)

    pad_r = kernel_size[0] // 2
    pad_c = kernel_size[1] // 2
    padded = np.pad(
        image, ((pad_r, pad_r), (pad_c, pad_c)), mode="constant", constant_values=np.nan
    )

    result = np.full_like(image, np.nan)
    rows, cols = image.shape
    for r in range(rows):
        for c in range(cols):
            window = padded[r : r + kernel_size[0], c : c + kernel_size[1]]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                result[r, c] = np.median(valid)
    return result
