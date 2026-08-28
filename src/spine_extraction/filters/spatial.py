"""Spatial filtering operations.

Ports spatial filtering from localizeSources_vIM.m:
- Difference of Gaussians (DoG)
- Gaussian filtering
- NaN-tolerant median filtering
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


def _gaussian_nanaware(
    data: np.ndarray, sigma_spec, nan_mask: np.ndarray,
) -> np.ndarray:
    """Gaussian filter with NaN normalization.

    Instead of treating NaN as 0 (which attenuates values near NaN borders),
    normalizes each pixel by the local fraction of valid samples.  This
    matches the behaviour of MATLAB ``imgaussfilt`` on data where NaN pixels
    have already been zeroed out, *provided the NaN pattern is spatially
    smooth* (which it is for motion-induced border NaN).

    G_norm(x) = G(data_filled)(x) / G(valid_mask)(x)
    """
    data_filled = np.where(nan_mask, 0.0, data)
    valid = (~nan_mask).astype(data.dtype)

    mode = "constant"
    smoothed = gaussian_filter(data_filled, sigma=sigma_spec, mode=mode, cval=0.0)
    weight = gaussian_filter(valid, sigma=sigma_spec, mode=mode, cval=0.0)

    # Avoid division by zero where all neighbours are NaN
    weight = np.maximum(weight, 1e-10)
    return smoothed / weight


def difference_of_gaussians(
    data: np.ndarray, sigma: float, nan_mask: np.ndarray | None = None
) -> np.ndarray:
    """Difference of Gaussians filter: G(sigma) - G(5*sigma).

    Ports localizeSources_vIM.m lines 83-87:
        IMf(nans) = 0;
        IMf = imgaussfilt(IMf, [sigma sigma]);
        IMf = IMf - imgaussfilt(IMf, 5*[sigma sigma]);
        IMf(nans) = nan;

    When *nan_mask* is provided the Gaussian is normalized by the local
    fraction of valid pixels so that values near NaN borders are not
    attenuated.  This prevents false peaks at NaN boundaries.

    Args:
        data: 2D or 3D array. For 3D, filters each frame independently.
        sigma: Small Gaussian sigma in pixels.
        nan_mask: Boolean mask of NaN locations (optional).

    Returns:
        DoG-filtered data.
    """
    # MATLAB imgaussfilt defaults:
    #   kernel radius = ceil(2*sigma)  →  kernel size = 2*ceil(2*sigma)+1
    #   boundary = 'replicate'         →  mode='nearest' in scipy
    # scipy computes radius = int(truncate * sigma + 0.5), so we set
    # truncate per-sigma to match MATLAB's ceil(2*sigma) exactly.
    import math

    def _matlab_truncate(s):
        """Compute scipy truncate to match MATLAB imgaussfilt kernel size."""
        matlab_radius = math.ceil(2.0 * s)
        return (matlab_radius - 0.5 + 1e-9) / s

    _trunc_small = _matlab_truncate(sigma)
    _trunc_large = _matlab_truncate(5 * sigma)
    _mode = "nearest"

    if nan_mask is not None:
        # Match MATLAB exactly: zero out NaN, filter, restore NaN.
        # MATLAB (localizeSources_vIM.m lines 96-99):
        #   IMf(nans) = 0;
        #   IMf = imgaussfilt(IMf, [sigma sigma]);
        #   IMf = IMf - imgaussfilt(IMf, 5*[sigma sigma]);
        #   IMf(nans) = nan;
        # This intentionally attenuates border values (convolved with
        # zeros), which suppresses false peaks at NaN boundaries.
        data_filled = np.where(nan_mask, 0.0, data)
        if data.ndim == 3:
            small = gaussian_filter(data_filled, sigma=[sigma, sigma, 0], truncate=_trunc_small, mode=_mode)
            large = gaussian_filter(small, sigma=[5 * sigma, 5 * sigma, 0], truncate=_trunc_large, mode=_mode)
        else:
            small = gaussian_filter(data_filled, sigma=sigma, truncate=_trunc_small, mode=_mode)
            large = gaussian_filter(small, sigma=5 * sigma, truncate=_trunc_large, mode=_mode)
        result = small - large
        result[nan_mask] = np.nan
    else:
        if data.ndim == 3:
            small = gaussian_filter(data, sigma=[sigma, sigma, 0], truncate=_trunc_small, mode=_mode)
            large = gaussian_filter(small, sigma=[5 * sigma, 5 * sigma, 0], truncate=_trunc_large, mode=_mode)
        else:
            small = gaussian_filter(data, sigma=sigma, truncate=_trunc_small, mode=_mode)
            large = gaussian_filter(small, sigma=5 * sigma, truncate=_trunc_large, mode=_mode)
        result = small - large

    return result


def nanmedfilt2(image: np.ndarray, kernel_size: int | tuple[int, int]) -> np.ndarray:
    """2D median filter that ignores NaN values.

    Ports nanmedfilt2.m - handles NaN pixels by excluding them from the
    local median computation.

    Args:
        image: 2D array (may contain NaN).
        kernel_size: Filter kernel size.

    Returns:
        Median-filtered image.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    # For images without NaNs, use scipy directly
    if not np.any(np.isnan(image)):
        return median_filter(image, size=kernel_size)

    # Pad image
    pad_r = kernel_size[0] // 2
    pad_c = kernel_size[1] // 2
    padded = np.pad(image, ((pad_r, pad_r), (pad_c, pad_c)), mode="constant", constant_values=np.nan)

    result = np.full_like(image, np.nan)
    rows, cols = image.shape

    for r in range(rows):
        for c in range(cols):
            window = padded[r : r + kernel_size[0], c : c + kernel_size[1]]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                result[r, c] = np.median(valid)

    return result
