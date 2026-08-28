"""Spatial footprint initialization and Gaussian smoothing.

Ports the spatial footprint logic from extractTrial.m lines 154-166 and
setParamsExtractTrial.m lines 3-8.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation


def build_gaussian_filter_kernel(sigma_px: float) -> np.ndarray:
    """Build the Gaussian filter kernel for Hs -> H conversion.

    Ports setParamsExtractTrial.m lines 4-6:
        tmp = zeros(4*ceil(sigma_px)+1);
        tmp(ceil(end/2), ceil(end/2)) = 1;
        Hfilter = imgaussfilt(tmp, max(0.5, 0.9*sigma_px), 'FilterSize', size(tmp,1));

    Args:
        sigma_px: PSF sigma in pixels.

    Returns:
        2D Gaussian kernel for convolution.
    """
    size = 4 * int(np.ceil(sigma_px)) + 1
    center = size // 2
    tmp = np.zeros((size, size))
    tmp[center, center] = 1.0
    sigma_eff = max(0.5, 0.9 * sigma_px)
    kernel = gaussian_filter(tmp, sigma=sigma_eff)
    return kernel


def initialize_footprints(
    source_rows: np.ndarray,
    source_cols: np.ndarray,
    sel_pix: np.ndarray,
    sigma_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize spatial footprints H, Hs, and validity masks.

    Ports extractTrial.m lines 154-166:
    For each source:
    1. Place a point at the source center
    2. Dilate with 3x3 kernel -> valid_kernel (where intensity can be placed)
    3. Dilate with disk(1) -> Hs (super-resolution estimate, normalized)
    4. Gaussian blur -> H (smoothed footprint)

    Args:
        source_rows: Row coordinates of source centers.
        source_cols: Column coordinates of source centers.
        sel_pix: 2D boolean mask of selected pixels.
        sigma_px: PSF Gaussian sigma.

    Returns:
        Tuple of (H, Hs, H_valid):
        - H: Smoothed footprints, shape (n_selected_pixels, n_sources).
        - Hs: Super-resolution footprints, shape (n_selected_pixels, n_sources).
        - H_valid: Validity mask, shape (n_selected_pixels, n_sources).
    """
    h, w = sel_pix.shape
    n_sources = len(source_rows)
    n_pixels = np.sum(sel_pix)

    H = np.zeros((n_pixels, n_sources), dtype=np.float32)
    Hs = np.zeros((n_pixels, n_sources), dtype=np.float32)
    H_valid = np.zeros((n_pixels, n_sources), dtype=bool)

    valid_kernel = np.ones((3, 3), dtype=bool)
    disk_se = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)  # disk(1)

    for i in range(n_sources):
        r, c = int(round(source_rows[i])), int(round(source_cols[i]))

        # Valid kernel: 3x3 neighborhood around center
        tmp_valid = np.zeros((h, w), dtype=bool)
        tmp_valid[r, c] = True
        tmp_valid = binary_dilation(tmp_valid, structure=valid_kernel)

        # Super-resolution estimate: disk(1) dilation, normalized
        tmp_hs = np.zeros((h, w), dtype=np.float64)
        tmp_hs[r, c] = 1.0
        tmp_hs_dilated = binary_dilation(tmp_hs > 0, structure=disk_se).astype(np.float64)
        tmp_hs_dilated /= max(tmp_hs_dilated.sum(), 1.0)

        # Smoothed footprint: Gaussian blur of Hs
        tmp_h = gaussian_filter(tmp_hs_dilated, sigma=sigma_px)

        # Extract selected pixels
        Hs[:, i] = tmp_hs_dilated[sel_pix].astype(np.float32)
        H[:, i] = tmp_h[sel_pix].astype(np.float32)
        H_valid[:, i] = tmp_valid[sel_pix]

    return H, Hs, H_valid


def smooth_hs_to_h(
    Hs: np.ndarray,
    sel_pix: np.ndarray,
    h_filter: np.ndarray,
    n_sources: int,
) -> np.ndarray:
    """Convert super-resolution Hs to smoothed H via convolution.

    Ports extractTrial.m lines 282-285:
        tmp = zeros([sz num_sources]);
        tmp(repmat(selPix,1,1,num_sources)) = Hs_est;
        tmp = reshape(convn(tmp, Hfilter, 'same'), numel(selPix), num_sources);
        H_est = tmp(selPix,:);

    Vectorized: all sources are scattered into a 3D volume and convolved
    with a single fftconvolve call (kernel broadcasts over the source axis).

    Args:
        Hs: Super-resolution footprints, (n_pixels, n_sources).
        sel_pix: 2D boolean mask.
        h_filter: 2D Gaussian convolution kernel.
        n_sources: Number of sources.

    Returns:
        Smoothed H, (n_pixels, n_sources).
    """
    from scipy.signal import fftconvolve

    h, w = sel_pix.shape

    # Scatter all sources into a 3D volume at once: (h, w, n_sources)
    vol = np.zeros((h, w, n_sources), dtype=np.float64)
    vol[sel_pix] = Hs  # broadcasts (n_sel, n_sources) into True positions

    # Single batched convolution — kernel (kh, kw) broadcasts over axis 2
    filtered = fftconvolve(vol, h_filter[:, :, np.newaxis], mode="same")

    # Gather selected pixels back
    H_new = filtered[sel_pix].astype(np.float32)  # (n_sel, n_sources)
    return H_new
