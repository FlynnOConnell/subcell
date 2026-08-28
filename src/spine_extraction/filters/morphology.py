"""Morphological operations for source detection.

Ports the non-maximum suppression and dilation logic from localizeSources_vIM.m.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter, binary_dilation, generate_binary_structure


def local_maxima_3x3(image: np.ndarray) -> np.ndarray:
    """Find 3x3 local maxima in a 2D image.

    Matches MATLAB: ``image == ordfilt2(image, 9, ones(3))``

    MATLAB's ``ordfilt2`` propagates NaN: if *any* pixel in the 3×3
    window is NaN the result is NaN, so ``value == NaN`` → false and
    the pixel cannot be a peak.  ``scipy.ndimage.maximum_filter`` does
    **not** propagate NaN, so we add an explicit check: any pixel whose
    3×3 neighbourhood contains a NaN is excluded from peak candidates.
    This provides an implicit 1-pixel erosion from NaN borders and
    prevents false peaks at motion-induced NaN boundaries.

    Args:
        image: 2D array (may contain NaN).

    Returns:
        Boolean mask of local maximum pixels.
    """
    max_filt = maximum_filter(image, size=3)
    is_max = image == max_filt

    # Propagate NaN like MATLAB ordfilt2: suppress peaks near NaN
    if np.any(np.isnan(image)):
        nan_neighbor = maximum_filter(np.isnan(image).astype(np.float64), size=3) > 0
        is_max[nan_neighbor] = False

    return is_max


def spatiotemporal_nms(
    movie: np.ndarray, tau_frames: float,
    nan_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Non-maximum suppression in space (3x3) and time (t-1, t, t+1).

    Ports localizeSources_vIM.m lines 101-115:
    For each frame (excluding last 1.5*tau frames):
    - Find 3x3 spatial local maxima
    - Require value > previous frame AND >= next frame
    - Accumulate squared peak values into activity image
    - Normalize by (regularizer + number of valid observations per pixel)

    Vectorized: processes temporal chunks to balance memory and speed.

    Args:
        movie: Filtered movie, shape (rows, cols, time).
        tau_frames: Decay time in frames (used to skip uncertain end frames).
        nan_mask: Boolean mask of NaN positions, same shape as movie.
            Used to count valid observations per pixel for normalization.

    Returns:
        Tuple of (activity_image, None):
        - activity_image: Observation-normalized sum of squared local maxima
          at each pixel, shape (rows, cols).
    """
    rows, cols, n_frames = movie.shape
    skip_end = int(np.ceil(1.5 * tau_frames))
    end = n_frames - skip_end  # exclusive upper bound for valid frames

    activity_image = np.zeros((rows, cols), dtype=np.float64)

    # Process in temporal chunks to limit memory
    chunk_size = 2000
    for start in range(1, end, chunk_size):
        stop = min(start + chunk_size, end)

        cur = movie[:, :, start:stop]
        prev = movie[:, :, start - 1 : stop - 1]
        nxt = movie[:, :, start + 1 : stop + 1]

        # Spatial: 3x3 local max per frame (size=1 along time = no temporal mixing)
        spatial_max = maximum_filter(cur, size=(3, 3, 1))
        is_spatial = cur == spatial_max
        del spatial_max

        # Propagate NaN like MATLAB ordfilt2: suppress peaks near NaN
        # MATLAB's ordfilt2 returns NaN if any pixel in the 3x3 window
        # is NaN, so border pixels adjacent to NaN can never be peaks.
        # scipy's maximum_filter does NOT propagate NaN, so we must
        # explicitly suppress peaks whose 3x3 neighbourhood has NaN.
        nan_cur = np.isnan(cur)
        if np.any(nan_cur):
            nan_neighbor = maximum_filter(
                nan_cur.astype(np.float32), size=(3, 3, 1)
            ) > 0
            is_spatial[nan_neighbor] = False
            del nan_neighbor
        del nan_cur

        # Temporal: must be > previous and >= next
        is_peak = is_spatial & (cur > prev) & (cur >= nxt)
        del is_spatial

        # Accumulate squared peak values
        activity_image += np.sum(
            np.where(is_peak, cur * cur, 0.0), axis=2
        )
        del is_peak

    # Normalize by number of valid observations per pixel (with regularizer)
    # MATLAB: skIm = skIm ./ (300 + sum(~nans(:,:, 2:end-ceil(1.5*tau)), 3))
    if nan_mask is not None:
        n_valid = np.sum(~nan_mask[:, :, 1:end], axis=2).astype(np.float64)
    else:
        n_valid = float(max(0, end - 1))
    activity_image /= (300.0 + n_valid)

    return activity_image, None


def iterative_nms_peaks(
    image: np.ndarray,
    dilation_size: int = 5,
) -> np.ndarray:
    """Find peaks via iterative non-maximum suppression.

    Ports localizeSources_vIM.m lines 113-129 and summarize_LoCo.m:
    Repeatedly:
    1. Find 3x3 local maxima > 0
    2. Add to peak mask
    3. Dilate found peaks and zero out neighborhood
    4. Repeat until no new peaks

    Args:
        image: 2D array (activity image with local baseline subtracted).
        dilation_size: Size of dilation kernel to suppress around found peaks.

    Returns:
        Boolean mask of peak locations.
    """
    explored = image.copy()
    peak_mask = np.zeros_like(image, dtype=bool)
    se = np.ones((dilation_size, dilation_size), dtype=bool)

    while True:
        candidates = (explored > 0) & local_maxima_3x3(explored)
        if not np.any(candidates):
            break
        peak_mask |= candidates
        # Zero out neighborhood around found peaks
        dilated = binary_dilation(candidates, structure=se)
        explored[dilated] = 0

    return peak_mask


def apply_density_threshold(
    image: np.ndarray,
    peak_mask: np.ndarray,
    max_density: float,
    n_time_points: int | None = None,
    align_hz: float | None = None,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply maximum synapse density threshold to peak selection.

    Ports localizeSources_vIM.m lines 128-130:
        threshP = 1.5 * sortedP(min(end, ceil(totalPix * maxSynapseDensity *
                  (1-exp(-nTimePoints/alignHz/10)))));

    Args:
        image: Activity image.
        peak_mask: Boolean mask of candidate peaks.
        max_density: Maximum synapses per valid pixel.
        n_time_points: Number of time points (for time-dependent threshold).
        align_hz: Alignment sampling rate.
        valid_mask: Mask of valid pixels.

    Returns:
        Tuple of (rows, cols, values) for selected peaks.
    """
    peak_vals = image[peak_mask]
    if len(peak_vals) == 0:
        return np.array([]), np.array([]), np.array([])

    sorted_p = np.sort(peak_vals)[::-1]
    total_pix = np.sum(~np.isnan(image)) if valid_mask is None else np.sum(valid_mask)

    # Time-dependent density adjustment
    if n_time_points is not None and align_hz is not None:
        time_factor = 1.0 - np.exp(-n_time_points / align_hz / 10.0)
    else:
        time_factor = 1.0

    max_peaks = max(1, int(np.ceil(total_pix * max_density * time_factor)))
    thresh_idx = min(len(sorted_p) - 1, max_peaks - 1)
    threshold = 1.5 * sorted_p[thresh_idx]

    # Apply threshold
    thresholded = image.copy()
    thresholded[~peak_mask] = 0
    thresholded[thresholded < threshold] = 0

    rows, cols = np.where(thresholded > 0)
    vals = thresholded[rows, cols]

    return rows, cols, vals
