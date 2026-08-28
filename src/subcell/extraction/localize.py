"""Source localization from downsampled registered movies.

Ports localizeSources_vIM.m - detects synaptic source locations via:
1. Temporal baseline subtraction and noise normalization
2. Exponential matched filter (backward causal)
3. Difference of Gaussians spatial filter
4. Spatiotemporal non-maximum suppression
5. Activity image computation (sum of squared peaks)
6. Peak detection with density thresholding
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from subcell.config import ExtractionConfig
from subcell.filters.morphology import (
    apply_density_threshold,
    local_maxima_3x3,
    spatiotemporal_nms,
)
from subcell.filters.spatial import difference_of_gaussians, nanmedfilt2
from subcell.filters.temporal import (
    exponential_matched_filter,
    moving_mad_noise,
    moving_mean,
    moving_median_baseline,
)

logger = logging.getLogger(__name__)


@dataclass
class PeakSet:
    """Detected peak locations and strengths."""

    row: np.ndarray = field(default_factory=lambda: np.array([]))
    col: np.ndarray = field(default_factory=lambda: np.array([]))
    val: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_image: np.ndarray | None = None


def localize_sources(
    movie: np.ndarray,
    config: ExtractionConfig,
    align_hz: float,
    variance_image: np.ndarray | None = None,
) -> tuple[np.ndarray, PeakSet]:
    """
    Detect synaptic source locations from a single-trial movie.

    Ports localizeSources_vIM.m for the Bergamo pathway.

    Parameters
    ----------
    movie : np.ndarray
        3D array (rows, cols, time). Single channel, activity channel.
    config : ExtractionConfig
        Extraction configuration.
    align_hz : float
        Alignment sampling rate in Hz.
    variance_image : np.ndarray, optional
        Variance multiplier image (for SLAP2; None for Bergamo).

    Returns
    -------
    tuple[np.ndarray, PeakSet]
        Tuple of (activity_image, peaks)
        - activity_image
        2D array of summed squared peak values per pixel.
        - peaks
        PeakSet with detected source locations.
    """
    tau = config.tau_s * align_hz
    sigma = config.sigma_px
    baseline_window = int(math.ceil(config.baseline_window_glu_s * align_hz))
    denoise_window = int(math.ceil(config.denoise_window_s * align_hz))

    n_time = movie.shape[2]
    nan_mask = np.isnan(movie)

    nan_frac_spatial = np.mean(nan_mask, axis=2)
    valid = nan_frac_spatial < config.nan_thresh

    if not np.any(valid):
        logger.warning("No valid pixels found (too much motion)")
        return np.full(movie.shape[:2], np.nan), PeakSet()

    IMf = movie.copy()
    invalid_3d = np.broadcast_to(~valid[:, :, np.newaxis], IMf.shape)
    IMf[invalid_3d] = np.nan
    nan_mask = np.isnan(IMf)

    # NOTE: no NaN interpolation here, matching MATLAB; border peaks are
    # suppressed by NaN-propagating peak detection instead.

    # Bergamo pathway (matches MATLAB localizeSources_vIM.m lines 70-78)
    IMfden = moving_mean(IMf, window=denoise_window, axis=2)
    IMb = moving_median_baseline(IMfden, window=baseline_window, axis=2)
    IMf = IMf - IMb  # subtract smoothed baseline from raw signal

    std_IM = moving_mad_noise(IMfden - IMb, window=baseline_window, axis=2)
    std_IM = std_IM * denoise_window
    del IMfden, IMb

    with np.errstate(divide="ignore", invalid="ignore"):
        IMf = IMf / std_IM
    del std_IM
    non_finite = ~np.isfinite(IMf)
    new_nans = non_finite & ~nan_mask
    if np.any(new_nans):
        logger.debug(
            "Z-score produced %d non-finite values (border pixels with zero MAD)",
            int(np.sum(new_nans)),
        )
        IMf[non_finite] = np.nan
        nan_mask = nan_mask | non_finite
    del non_finite, new_nans

    IMf = exponential_matched_filter(IMf, tau)
    IMf[nan_mask] = np.nan

    IMf = difference_of_gaussians(IMf, sigma, nan_mask=nan_mask)

    activity_image, _ = spatiotemporal_nms(IMf, tau, nan_mask=nan_mask)
    del IMf

    activity_image[~valid] = np.nan
    med_filt = nanmedfilt2(activity_image, 5)
    activity_image = activity_image - med_filt
    activity_image[~valid] = np.nan

    # Find peaks — simple 3x3 local maxima, matching MATLAB
    peak_mask = local_maxima_3x3(activity_image)
    rows, cols, vals = apply_density_threshold(
        activity_image,
        peak_mask,
        config.max_synapse_density,
        n_time_points=n_time,
        align_hz=align_hz,
        valid_mask=valid,
    )

    refined_rows = np.zeros_like(rows, dtype=np.float64)
    refined_cols = np.zeros_like(cols, dtype=np.float64)
    h, w = activity_image.shape

    for i in range(len(rows)):
        r, c = int(rows[i]), int(cols[i])
        if 0 < r < h - 1:
            v_prev = activity_image[r - 1, c]
            v_curr = activity_image[r, c]
            v_next = activity_image[r + 1, c]
            denom = v_curr - v_next
            if abs(denom) > 0:
                ratio = min(1e6, (v_curr - v_prev) / denom)
                d_r = (1 - ratio) / (1 + ratio) / 2
            else:
                d_r = 0.0
            refined_rows[i] = r - d_r
        else:
            refined_rows[i] = r

        if 0 < c < w - 1:
            v_prev = activity_image[r, c - 1]
            v_curr = activity_image[r, c]
            v_next = activity_image[r, c + 1]
            denom = v_curr - v_next
            if abs(denom) > 0:
                ratio = min(1e6, (v_curr - v_prev) / denom)
                d_c = (1 - ratio) / (1 + ratio) / 2
            else:
                d_c = 0.0
            refined_cols[i] = c - d_c
        else:
            refined_cols[i] = c

    peaks = PeakSet(
        row=refined_rows,
        col=refined_cols,
        val=vals,
        peak_image=activity_image,
    )

    logger.info("Localized %d sources", len(peaks.row))
    return activity_image, peaks
