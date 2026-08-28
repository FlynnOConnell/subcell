"""Source selection from averaged activity images.

Ports summarize_LoCo.m lines 220-275: find local maxima in the cross-trial
averaged activity image, mask soma regions, apply density threshold, and
dilate pixels around each source.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

from subcell.config import ExtractionConfig
from subcell.filters.morphology import iterative_nms_peaks
from subcell.filters.spatial import nanmedfilt2

logger = logging.getLogger(__name__)


@dataclass
class SourceSet:
    """Selected synaptic sources with spatial footprints."""

    rows: np.ndarray  # (n_sources,) row coordinates
    cols: np.ndarray  # (n_sources,) col coordinates
    values: np.ndarray  # (n_sources,) peak strengths
    sel_pix: np.ndarray  # (H, W, n_sources) boolean pixel selection
    n_sources: int = 0


def select_sources(
    aligned_activity: np.ndarray,
    valid_trials: np.ndarray,
    config: ExtractionConfig,
    soma_mask: np.ndarray | None = None,
    valid_pixel_mask: np.ndarray | None = None,
) -> SourceSet:
    """
    Select synaptic sources from cross-trial averaged activity image.

    Ports summarize_LoCo.m lines 220-275.

    Parameters
    ----------
    aligned_activity : np.ndarray
        Aligned activity images, shape (H, W, n_trials).
    valid_trials : np.ndarray
        Indices of valid trials.
    config : ExtractionConfig
        Extraction configuration.
    soma_mask : np.ndarray, optional
        Boolean mask of soma regions to exclude (optional).
    valid_pixel_mask : np.ndarray, optional
        Boolean mask of always-valid pixels (optional).

    Returns
    -------
    SourceSet
        SourceSet with selected sources and pixel selections.
    """
    act_img = np.mean(aligned_activity[:, :, valid_trials], axis=2)
    h, w = act_img.shape

    kernel_size = 2 * int(np.ceil(1.5 * config.dXY)) + 1
    med_baseline = nanmedfilt2(act_img, kernel_size)
    act_img = act_img - med_baseline

    peak_mask = iterative_nms_peaks(act_img, dilation_size=5)

    if soma_mask is not None:
        peak_mask[soma_mask] = False

    peak_vals = act_img[peak_mask]
    if len(peak_vals) == 0:
        logger.warning("No peaks found in averaged activity image")
        return SourceSet(
            rows=np.array([]),
            cols=np.array([]),
            values=np.array([]),
            sel_pix=np.zeros((h, w, 0), dtype=bool),
            n_sources=0,
        )

    sorted_p = np.sort(peak_vals)[::-1]
    total_pix = np.sum(~np.isnan(act_img))
    if soma_mask is not None:
        total_pix -= np.sum(soma_mask & ~np.isnan(act_img))

    max_sources = max(1, int(np.ceil(total_pix * config.max_synapse_density)))
    thresh_idx = min(len(sorted_p) - 1, max_sources - 1)
    threshold = 2.0 * sorted_p[thresh_idx]

    thresholded = act_img.copy()
    thresholded[~peak_mask] = 0
    thresholded[thresholded < threshold] = 0

    source_rows, source_cols = np.where(thresholded > 0)
    source_vals = thresholded[source_rows, source_cols]

    sort_idx = np.argsort(source_vals)[::-1]
    source_rows = source_rows[sort_idx]
    source_cols = source_cols[sort_idx]
    source_vals = source_vals[sort_idx]

    n_sources = len(source_rows)
    logger.info("Selected %d sources above density threshold", n_sources)

    sel_radius = config.sel_radius
    sel_pix = np.zeros((h, w, n_sources), dtype=bool)

    y, x = np.mgrid[-sel_radius : sel_radius + 1, -sel_radius : sel_radius + 1]
    disk_se = (x**2 + y**2) <= sel_radius**2

    for i in range(n_sources):
        r, c = int(round(source_rows[i])), int(round(source_cols[i]))
        center_mask = np.zeros((h, w), dtype=bool)
        center_mask[r, c] = True
        sel_pix[:, :, i] = binary_dilation(center_mask, structure=disk_se)

    if valid_pixel_mask is not None:
        for i in range(n_sources):
            sel_pix[:, :, i] &= valid_pixel_mask

    keep = np.sum(sel_pix, axis=(0, 1)) > 5
    source_rows = source_rows[keep]
    source_cols = source_cols[keep]
    source_vals = source_vals[keep]
    sel_pix = sel_pix[:, :, keep]
    n_sources = int(np.sum(keep))

    logger.info("After pruning: %d sources with > 5 pixels each", n_sources)

    return SourceSet(
        rows=source_rows,
        cols=source_cols,
        values=source_vals,
        sel_pix=sel_pix,
        n_sources=n_sources,
    )
