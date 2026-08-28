"""Cross-trial alignment and valid trial selection.

Ports the cross-trial alignment logic from summarize_LoCo.m lines 162-192.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from subcell.config import ExtractionConfig
from subcell.registration.interpolation import apply_shift_numpy
from subcell.registration.template import make_template_multi_trial
from subcell.registration.xcorr_nans import xcorr2_nans

logger = logging.getLogger(__name__)


@dataclass
class CrossTrialResult:
    """Results of cross-trial alignment."""

    template: np.ndarray
    motion: np.ndarray  # (2, n_trials) alignment offsets
    corr_coeffs: np.ndarray  # (n_trials,) correlation with template
    valid_trials: np.ndarray  # Indices of valid trials
    aligned_mean: np.ndarray  # (H, W, C, n_trials) aligned mean images
    aligned_activity: np.ndarray  # (H, W, n_trials) aligned activity images


def align_trials_cross(
    mean_images: np.ndarray,
    activity_images: np.ndarray,
    config: ExtractionConfig,
    keep_trials: np.ndarray,
) -> CrossTrialResult:
    """
    Align all trials to a common reference and identify valid trials.

    Ports summarize_LoCo.m lines 162-210.

    Parameters
    ----------
    mean_images : np.ndarray
        Per-trial mean images, shape (H, W, C, n_trials).
    activity_images : np.ndarray
        Per-trial activity images, shape (H, W, n_trials).
    config : ExtractionConfig
        Extraction configuration.
    keep_trials : np.ndarray
        Boolean mask of trials that passed verification.

    Returns
    -------
    CrossTrialResult
        CrossTrialResult with aligned data and valid trial indices.
    """
    maxshift = config.cross_trial_maxshift
    n_trials = mean_images.shape[3]
    h, w = mean_images.shape[:2]
    n_ch = mean_images.shape[2]

    M_all = np.sum(mean_images, axis=2)  # (h, w, n_trials)

    M_kept = np.sum(mean_images[:, :, :, keep_trials], axis=2)
    template, _, _ = make_template_multi_trial(M_kept, maxshift)

    t_h, t_w = template.shape
    Mpad = np.full((t_h, t_w, n_trials), np.nan, dtype=np.float32)
    Mpad[maxshift : maxshift + h, maxshift : maxshift + w, :] = M_all

    corr_coeffs = np.full(n_trials, np.nan)
    mot_output = np.full((2, n_trials), np.nan)
    aligned_mean = np.full((h, w, n_ch, n_trials), np.nan, dtype=np.float32)
    aligned_activity = np.full((h, w, n_trials), np.nan, dtype=np.float32)

    view_r, view_c = np.mgrid[0:h, 0:w].astype(np.float64)

    trial_indices = np.where(keep_trials)[0]
    for trial_ix in trial_indices:
        act_trial = activity_images[:, :, trial_ix]
        if np.all(np.isnan(act_trial)):
            logger.info("Skipping trial %d (all NaN activity)", trial_ix)
            continue

        m1, _ = xcorr2_nans(Mpad[:, :, trial_ix], template, np.array([0, 0]), maxshift)
        mot, corr = xcorr2_nans(
            Mpad[:, :, trial_ix],
            template,
            np.round(m1).astype(int),
            maxshift,
        )

        mot_output[:, trial_ix] = mot
        corr_coeffs[trial_ix] = corr

        for ch in range(n_ch):
            aligned_mean[:, :, ch, trial_ix] = apply_shift_numpy(
                mean_images[:, :, ch, trial_ix],
                mot[0],
                mot[1],
                view_r,
                view_c,
            )

        aligned_activity[:, :, trial_ix] = apply_shift_numpy(
            act_trial,
            mot[0],
            mot[1],
            view_r,
            view_c,
        )

    valid_corrs = corr_coeffs[~np.isnan(corr_coeffs)]
    if len(valid_corrs) > 1:
        corr_thresh = min(
            config.valid_trial_corr_min,
            np.median(valid_corrs) - 2 * np.std(valid_corrs),
        )
    else:
        corr_thresh = config.valid_trial_corr_min

    act_valid_pix = np.mean(~np.isnan(aligned_activity[:, :, :]), axis=(0, 1))
    valid_pix_thresh = np.mean(act_valid_pix[~np.isnan(act_valid_pix)]) / 2

    valid_mask = (
        ~np.isnan(corr_coeffs)
        & (corr_coeffs > corr_thresh)
        & (act_valid_pix > valid_pix_thresh)
    )
    valid_trials = np.where(valid_mask)[0]

    logger.info(
        "Cross-trial alignment: %d/%d trials valid (corr_thresh=%.4f)",
        len(valid_trials),
        n_trials,
        corr_thresh,
    )

    return CrossTrialResult(
        template=template,
        motion=mot_output,
        corr_coeffs=corr_coeffs,
        valid_trials=valid_trials,
        aligned_mean=aligned_mean,
        aligned_activity=aligned_activity,
    )
