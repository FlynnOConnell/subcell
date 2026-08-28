"""Initial template creation via hierarchical clustering.

Ports the template creation logic from stripRegBergamo.m lines 133-184 and
makeTemplateMultiRoi.m for cross-trial alignment.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from spine_extraction.registration.dft_registration import (
    dft_registration_clipped,
    RegistrationResult,
)
from spine_extraction.registration.interpolation import apply_shift_numpy
from spine_extraction.registration.xcorr_nans import xcorr2_nans

logger = logging.getLogger(__name__)


def create_initial_template(
    frames_ds: np.ndarray,
    maxshift: int = 50,
    min_cluster_size: int = 100,
    max_cutoff: float = 2.0,
    device: "torch.device | None" = None,
) -> np.ndarray:
    """Create initial template via hierarchical clustering and rigid alignment.

    Ports stripRegBergamo.m lines 133-184:
    1. Compute pairwise correlation on flattened frames
    2. Hierarchical clustering (average linkage on 1-correlation distance)
    3. Increase distance cutoff until a cluster with >= min_cluster_size frames
    4. Select the cluster with highest mean correlation
    5. Rigid-align ALL frames to the cluster mean and average
    6. Pad with NaN borders of width maxshift

    Args:
        frames_ds: Downsampled frames, shape (rows, cols, n_frames).
            Should be the sum across channels (single-channel representation).
        maxshift: Maximum shift for padding and registration.
        min_cluster_size: Minimum frames in a valid cluster.
        max_cutoff: Maximum distance cutoff for clustering.

    Returns:
        Template padded with NaN borders, shape (rows+2*maxshift, cols+2*maxshift).
    """
    rows, cols, n_frames = frames_ds.shape
    logger.info("Creating initial template from %d frames", n_frames)

    # Compute correlation matrix
    flat = frames_ds.reshape(-1, n_frames)  # (pixels, frames)
    rho = np.corrcoef(flat.T)  # (n_frames, n_frames)
    dist_matrix = 1.0 - rho
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = np.maximum(dist_matrix, 0)  # Ensure non-negative
    # Force exact symmetry (corrcoef can have tiny float asymmetries)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    # Hierarchical clustering
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="average")

    # Find cluster with >= min_cluster_size frames
    # MATLAB increments cutoff BEFORE testing, so first test is at 0.02
    cutoff = 0.01
    best_cluster = None
    while True:
        cutoff += 0.01
        if cutoff > max_cutoff:
            break

        T = fcluster(Z, t=cutoff, criterion="distance")
        clusters = [np.where(T == label)[0] for label in np.unique(T)]

        # Find best cluster (highest mean correlation, >= min_cluster_size)
        max_mean_corr = -np.inf
        for cluster_indices in clusters:
            if len(cluster_indices) >= min_cluster_size:
                sub_rho = rho[np.ix_(cluster_indices, cluster_indices)]
                mean_corr = np.mean(sub_rho)
                if mean_corr > max_mean_corr:
                    max_mean_corr = mean_corr
                    best_cluster = cluster_indices

        if best_cluster is not None:
            break

    if best_cluster is None:
        logger.warning(
            "Could not find cluster with >= %d frames. Using all frames.",
            min_cluster_size,
        )
        best_cluster = np.arange(n_frames)

    logger.info(
        "Selected cluster with %d frames (mean correlation %.4f)",
        len(best_cluster),
        max_mean_corr,
    )

    # Rigid alignment of ALL frames to the best-cluster mean.
    # MATLAB: normcorre(Yhp, options, mean(Yhp(:,:,best_cluster),3)); F = mean(F,3);
    # The cluster mean is only used as the alignment target — all frames
    # contribute to the final averaged template F.
    mean_template = np.mean(frames_ds[:, :, best_cluster], axis=2)

    import torch

    if device is None:
        device = torch.device("cpu")

    aligned = np.zeros((rows, cols, n_frames), dtype=np.float32)
    template_t = torch.from_numpy(mean_template.astype(np.float32)).to(device)
    template_fft = torch.fft.fft2(template_t)
    for i in range(n_frames):
        frame = frames_ds[:, :, i]
        frame_t = torch.from_numpy(frame.astype(np.float32)).to(device)
        result = dft_registration_clipped(
            torch.fft.fft2(frame_t),
            template_fft,
            usfac=4,
            clip=maxshift,
        )
        aligned[:, :, i] = apply_shift_numpy(frame, result.row_shift, result.col_shift)

    F = np.nanmean(aligned, axis=2)

    # Pad template with NaN borders
    template = np.full(
        (2 * maxshift + rows, 2 * maxshift + cols), np.nan, dtype=np.float32
    )
    template[maxshift : maxshift + rows, maxshift : maxshift + cols] = F

    logger.info("Initial template created: shape %s", template.shape)
    return template


def make_template_multi_trial(
    mean_images: np.ndarray,
    maxshift: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create cross-trial template from mean images.

    Ports makeTemplateMultiRoi.m:
    1. Compute all pairwise motions via xcorr2_nans
    2. Find best reference trial (highest median correlation)
    3. Align best-correlated trials and average

    Args:
        mean_images: Stack of mean images, shape (rows, cols, n_trials).
        maxshift: Maximum shift for alignment.

    Returns:
        Tuple of (template, motions, corr_matrix):
        - template: Aligned template with NaN padding.
        - motions: Motion vectors (2, n_trials, n_trials).
        - corr_matrix: Pairwise correlation coefficients (n_trials, n_trials).
    """
    rows, cols, n_trials = mean_images.shape
    logger.info("Building cross-trial template from %d trials", n_trials)

    R = np.ones((n_trials, n_trials))
    motion = np.zeros((2, n_trials, n_trials))

    for f1 in range(n_trials):
        for f2 in range(f1 + 1, n_trials):
            # Coarse alignment
            m1, _ = xcorr2_nans(
                mean_images[:, :, f2],
                mean_images[:, :, f1],
                np.array([0, 0]),
                maxshift,
            )
            # Fine alignment
            mot, r = xcorr2_nans(
                mean_images[:, :, f2],
                mean_images[:, :, f1],
                np.round(m1).astype(int),
                maxshift,
            )
            motion[:, f1, f2] = mot
            motion[:, f2, f1] = -mot
            R[f1, f2] = r
            R[f2, f1] = r

    # Find best reference trial
    median_R = np.median(R, axis=0)
    max_ind = np.argmax(median_R)
    best_R = median_R[max_ind]
    frame_inds = np.where(R[:, max_ind] >= best_R)[0]

    logger.info(
        "Best reference trial: %d (median corr %.4f, %d aligned trials)",
        max_ind,
        best_R,
        len(frame_inds),
    )

    # Build template by aligning selected frames to best trial
    view_r, view_c = np.mgrid[
        -maxshift : rows + maxshift, -maxshift : cols + maxshift
    ].astype(np.float64)

    aligned = np.full(
        (rows + 2 * maxshift, cols + 2 * maxshift, len(frame_inds)),
        np.nan,
        dtype=np.float32,
    )

    for i, fix in enumerate(frame_inds):
        mot_r = motion[0, fix, max_ind]
        mot_c = motion[1, fix, max_ind]
        aligned[:, :, i] = apply_shift_numpy(
            mean_images[:, :, fix], -mot_r, -mot_c, view_r, view_c
        )

    template = np.nanmean(aligned, axis=2)
    # NaN out places with few measurements
    nan_count = np.sum(np.isnan(aligned), axis=2)
    template[nan_count > len(frame_inds) // 2] = np.nan

    return template, motion, R
