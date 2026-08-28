"""Pipeline orchestration: build trial table, register, extract."""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
from scipy.ndimage import binary_closing, binary_dilation, median_filter

from subcell._utils.torch_helpers import get_device
from subcell.config import ExtractionConfig, PipelineConfig
from subcell.extraction.assemble import censor_frames, compute_denoised
from subcell.extraction.cross_trial_align import align_trials_cross
from subcell.extraction.extract_trial import extract_trial
from subcell.extraction.localize import localize_sources
from subcell.extraction.source_selection import select_sources
from subcell.io.trial_table import TrialTable
from subcell.io.zarr_store import AlignmentData, ExperimentStore
from subcell.registration.bergamo import register_bergamo

logger = logging.getLogger(__name__)


def upsample_discard_mask(discard_ds: np.ndarray, n_frames: int) -> np.ndarray:
    """
    Expand a downsampled discard mask to full resolution.

    Each downsampled frame censors the whole block of full-resolution frames it
    was built from, not just one representative frame.

    Parameters
    ----------
    discard_ds : np.ndarray
        Boolean mask over downsampled frames.
    n_frames : int
        Number of full-resolution frames.

    Returns
    -------
    np.ndarray
        Boolean mask of length ``n_frames``.
    """
    if len(discard_ds) == 0:
        return np.zeros(n_frames, dtype=bool)
    if len(discard_ds) >= n_frames:
        return discard_ds[:n_frames].astype(bool)

    ds_factor = max(1, int(round(n_frames / len(discard_ds))))
    full = np.repeat(discard_ds.astype(bool), ds_factor)
    if len(full) < n_frames:
        full = np.concatenate([full, np.zeros(n_frames - len(full), dtype=bool)])
    return full[:n_frames]


def detect_motion_frames(adata: AlignmentData, config: ExtractionConfig) -> np.ndarray:
    """
    Flag motion-corrupted frames from the registration error trace.

    Ports loadAndProcessTrialAsync.m: the reconstruction error is detrended by a
    median filter, normalized by its 5th percentile, then thresholded with
    hysteresis and closed over a short window.

    Returns
    -------
    np.ndarray
        Boolean mask over downsampled frames.
    """
    rec_err = adata.rec_neg_err
    if len(rec_err) == 0:
        return np.zeros(0, dtype=bool)

    med_size = max(3, int(round(4 * adata.align_hz)))
    tmp = rec_err - median_filter(rec_err, size=med_size, mode="constant", cval=0.0)

    valid = tmp[~np.isnan(tmp)]
    baseline = min(-0.005, np.percentile(valid, 5) if len(valid) > 0 else -0.005)
    tmp = -tmp / baseline

    thresh = config.motion_thresh
    window = 2 * math.ceil(0.025 * adata.align_hz) + 1
    se = np.ones(window, dtype=bool)
    combined = binary_dilation(tmp > thresh, structure=se) & (tmp > thresh / 2)
    discard = binary_closing(combined, structure=se)

    n_init = math.ceil(config.discard_initial_s * adata.align_hz)
    if n_init > 0:
        discard[:n_init] = True
    return discard


def run_full_pipeline(
    config: PipelineConfig,
    device: torch.device | None = None,
) -> None:
    """
    Run the complete pipeline: trial table, registration, extraction.

    Raises
    ------
    ValueError
        If registration would not save the full-resolution data that extraction
        needs, checked before the expensive registration stage runs.
    """
    if device is None:
        device = get_device(config.device)

    if not config.registration.save_full_resolution:
        raise ValueError(
            "Extraction requires full-resolution registered data, but "
            "registration.save_full_resolution is False. Set it to True (the "
            "output store grows by the temporal downsampling factor), or run "
            "the register and extract stages separately."
        )

    data_dir = config.data_directory
    out_dir = config.get_output_directory()
    out_dir.mkdir(parents=True, exist_ok=True)

    tt_path = data_dir / "trial_table.json"
    if tt_path.exists():
        logger.info("Loading existing trial table from %s", tt_path)
        tt = TrialTable.load(tt_path)
    else:
        logger.info("Building trial table from %s", data_dir)
        tt = TrialTable.from_directory(data_dir)
        tt.save(tt_path)

    store = ExperimentStore(out_dir / "experiment.zarr")

    logger.info("Starting registration...")
    tt = register_bergamo(tt, config.registration, store, device=device)
    tt.save(tt_path)

    logger.info("Starting signal extraction...")
    run_extraction(tt, config.extraction, store, device=device)
    logger.info("Full pipeline complete.")


def run_extraction(
    trial_table: TrialTable,
    config: ExtractionConfig,
    store: ExperimentStore,
    device: torch.device | None = None,
) -> None:
    """
    Localize sources, align across trials, select, and extract signals.

    Parameters
    ----------
    trial_table : TrialTable
        Trials whose registration outputs are present in ``store``.
    device : torch.device, optional
        Resolved once here and reused by every subproblem solve.

    Raises
    ------
    ValueError
        If no trial has full-resolution registered data.
    """
    if device is None:
        device = get_device(config.device)

    n_trials = trial_table.n_trials
    logger.info("Step 1: Per-trial source localization...")

    mean_images = []
    activity_images = []
    alignment_data = []
    keep_trials = np.ones(n_trials, dtype=bool)

    for i, entry in enumerate(trial_table.entries):
        try:
            adata = store.load_alignment_data(entry.trial_index)
            reg_ds = store.load_registered_ds(entry.trial_index)
            num_ch = adata.num_channels

            n_ds_frames = reg_ds.shape[2] // num_ch
            movie_4d = reg_ds.reshape(
                reg_ds.shape[0], reg_ds.shape[1], n_ds_frames, num_ch
            ).transpose(0, 1, 3, 2)

            mean_im = np.nanmean(movie_4d, axis=3)
            movie_act = movie_4d[:, :, config.activity_channel - 1, :]
            act_img, _ = localize_sources(movie_act, config, adata.align_hz)

            store.save_activity_image(entry.trial_index, act_img)
            store.save_mean_image(entry.trial_index, mean_im)

            alignment_data.append(adata)
            mean_images.append(mean_im)
            activity_images.append(act_img)
        except Exception:
            logger.exception("Failed to process trial %d", entry.trial_index)
            keep_trials[i] = False
            mean_images.append(None)
            activity_images.append(None)
            alignment_data.append(None)

    valid_indices = np.where(keep_trials)[0]
    if len(valid_indices) == 0:
        logger.error("No valid trials found")
        return

    h, w, n_ch = mean_images[valid_indices[0]].shape
    mean_stack = np.full((h, w, n_ch, n_trials), np.nan, dtype=np.float32)
    act_stack = np.full((h, w, n_trials), np.nan, dtype=np.float32)
    for i in valid_indices:
        mean_stack[:, :, :, i] = mean_images[i]
        act_stack[:, :, i] = activity_images[i]

    logger.info("Step 2: Cross-trial alignment...")
    cross_result = align_trials_cross(mean_stack, act_stack, config, keep_trials)

    logger.info("Step 3: Source selection...")
    valid_pix = np.mean(
        ~np.isnan(cross_result.aligned_mean[:, :, 0, cross_result.valid_trials]),
        axis=2,
    ) > (1 - config.nan_thresh)

    sources = select_sources(
        cross_result.aligned_activity,
        cross_result.valid_trials,
        config,
        valid_pixel_mask=valid_pix,
    )
    if sources.n_sources == 0:
        logger.warning("No sources found")
        return
    logger.info("Found %d sources", sources.n_sources)

    extractable = [
        ix for ix in cross_result.valid_trials if store.has_registered_raw(ix)
    ]
    if not extractable:
        raise ValueError(
            "No trial has full-resolution registered data, which signal "
            "extraction requires. Re-run registration with "
            "registration.save_full_resolution set to True."
        )
    if len(extractable) < len(cross_result.valid_trials):
        logger.warning(
            "Skipping %d trial(s) without full-resolution data",
            len(cross_result.valid_trials) - len(extractable),
        )

    logger.info("Step 4: High-resolution signal extraction...")
    sel_pix_union = np.any(sources.sel_pix, axis=2)

    for trial_ix in extractable:
        entry = trial_table.entries[trial_ix]
        logger.info("Extracting trial %d...", entry.trial_index)

        try:
            adata = alignment_data[trial_ix]
            full_hz = 1.0 / adata.frame_time if adata.frame_time > 0 else adata.align_hz

            Y_sel = store.load_registered_raw_pixels(
                entry.trial_index,
                sel_pix_union,
                adata.num_channels,
                config.activity_channel - 1,
                block_size=config.block_size,
            )
            n_frames = Y_sel.shape[1]

            discard_full = upsample_discard_mask(
                detect_motion_frames(adata, config), n_frames
            )
            Y_sel[:, discard_full] = np.nan

            result = extract_trial(
                Y_sel,
                np.ones_like(Y_sel),
                sources.rows,
                sources.cols,
                sel_pix_union,
                config,
                full_hz,
                device=device,
            )

            denoised = compute_denoised(result.S, config.tau_s, full_hz)
            store.save_extraction_results(
                entry.trial_index,
                footprints=result.H,
                events=censor_frames(result.S, discard_full),
                denoised=censor_frames(denoised, discard_full),
                ls=censor_frames(result.LS, discard_full),
                f0=result.F0,
                snr=result.SNR,
                discard_frames=discard_full,
            )
        except Exception:
            logger.exception("Failed to extract trial %d", entry.trial_index)

    avg_mean = np.nanmean(
        cross_result.aligned_mean[:, :, :, cross_result.valid_trials], axis=3
    )
    avg_act = np.nanmean(
        cross_result.aligned_activity[:, :, cross_result.valid_trials], axis=2
    )
    store.save_summary(
        mean_image=avg_mean,
        activity_image=avg_act,
        source_locations_r=sources.rows,
        source_locations_c=sources.cols,
        sel_pix=sel_pix_union,
        valid_trials=cross_result.valid_trials,
        trial_offsets=cross_result.motion,
        trial_corr_coeffs=cross_result.corr_coeffs,
        params=config.model_dump(),
    )

    logger.info(
        "Extraction complete: %d sources across %d trials",
        sources.n_sources,
        len(extractable),
    )
