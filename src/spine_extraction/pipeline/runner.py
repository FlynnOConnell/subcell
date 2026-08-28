"""Full pipeline orchestration.

Wires together all stages: build trial table -> register -> extract.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from spine_extraction.config import PipelineConfig, ExtractionConfig
from spine_extraction.io.trial_table import TrialTable
from spine_extraction.io.zarr_store import ExperimentStore
from spine_extraction.registration.bergamo import register_bergamo
from spine_extraction._utils.torch_helpers import get_device
from spine_extraction._utils.logging import setup_logging

logger = logging.getLogger(__name__)


def run_full_pipeline(
    config: PipelineConfig,
    device: torch.device | None = None,
) -> None:
    """Run the complete pipeline: build trial table -> register -> extract.

    Args:
        config: Full pipeline configuration.
        device: PyTorch device.
    """
    if device is None:
        device = get_device(config.device)

    data_dir = config.data_directory
    out_dir = config.get_output_directory()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Build trial table
    tt_path = data_dir / "trial_table.json"
    if tt_path.exists():
        logger.info("Loading existing trial table from %s", tt_path)
        tt = TrialTable.load(tt_path)
    else:
        logger.info("Building trial table from %s", data_dir)
        tt = TrialTable.from_directory(data_dir)
        tt.save(tt_path)

    # Stage 2: Registration
    store = ExperimentStore(out_dir / "experiment.zarr")
    logger.info("Starting registration...")
    tt = register_bergamo(tt, config.registration, store, device=device)
    tt.save(tt_path)
    logger.info("Registration complete.")

    # Stage 3: Extraction
    logger.info("Starting signal extraction...")
    run_extraction(tt, config.extraction, store)
    logger.info("Full pipeline complete.")


def run_extraction(
    trial_table: TrialTable,
    config: ExtractionConfig,
    store: ExperimentStore,
) -> None:
    """Run signal extraction on registered data.

    Steps:
    1. Per-trial source localization
    2. Cross-trial alignment
    3. Source selection
    4. High-resolution signal extraction

    Args:
        trial_table: Trial table with registration paths.
        config: Extraction configuration.
        store: Zarr experiment store.
    """
    from spine_extraction.extraction.localize import localize_sources
    from spine_extraction.extraction.cross_trial_align import align_trials_cross
    from spine_extraction.extraction.source_selection import select_sources
    from spine_extraction.extraction.extract_trial import extract_trial
    from spine_extraction.extraction.assemble import compute_denoised, censor_frames
    from spine_extraction.extraction.nmf.objective import convolve_with_kernel, build_decay_kernel

    n_trials = trial_table.n_trials

    # Step 1: Per-trial localization
    logger.info("Step 1: Per-trial source localization...")
    mean_images = []
    activity_images = []
    alignment_data = []
    keep_trials = np.ones(n_trials, dtype=bool)

    for i, entry in enumerate(trial_table.entries):
        try:
            adata = store.load_alignment_data(entry.trial_index)
            alignment_data.append(adata)

            # Load downsampled registered data
            reg_ds = store.load_registered_ds(entry.trial_index)
            num_ch = adata.num_channels

            # Deinterleave channels (stored interleaved: ch0_t0, ch1_t0, ch0_t1, ...)
            n_ds_frames = reg_ds.shape[2] // num_ch
            movie_4d = reg_ds.reshape(
                reg_ds.shape[0], reg_ds.shape[1], n_ds_frames, num_ch
            ).transpose(0, 1, 3, 2)

            # Compute mean image
            mean_im = np.nanmean(movie_4d, axis=3)
            mean_images.append(mean_im)

            # Extract activity channel and localize
            act_ch = config.activity_channel - 1  # 0-indexed
            movie_act = movie_4d[:, :, act_ch, :]

            act_img, peaks = localize_sources(movie_act, config, adata.align_hz)
            activity_images.append(act_img)
            store.save_activity_image(entry.trial_index, act_img)
            store.save_mean_image(entry.trial_index, mean_im)

        except Exception:
            logger.exception("Failed to process trial %d", entry.trial_index)
            keep_trials[i] = False
            mean_images.append(None)
            activity_images.append(None)
            alignment_data.append(None)

    # Stack valid images
    valid_indices = np.where(keep_trials)[0]
    if len(valid_indices) == 0:
        logger.error("No valid trials found")
        return

    h, w = mean_images[valid_indices[0]].shape[:2]
    n_ch = mean_images[valid_indices[0]].shape[2]

    mean_stack = np.full((h, w, n_ch, n_trials), np.nan, dtype=np.float32)
    act_stack = np.full((h, w, n_trials), np.nan, dtype=np.float32)
    for i in valid_indices:
        mean_stack[:, :, :, i] = mean_images[i]
        act_stack[:, :, i] = activity_images[i]

    # Step 2: Cross-trial alignment
    logger.info("Step 2: Cross-trial alignment...")
    cross_result = align_trials_cross(mean_stack, act_stack, config, keep_trials)

    # Step 3: Source selection
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

    # Step 4: High-resolution signal extraction
    logger.info("Step 4: High-resolution signal extraction...")
    analyze_hz = alignment_data[valid_indices[0]].align_hz * config.extraction.ds_factor if hasattr(config, 'extraction') else alignment_data[valid_indices[0]].align_hz

    # Use the alignment Hz for now (full framerate would need registration config)
    adata0 = alignment_data[valid_indices[0]]
    full_hz = 1.0 / adata0.frame_time if adata0.frame_time > 0 else adata0.align_hz

    sel_pix_union = np.any(sources.sel_pix, axis=2)

    for trial_ix in cross_result.valid_trials:
        entry = trial_table.entries[trial_ix]
        logger.info("Extracting trial %d...", entry.trial_index)

        try:
            # Load full-res registered data
            reg_raw = store.load_registered_raw(entry.trial_index)
            adata = alignment_data[trial_ix]
            num_ch = adata.num_channels

            # Deinterleave and extract activity channel (stored interleaved)
            n_frames = reg_raw.shape[2] // num_ch
            movie_4d = reg_raw.reshape(
                reg_raw.shape[0], reg_raw.shape[1], n_frames, num_ch
            ).transpose(0, 1, 3, 2)
            act_ch = config.activity_channel - 1
            Y_full = movie_4d[:, :, act_ch, :]

            # Extract selected pixels
            Y_sel = Y_full[sel_pix_union, :]

            # Create Finv (for Bergamo, uniform = 1.0)
            Finv = np.ones_like(Y_sel)

            # Detect and censor motion frames
            discard = _detect_motion_frames(adata, config)
            if len(discard) < n_frames:
                discard_full = np.zeros(n_frames, dtype=bool)
                # Upsample discard mask
                ds_to_full = np.round(
                    np.linspace(0, n_frames - 1, len(discard))
                ).astype(int)
                for d_ix, full_ix in enumerate(ds_to_full):
                    if discard[d_ix]:
                        discard_full[full_ix] = True
            else:
                discard_full = discard[:n_frames]

            Y_sel[:, discard_full] = np.nan

            # Run NMF extraction
            result = extract_trial(
                Y_sel, Finv,
                sources.rows, sources.cols,
                sel_pix_union,
                config, full_hz,
            )

            # Compute denoised
            denoised = compute_denoised(result.S, config.tau_s, full_hz)

            # Censor discarded frames
            events = censor_frames(result.S, discard_full)
            denoised = censor_frames(denoised, discard_full)
            ls = censor_frames(result.LS, discard_full)

            # Save results
            store.save_extraction_results(
                entry.trial_index,
                footprints=result.H,
                events=events,
                denoised=denoised,
                ls=ls,
                f0=result.F0,
                snr=result.SNR,
                discard_frames=discard_full,
            )

        except Exception:
            logger.exception("Failed to extract trial %d", entry.trial_index)

    # Save experiment summary
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

    logger.info("Extraction complete: %d sources across %d valid trials",
                sources.n_sources, len(cross_result.valid_trials))


def _detect_motion_frames(adata, config: ExtractionConfig) -> np.ndarray:
    """Detect motion-corrupted frames from alignment data.

    Ports loadAndProcessTrialAsync.m lines 26-31:
        tmp = recNegErr - medfilt1(recNegErr, round(4*alignHz));
        tmp = -tmp ./ min(-0.005, prctile(tmp, 5));
        window = 2*ceil(0.025*alignHz) + 1;
        discardFrames = imclose(
            imdilate(tmp>thresh, ones(window,1)) & (tmp>thresh/2),
            ones(window,1));
        discardFrames(1:nInitFrames) = true;
    """
    import math
    from scipy.ndimage import median_filter, binary_dilation, binary_closing

    rec_err = adata.rec_neg_err
    if len(rec_err) == 0:
        return np.zeros(0, dtype=bool)

    # Deviation from smoothed baseline
    # MATLAB: medfilt1(recNegErr, round(4*alignHz))
    med_size = max(3, int(round(4 * adata.align_hz)))
    # medfilt1 zero-pads; use mode='constant' to match
    smoothed = median_filter(rec_err, size=med_size, mode="constant", cval=0.0)
    tmp = rec_err - smoothed

    # Normalize: -tmp / min(-0.005, prctile(tmp, 5))
    valid = tmp[~np.isnan(tmp)]
    baseline = np.percentile(valid, 5) if len(valid) > 0 else -0.005
    baseline = min(-0.005, baseline)
    tmp = -tmp / baseline

    thresh = config.motion_thresh

    # Temporal censoring window (~50 ms around motion events)
    # MATLAB: window = 2*ceil(0.025*alignHz) + 1
    window = 2 * math.ceil(0.025 * adata.align_hz) + 1
    se = np.ones(window, dtype=bool)

    # MATLAB: imclose(imdilate(tmp>thresh, ones(window,1)) & (tmp>thresh/2), ones(window,1))
    hard_mask = tmp > thresh
    dilated = binary_dilation(hard_mask, structure=se)
    soft_mask = tmp > (thresh / 2)
    combined = dilated & soft_mask
    discard = binary_closing(combined, structure=se)

    # Discard initial frames
    # MATLAB: discardFrames(1:nInitFrames) = true
    n_init = math.ceil(config.discard_initial_s * adata.align_hz)
    if n_init > 0:
        discard[:n_init] = True

    return discard
