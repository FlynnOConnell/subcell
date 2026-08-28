"""Bergamo registration orchestrator.

Ports stripRegBergamo.m - the main motion correction pipeline for Bergamo
multi-page TIFF recordings. Handles a single trial registration and the
parallel dispatch across all trials.
"""

from __future__ import annotations

import logging
import math
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from spine_extraction.config import RegistrationConfig
from spine_extraction.io.tiff_reader import (
    MmapTiffReader,
    ScanImageMetadata,
    read_scanimage_tiff,
    reshape_interleaved,
)
from spine_extraction.io.trial_table import TrialTable
from spine_extraction.io.zarr_store import AlignmentData, ExperimentStore
from spine_extraction.registration.dft_registration import (
    dft_registration_clipped,
    dft_registration_clipped_numpy,
)
from spine_extraction.registration.downsample import downsample_time, downsample_space
from spine_extraction.registration.interpolation import apply_shift_numpy, apply_shifts_batch
from spine_extraction.registration.motion_upsample import upsample_motion
from spine_extraction.registration.quality import compute_alignment_quality
from spine_extraction.registration.template import create_initial_template
from spine_extraction.registration.xcorr_nans import xcorr2_nans
from spine_extraction._utils.parallel import compute_max_workers, estimate_file_size

logger = logging.getLogger(__name__)


def register_bergamo(
    trial_table: TrialTable,
    config: RegistrationConfig,
    store: ExperimentStore,
    device: torch.device | None = None,
    raw_data: np.ndarray | None = None,
    metadata: ScanImageMetadata | None = None,
) -> TrialTable:
    """Register all trials in parallel.

    Args:
        trial_table: Trial table with file paths.
        config: Registration configuration.
        store: Zarr output store.
        device: PyTorch device for GPU operations.
        raw_data: Pre-loaded TIFF data array (rows, cols, total_pages). If provided
            together with metadata, the TIFF file will not be re-read from disk.
            Only valid for single-trial tables.
        metadata: Pre-loaded ScanImage metadata. Must be provided together with raw_data.

    Returns:
        Updated trial table with registration output paths.
    """
    if device is None or device == "auto":
        from spine_extraction._utils.torch_helpers import get_device
        device = get_device("auto")
    elif isinstance(device, str):
        device = torch.device(device)

    # Validate pre-loaded data arguments
    if (raw_data is None) != (metadata is None):
        raise ValueError("raw_data and metadata must both be provided or both be None")
    if raw_data is not None and trial_table.n_trials > 1:
        raise ValueError("Pre-loaded raw_data is only supported for single-trial tables")

    n_trials = trial_table.n_trials
    data_dir = Path(trial_table.directory)

    avg_file_size = estimate_file_size(data_dir, "*.tif")
    n_workers = compute_max_workers(n_trials, avg_file_size, config.n_workers)

    logger.info("Registering %d trials with %d workers", n_trials, n_workers)

    if n_workers <= 1:
        # Sequential processing
        for entry in trial_table.entries:
            _register_single_trial(
                data_dir, entry.filename, entry.trial_index, config, store, device,
                raw_data=raw_data, metadata=metadata,
            )
    else:
        # Parallel processing (GPU not shared across processes — use CPU)
        if device.type != "cpu":
            logger.warning("Multi-worker registration uses CPU; GPU used for single-worker only")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for entry in trial_table.entries:
                future = pool.submit(
                    _register_single_trial,
                    data_dir,
                    entry.filename,
                    entry.trial_index,
                    config,
                    store,
                    torch.device("cpu"),
                )
                futures[future] = entry

            for future in as_completed(futures):
                entry = futures[future]
                try:
                    future.result()
                    logger.info("Completed registration for trial %d", entry.trial_index)
                except Exception:
                    logger.exception(
                        "Failed to register trial %d (%s)",
                        entry.trial_index,
                        entry.filename,
                    )

    trial_table.align_params = config.model_dump()
    return trial_table


def _compute_nan_borders(
    out_h: int,
    out_w: int,
    max_shift_r: float,
    max_shift_c: float,
    src_h: int,
    src_w: int,
    motion_r: np.ndarray,
    motion_c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Determine which output rows/cols are always out-of-bounds for all frames.

    A row is always NaN when no frame's shift can place it within the source image.
    Avoids the need to allocate the full output array just to scan for NaN borders.
    """
    min_mr, max_mr = np.nanmin(motion_r), np.nanmax(motion_r)
    min_mc, max_mc = np.nanmin(motion_c), np.nanmax(motion_c)

    row_idx = np.arange(out_h, dtype=np.float64)
    # Source row for output row i: i - max_shift_r + shift
    # Valid when 0 <= source <= src_h - 1 for at least one shift
    nan_rows = (
        (row_idx - max_shift_r + max_mr < 0)
        | (row_idx - max_shift_r + min_mr > src_h - 1)
    )

    col_idx = np.arange(out_w, dtype=np.float64)
    nan_cols = (
        (col_idx - max_shift_c + max_mc < 0)
        | (col_idx - max_shift_c + min_mc > src_w - 1)
    )

    return nan_rows, nan_cols


def _register_single_trial(
    data_dir: Path,
    filename: str,
    trial_idx: int,
    config: RegistrationConfig,
    store: ExperimentStore,
    device: torch.device | None = None,
    raw_data: np.ndarray | None = None,
    metadata: ScanImageMetadata | None = None,
) -> None:
    """Register a single trial.

    Ports the alignAsync inner function from stripRegBergamo.m.

    Args:
        data_dir: Directory containing TIFF files.
        filename: TIFF filename.
        trial_idx: Trial index.
        config: Registration configuration.
        store: Zarr output store.
        device: PyTorch device.
        raw_data: Pre-loaded TIFF data (rows, cols, total_pages). Skips file read if provided.
        metadata: Pre-loaded ScanImage metadata. Required if raw_data is provided.
    """
    if device is None or device == "auto":
        from spine_extraction._utils.torch_helpers import get_device
        device = get_device("auto")
    elif isinstance(device, str):
        device = torch.device(device)
    maxshift = config.maxshift
    ds_factor = config.ds_factor
    ds_time = config.ds_time
    filepath = data_dir / filename

    import time as _time
    _phase_times = {}  # phase_name → elapsed seconds
    _t_phase = _time.perf_counter()

    logger.info("Aligning trial %d: %s", trial_idx, filename)

    # Phase 1: Load and parse metadata
    _mmap_reader = None  # keep reference alive for mmap-backed Ad
    if raw_data is None:
        # Try zero-copy mmap reader first; fall back to full load on failure
        try:
            _mmap_reader = MmapTiffReader(filepath, remove_lines=config.remove_lines)
            Ad = _mmap_reader.data  # (rows, cols, channels, frames) view into mmap
            metadata = _mmap_reader.metadata
            logger.info("Using mmap reader (zero-copy)")
        except (ValueError, OSError) as exc:
            logger.info("Mmap reader unavailable (%s), falling back to full load", exc)
            raw_data, metadata = read_scanimage_tiff(filepath, dtype=None)
            Ad = reshape_interleaved(raw_data, metadata.num_channels, config.remove_lines)
            del raw_data
    else:
        logger.info("Using pre-loaded data, skipping TIFF read")
        Ad = reshape_interleaved(raw_data, metadata.num_channels, config.remove_lines)
        del raw_data
    num_channels = metadata.num_channels
    rows, cols, n_ch, n_raw_frames = Ad.shape
    logger.info("Ad shape: %s, dtype: %s", Ad.shape, Ad.dtype)

    # Determine frametime
    if metadata.frame_time > 0:
        frametime = metadata.frame_time
    elif config.frame_rate > 0:
        frametime = 1.0 / config.frame_rate
        logger.warning("Using user-supplied frame_rate=%f", config.frame_rate)
    else:
        frametime = 0.0023
        logger.warning("Using default frametime=0.0023")

    align_hz = 1.0 / frametime / ds_factor

    # Phase 2: Compute baseline (deferred subtraction — avoids full float32 copy)
    _phase_times["reshape"] = _time.perf_counter() - _t_phase; _t_phase = _time.perf_counter()
    n_sample = min(500, n_raw_frames)
    sample_idx = np.round(np.linspace(0, n_raw_frames - 1, n_sample)).astype(int)
    sample_data = Ad[:, :, :, sample_idx].astype(np.float32)
    bg = np.percentile(sample_data, 10, axis=(0, 1, 3), keepdims=True).astype(np.float32)
    bg = np.minimum(bg, 1000.0)  # (1, 1, n_ch, 1)
    # bg subtraction is deferred to per-chunk processing in M_all and write phases
    del sample_data

    # Phase 3: Create initial template
    _phase_times["baseline_subtract"] = _time.perf_counter() - _t_phase; _t_phase = _time.perf_counter()
    init_frames = min(config.init_frames, n_raw_frames // ds_factor)
    frames_to_read = init_frames * ds_factor
    if frames_to_read > n_raw_frames:
        logger.warning("File too short for %d init frames, using all", init_frames)
        frames_to_read = (n_raw_frames // ds_factor) * ds_factor
        init_frames = frames_to_read // ds_factor

    # Use int32 to prevent overflow during pairwise summation (int16 max 32767,
    # but 8 frames of ~4000 counts = 32000 which is borderline).
    Y = downsample_time(Ad[:, :, :, :frames_to_read].astype(np.int32), ds_time)
    # Subtract deferred baseline and sum across channels
    Y = Y.astype(np.float32) - bg
    Yhp = np.sum(Y, axis=2)  # (rows, cols, n_ds_init_frames)

    template = create_initial_template(
        Yhp, maxshift=maxshift, min_cluster_size=config.min_cluster_size,
        device=device,
    )
    del Y, Yhp

    # Compute bg correction scalar (deferred baseline summed across channels)
    _phase_times["template_creation"] = _time.perf_counter() - _t_phase; _t_phase = _time.perf_counter()
    n_ds_frames = n_raw_frames // ds_factor
    bg_correction = float(bg[0, 0, :, 0].sum())  # sum of per-channel bg
    logger.info("Will compute %d downsampled frames on-the-fly", n_ds_frames)

    # Initialize tracking arrays
    motion_ds_r = np.full(n_ds_frames, np.nan)
    motion_ds_c = np.full(n_ds_frames, np.nan)

    init_r, init_c = 0, 0
    template_full = template.copy()
    template_ct = np.zeros_like(template)
    T0 = template.copy()  # Initial template (fixed)
    T00 = np.zeros_like(template)  # Zero bias term (MATLAB T00)

    # View matrices for interpolation
    view_r, view_c = np.mgrid[
        -maxshift : rows + maxshift, -maxshift : cols + maxshift
    ].astype(np.float64)

    # Spatial downsampling for quality assessment
    ds_space = 2
    ds_sz = (
        (2 * maxshift + rows) // (2**ds_space),
        (2 * maxshift + cols) // (2**ds_space),
    )
    A_ds = np.full((*ds_sz, n_ds_frames), np.nan, dtype=np.float32)

    # Phase 4: Frame-by-frame registration
    _phase_times["precompute_ds"] = _time.perf_counter() - _t_phase; _t_phase = _time.perf_counter()
    # Per-frame DFT registration: use numpy (CPU) for small images because
    # per-frame GPU transfer overhead dominates at these sizes (~146×228).
    # GPU is still used for batch interpolation in Phases 7/9.
    use_gpu_dft = device.type != "cpu" and rows * cols > 500 * 500
    logger.info(
        "Registering %d downsampled frames (%s DFT, %s interpolation)...",
        n_ds_frames,
        "GPU" if use_gpu_dft else "CPU",
        "GPU" if device.type != "cpu" else "CPU",
    )

    # Cache for template blending — only recompute when template changes
    _prev_init_r, _prev_init_c = None, None
    _Ttmp = None
    _T_crop_fft = None  # Cached FFT of template crop (numpy complex128 or torch)
    _template_changed = True

    for ds_frame in range(n_ds_frames):
        # Compute downsampled frame on-the-fly: sum ds_factor raw frames across channels
        raw_start = ds_frame * ds_factor
        M = (
            Ad[:, :, :, raw_start : raw_start + ds_factor]
            .sum(axis=(2, 3), dtype=np.float32)
            / (2.0**ds_time)
            - bg_correction
        )

        if ds_frame % 1000 == 0 and ds_frame > 0:
            logger.info("  Frame %d / %d", ds_frame, n_ds_frames)

        # Recompute blended template only when template or crop position changed
        if _template_changed or init_r != _prev_init_r or init_c != _prev_init_c:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                _Ttmp = np.nanmean(np.stack([T0, T00, template], axis=2), axis=2)
            r_start = maxshift - init_r
            c_start = maxshift - init_c
            T_crop = _Ttmp[r_start : r_start + rows, c_start : c_start + cols]

            if use_gpu_dft:
                _T_crop_fft = torch.fft.fft2(
                    torch.from_numpy(T_crop.astype(np.float32)).to(device)
                )
            else:
                _T_crop_fft = np.fft.fft2(T_crop.astype(np.float32))

            _prev_init_r, _prev_init_c = init_r, init_c
            _template_changed = False

        # FFT cross-correlation
        # MATLAB convention: buf1ft=fft2(frame), buf2ft=fft2(template)
        # The shift output is from template toward frame (i.e. where the frame is).
        if use_gpu_dft:
            M_t = torch.from_numpy(M.astype(np.float32)).to(device)
            result = dft_registration_clipped(
                torch.fft.fft2(M_t), _T_crop_fft, usfac=4, clip=config.clip_shift
            )
        else:
            M_fft = np.fft.fft2(M.astype(np.float32))
            result = dft_registration_clipped_numpy(
                M_fft, _T_crop_fft, usfac=4, clip=config.clip_shift
            )

        motion_ds_r[ds_frame] = init_r + result.row_shift
        motion_ds_c[ds_frame] = init_c + result.col_shift

        # Fallback to xcorr2_nans if DFT registration hit the clip boundary,
        # indicating the true shift may exceed the search range.
        # The original MATLAB check (rel_motion > 0.75^2) uses total motion
        # relative to image size, which triggers too often on small images.
        residual_at_clip = (
            abs(result.row_shift) >= config.clip_shift - 0.5
            or abs(result.col_shift) >= config.clip_shift - 0.5
        )
        if residual_at_clip:
            M_full = apply_shift_numpy(M, 0, 0, view_r, view_c)
            motion_vec, _ = xcorr2_nans(
                M_full, _Ttmp, np.array([init_r, init_c]), maxshift
            )
            motion_ds_r[ds_frame] = motion_vec[0]
            motion_ds_c[ds_frame] = motion_vec[1]

        # Apply shift and update template
        if (
            abs(motion_ds_r[ds_frame]) < maxshift
            and abs(motion_ds_c[ds_frame]) < maxshift
        ):
            # Integer-shift via array slicing (template averaging doesn't
            # need sub-pixel precision; sub-pixel motion is saved separately).
            ir = int(math.floor(motion_ds_r[ds_frame] + 0.5))
            ic = int(math.floor(motion_ds_c[ds_frame] + 0.5))
            A = np.full((rows + 2 * maxshift, cols + 2 * maxshift), np.nan, dtype=np.float32)
            r0 = maxshift - ir
            c0 = maxshift - ic
            A[r0:r0 + rows, c0:c0 + cols] = M

            # Spatial downsampling for quality assessment
            A_ds[:, :, ds_frame] = downsample_space(A, ds_space)[: ds_sz[0], : ds_sz[1]]

            # Running template update
            valid = ~np.isnan(A)
            combined = np.where(
                valid,
                np.where(
                    np.isnan(template_full),
                    A,
                    (template_full * template_ct + A),
                ),
                template_full * template_ct,
            )
            template_ct = template_ct + valid.astype(np.float32)
            template_ct_safe = np.maximum(template_ct, 1)
            template_full = combined / template_ct_safe
            template = template_full.copy()
            template[template_ct < config.template_min_count] = np.nan

            # MATLAB rounds half-away-from-zero; Python's round() uses
            # banker's rounding (half-to-even).  Use floor(x + 0.5) to match.
            init_r = int(math.floor(motion_ds_r[ds_frame] + 0.5))
            init_c = int(math.floor(motion_ds_c[ds_frame] + 0.5))
            _template_changed = True
        else:
            motion_ds_r[ds_frame] = init_r
            motion_ds_c[ds_frame] = init_c

    _phase_times["registration_loop"] = _time.perf_counter() - _t_phase; _t_phase = _time.perf_counter()

    # Phase 5: Alignment quality
    # MATLAB: offset = max(0, prctile(template(:), 25))  — uses running template
    rec_neg_err = compute_alignment_quality(A_ds, align_hz, running_template=template)

    # Phase 6: Upsample motion vectors
    motion_r = upsample_motion(motion_ds_r, ds_factor, ds_time)
    motion_c = upsample_motion(motion_ds_c, ds_factor, ds_time)

    # Trim to actual number of raw frames
    n_total = min(len(motion_r), n_raw_frames)
    motion_r = motion_r[:n_total]
    motion_c = motion_c[:n_total]

    # Compute tighter FOV — output grid dimensions without allocating mgrid
    max_shift_c = max(abs(motion_c)) if len(motion_c) > 0 else maxshift
    max_shift_r = max(abs(motion_r)) if len(motion_r) > 0 else maxshift
    out_rows = len(np.arange(-max_shift_r, rows + max_shift_r))
    out_cols = len(np.arange(-max_shift_c, cols + max_shift_c))

    # Determine which border rows/cols are always out-of-bounds (NaN for all frames)
    nan_rows, nan_cols = _compute_nan_borders(
        out_rows, out_cols, max_shift_r, max_shift_c, rows, cols,
        motion_r, motion_c,
    )
    out_h = int(np.sum(~nan_rows))
    out_w = int(np.sum(~nan_cols))

    # Temporal chunk size for streaming to zarr
    CHUNK_T = 500

    def _zarr_write(arr, out_s, out_e, data):
        """Write a chunk to zarr (runs in background thread)."""
        arr[:, :, out_s:out_e] = data

    # Phases 7-9: Merged write loop (ds + fullres in a single pass over Ad)
    # Iterates in ds-chunk units; for each chunk:
    #   1. Downsample → shift → write to ds zarr + accumulate mean image
    #   2. (If save_full_resolution) shift raw frames in sub-chunks → write to fullres zarr
    _phase_times["quality_upsample"] = _time.perf_counter() - _t_phase; _t_phase = _time.perf_counter()

    # Device for batch interpolation: CPU for small images where PCIe overhead dominates
    interp_device = torch.device("cpu") if rows * cols < 500 * 500 else device

    CHUNK_T_DS = 500   # ds frames per outer iteration
    CHUNK_T_RAW = 500  # raw frames per fullres sub-chunk

    ds_arr = store.create_registered_ds(
        trial_idx,
        shape=(out_h, out_w, n_ds_frames * num_channels),
        num_channels=num_channels,
    )
    raw_arr = None
    if config.save_full_resolution:
        raw_arr = store.create_registered_raw(
            trial_idx,
            shape=(out_h, out_w, n_total * num_channels),
            num_channels=num_channels,
        )

    B_sum = np.zeros((out_rows, out_cols, num_channels), dtype=np.float64)
    B_count = np.zeros((out_rows, out_cols, num_channels), dtype=np.float64)

    logger.info(
        "Writing registered data (%d ds + %s raw frames, %s interpolation)...",
        n_ds_frames,
        str(n_total) if config.save_full_resolution else "0",
        interp_device.type,
    )

    with ThreadPoolExecutor(max_workers=1) as writer:
        pending = None
        for t_ds_start in range(0, n_ds_frames, CHUNK_T_DS):
            t_ds_end = min(t_ds_start + CHUNK_T_DS, n_ds_frames)
            n_ds_chunk = t_ds_end - t_ds_start
            raw_start = t_ds_start * ds_factor
            raw_end = min(t_ds_end * ds_factor, n_total)

            # --- DS: compute downsampled chunk via strided sums (avoids reshape copy) ---
            chunk_ds = np.zeros((rows, cols, n_ch, n_ds_chunk), dtype=np.float32)
            for f in range(ds_factor):
                chunk_ds += Ad[:, :, :, raw_start + f:raw_end:ds_factor].astype(np.float32)
            chunk_ds /= 2.0**ds_time
            chunk_ds -= bg

            ds_chunk_out = np.full(
                (out_h, out_w, n_ds_chunk * num_channels), np.nan, dtype=np.float32
            )
            for ch in range(num_channels):
                B = apply_shifts_batch(
                    chunk_ds[:, :, ch, :],
                    motion_ds_r[t_ds_start:t_ds_end],
                    motion_ds_c[t_ds_start:t_ds_end],
                    out_rows, out_cols, -max_shift_r, -max_shift_c,
                    device=interp_device,
                )
                valid = ~np.isnan(B)
                B_count[:, :, ch] += np.sum(valid, axis=2)
                B_sum[:, :, ch] += np.nansum(B, axis=2)
                ds_chunk_out[:, :, ch::num_channels] = B[~nan_rows, :, :][:, ~nan_cols, :]
                del B
            del chunk_ds

            # Submit ds write
            if pending is not None:
                pending.result()
            ds_out_s = t_ds_start * num_channels
            ds_out_e = t_ds_end * num_channels
            pending = writer.submit(_zarr_write, ds_arr, ds_out_s, ds_out_e, ds_chunk_out)

            # --- Fullres: shift raw frames in sub-chunks ---
            if config.save_full_resolution:
                for raw_sub_start in range(raw_start, raw_end, CHUNK_T_RAW):
                    raw_sub_end = min(raw_sub_start + CHUNK_T_RAW, raw_end)
                    n_sub = raw_sub_end - raw_sub_start

                    # Convert sub-chunk to float32 with bg subtraction
                    sub_data = Ad[:, :, :, raw_sub_start:raw_sub_end].astype(np.float32) - bg

                    raw_chunk_out = np.full(
                        (out_h, out_w, n_sub * num_channels), np.nan, dtype=np.float32
                    )
                    for ch in range(num_channels):
                        B = apply_shifts_batch(
                            sub_data[:, :, ch, :],
                            motion_r[raw_sub_start:raw_sub_end],
                            motion_c[raw_sub_start:raw_sub_end],
                            out_rows, out_cols, -max_shift_r, -max_shift_c,
                            device=interp_device,
                        )
                        raw_chunk_out[:, :, ch::num_channels] = B[~nan_rows, :, :][:, ~nan_cols, :]
                        del B
                    del sub_data

                    if pending is not None:
                        pending.result()
                    raw_out_s = raw_sub_start * num_channels
                    raw_out_e = raw_sub_end * num_channels
                    pending = writer.submit(_zarr_write, raw_arr, raw_out_s, raw_out_e, raw_chunk_out)

            if t_ds_start % 5000 == 0 and t_ds_start > 0:
                logger.info("  Write: %d / %d ds frames", t_ds_start, n_ds_frames)

        if pending is not None:
            pending.result()

    # Phase 8: Save average images per channel
    for ch in range(num_channels):
        B_mean = B_sum[:, :, ch] / np.maximum(B_count[:, :, ch], 1)
        B_mean[B_count[:, :, ch] == 0] = np.nan
        valid_vals = B_mean[~np.isnan(B_mean)]
        if len(valid_vals) > 0:
            min_v = np.percentile(valid_vals, 10)
            max_v = np.percentile(valid_vals, 99.9)
            B_norm = np.clip((B_mean - min_v) / max(max_v - min_v, 1e-10), 0, 1)
            B_norm = np.nan_to_num(B_norm, nan=0.0)
            B_8bit = (255 * np.sqrt(B_norm)).astype(np.uint8)
        else:
            B_8bit = np.zeros((out_rows, out_cols), dtype=np.uint8)
        B_8bit = B_8bit[~nan_rows, :][:, ~nan_cols]
        store.save_avg_image(trial_idx, ch + 1, B_8bit)
    del B_sum, B_count

    del Ad
    if _mmap_reader is not None:
        _mmap_reader.close()
        _mmap_reader = None
    _phase_times["write_registered"] = _time.perf_counter() - _t_phase; _t_phase = _time.perf_counter()

    # Phase 10: Save alignment metadata
    adata = AlignmentData(
        num_channels=num_channels,
        frame_time=frametime,
        align_hz=align_hz,
        motion_r=motion_r,
        motion_c=motion_c,
        motion_ds_r=motion_ds_r,
        motion_ds_c=motion_ds_c,
        rec_neg_err=rec_neg_err,
    )
    store.save_alignment_data(trial_idx, adata)

    total = sum(_phase_times.values())
    logger.info(
        "Registration complete for trial %d — phase times: %s (total %.1fs)",
        trial_idx,
        ", ".join(f"{k}={v:.1f}s" for k, v in _phase_times.items()),
        total,
    )
