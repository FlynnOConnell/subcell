"""Bergamo registration orchestrator.

Ports stripRegBergamo.m - the main motion correction pipeline for Bergamo
multi-page TIFF recordings. Handles a single trial registration and the
parallel dispatch across all trials.
"""

from __future__ import annotations

import logging
import math
import time as _time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from subcell._utils.parallel import compute_max_workers, estimate_file_size
from subcell._utils.torch_helpers import get_device
from subcell.config import RegistrationConfig
from subcell.io.lazy_array import ArrayTrialSource
from subcell.io.tiff_reader import (
    MmapTiffReader,
    ScanImageMetadata,
    read_scanimage_tiff,
    reshape_interleaved,
)
from subcell.io.trial_table import TrialTable
from subcell.io.zarr_store import AlignmentData, ExperimentStore
from subcell.registration.dft_registration import (
    dft_registration_clipped,
    dft_registration_clipped_numpy,
)
from subcell.registration.downsample import downsample_space, downsample_time
from subcell.registration.interpolation import apply_shift_numpy, apply_shifts_batch
from subcell.registration.motion_upsample import upsample_motion
from subcell.registration.quality import compute_alignment_quality
from subcell.registration.template import create_initial_template
from subcell.registration.xcorr_nans import xcorr2_nans

logger = logging.getLogger(__name__)

_CHUNK_T_DS = 500
_CHUNK_T_RAW = 500


def _write_zarr_chunk(arr, out_start: int, out_end: int, data: np.ndarray) -> None:
    """Write one temporal chunk to a zarr array, from the writer thread."""
    arr[:, :, out_start:out_end] = data


def register_bergamo(
    trial_table: TrialTable,
    config: RegistrationConfig,
    store: ExperimentStore,
    device: torch.device | None = None,
    raw_data: np.ndarray | None = None,
    metadata: ScanImageMetadata | None = None,
    source: ArrayTrialSource | None = None,
) -> TrialTable:
    """
    Register every trial in the table, writing results into the store.

    Parameters
    ----------
    device : torch.device, optional
        Used only when registering on a single worker; the process pool always
        runs on CPU because a GPU context cannot be shared across processes.
    raw_data : np.ndarray, optional
        Pre-loaded TIFF data, (rows, cols, total_pages), to skip re-reading from
        disk. Only valid for a single-trial table and requires ``metadata``.
    metadata : ScanImageMetadata, optional
        Pre-loaded metadata; must accompany ``raw_data``.
    source : ArrayTrialSource, optional
        Read trials from an mbo_utilities LazyArray rather than from TIFFs on
        disk. Registration runs sequentially in this case: the array is not
        picklable across a process pool, and its reads are already lazy.

    Returns
    -------
    TrialTable
        The input table with ``align_params`` recorded.
    """
    if device is None or device == "auto":
        device = get_device("auto")
    elif isinstance(device, str):
        device = torch.device(device)

    if (raw_data is None) != (metadata is None):
        raise ValueError("raw_data and metadata must both be provided or both be None")
    if raw_data is not None and trial_table.n_trials > 1:
        raise ValueError(
            "Pre-loaded raw_data is only supported for single-trial tables"
        )

    entries = trial_table.entries
    if not config.overwrite_existing:
        pending = [e for e in entries if not store.has_alignment_data(e.trial_index)]
        if len(pending) < len(entries):
            logger.info(
                "Skipping %d already-registered trial(s); set overwrite_existing "
                "to re-register them",
                len(entries) - len(pending),
            )
        entries = pending

    if not entries:
        trial_table.align_params = config.model_dump()
        return trial_table

    data_dir = Path(trial_table.directory)

    if source is not None:
        logger.info("Registering %d trials from array source", len(entries))
        index_of = {e.trial_index: i for i, e in enumerate(trial_table.entries)}
        for entry in entries:
            _register_single_trial(
                data_dir,
                entry.filename,
                entry.trial_index,
                config,
                store,
                device,
                metadata=source.metadata,
                frames=source.read_trial(index_of[entry.trial_index]),
            )
        trial_table.align_params = config.model_dump()
        return trial_table

    avg_file_size = estimate_file_size(data_dir, "*.tif")
    n_workers = compute_max_workers(len(entries), avg_file_size, config.n_workers)

    logger.info("Registering %d trials with %d workers", len(entries), n_workers)

    if n_workers <= 1:
        for entry in entries:
            _register_single_trial(
                data_dir,
                entry.filename,
                entry.trial_index,
                config,
                store,
                device,
                raw_data=raw_data,
                metadata=metadata,
            )
    else:
        if device.type != "cpu":
            logger.warning(
                "Multi-worker registration uses CPU; GPU used for single-worker only"
            )
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for entry in entries:
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
                    logger.info(
                        "Completed registration for trial %d", entry.trial_index
                    )
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
    """
    Determine which output rows/cols are always out-of-bounds for all frames.

    A row is always NaN when no frame's shift can place it within the source image.
    Avoids the need to allocate the full output array just to scan for NaN borders.
    """
    min_mr, max_mr = np.nanmin(motion_r), np.nanmax(motion_r)
    min_mc, max_mc = np.nanmin(motion_c), np.nanmax(motion_c)

    row_idx = np.arange(out_h, dtype=np.float64)
    nan_rows = (row_idx - max_shift_r + max_mr < 0) | (
        row_idx - max_shift_r + min_mr > src_h - 1
    )

    col_idx = np.arange(out_w, dtype=np.float64)
    nan_cols = (col_idx - max_shift_c + max_mc < 0) | (
        col_idx - max_shift_c + min_mc > src_w - 1
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
    frames: np.ndarray | None = None,
) -> None:
    """
    Register a single trial.

    Ports the alignAsync inner function from stripRegBergamo.m.

    Parameters
    ----------
    data_dir : Path
        Directory containing TIFF files.
    filename : str
        TIFF filename.
    trial_idx : int
        Trial index.
    config : RegistrationConfig
        Registration configuration.
    store : ExperimentStore
        Zarr output store.
    device : torch.device, optional
        PyTorch device.
    raw_data : np.ndarray, optional
        Pre-loaded TIFF data (rows, cols, total_pages). Skips file read if provided.
    metadata : ScanImageMetadata, optional
        Pre-loaded ScanImage metadata. Required if raw_data or frames is given.
    frames : np.ndarray, optional
        Already-deinterleaved (rows, cols, channels, frames) block, as supplied
        by an mbo_utilities LazyArray source. Skips both the TIFF read and the
        interleave reshape. Requires ``metadata``.
    """
    if device is None or device == "auto":
        device = get_device("auto")
    elif isinstance(device, str):
        device = torch.device(device)
    maxshift = config.maxshift
    ds_factor = config.ds_factor
    ds_time = config.ds_time
    filepath = data_dir / filename

    _phase_times = {}  # phase_name → elapsed seconds
    _t_phase = _time.perf_counter()

    logger.info("Aligning trial %d: %s", trial_idx, filename)

    _mmap_reader = None  # keep reference alive for mmap-backed Ad
    if frames is not None:
        if metadata is None:
            raise ValueError("frames requires metadata")
        logger.info("Using pre-deinterleaved frames from array source")
        Ad = frames
    elif raw_data is None:
        try:
            _mmap_reader = MmapTiffReader(filepath, remove_lines=config.remove_lines)
            Ad = _mmap_reader.data  # (rows, cols, channels, frames) view into mmap
            metadata = _mmap_reader.metadata
            logger.info("Using mmap reader (zero-copy)")
        except (ValueError, OSError) as exc:
            logger.info("Mmap reader unavailable (%s), falling back to full load", exc)
            raw_data, metadata = read_scanimage_tiff(filepath, dtype=None)
            Ad = reshape_interleaved(
                raw_data, metadata.num_channels, config.remove_lines
            )
            del raw_data
    else:
        logger.info("Using pre-loaded data, skipping TIFF read")
        Ad = reshape_interleaved(raw_data, metadata.num_channels, config.remove_lines)
        del raw_data
    num_channels = metadata.num_channels
    rows, cols, n_ch, n_raw_frames = Ad.shape
    logger.info("Ad shape: %s, dtype: %s", Ad.shape, Ad.dtype)

    if metadata.frame_time > 0:
        frametime = metadata.frame_time
    elif config.frame_rate > 0:
        frametime = 1.0 / config.frame_rate
        logger.warning("Using user-supplied frame_rate=%f", config.frame_rate)
    else:
        frametime = 0.0023
        logger.warning("Using default frametime=0.0023")

    align_hz = 1.0 / frametime / ds_factor

    _phase_times["reshape"] = _time.perf_counter() - _t_phase
    _t_phase = _time.perf_counter()
    n_sample = min(500, n_raw_frames)
    sample_idx = np.round(np.linspace(0, n_raw_frames - 1, n_sample)).astype(int)
    sample_data = Ad[:, :, :, sample_idx].astype(np.float32)
    bg = np.percentile(sample_data, 10, axis=(0, 1, 3), keepdims=True).astype(
        np.float32
    )
    bg = np.minimum(bg, 1000.0)  # (1, 1, n_ch, 1)
    del sample_data

    _phase_times["baseline_subtract"] = _time.perf_counter() - _t_phase
    _t_phase = _time.perf_counter()
    init_frames = min(config.init_frames, n_raw_frames // ds_factor)
    frames_to_read = init_frames * ds_factor
    if frames_to_read > n_raw_frames:
        logger.warning("File too short for %d init frames, using all", init_frames)
        frames_to_read = (n_raw_frames // ds_factor) * ds_factor
        init_frames = frames_to_read // ds_factor

    Y = downsample_time(Ad[:, :, :, :frames_to_read].astype(np.int32), ds_time)
    Y = Y.astype(np.float32) - bg
    Yhp = np.sum(Y, axis=2)  # (rows, cols, n_ds_init_frames)

    template = create_initial_template(
        Yhp,
        maxshift=maxshift,
        min_cluster_size=config.min_cluster_size,
        device=device,
    )
    del Y, Yhp

    _phase_times["template_creation"] = _time.perf_counter() - _t_phase
    _t_phase = _time.perf_counter()
    n_ds_frames = n_raw_frames // ds_factor
    bg_correction = float(bg[0, 0, :, 0].sum())  # sum of per-channel bg
    logger.info("Will compute %d downsampled frames on-the-fly", n_ds_frames)

    motion_ds_r = np.full(n_ds_frames, np.nan)
    motion_ds_c = np.full(n_ds_frames, np.nan)

    init_r, init_c = 0, 0
    template_full = template.copy()
    template_ct = np.zeros_like(template)
    T0 = template.copy()
    T00 = np.zeros_like(template)  # Zero bias term (MATLAB T00)

    view_r, view_c = np.mgrid[
        -maxshift : rows + maxshift, -maxshift : cols + maxshift
    ].astype(np.float64)

    ds_space = 2
    ds_sz = (
        (2 * maxshift + rows) // (2**ds_space),
        (2 * maxshift + cols) // (2**ds_space),
    )
    A_ds = np.full((*ds_sz, n_ds_frames), np.nan, dtype=np.float32)

    _phase_times["precompute_ds"] = _time.perf_counter() - _t_phase
    _t_phase = _time.perf_counter()
    use_gpu_dft = device.type != "cpu" and rows * cols > 500 * 500
    logger.info(
        "Registering %d downsampled frames (%s DFT, %s interpolation)...",
        n_ds_frames,
        "GPU" if use_gpu_dft else "CPU",
        "GPU" if device.type != "cpu" else "CPU",
    )

    _prev_init_r, _prev_init_c = None, None
    _Ttmp = None
    _T_crop_fft = None  # Cached FFT of template crop (numpy complex128 or torch)
    _template_changed = True

    for ds_frame in range(n_ds_frames):
        raw_start = ds_frame * ds_factor
        M = (
            Ad[:, :, :, raw_start : raw_start + ds_factor].sum(
                axis=(2, 3), dtype=np.float32
            )
            / (2.0**ds_time)
            - bg_correction
        )

        if ds_frame % 1000 == 0 and ds_frame > 0:
            logger.info("  Frame %d / %d", ds_frame, n_ds_frames)

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

        if (
            abs(motion_ds_r[ds_frame]) < maxshift
            and abs(motion_ds_c[ds_frame]) < maxshift
        ):
            ir = int(math.floor(motion_ds_r[ds_frame] + 0.5))
            ic = int(math.floor(motion_ds_c[ds_frame] + 0.5))
            A = np.full(
                (rows + 2 * maxshift, cols + 2 * maxshift), np.nan, dtype=np.float32
            )
            r0 = maxshift - ir
            c0 = maxshift - ic
            A[r0 : r0 + rows, c0 : c0 + cols] = M

            A_ds[:, :, ds_frame] = downsample_space(A, ds_space)[: ds_sz[0], : ds_sz[1]]

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

            # MATLAB rounds half away from zero; Python's round() is half-to-even
            init_r = int(math.floor(motion_ds_r[ds_frame] + 0.5))
            init_c = int(math.floor(motion_ds_c[ds_frame] + 0.5))
            _template_changed = True
        else:
            motion_ds_r[ds_frame] = init_r
            motion_ds_c[ds_frame] = init_c

    _phase_times["registration_loop"] = _time.perf_counter() - _t_phase
    _t_phase = _time.perf_counter()

    rec_neg_err = compute_alignment_quality(A_ds, align_hz, running_template=template)

    motion_r = upsample_motion(motion_ds_r, ds_factor, ds_time)
    motion_c = upsample_motion(motion_ds_c, ds_factor, ds_time)

    n_total = min(len(motion_r), n_raw_frames)
    motion_r = motion_r[:n_total]
    motion_c = motion_c[:n_total]

    max_shift_c = max(abs(motion_c)) if len(motion_c) > 0 else maxshift
    max_shift_r = max(abs(motion_r)) if len(motion_r) > 0 else maxshift
    out_rows = len(np.arange(-max_shift_r, rows + max_shift_r))
    out_cols = len(np.arange(-max_shift_c, cols + max_shift_c))

    nan_rows, nan_cols = _compute_nan_borders(
        out_rows,
        out_cols,
        max_shift_r,
        max_shift_c,
        rows,
        cols,
        motion_r,
        motion_c,
    )
    out_h = int(np.sum(~nan_rows))
    out_w = int(np.sum(~nan_cols))

    _phase_times["quality_upsample"] = _time.perf_counter() - _t_phase
    _t_phase = _time.perf_counter()

    interp_device = torch.device("cpu") if rows * cols < 500 * 500 else device

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
        for t_ds_start in range(0, n_ds_frames, _CHUNK_T_DS):
            t_ds_end = min(t_ds_start + _CHUNK_T_DS, n_ds_frames)
            n_ds_chunk = t_ds_end - t_ds_start
            raw_start = t_ds_start * ds_factor
            raw_end = min(t_ds_end * ds_factor, n_total)

            chunk_ds = np.zeros((rows, cols, n_ch, n_ds_chunk), dtype=np.float32)
            for f in range(ds_factor):
                chunk_ds += Ad[:, :, :, raw_start + f : raw_end : ds_factor].astype(
                    np.float32
                )
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
                    out_rows,
                    out_cols,
                    -max_shift_r,
                    -max_shift_c,
                    device=interp_device,
                )
                valid = ~np.isnan(B)
                B_count[:, :, ch] += np.sum(valid, axis=2)
                B_sum[:, :, ch] += np.nansum(B, axis=2)
                ds_chunk_out[:, :, ch::num_channels] = B[~nan_rows, :, :][
                    :, ~nan_cols, :
                ]
                del B
            del chunk_ds

            if pending is not None:
                pending.result()
            ds_out_s = t_ds_start * num_channels
            ds_out_e = t_ds_end * num_channels
            pending = writer.submit(
                _write_zarr_chunk, ds_arr, ds_out_s, ds_out_e, ds_chunk_out
            )

            if config.save_full_resolution:
                for raw_sub_start in range(raw_start, raw_end, _CHUNK_T_RAW):
                    raw_sub_end = min(raw_sub_start + _CHUNK_T_RAW, raw_end)
                    n_sub = raw_sub_end - raw_sub_start

                    sub_data = (
                        Ad[:, :, :, raw_sub_start:raw_sub_end].astype(np.float32) - bg
                    )

                    raw_chunk_out = np.full(
                        (out_h, out_w, n_sub * num_channels), np.nan, dtype=np.float32
                    )
                    for ch in range(num_channels):
                        B = apply_shifts_batch(
                            sub_data[:, :, ch, :],
                            motion_r[raw_sub_start:raw_sub_end],
                            motion_c[raw_sub_start:raw_sub_end],
                            out_rows,
                            out_cols,
                            -max_shift_r,
                            -max_shift_c,
                            device=interp_device,
                        )
                        raw_chunk_out[:, :, ch::num_channels] = B[~nan_rows, :, :][
                            :, ~nan_cols, :
                        ]
                        del B
                    del sub_data

                    if pending is not None:
                        pending.result()
                    raw_out_s = raw_sub_start * num_channels
                    raw_out_e = raw_sub_end * num_channels
                    pending = writer.submit(
                        _write_zarr_chunk, raw_arr, raw_out_s, raw_out_e, raw_chunk_out
                    )

            if t_ds_start % 5000 == 0 and t_ds_start > 0:
                logger.info("  Write: %d / %d ds frames", t_ds_start, n_ds_frames)

        if pending is not None:
            pending.result()

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
    _phase_times["write_registered"] = _time.perf_counter() - _t_phase
    _t_phase = _time.perf_counter()

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
