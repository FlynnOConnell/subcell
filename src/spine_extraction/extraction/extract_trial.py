"""Per-trial signal extraction orchestration.

Ports extractTrial.m - decomposes the extraction into independent spatial
subproblems (connected components) and solves each via constrained NMF.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.ndimage import binary_dilation, label as ndimage_label

from spine_extraction.config import ExtractionConfig
from spine_extraction.extraction.nmf.solver import solve_nmf_subproblem, ExtractionResult

logger = logging.getLogger(__name__)


def _get_available_memory_gb() -> float:
    """Return available system memory in GB."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        return float("nan")


def _estimate_subproblem_memory_gb(
    n_pixels: int, n_time: int, n_sources: int, h_crop: int, w_crop: int,
) -> float:
    """Estimate peak memory usage for one subproblem in GB.

    Accounts for:
    - Input arrays: Y_sub, F_sub (float32)
    - Torch tensors (CPU): Y_t, Finv_t, H_t, B_t, S_t
    - L-BFGS history for S optimizer: history_size=20 pairs of (n_sources, n_time)
    - _fit_baseline: hp_full (h*w*n_time*4), HP_surround (n_pix*n_time*3*4)
    - FFT chunks are adaptive and not counted here
    """
    bytes_per_float32 = 4
    # Core arrays: Y_filled + Y_obs + Finv + B_np = 4 arrays of (n_pixels, n_time)
    core_arrays = 4 * n_pixels * n_time * bytes_per_float32
    # hp_full in get_surround: (h_crop, w_crop, n_time)
    hp_full = h_crop * w_crop * n_time * bytes_per_float32
    # HP_surround: (n_pixels, n_time, 3)
    hp_surround = n_pixels * n_time * 3 * bytes_per_float32
    # S, X arrays: (n_sources, n_time)
    temporal = 2 * n_sources * n_time * bytes_per_float32
    # L-BFGS history for S: 20 pairs × 2 vectors × (n_sources × n_time × 4)
    lbfgs_history = 20 * 2 * n_sources * n_time * bytes_per_float32
    total = core_arrays + hp_full + hp_surround + temporal + lbfgs_history
    return total / (1024**3)


def extract_trial(
    Y_obs: np.ndarray,
    Finv: np.ndarray,
    source_rows: np.ndarray,
    source_cols: np.ndarray,
    sel_pix: np.ndarray,
    config: ExtractionConfig,
    analyze_hz: float,
) -> ExtractionResult:
    """Extract signals from a single trial via constrained NMF.

    Ports extractTrial.m lines 1-304.
    Decomposes into independent subproblems based on spatial connectivity,
    solves each, and assembles results.

    Args:
        Y_obs: Observations for selected pixels, (n_sel_pixels, n_time).
        Finv: Inverse freshness, (n_sel_pixels, n_time).
        source_rows: Row coordinates of all sources.
        source_cols: Column coordinates of all sources.
        sel_pix: 2D boolean mask of selected pixels (union across all sources).
        config: Extraction configuration.
        analyze_hz: Analysis framerate.

    Returns:
        ExtractionResult with assembled results across all subproblems.
    """
    h, w = sel_pix.shape
    n_sources = len(source_rows)
    n_time = Y_obs.shape[1]
    n_sel_pixels = Y_obs.shape[0]

    # Build pixel index map: sel_pix pixels -> 0..n_sel_pixels-1
    sel_idxs = np.full((h, w), -1, dtype=np.int32)
    sel_idxs[sel_pix] = np.arange(n_sel_pixels)

    # Create zones: dilate around each source center
    sigma_radius = int(np.ceil(1.5 * config.sigma_px + 1))
    zones = np.zeros((h, w), dtype=bool)
    for i in range(n_sources):
        r, c = int(round(source_rows[i])), int(round(source_cols[i]))
        if 0 <= r < h and 0 <= c < w:
            zones[r, c] = True

    # Dilate zones
    y, x = np.mgrid[-sigma_radius:sigma_radius + 1, -sigma_radius:sigma_radius + 1]
    disk = (x**2 + y**2) <= sigma_radius**2
    zones = binary_dilation(zones, structure=disk)

    # Find connected components
    labeled, n_problems = ndimage_label(zones)

    avail_gb = _get_available_memory_gb()
    logger.info(
        "Decomposed into %d independent subproblems (%.1f GB available)",
        n_problems, avail_gb,
    )

    # Initialize output arrays
    H_all = np.full((n_sel_pixels, n_sources), np.nan, dtype=np.float32)
    B_all = np.full((n_sel_pixels, n_time), np.nan, dtype=np.float32)
    S_all = np.full((n_sources, n_time), np.nan, dtype=np.float32)
    LS_all = np.full((n_sources, n_time), np.nan, dtype=np.float32)
    F0_all = np.full((n_sources, n_time), np.nan, dtype=np.float32)
    SNR_all = np.full(n_sources, np.nan, dtype=np.float32)

    # Prepare subproblem metadata (lightweight — pixel data extracted on-demand)
    subproblems = []
    subproblem_estimates = []
    for prob_ix in range(1, n_problems + 1):
        prob_mask = labeled == prob_ix
        source_indices = []
        for i in range(n_sources):
            r, c = int(round(source_rows[i])), int(round(source_cols[i]))
            if 0 <= r < h and 0 <= c < w and prob_mask[r, c]:
                source_indices.append(i)

        if len(source_indices) == 0:
            continue

        source_indices = np.array(source_indices)
        sub_rows = source_rows[source_indices]
        sub_cols = source_cols[source_indices]

        # Build local sel_pix for this subproblem
        sub_sel = np.zeros((h, w), dtype=bool)
        for i in source_indices:
            r, c = int(round(source_rows[i])), int(round(source_cols[i]))
            sub_sel[r, c] = True
        sub_sel = binary_dilation(sub_sel, structure=disk) & sel_pix

        # Extract pixel indices (small integer array, not the actual data)
        sub_pixel_idxs = sel_idxs[sub_sel]
        valid_mask = sub_pixel_idxs >= 0
        sub_pixel_idxs = sub_pixel_idxs[valid_mask]

        if len(sub_pixel_idxs) == 0:
            continue

        # Crop to bounding box
        rows_any = np.any(sub_sel, axis=1)
        cols_any = np.any(sub_sel, axis=0)
        r_min, r_max = np.where(rows_any)[0][[0, -1]]
        c_min, c_max = np.where(cols_any)[0][[0, -1]]

        sub_sel_cropped = sub_sel[r_min:r_max + 1, c_min:c_max + 1]
        sub_rows_local = sub_rows - r_min
        sub_cols_local = sub_cols - c_min

        subproblems.append((
            prob_ix, source_indices, sub_pixel_idxs,
            sub_rows_local, sub_cols_local, sub_sel_cropped,
        ))
        h_crop, w_crop = sub_sel_cropped.shape
        subproblem_estimates.append(
            _estimate_subproblem_memory_gb(
                len(sub_pixel_idxs), n_time, len(source_indices), h_crop, w_crop,
            )
        )

    # Determine how many subproblems can run concurrently
    n_cpus = os.cpu_count() or 4
    if not np.isnan(avail_gb) and subproblem_estimates:
        max_est = max(subproblem_estimates)
        peak_per_sp = max_est * 1.5  # safety factor for split_freq/FFT temporaries
        max_concurrent = max(1, int(avail_gb * 0.9 / peak_per_sp))
        max_concurrent = min(max_concurrent, len(subproblems))
        # Cap by CPU: each subproblem gets cpu_count/N FFT workers,
        # diminishing returns below ~4 workers per subproblem.
        max_concurrent = min(max_concurrent, max(1, n_cpus // 4))
    else:
        max_concurrent = 1

    fft_workers_per = max(1, n_cpus // max_concurrent)
    logger.info(
        "Scheduling: max_concurrent=%d (est %.1f GB/subproblem peak, "
        "%.1f GB available, %d FFT workers each)",
        max_concurrent, peak_per_sp if subproblem_estimates else 0,
        avail_gb, fft_workers_per,
    )

    # Resolve device once in the main thread.  This eagerly initializes
    # the CUDA context (avoids concurrent init from worker threads which
    # can crash the Windows driver) and avoids repeated log messages.
    from spine_extraction._utils.torch_helpers import get_device
    device = get_device(config.device)

    def _solve_one(sp_idx, args):
        """Solve a single subproblem with memory and timing logging."""
        (prob_ix, source_indices, sub_pixel_idxs,
         sub_rows_local, sub_cols_local, sub_sel_cropped) = args

        h_crop, w_crop = sub_sel_cropped.shape
        n_pix = len(sub_pixel_idxs)
        n_src = len(source_indices)
        est_gb = _estimate_subproblem_memory_gb(n_pix, n_time, n_src, h_crop, w_crop)
        cur_avail = _get_available_memory_gb()

        logger.info(
            "Subproblem %d/%d (id=%d): %d sources, %d pixels, "
            "crop %dx%d — est %.1f GB, avail %.1f GB",
            sp_idx + 1, len(subproblems), prob_ix,
            n_src, n_pix, h_crop, w_crop, est_gb, cur_avail,
        )

        # Extract pixel data on-demand
        Y_sub = Y_obs[sub_pixel_idxs, :]
        F_sub = Finv[sub_pixel_idxs, :]

        t0 = time.perf_counter()
        try:
            result = solve_nmf_subproblem(
                Y_sub, F_sub,
                sub_rows_local, sub_cols_local,
                sub_sel_cropped,
                config, analyze_hz,
                n_concurrent=max_concurrent,
                device=device,
            )
            # Assemble into global arrays (disjoint indices — thread-safe)
            H_all[sub_pixel_idxs[:, np.newaxis], source_indices[np.newaxis, :]] = result.H
            B_all[sub_pixel_idxs, :] = result.B
            S_all[source_indices, :] = result.S
            LS_all[source_indices, :] = result.LS
            F0_all[source_indices, :] = result.F0
            SNR_all[source_indices] = result.SNR
            elapsed = time.perf_counter() - t0
            logger.info(
                "  Subproblem %d/%d done in %.1fs (avail %.1f GB)",
                sp_idx + 1, len(subproblems), elapsed, _get_available_memory_gb(),
            )
        except Exception:
            elapsed = time.perf_counter() - t0
            logger.exception(
                "Failed to solve subproblem %d after %.1fs", prob_ix, elapsed,
            )

    t_total = time.perf_counter()
    if max_concurrent <= 1:
        # Sequential execution
        for i, sp in enumerate(subproblems):
            _solve_one(i, sp)
    else:
        # Parallel execution with memory-aware concurrency limit
        with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            futures = {
                pool.submit(_solve_one, i, sp): i
                for i, sp in enumerate(subproblems)
            }
            for future in as_completed(futures):
                # Re-raise any exception from the worker thread
                exc = future.exception()
                if exc is not None:
                    logger.error("Subproblem thread raised: %s", exc)

    logger.info(
        "All %d subproblems completed in %.1fs",
        len(subproblems), time.perf_counter() - t_total,
    )

    return ExtractionResult(
        H=H_all,
        S=S_all,
        B=B_all,
        LS=LS_all,
        SNR=SNR_all,
        F0=F0_all,
    )
