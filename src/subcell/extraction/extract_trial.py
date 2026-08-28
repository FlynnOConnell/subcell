"""Per-trial signal extraction, porting extractTrial.m.

The extraction is split into independent spatial subproblems (connected
components of the dilated source mask), each solved via constrained NMF.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import psutil
import torch
from scipy.ndimage import binary_dilation
from scipy.ndimage import label as ndimage_label

from subcell._utils.torch_helpers import get_device
from subcell.config import ExtractionConfig
from subcell.extraction.nmf.solver import ExtractionResult, solve_nmf_subproblem

logger = logging.getLogger(__name__)

_BYTES_PER_FLOAT32 = 4
_LBFGS_HISTORY_SIZE = 20


@dataclass
class Subproblem:
    """One spatially independent group of sources and its selected pixels."""

    label: int
    source_indices: np.ndarray
    pixel_idxs: np.ndarray
    rows_local: np.ndarray
    cols_local: np.ndarray
    sel_cropped: np.ndarray


def _available_memory_gb() -> float:
    """Available system memory in GB."""
    return psutil.virtual_memory().available / (1024**3)


def _estimate_memory_gb(
    n_pixels: int,
    n_time: int,
    n_sources: int,
    h_crop: int,
    w_crop: int,
) -> float:
    """
    Peak memory of one subproblem in GB.

    Counts the four (n_pixels, n_time) core arrays, the baseline fit's
    high-pass buffers, the temporal arrays, and the L-BFGS history. FFT
    chunks adapt at runtime and are excluded.
    """
    core = 4 * n_pixels * n_time * _BYTES_PER_FLOAT32
    hp_full = h_crop * w_crop * n_time * _BYTES_PER_FLOAT32
    hp_surround = n_pixels * n_time * 3 * _BYTES_PER_FLOAT32
    temporal = 2 * n_sources * n_time * _BYTES_PER_FLOAT32
    lbfgs = _LBFGS_HISTORY_SIZE * 2 * n_sources * n_time * _BYTES_PER_FLOAT32
    return (core + hp_full + hp_surround + temporal + lbfgs) / (1024**3)


def _build_subproblems(
    labeled: np.ndarray,
    n_problems: int,
    source_rows: np.ndarray,
    source_cols: np.ndarray,
    sel_pix: np.ndarray,
    sel_idxs: np.ndarray,
    disk: np.ndarray,
) -> list[Subproblem]:
    """
    Group sources into subproblems by connected component.

    Parameters
    ----------
    labeled : np.ndarray
        Connected-component labels over the dilated source mask, (h, w).
    n_problems : int
        Number of components in ``labeled``.
    sel_idxs : np.ndarray
        Map from (h, w) pixel to row index in the observation matrix, -1 where invalid.
    disk : np.ndarray
        Structuring element used to grow each source into its pixel set.

    Returns
    -------
    list of Subproblem
        Components that contain at least one source and one selected pixel.
    """
    h, w = sel_pix.shape
    n_sources = len(source_rows)
    subproblems = []

    for label in range(1, n_problems + 1):
        prob_mask = labeled == label
        source_indices = [
            i
            for i in range(n_sources)
            if 0 <= round(source_rows[i]) < h
            and 0 <= round(source_cols[i]) < w
            and prob_mask[int(round(source_rows[i])), int(round(source_cols[i]))]
        ]
        if not source_indices:
            continue
        source_indices = np.array(source_indices)

        sub_sel = np.zeros((h, w), dtype=bool)
        for i in source_indices:
            sub_sel[int(round(source_rows[i])), int(round(source_cols[i]))] = True
        sub_sel = binary_dilation(sub_sel, structure=disk) & sel_pix

        pixel_idxs = sel_idxs[sub_sel]
        pixel_idxs = pixel_idxs[pixel_idxs >= 0]
        if len(pixel_idxs) == 0:
            continue

        rows_any = np.any(sub_sel, axis=1)
        cols_any = np.any(sub_sel, axis=0)
        r_min, r_max = np.where(rows_any)[0][[0, -1]]
        c_min, c_max = np.where(cols_any)[0][[0, -1]]

        subproblems.append(
            Subproblem(
                label=label,
                source_indices=source_indices,
                pixel_idxs=pixel_idxs,
                rows_local=source_rows[source_indices] - r_min,
                cols_local=source_cols[source_indices] - c_min,
                sel_cropped=sub_sel[r_min : r_max + 1, c_min : c_max + 1],
            )
        )
    return subproblems


def _solve_subproblem(
    sub: Subproblem,
    Y_obs: np.ndarray,
    Finv: np.ndarray,
    config: ExtractionConfig,
    analyze_hz: float,
    n_concurrent: int,
    device: torch.device,
) -> ExtractionResult | None:
    """
    Solve one subproblem, returning None if it fails.

    Returns
    -------
    ExtractionResult or None
        None when the NMF solve raised; the failure is logged.
    """
    h_crop, w_crop = sub.sel_cropped.shape
    n_time = Y_obs.shape[1]
    logger.info(
        "Subproblem %d: %d sources, %d pixels, crop %dx%d - est %.1f GB, avail %.1f GB",
        sub.label,
        len(sub.source_indices),
        len(sub.pixel_idxs),
        h_crop,
        w_crop,
        _estimate_memory_gb(
            len(sub.pixel_idxs), n_time, len(sub.source_indices), h_crop, w_crop
        ),
        _available_memory_gb(),
    )

    t0 = time.perf_counter()
    try:
        result = solve_nmf_subproblem(
            Y_obs[sub.pixel_idxs, :],
            Finv[sub.pixel_idxs, :],
            sub.rows_local,
            sub.cols_local,
            sub.sel_cropped,
            config,
            analyze_hz,
            n_concurrent=n_concurrent,
            device=device,
        )
    except Exception:
        logger.exception(
            "Subproblem %d failed after %.1fs", sub.label, time.perf_counter() - t0
        )
        return None

    logger.info(
        "Subproblem %d done in %.1fs (avail %.1f GB)",
        sub.label,
        time.perf_counter() - t0,
        _available_memory_gb(),
    )
    return result


def extract_trial(
    Y_obs: np.ndarray,
    Finv: np.ndarray,
    source_rows: np.ndarray,
    source_cols: np.ndarray,
    sel_pix: np.ndarray,
    config: ExtractionConfig,
    analyze_hz: float,
    device: torch.device | None = None,
) -> ExtractionResult:
    """
    Extract signals from a single trial via constrained NMF.

    Parameters
    ----------
    Y_obs : np.ndarray
        Observations for the selected pixels, (n_sel_pixels, n_time).
    Finv : np.ndarray
        Inverse freshness, same shape as ``Y_obs``.
    source_rows, source_cols : np.ndarray
        Source coordinates in the full field of view.
    sel_pix : np.ndarray
        Boolean mask of selected pixels, (h, w). Its True count must equal
        ``Y_obs.shape[0]``.
    analyze_hz : float
        Framerate of ``Y_obs``.
    device : torch.device, optional
        Resolved once in the calling thread. Defaults to ``config.device``.

    Returns
    -------
    ExtractionResult
        Results assembled across all subproblems, NaN where a subproblem failed.
    """
    h, w = sel_pix.shape
    n_sources = len(source_rows)
    n_time = Y_obs.shape[1]
    n_sel_pixels = Y_obs.shape[0]

    if int(sel_pix.sum()) != n_sel_pixels:
        raise ValueError(
            f"sel_pix has {int(sel_pix.sum())} True pixels but Y_obs has "
            f"{n_sel_pixels} rows"
        )

    if device is None:
        device = get_device(config.device)

    sel_idxs = np.full((h, w), -1, dtype=np.int32)
    sel_idxs[sel_pix] = np.arange(n_sel_pixels)

    zones = np.zeros((h, w), dtype=bool)
    for i in range(n_sources):
        r, c = int(round(source_rows[i])), int(round(source_cols[i]))
        if 0 <= r < h and 0 <= c < w:
            zones[r, c] = True

    sigma_radius = int(np.ceil(1.5 * config.sigma_px + 1))
    y, x = np.mgrid[-sigma_radius : sigma_radius + 1, -sigma_radius : sigma_radius + 1]
    disk = (x**2 + y**2) <= sigma_radius**2
    labeled, n_problems = ndimage_label(binary_dilation(zones, structure=disk))

    subproblems = _build_subproblems(
        labeled, n_problems, source_rows, source_cols, sel_pix, sel_idxs, disk
    )
    avail_gb = _available_memory_gb()
    logger.info(
        "Decomposed into %d subproblems (%.1f GB available)", len(subproblems), avail_gb
    )

    n_cpus = os.cpu_count() or 4
    if subproblems:
        peak_per_sp = 1.5 * max(
            _estimate_memory_gb(
                len(sub.pixel_idxs),
                n_time,
                len(sub.source_indices),
                *sub.sel_cropped.shape,
            )
            for sub in subproblems
        )
        max_concurrent = max(1, int(avail_gb * 0.9 / peak_per_sp))
        max_concurrent = min(
            max_concurrent,
            len(subproblems),
            max(1, n_cpus // 4),
            config.n_parallel_workers,
        )
    else:
        peak_per_sp = 0.0
        max_concurrent = 1

    logger.info(
        "Scheduling: max_concurrent=%d (est %.1f GB/subproblem peak, %.1f GB available)",
        max_concurrent,
        peak_per_sp,
        avail_gb,
    )

    H_all = np.full((n_sel_pixels, n_sources), np.nan, dtype=np.float32)
    B_all = np.full((n_sel_pixels, n_time), np.nan, dtype=np.float32)
    S_all = np.full((n_sources, n_time), np.nan, dtype=np.float32)
    LS_all = np.full((n_sources, n_time), np.nan, dtype=np.float32)
    F0_all = np.full((n_sources, n_time), np.nan, dtype=np.float32)
    SNR_all = np.full(n_sources, np.nan, dtype=np.float32)

    t_total = time.perf_counter()
    solved = []
    if max_concurrent <= 1:
        for sub in subproblems:
            solved.append(
                (
                    sub,
                    _solve_subproblem(sub, Y_obs, Finv, config, analyze_hz, 1, device),
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            futures = {
                pool.submit(
                    _solve_subproblem,
                    sub,
                    Y_obs,
                    Finv,
                    config,
                    analyze_hz,
                    max_concurrent,
                    device,
                ): sub
                for sub in subproblems
            }
            for future in as_completed(futures):
                solved.append((futures[future], future.result()))

    for sub, result in solved:
        if result is None:
            continue
        H_all[sub.pixel_idxs[:, np.newaxis], sub.source_indices[np.newaxis, :]] = (
            result.H
        )
        B_all[sub.pixel_idxs, :] = result.B
        S_all[sub.source_indices, :] = result.S
        LS_all[sub.source_indices, :] = result.LS
        F0_all[sub.source_indices, :] = result.F0
        SNR_all[sub.source_indices] = result.SNR

    n_failed = sum(1 for _, result in solved if result is None)
    logger.info(
        "%d/%d subproblems completed in %.1fs",
        len(solved) - n_failed,
        len(solved),
        time.perf_counter() - t_total,
    )

    return ExtractionResult(
        H=H_all,
        S=S_all,
        B=B_all,
        LS=LS_all,
        SNR=SNR_all,
        F0=F0_all,
    )
