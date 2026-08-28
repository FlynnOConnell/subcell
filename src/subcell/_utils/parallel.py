"""Memory-aware parallel worker pool management."""

from __future__ import annotations

import logging
from pathlib import Path

import psutil

logger = logging.getLogger(__name__)


def compute_max_workers(
    n_trials: int,
    file_size_bytes: int,
    requested_workers: int,
    memory_fraction: float = 0.13,
) -> int:
    """
    Compute the maximum number of parallel workers based on available memory.

    Mirrors MATLAB logic: min(nTrials, floor(fraction * memAvailable / fileSize)).

    Parameters
    ----------
    n_trials : int
        Total number of trials to process.
    file_size_bytes : int
        Approximate size of each trial file in bytes.
    requested_workers : int
        Number of workers requested by user/config.
    memory_fraction : float
        Fraction of available memory to use (default 0.13).
    """
    mem_available = psutil.virtual_memory().available
    if file_size_bytes > 0:
        mem_workers = int(memory_fraction * mem_available / file_size_bytes)
    else:
        mem_workers = requested_workers

    n_workers = min(n_trials, requested_workers, max(1, mem_workers))
    logger.info(
        "Using %d workers (requested=%d, memory-limited=%d, trials=%d)",
        n_workers,
        requested_workers,
        mem_workers,
        n_trials,
    )
    return n_workers


def estimate_file_size(directory: Path, pattern: str = "*.tif") -> int:
    """Estimate the average file size for files matching a pattern."""
    files = list(directory.glob(pattern))
    if not files:
        return 0
    total = sum(f.stat().st_size for f in files)
    return total // len(files)
