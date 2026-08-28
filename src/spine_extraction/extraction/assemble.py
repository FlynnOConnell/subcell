"""Assemble extraction results and compute derived signals.

Post-processes the raw NMF outputs to produce the final signal variants:
events, denoised, least-squares, F0, SNR.
"""

from __future__ import annotations

import math

import numpy as np


def compute_denoised(
    events: np.ndarray,
    tau_s: float,
    analyze_hz: float,
) -> np.ndarray:
    """Compute denoised activity by convolving events with decay kernel.

    dF.denoised = conv(S, kernel, 'same')

    Args:
        events: Event matrix (n_sources, n_time).
        tau_s: Decay time constant in seconds.
        analyze_hz: Analysis framerate.

    Returns:
        Denoised activity, same shape as events.
    """
    tau_frames = tau_s * analyze_hz
    half_len = int(math.ceil(6 * tau_frames))
    t = np.arange(half_len + 1)
    right = np.exp(-t / tau_frames)
    left = np.zeros(half_len)
    kernel = np.concatenate([left, right])
    kernel = kernel / kernel.sum()

    # Convolve each source
    denoised = np.zeros_like(events)
    for i in range(events.shape[0]):
        denoised[i] = np.convolve(events[i], kernel, mode="same")

    return denoised


def censor_frames(
    data: np.ndarray,
    discard_frames: np.ndarray,
) -> np.ndarray:
    """Apply NaN to discarded frames.

    Args:
        data: (n_sources, n_time) or (n_pixels, n_time).
        discard_frames: Boolean mask of frames to censor.

    Returns:
        Data with censored frames set to NaN.
    """
    result = data.copy()
    result[:, discard_frames] = np.nan
    return result
