"""Upsample motion vectors from downsampled to original framerate.

Ports stripRegBergamo.m lines 293-296: PCHIP interpolation of motion vectors.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


def upsample_motion(
    motion_ds: np.ndarray,
    ds_factor: int,
    ds_time: int,
) -> np.ndarray:
    """
    Upsample motion vectors via PCHIP interpolation.

    Ports MATLAB:
        tDS = (1:nDSframes) * dsFac - 2^(ds_time-1) + 0.5;
        motion = interp1(tDS, motionDS, 1:nTotal, 'pchip', 'extrap');

    Parameters
    ----------
    motion_ds : np.ndarray
        Downsampled motion vector, shape (n_ds_frames,).
    ds_factor : int
        Downsampling factor (2^ds_time).
    ds_time : int
        Number of halvings.

    Returns
    -------
    np.ndarray
        Upsampled motion vector, shape (n_ds_frames * ds_factor,).
    """
    n_ds = len(motion_ds)
    # Time points for downsampled frames (MATLAB 1-indexed, Python 0-indexed)
    t_ds = np.arange(1, n_ds + 1) * ds_factor - 2 ** (ds_time - 1) + 0.5
    t_full = np.arange(1, n_ds * ds_factor + 1)

    interp = PchipInterpolator(t_ds, motion_ds, extrapolate=True)
    return interp(t_full)
