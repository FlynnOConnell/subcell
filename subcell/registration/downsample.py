"""Temporal downsampling via recursive pairwise averaging.

Ports the downsampleTime function from stripRegBergamo.m lines 421-426.
"""

from __future__ import annotations

import numpy as np


def downsample_time(data: np.ndarray, ds_time: int) -> np.ndarray:
    """
    Downsample the last axis by factor 2^ds_time via recursive pairwise averaging.

    Ports MATLAB:
        for ix = 1:ds_time
            Y = Y(:,:,:,1:2:2*floor(end/2)) + Y(:,:,:,2:2:end);
        end
        Y = Y ./ 2^ds_time;

    Parameters
    ----------
    data : np.ndarray
        Array of shape (..., T) where T is the temporal dimension (last axis).
    ds_time : int
        Number of halvings. Total factor = 2^ds_time.

    Returns
    -------
    np.ndarray
        Downsampled array with last dimension T // 2^ds_time.
    """
    for _ in range(ds_time):
        n = data.shape[-1]
        n_even = 2 * (n // 2)
        data = data[..., 0:n_even:2] + data[..., 1:n_even:2]
    data = data / (2.0**ds_time)
    return data


def downsample_space(data: np.ndarray, ds_times: int = 2) -> np.ndarray:
    """
    Downsample the first two spatial dimensions by 2^ds_times via 2x2 block sums.

    Ports MATLAB spatial downsampling from stripRegBergamo.m lines 251-253:
        for dsIx = 1:dsTimes
            dsTmp = dsTmp(1:2:2*floor(end/2), 1:2:2*floor(end/2)) + ...
        end

    Parameters
    ----------
    data : np.ndarray
        2D or 3D array with spatial dims as first two axes.
    ds_times : int
        Number of halvings in each spatial dimension.

    Returns
    -------
    np.ndarray
        Spatially downsampled array.
    """
    for _ in range(ds_times):
        r = 2 * (data.shape[0] // 2)
        c = 2 * (data.shape[1] // 2)
        data = (
            data[0:r:2, 0:c:2, ...]
            + data[0:r:2, 1:c:2, ...]
            + data[1:r:2, 0:c:2, ...]
            + data[1:r:2, 1:c:2, ...]
        )
    return data
