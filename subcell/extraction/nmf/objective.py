"""NMF objective function with PyTorch autograd.

Ports the loss function from extractTrial.m:
    L = sum((Y - T)^2 / (Finv * (T + lambda)))
where T = H @ X + B, X = conv1d(S, kernel, 'same')

PyTorch autograd replaces the manual gradient and Hessian-vector product
implementations (objfun_S_wrapper, hessmult_S_wrapper, objfun_Hs_wrapper,
hessmult_Hs_wrapper).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def build_decay_kernel(tau_frames: float) -> torch.Tensor:
    """
    Build the normalized exponential decay kernel.

    Ports setParamsExtractTrial.m lines 23-24:
        k = [zeros(1,ceil(6*tau)) exp(-(0:ceil(6*tau))/tau)];
        k = k./sum(k);

    Parameters
    ----------
    tau_frames : float
        Decay time constant in frames.

    Returns
    -------
    torch.Tensor
        1D kernel tensor, shape (1, 1, kernel_len), normalized to sum=1.
    """
    half_len = int(math.ceil(6 * tau_frames))
    t = torch.arange(half_len + 1, dtype=torch.float64)
    right = torch.exp(-t / tau_frames)
    left = torch.zeros(half_len, dtype=torch.float64)
    kernel = torch.cat([left, right])
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0).float()  # (1, 1, kernel_len)


def convolve_with_kernel(S: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Convolve S with decay kernel (same padding).

    X = conv1d(S, kernel, padding='same')

    Parameters
    ----------
    S : torch.Tensor
        Activity tensor, shape (n_sources, n_time).
    kernel : torch.Tensor
        Decay kernel, shape (1, 1, kernel_len).

    Returns
    -------
    torch.Tensor
        X
        Convolved tensor, shape (n_sources, n_time).
    """
    _, n_time = S.shape
    pad = kernel.shape[2] // 2
    S_3d = S.unsqueeze(1)  # (n_sources, 1, n_time)
    X = F.conv1d(S_3d, kernel, padding=pad)
    if X.shape[2] > n_time:
        X = X[:, :, :n_time]
    return X.squeeze(1)  # (n_sources, n_time)


def nmf_loss(
    Y_obs: torch.Tensor,
    H: torch.Tensor,
    X: torch.Tensor,
    B: torch.Tensor,
    Finv: torch.Tensor,
    lam: float = 1.0,
) -> torch.Tensor:
    """
    Compute the weighted Poisson-Gaussian NMF loss.

    L = sum((Y - T)^2 / (Finv * (T + lambda)))
    where T = H @ X + B

    Parameters
    ----------
    Y_obs : torch.Tensor
        Observations, (n_pixels, n_time).
    H : torch.Tensor
        Spatial footprints, (n_pixels, n_sources).
    X : torch.Tensor
        Temporal convolved activity, (n_sources, n_time).
    B : torch.Tensor
        Baseline, (n_pixels, n_time).
    Finv : torch.Tensor
        Inverse freshness, (n_pixels, n_time).
    lam : float
        Regularization parameter (after normalization, typically 1.0).

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    T = H @ X + B
    residual = Y_obs - T
    variance = Finv * (T + lam)
    loss = (residual**2 / variance.clamp(min=1e-8)).sum()
    return loss


def compute_snr(
    Y_obs: torch.Tensor,
    H: torch.Tensor,
    X: torch.Tensor,
    B: torch.Tensor,
    Finv: torch.Tensor,
    tau_full: float,
) -> torch.Tensor:
    """
    Compute per-source signal-to-noise ratio.

    Ports extractTrial.m lines 245-254.

    Parameters
    ----------
    Y_obs : torch.Tensor
        Observations.
    H : torch.Tensor
        Spatial footprints.
    X : torch.Tensor
        Convolved activity.
    B : torch.Tensor
        Baseline.
    Finv : torch.Tensor
        Inverse freshness.
    tau_full : float
        Decay constant in frames.

    Returns
    -------
    torch.Tensor
        SNR per source, shape (n_sources,).
    """
    resid = Y_obs - (B + H @ X)

    valid = ~torch.isnan(resid)  # (n_pixels, n_time)
    resid = torch.where(valid, resid, torch.zeros_like(resid))

    resid_weights = 1.0 / Finv.clamp(min=1e-8)
    resid_weights[resid >= 0] = 0
    resid_weights[~valid] = 0  # zero weight for censored frames
    resid_var = (resid**2 * resid_weights).sum(dim=1) / resid_weights.sum(dim=1).clamp(
        min=1e-8
    )

    W = torch.diag(1.0 / resid_var.clamp(min=1e-8))
    HtWH = H.T @ W @ H
    HtWH += 1e-6 * torch.eye(HtWH.shape[0])
    cov_X = torch.linalg.inv(HtWH)
    X_noise = torch.sqrt(torch.abs(torch.diag(cov_X)) / max(tau_full, 1.0))

    X_std = X.std(dim=1)
    snr = X_std / X_noise.clamp(min=1e-8)
    return snr, X_noise
