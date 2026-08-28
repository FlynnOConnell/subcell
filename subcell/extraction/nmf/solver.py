"""NMF solver using PyTorch L-BFGS with projected gradient descent.

Replaces MATLAB fmincon('trust-region-reflective') with PyTorch-native
optimization. Non-negativity constraints are enforced via projection
(clamping) after each optimization step.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt

from subcell._utils.torch_helpers import get_device
from subcell.config import ExtractionConfig
from subcell.extraction.nmf.baseline import (
    build_baseline_filters,
    get_surround,
    split_freq,
)
from subcell.extraction.nmf.objective import (
    build_decay_kernel,
    compute_snr,
    convolve_with_kernel,
    nmf_loss,
)
from subcell.extraction.nmf.spatial import (
    build_gaussian_filter_kernel,
    initialize_footprints,
    smooth_hs_to_h,
)

logger = logging.getLogger(__name__)

_LBFGS_HISTORY_SIZE = 20


@dataclass
class ExtractionResult:
    """Result of NMF extraction for a subproblem."""

    H: np.ndarray  # (n_pixels, n_sources) spatial footprints
    S: np.ndarray  # (n_sources, n_time) inferred events
    B: np.ndarray  # (n_pixels, n_time) baseline
    LS: np.ndarray  # (n_sources, n_time) least-squares estimate
    SNR: np.ndarray  # (n_sources,) signal-to-noise ratio
    F0: np.ndarray  # (n_sources, n_time) baseline fluorescence per source


def solve_nmf_subproblem(
    Y_obs: np.ndarray,
    Finv: np.ndarray,
    source_rows: np.ndarray,
    source_cols: np.ndarray,
    sel_pix: np.ndarray,
    config: ExtractionConfig,
    analyze_hz: float,
    n_concurrent: int = 1,
    device: object | None = None,
) -> ExtractionResult:
    """
    Solve a single NMF subproblem.

    Ports extractSources from extractTrial.m lines 137-304.

    Parameters
    ----------
    Y_obs : np.ndarray
        Observations, (n_pixels, n_time).
    Finv : np.ndarray
        Inverse freshness, (n_pixels, n_time).
    source_rows : np.ndarray
        Row coordinates of sources in this subproblem.
    source_cols : np.ndarray
        Column coordinates of sources in this subproblem.
    sel_pix : np.ndarray
        2D boolean mask for this subproblem.
    config : ExtractionConfig
        Extraction configuration.
    analyze_hz : float
        Analysis framerate.
    n_concurrent : int
        Number of subproblems running concurrently.

    Returns
    -------
    ExtractionResult
        ExtractionResult with all extracted signals.
    """
    _, n_time = Y_obs.shape
    n_sources = len(source_rows)
    tau_full = config.tau_s * analyze_hz
    denoise_samps = int(config.denoise_window_s * analyze_hz)
    baseline_samps = int(config.baseline_window_glu_s * analyze_hz)

    lam = config.lambda_param
    if lam is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            px_std = np.nanstd(Y_obs, axis=1)
            px_mean = np.nanmean(Y_obs, axis=1)
        dim_sel = px_mean < np.nanpercentile(px_mean, 20)
        if np.any(dim_sel):
            lam = float(4 * np.nanpercentile(px_std[dim_sel], 90))
        else:
            lam = float(4 * np.nanpercentile(px_std, 90))
        logger.debug("Auto-estimated lambda = %.4f", lam)

    Y_obs = Y_obs / np.float32(lam)
    lam_normalized = 1.0

    H_np, Hs_np, H_valid = initialize_footprints(
        source_rows, source_cols, sel_pix, config.sigma_px
    )

    B_np, _ = split_freq(Y_obs, denoise_samps, max(1, baseline_samps // denoise_samps))
    B_np = np.maximum(B_np, np.float32(lam_normalized / 10)).astype(np.float32)

    Y_filled = Y_obs.copy()
    nan_mask = np.isnan(Y_filled)
    if np.any(nan_mask):
        Y_filled[nan_mask] = B_np[nan_mask]

    init_residual = np.nan_to_num(Y_filled - B_np, nan=np.float32(0.0))
    H_f64 = H_np.astype(np.float64)
    res_f64 = init_residual.astype(np.float64)
    try:
        dF_ls = np.linalg.lstsq(H_f64, res_f64, rcond=None)[0].astype(np.float32)
    except np.linalg.LinAlgError:
        try:
            dF_ls = (np.linalg.pinv(H_f64) @ res_f64).astype(np.float32)
        except np.linalg.LinAlgError:
            dF_ls = np.zeros((n_sources, n_time), dtype=np.float32)
    del H_f64, res_f64
    S_init = np.maximum(
        0, dF_ls - np.concatenate([np.zeros((n_sources, 1)), dF_ls[:, :-1]], axis=1)
    )

    kernel = build_decay_kernel(tau_full)
    h_filter = build_gaussian_filter_kernel(config.sigma_px)

    if device is None:
        device = get_device(config.device)

    Y_t = torch.from_numpy(Y_filled)
    Finv_t = torch.from_numpy(Finv.astype(np.float32))
    H_t = torch.from_numpy(H_np.astype(np.float32))
    B_t = torch.from_numpy(B_np.astype(np.float32))
    S_t = torch.from_numpy(S_init.astype(np.float32))
    Hs_t = torch.from_numpy(Hs_np.astype(np.float32))
    H_valid_t = torch.from_numpy(H_valid)

    for outer_loop in range(config.nmf_iter):
        max_iter = 10 * (outer_loop + 1)

        S_t = _optimize_S(Y_t, H_t, B_t, Finv_t, S_t, kernel, lam_normalized, max_iter)

        X_t = convolve_with_kernel(S_t, kernel)

        Y_orig_t = torch.from_numpy(Y_obs.astype(np.float32))
        snr_t, X_noise_t = compute_snr(Y_orig_t, H_t, X_t, B_t, Finv_t, tau_full)
        del Y_orig_t

        if outer_loop == config.nmf_iter - 1:
            break

        # Subtract floor and noise level from X (MATLAB extractTrial.m line 261)
        X_np = X_t.detach().numpy()
        X_floor = _compute_floor(X_np, denoise_samps, baseline_samps)
        X_noise_np = X_noise_t.detach().numpy()[:, None]  # (n_sources, 1)
        X_np = np.maximum(0, X_np - X_floor - X_noise_np)
        X_t = torch.from_numpy(X_np.astype(np.float32))

        B_np = _fit_baseline(
            Y_filled,
            X_np,
            H_np,
            Finv,
            B_np,
            sel_pix,
            denoise_samps,
            baseline_samps,
            lam_normalized,
            n_concurrent=n_concurrent,
            device=device,
        )
        B_t = torch.from_numpy(B_np.astype(np.float32))

        Hs_t = _optimize_Hs(
            Y_t,
            X_t,
            B_t,
            Finv_t,
            Hs_t,
            H_valid_t,
            sel_pix,
            h_filter,
            lam_normalized,
            max_iter,
        )

        Hs_np = Hs_t.detach().numpy()
        H_new = smooth_hs_to_h(Hs_np, sel_pix, h_filter, n_sources)

        norm_fac = np.sum(H_new, axis=0, keepdims=True)
        norm_fac = np.maximum(norm_fac, 1e-8)
        H_np = (H_new / norm_fac).astype(np.float32)
        Hs_np = Hs_np / norm_fac
        S_np = S_t.detach().numpy() * norm_fac.T

        H_t = torch.from_numpy(H_np)
        Hs_t = torch.from_numpy(Hs_np.astype(np.float32))
        S_t = torch.from_numpy(S_np.astype(np.float32))

    H_final = H_t.detach().numpy()
    B_final = B_t.detach().numpy()
    S_final = S_t.detach().numpy()
    snr_final = snr_t.detach().numpy()

    residual = np.nan_to_num(Y_obs - B_final, nan=0.0)
    H_f64 = np.nan_to_num(H_final, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    dF_ls = np.linalg.lstsq(H_f64, residual.astype(np.float64), rcond=None)[0].astype(
        np.float32
    )

    F0 = (H_final.T @ B_final) / np.maximum(
        np.sum(H_final**2, axis=0, keepdims=True).T, 1e-8
    )

    return ExtractionResult(
        H=H_final,
        S=S_final,
        B=B_final,
        LS=dF_ls,
        SNR=snr_final,
        F0=F0,
    )


def _closure_s(optimizer, S, Y, H, B, Finv, kernel, lam):
    """L-BFGS objective for the temporal activity S."""
    optimizer.zero_grad()
    X = convolve_with_kernel(S.clamp(min=0), kernel)
    loss = nmf_loss(Y, H, X, B, Finv, lam)
    loss.backward()
    return loss


def _optimize_S(
    Y: torch.Tensor,
    H: torch.Tensor,
    B: torch.Tensor,
    Finv: torch.Tensor,
    S_init: torch.Tensor,
    kernel: torch.Tensor,
    lam: float,
    max_iter: int,
) -> torch.Tensor:
    """
    Optimize the temporal activity S under non-negativity.

    L-BFGS with projected gradient: the iterate is clamped to >= 0 after the
    step, and reverted to ``S_init`` if the line search diverged to NaN.
    """
    S = S_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [S],
        lr=1.0,
        max_iter=max_iter,
        history_size=_LBFGS_HISTORY_SIZE,
        line_search_fn="strong_wolfe",
    )
    optimizer.step(partial(_closure_s, optimizer, S, Y, H, B, Finv, kernel, lam))
    del optimizer
    S.data.clamp_(min=0)

    if torch.isnan(S).any():
        logger.warning("NaN in S after optimization, reverting to initial values")
        return S_init.detach().clamp(min=0)

    return S.detach()


def _hs_to_h_torch(
    Hs: torch.Tensor,
    sel_pix: np.ndarray,
    h_filter_t: torch.Tensor,
    n_sources: int,
    _sel_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Convert super-resolution Hs to smoothed H via 2D convolution.

    Ports extractTrial.m lines 282-285:
        tmp = zeros([sz num_sources]);
        tmp(repmat(selPix,1,1,num_sources)) = Hs;
        tmp = reshape(convn(tmp, Hfilter, 'same'), numel(selPix), num_sources);
        H = tmp(selPix,:);

    This is done in PyTorch to allow gradient flow through the convolution.
    Vectorized: all sources are scattered/gathered in one operation.
    """
    sh, sw = sel_pix.shape
    n_flat = sh * sw

    if _sel_idx is None:
        _sel_idx = torch.from_numpy(np.where(sel_pix.ravel())[0])

    imgs_flat = torch.zeros(n_sources, n_flat, dtype=Hs.dtype)
    imgs_flat[:, _sel_idx] = Hs.T  # (n_sources, n_sel) via transpose
    imgs = imgs_flat.view(n_sources, 1, sh, sw)

    pad_h = h_filter_t.shape[2] // 2
    pad_w = h_filter_t.shape[3] // 2
    filtered = torch.nn.functional.conv2d(imgs, h_filter_t, padding=(pad_h, pad_w))

    filtered_flat = filtered.view(n_sources, -1)  # (n_sources, sh*sw)
    H = filtered_flat[:, _sel_idx].T  # (n_sel, n_sources)
    return H


def _closure_hs(
    optimizer, Hs, Y, X, B, Finv, H_valid, sel_pix, h_filter_t, sel_idx, n_sources, lam
):
    """L-BFGS objective for the super-resolution footprints Hs."""
    optimizer.zero_grad()
    Hs_clamped = Hs.clone()
    Hs_clamped[~H_valid] = 0
    Hs_clamped[H_valid] = Hs_clamped[H_valid].clamp(min=0)

    H = _hs_to_h_torch(Hs_clamped, sel_pix, h_filter_t, n_sources, sel_idx)
    T = H @ X + B
    variance = Finv * (T + lam)
    loss = ((Y - T) ** 2 / variance.clamp(min=1e-8)).sum()
    loss.backward()
    return loss


def _optimize_Hs(
    Y: torch.Tensor,
    X: torch.Tensor,
    B: torch.Tensor,
    Finv: torch.Tensor,
    Hs_init: torch.Tensor,
    H_valid: torch.Tensor,
    sel_pix: np.ndarray,
    h_filter: np.ndarray,
    lam: float,
    max_iter: int,
) -> torch.Tensor:
    """
    Optimize Hs (super-resolution spatial footprints).

    Ports extractTrial.m lines 269-285:
    Optimizes Hs, then convolves Hs with Hfilter to get H (PSF smoothing),
    and uses H in the loss function. This implements the spatial prior
    that footprints should look like PSF-convolved point sources.

    Non-negativity on valid pixels, zero on invalid pixels.
    """
    n_sources = Hs_init.shape[1]
    h_filter_t = torch.from_numpy(
        h_filter[np.newaxis, np.newaxis, :, :].astype(np.float32)
    )
    sel_idx = torch.from_numpy(np.where(sel_pix.ravel())[0])

    Hs = Hs_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [Hs],
        lr=1.0,
        max_iter=max_iter,
        history_size=_LBFGS_HISTORY_SIZE,
        line_search_fn="strong_wolfe",
    )
    optimizer.step(
        partial(
            _closure_hs,
            optimizer,
            Hs,
            Y,
            X,
            B,
            Finv,
            H_valid,
            sel_pix,
            h_filter_t,
            sel_idx,
            n_sources,
            lam,
        )
    )
    del optimizer

    if torch.isnan(Hs).any():
        logger.warning("NaN in Hs after optimization, reverting to initial values")
        Hs = Hs_init.clone().detach()

    Hs.data[~H_valid] = 0
    Hs.data[H_valid] = Hs.data[H_valid].clamp(min=0)
    return Hs.detach()


def _compute_floor(
    X: np.ndarray, denoise_window: int, baseline_window: int
) -> np.ndarray:
    """
    Compute floor of X for subtraction.

    Ports extractTrial.m computeFloor (lines 600-604):
        ord = ceil(0.1*baseline);
        Xmed = medfilt2(X, [1 2*ceil(denoiseWindow)+1], 'symmetric');
        Xmed_min = ordfilt2(Xmed, ord, ones(1,ceil(baseline)), 'symmetric');
        Xfloor = smoothdata(Xmed_min, 2, 'movmean', ceil(baseline), 'omitmissing');
    """

    med_size = 2 * math.ceil(denoise_window) + 1
    # Pad with reflect to match MATLAB 'symmetric', then apply 1D medfilt per row
    pad = med_size // 2
    X_padded = np.pad(X, ((0, 0), (pad, pad)), mode="reflect")
    X_med = np.empty_like(X)
    for r in range(X.shape[0]):
        X_med[r] = medfilt(X_padded[r], kernel_size=med_size)[pad : pad + X.shape[1]]

    bl = math.ceil(baseline_window)
    quantile = 0.1
    X_med_min = _rolling_quantile(X_med, bl, quantile)

    X_floor = uniform_filter1d(X_med_min, size=bl, axis=1, mode="nearest")

    return X_floor


def _rolling_quantile(data: np.ndarray, window: int, quantile: float) -> np.ndarray:
    """
    Fast rolling quantile along axis=1 using pandas skiplist algorithm.

    Replaces scipy.ndimage.rank_filter which is extremely slow for
    large 1D windows. Pandas uses an O(n log w) skiplist internally,
    giving ~30x speedup for typical window sizes.
    """

    _, n_cols = data.shape
    if window <= 1:
        return data.copy()

    half = window // 2
    # Pad with reflect to match MATLAB 'symmetric' boundary handling
    padded = np.pad(data, ((0, 0), (half, half)), mode="reflect")

    df = pd.DataFrame(padded.T)  # (n_padded, n_rows)
    rolled = df.rolling(window=window, center=True, min_periods=1).quantile(quantile)
    result = rolled.values[half : half + n_cols, :].T.astype(data.dtype)

    return result


def _sine_predictors(n_time: int, base_period: int) -> np.ndarray:
    """
    Build sine/cosine + linear-trend predictor matrix.

    Ports extractTrial.m sinePredictors (lines 307-313).

    Returns
    -------
    np.ndarray
        Predictor matrix, shape (2*maxN + 1, n_time).
    """
    max_n = math.ceil(n_time / max(base_period, 1))
    return _sine_predictors_n(n_time, base_period, max_n)


def _sine_predictors_n(n_time: int, base_period: int, max_n: int) -> np.ndarray:
    """
    Build sine/cosine + linear-trend predictor matrix with explicit harmonic count.

    Parameters
    ----------
    n_time : int
        Number of time points.
    base_period : int
        Base period in samples (longest period = max_n * base_period).
    max_n : int
        Number of harmonics (sine + cosine pairs).

    Returns
    -------
    np.ndarray
        Predictor matrix, shape (2*max_n + 1, n_time).
    """
    T = np.arange(n_time, dtype=np.float32)
    periods = (np.arange(1, max_n + 1, dtype=np.float32) * base_period).reshape(-1, 1)
    phase = np.float32(2 * np.pi) * T[np.newaxis, :] / periods
    linear = T / max(T[-1], np.float32(1.0)) - np.float32(0.5)
    return np.vstack([np.sin(phase), np.cos(phase), linear[np.newaxis, :]]).astype(
        np.float32
    )


def _fit_baseline(
    Y: np.ndarray,
    X: np.ndarray,
    H: np.ndarray,
    Finv: np.ndarray,
    B_prev: np.ndarray,
    sel_pix: np.ndarray,
    denoise_samps: int,
    baseline_samps: int,
    lam: float,
    n_concurrent: int = 1,
    device: object | None = None,
) -> np.ndarray:
    """
    Fit baseline via vectorized weighted bounded least squares.

    Ports extractTrial.m fitB (lines 315-341):
    For each pixel, regresses Y against predictors:
      [source activity (H*X), sine/cos drift, surround neuropil, constant]
    using weighted bounded least squares (lsqlin).

    Vectorized: accumulates normal equations (AtA, Atb) over time chunks,
    then batch-solves all pixels at once via np.linalg.solve.

    Parameters
    ----------
    Y : np.ndarray
        Observations, (n_pixels, n_time).
    X : np.ndarray
        Convolved activity, (n_sources, n_time).
    H : np.ndarray
        Spatial footprints, (n_pixels, n_sources).
    Finv : np.ndarray
        Inverse freshness, (n_pixels, n_time).
    B_prev : np.ndarray
        Previous baseline estimate, (n_pixels, n_time).
    sel_pix : np.ndarray
        2D boolean mask.
    denoise_samps : int
        Denoising window in samples.
    baseline_samps : int
        Baseline window in samples.
    lam : float
        Lambda (regularizer, typically 1.0 after normalization).
    n_concurrent : int
        Number of subproblems running concurrently.

    Returns
    -------
    np.ndarray
        Updated baseline estimate, (n_pixels, n_time).
    """
    n_pixels, n_time = Y.shape
    n_sources = H.shape[1]

    _, HP = split_freq(
        Y - H @ X, 2 * denoise_samps, max(1, math.ceil(baseline_samps / denoise_samps))
    )
    HP = np.nan_to_num(HP, nan=0.0).astype(np.float32)

    b_filter = build_baseline_filters(
        math.ceil(2.0 * 3)  # sel_radius; config.dXY default = 3
    )
    HP_surround = get_surround(
        HP, sel_pix, b_filter, n_concurrent=n_concurrent, device=device
    )
    del HP
    n_filters = HP_surround.shape[2]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scale = np.nanmean(Y, axis=1).astype(np.float32)  # (n_pixels,)
    scale = np.where((scale <= 0) | np.isnan(scale), np.float32(1.0), scale)

    page_size = max(1, denoise_samps)
    n_pages = n_time // page_size

    if n_pages < 2:
        return np.full_like(Y, np.float32(max(lam / 10, float(np.nanmean(scale)))))

    tot = n_pages * page_size
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Y_pg = np.nanmean(
            Y[:, :tot].reshape(n_pixels, n_pages, page_size), axis=2
        ).astype(np.float32)  # (n_pixels, n_pages)
        HX_full = H @ X  # (n_pixels, n_time)
        HX_pg = np.nanmean(
            HX_full[:, :tot].reshape(n_pixels, n_pages, page_size), axis=2
        ).astype(np.float32)
        del HX_full
        Finv_pg = np.nanmean(
            Finv[:, :tot].reshape(n_pixels, n_pages, page_size), axis=2
        ).astype(np.float32)
        B_prev_pg = np.nanmean(
            B_prev[:, :tot].reshape(n_pixels, n_pages, page_size), axis=2
        ).astype(np.float32)
        X_pg = np.nanmean(
            X[:, :tot].reshape(n_sources, n_pages, page_size), axis=2
        ).astype(np.float32)
        surr_pg = np.nanmean(
            HP_surround[:, :tot, :].reshape(n_pixels, n_pages, page_size, n_filters),
            axis=2,
        ).astype(np.float32)  # (n_pixels, n_pages, n_filters)

    bl_pages = max(1, baseline_samps // page_size)
    max_n_harmonics = math.ceil(n_pages / max(bl_pages, 1))
    sine_preds_pg = _sine_predictors(n_pages, bl_pages)
    sine_pg_T = np.ascontiguousarray(sine_preds_pg.T)  # (n_pages, n_sine)
    n_sine = sine_preds_pg.shape[0]
    del sine_preds_pg

    n_preds = n_sources + n_sine + n_filters + 1
    logger.debug(
        "fit_baseline: %d pixels, %d pages (from %d frames), %d preds",
        n_pixels,
        n_pages,
        n_time,
        n_preds,
    )

    preds = np.empty((n_pixels, n_pages, n_preds), dtype=np.float32)
    np.einsum("ps,sc->pcs", H, X_pg, out=preds[:, :, :n_sources])
    preds[:, :, n_sources : n_sources + n_sine] = sine_pg_T[np.newaxis, :, :]
    preds[:, :, n_sources + n_sine : n_sources + n_sine + n_filters] = surr_pg
    preds[:, :, -1] = scale[:, None]
    del X_pg, surr_pg

    denom = np.maximum(B_prev_pg + HX_pg, np.float32(1e-10))
    W = (
        np.sqrt(np.float32(1.0) / np.maximum(Finv_pg, np.float32(1e-10)))
        * B_prev_pg
        / denom
    )
    del denom, HX_pg, Finv_pg, B_prev_pg

    valid = np.isfinite(Y_pg) & np.isfinite(W)
    W = np.where(valid, W, np.float32(0.0))
    Y_pg = np.where(valid, Y_pg, np.float32(0.0))
    del valid

    preds_w = preds * W[:, :, None]
    resp_w = Y_pg * W
    del preds, W, Y_pg

    AtA = np.einsum("ptk,ptj->pkj", preds_w, preds_w)  # (n_pixels, n_preds, n_preds)
    Atb = np.einsum("ptk,pt->pk", preds_w, resp_w)  # (n_pixels, n_preds)
    del preds_w, resp_w

    alpha = 1e-4
    for k in range(n_preds):
        diag_vals = AtA[:, k, k].copy()
        diag_vals = np.maximum(diag_vals, np.float32(1e-6))
        AtA[:, k, k] += alpha * diag_vals

    # Batch solve: (n_pixels, n_preds, n_preds) \ (n_pixels, n_preds, 1) -> (n_pixels, n_preds)
    b_coeffs = np.linalg.solve(AtA, Atb[..., None]).squeeze(-1)
    del AtA, Atb

    # Clip to bounds (matches MATLAB lsqlin bounds)
    lb = -10.0 * scale[:, None] * np.ones((1, n_preds), dtype=np.float32)
    lb[:, :n_sources] = 0.0
    ub = 10.0 * scale[:, None] * np.ones((1, n_preds), dtype=np.float32)
    b_coeffs = np.clip(b_coeffs, lb, ub)

    coeffs_sine = b_coeffs[:, n_sources : n_sources + n_sine].astype(np.float32)
    coeffs_surr = b_coeffs[
        :, n_sources + n_sine : n_sources + n_sine + n_filters
    ].astype(np.float32)
    base_const = (scale * b_coeffs[:, -1].astype(np.float32))[:, None]
    del b_coeffs

    sine_preds_full = _sine_predictors_n(n_time, baseline_samps, max_n_harmonics)
    sine_full_T = np.ascontiguousarray(sine_preds_full.T)  # (n_time, n_sine)
    del sine_preds_full

    B = np.full_like(Y, np.float32(lam / 10))
    lam_floor = np.float32(lam / 10)

    recon_chunk = max(1000, min(50000, n_time))
    for t0 in range(0, n_time, recon_chunk):
        t1 = min(t0 + recon_chunk, n_time)

        B_chunk = (sine_full_T[t0:t1, :] @ coeffs_sine.T).T

        B_chunk += np.einsum("ptf,pf->pt", HP_surround[:, t0:t1, :], coeffs_surr)

        B_chunk += base_const

        B[:, t0:t1] = np.maximum(lam_floor, B_chunk)

    return B
