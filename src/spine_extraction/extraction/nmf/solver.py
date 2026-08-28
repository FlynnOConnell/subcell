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

import numpy as np
import torch

from spine_extraction.config import ExtractionConfig
from spine_extraction.extraction.nmf.objective import (
    build_decay_kernel,
    convolve_with_kernel,
    nmf_loss,
    compute_snr,
)
from spine_extraction.extraction.nmf.spatial import (
    initialize_footprints,
    smooth_hs_to_h,
    build_gaussian_filter_kernel,
)
from spine_extraction.extraction.nmf.baseline import (
    split_freq,
    build_baseline_filters,
    get_surround,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of NMF extraction for a subproblem."""

    H: np.ndarray       # (n_pixels, n_sources) spatial footprints
    S: np.ndarray       # (n_sources, n_time) inferred events
    B: np.ndarray       # (n_pixels, n_time) baseline
    LS: np.ndarray      # (n_sources, n_time) least-squares estimate
    SNR: np.ndarray     # (n_sources,) signal-to-noise ratio
    F0: np.ndarray      # (n_sources, n_time) baseline fluorescence per source


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
    """Solve a single NMF subproblem.

    Ports extractSources from extractTrial.m lines 137-304.

    Args:
        Y_obs: Observations, (n_pixels, n_time).
        Finv: Inverse freshness, (n_pixels, n_time).
        source_rows: Row coordinates of sources in this subproblem.
        source_cols: Column coordinates of sources in this subproblem.
        sel_pix: 2D boolean mask for this subproblem.
        config: Extraction configuration.
        analyze_hz: Analysis framerate.
        n_concurrent: Number of subproblems running concurrently.

    Returns:
        ExtractionResult with all extracted signals.
    """
    n_pixels, n_time = Y_obs.shape
    n_sources = len(source_rows)
    tau_full = config.tau_s * analyze_hz
    denoise_samps = int(config.denoise_window_s * analyze_hz)
    baseline_samps = int(config.baseline_window_glu_s * analyze_hz)

    # Estimate lambda if needed
    lam = config.lambda_param
    if lam is None:
        # Some pixels may be entirely NaN (edge/motion), producing expected dof<=0 warnings
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

    # Normalize data by lambda (keep float32 to avoid doubling memory)
    Y_obs = Y_obs / np.float32(lam)
    lam_normalized = 1.0

    # Initialize footprints
    H_np, Hs_np, H_valid = initialize_footprints(
        source_rows, source_cols, sel_pix, config.sigma_px
    )

    # Initialize baseline (cast back to float32 — split_freq upcasts via interp1d)
    B_np, _ = split_freq(Y_obs, denoise_samps, max(1, baseline_samps // denoise_samps))
    B_np = np.maximum(B_np, np.float32(lam_normalized / 10)).astype(np.float32)

    # Replace NaN with baseline estimate for initialization (NaN frames get censored later)
    Y_filled = Y_obs.copy()  # float32
    nan_mask = np.isnan(Y_filled)
    if np.any(nan_mask):
        Y_filled[nan_mask] = B_np[nan_mask]

    # Initialize S from least-squares (use float64 for SVD stability)
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
    S_init = np.maximum(0, dF_ls - np.concatenate([np.zeros((n_sources, 1)), dF_ls[:, :-1]], axis=1))

    # Build kernel and filter
    kernel = build_decay_kernel(tau_full)
    h_filter = build_gaussian_filter_kernel(config.sigma_px)

    # Resolve device for GPU-accelerated operations (get_surround FFT)
    if device is None:
        from spine_extraction._utils.torch_helpers import get_device
        device = get_device(config.device)

    # Move to PyTorch (CPU)
    Y_t = torch.from_numpy(Y_filled)
    Finv_t = torch.from_numpy(Finv.astype(np.float32))
    H_t = torch.from_numpy(H_np.astype(np.float32))
    B_t = torch.from_numpy(B_np.astype(np.float32))
    S_t = torch.from_numpy(S_init.astype(np.float32))
    Hs_t = torch.from_numpy(Hs_np.astype(np.float32))
    H_valid_t = torch.from_numpy(H_valid)

    # Optimization loop
    for outer_loop in range(config.nmf_iter):
        max_iter = 10 * (outer_loop + 1)

        # SOLVE FOR S (temporal activity)
        S_t = _optimize_S(Y_t, H_t, B_t, Finv_t, S_t, kernel, lam_normalized, max_iter)

        # Update X = conv(S, kernel)
        X_t = convolve_with_kernel(S_t, kernel)

        # Compute SNR
        Y_orig_t = torch.from_numpy(Y_obs.astype(np.float32))
        snr_t, X_noise_t = compute_snr(Y_orig_t, H_t, X_t, B_t, Finv_t, tau_full)
        del Y_orig_t

        if outer_loop == config.nmf_iter - 1:
            break  # Stop after fitting S on last loop

        # Subtract floor and noise level from X (MATLAB extractTrial.m line 261)
        X_np = X_t.detach().numpy()
        X_floor = _compute_floor(X_np, denoise_samps, baseline_samps)
        X_noise_np = X_noise_t.detach().numpy()[:, None]  # (n_sources, 1)
        X_np = np.maximum(0, X_np - X_floor - X_noise_np)
        X_t = torch.from_numpy(X_np.astype(np.float32))

        # SOLVE FOR B (baseline)
        B_np = _fit_baseline(
            Y_filled, X_np, H_np,
            Finv, B_np, sel_pix,
            denoise_samps, baseline_samps, lam_normalized,
            n_concurrent=n_concurrent, device=device,
        )
        B_t = torch.from_numpy(B_np.astype(np.float32))

        # SOLVE FOR Hs (spatial footprints)
        Hs_t = _optimize_Hs(
            Y_t, X_t, B_t, Finv_t, Hs_t, H_valid_t,
            sel_pix, h_filter, lam_normalized, max_iter
        )

        # Update H from Hs
        Hs_np = Hs_t.detach().numpy()
        H_new = smooth_hs_to_h(Hs_np, sel_pix, h_filter, n_sources)

        # Normalize H and S
        norm_fac = np.sum(H_new, axis=0, keepdims=True)
        norm_fac = np.maximum(norm_fac, 1e-8)
        H_np = (H_new / norm_fac).astype(np.float32)
        Hs_np = Hs_np / norm_fac
        S_np = S_t.detach().numpy() * norm_fac.T

        H_t = torch.from_numpy(H_np)
        Hs_t = torch.from_numpy(Hs_np.astype(np.float32))
        S_t = torch.from_numpy(S_np.astype(np.float32))

    # Final outputs
    H_final = H_t.detach().numpy()
    B_final = B_t.detach().numpy()
    S_final = S_t.detach().numpy()
    snr_final = snr_t.detach().numpy()

    residual = np.nan_to_num(Y_obs - B_final, nan=0.0)
    H_f64 = np.nan_to_num(H_final, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    dF_ls = np.linalg.lstsq(H_f64, residual.astype(np.float64), rcond=None)[0].astype(np.float32)

    # Compute F0 = (H' @ B) / sum(H^2)
    F0 = (H_final.T @ B_final) / np.maximum(np.sum(H_final**2, axis=0, keepdims=True).T, 1e-8)

    return ExtractionResult(
        H=H_final,
        S=S_final,
        B=B_final,
        LS=dF_ls,
        SNR=snr_final,
        F0=F0,
    )


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
    """Optimize S (temporal activity) with non-negativity constraints.

    Uses L-BFGS with projected gradient (clamp to >= 0 after each step).
    """
    S = S_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [S], lr=1.0, max_iter=max_iter, history_size=20,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        X = convolve_with_kernel(S.clamp(min=0), kernel)
        loss = nmf_loss(Y, H, X, B, Finv, lam)
        loss.backward()
        return loss

    optimizer.step(closure)
    del optimizer  # free L-BFGS history buffers
    S.data.clamp_(min=0)

    # Guard against L-BFGS divergence
    if torch.isnan(S).any():
        logger.warning("NaN detected in S after optimization, reverting to initial values")
        return S_init.detach().clamp(min=0)

    return S.detach()


def _hs_to_h_torch(
    Hs: torch.Tensor,
    sel_pix: np.ndarray,
    h_filter_t: torch.Tensor,
    n_sources: int,
    _sel_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert super-resolution Hs to smoothed H via 2D convolution.

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

    # Pre-computed index of True pixels for scatter/gather
    if _sel_idx is None:
        _sel_idx = torch.from_numpy(np.where(sel_pix.ravel())[0])

    # Scatter: place all sources into 2D images at once
    # imgs shape: (n_sources, 1, sh*sw) — flat, then reshape
    imgs_flat = torch.zeros(n_sources, n_flat, dtype=Hs.dtype)
    imgs_flat[:, _sel_idx] = Hs.T  # (n_sources, n_sel) via transpose
    imgs = imgs_flat.view(n_sources, 1, sh, sw)

    # Convolve with Gaussian filter (batched over sources)
    pad_h = h_filter_t.shape[2] // 2
    pad_w = h_filter_t.shape[3] // 2
    filtered = torch.nn.functional.conv2d(imgs, h_filter_t, padding=(pad_h, pad_w))

    # Gather: extract selected pixels for all sources at once
    filtered_flat = filtered.view(n_sources, -1)  # (n_sources, sh*sw)
    H = filtered_flat[:, _sel_idx].T  # (n_sel, n_sources)
    return H


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
    """Optimize Hs (super-resolution spatial footprints).

    Ports extractTrial.m lines 269-285:
    Optimizes Hs, then convolves Hs with Hfilter to get H (PSF smoothing),
    and uses H in the loss function. This implements the spatial prior
    that footprints should look like PSF-convolved point sources.

    Non-negativity on valid pixels, zero on invalid pixels.
    """
    n_sources = Hs_init.shape[1]

    # Prepare Gaussian filter for torch conv2d: shape (1, 1, kh, kw)
    h_filter_t = torch.from_numpy(
        h_filter[np.newaxis, np.newaxis, :, :].astype(np.float32)
    )

    # Pre-compute selected pixel indices once (used in every closure call)
    sel_idx = torch.from_numpy(np.where(sel_pix.ravel())[0])

    Hs = Hs_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [Hs], lr=1.0, max_iter=max_iter, history_size=20,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        # Apply validity mask
        Hs_clamped = Hs.clone()
        Hs_clamped[~H_valid] = 0
        Hs_clamped[H_valid] = Hs_clamped[H_valid].clamp(min=0)

        # Convert Hs -> H via convolution (matching MATLAB)
        H = _hs_to_h_torch(Hs_clamped, sel_pix, h_filter_t, n_sources, sel_idx)

        T = H @ X + B
        residual = Y - T
        variance = Finv * (T + lam)
        loss = (residual**2 / variance.clamp(min=1e-8)).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    del optimizer  # free L-BFGS history buffers

    # Guard against L-BFGS divergence
    if torch.isnan(Hs).any():
        logger.warning("NaN detected in Hs after optimization, reverting to initial values")
        Hs = Hs_init.clone().detach()

    # Project
    Hs.data[~H_valid] = 0
    Hs.data[H_valid] = Hs.data[H_valid].clamp(min=0)

    return Hs.detach()


def _compute_floor(X: np.ndarray, denoise_window: int, baseline_window: int) -> np.ndarray:
    """Compute floor of X for subtraction.

    Ports extractTrial.m computeFloor (lines 600-604):
        ord = ceil(0.1*baseline);
        Xmed = medfilt2(X, [1 2*ceil(denoiseWindow)+1], 'symmetric');
        Xmed_min = ordfilt2(Xmed, ord, ones(1,ceil(baseline)), 'symmetric');
        Xfloor = smoothdata(Xmed_min, 2, 'movmean', ceil(baseline), 'omitmissing');
    """
    from scipy.signal import medfilt
    from scipy.ndimage import uniform_filter1d

    # Step 1: 1D temporal median filter
    # MATLAB: medfilt2(X, [1, 2*ceil(denoiseWindow)+1], 'symmetric')
    med_size = 2 * math.ceil(denoise_window) + 1
    # Pad with reflect to match MATLAB 'symmetric', then apply 1D medfilt per row
    pad = med_size // 2
    X_padded = np.pad(X, ((0, 0), (pad, pad)), mode="reflect")
    X_med = np.empty_like(X)
    for r in range(X.shape[0]):
        X_med[r] = medfilt(X_padded[r], kernel_size=med_size)[pad:pad + X.shape[1]]

    # Step 2: Order statistic filter (10th percentile in local window)
    # MATLAB: ordfilt2(Xmed, ord, ones(1, ceil(baseline)), 'symmetric')
    # Use strided quantile — much faster than rank_filter for large windows.
    bl = math.ceil(baseline_window)
    quantile = 0.1
    X_med_min = _rolling_quantile(X_med, bl, quantile)

    # Step 3: Smooth with moving mean
    # MATLAB: smoothdata(Xmed_min, 2, 'movmean', ceil(baseline), 'omitmissing')
    X_floor = uniform_filter1d(X_med_min, size=bl, axis=1, mode="nearest")

    return X_floor


def _rolling_quantile(
    data: np.ndarray, window: int, quantile: float
) -> np.ndarray:
    """Fast rolling quantile along axis=1 using pandas skiplist algorithm.

    Replaces scipy.ndimage.rank_filter which is extremely slow for
    large 1D windows. Pandas uses an O(n log w) skiplist internally,
    giving ~30x speedup for typical window sizes.
    """
    import pandas as pd

    n_rows, n_cols = data.shape
    if window <= 1:
        return data.copy()

    half = window // 2
    # Pad with reflect to match MATLAB 'symmetric' boundary handling
    padded = np.pad(data, ((0, 0), (half, half)), mode="reflect")

    # pandas rolling quantile: operates on columns, so transpose
    df = pd.DataFrame(padded.T)  # (n_padded, n_rows)
    rolled = df.rolling(window=window, center=True, min_periods=1).quantile(quantile)
    result = rolled.values[half:half + n_cols, :].T.astype(data.dtype)

    return result


def _sine_predictors(n_time: int, base_period: int) -> np.ndarray:
    """Build sine/cosine + linear-trend predictor matrix.

    Ports extractTrial.m sinePredictors (lines 307-313).

    Returns:
        Predictor matrix, shape (2*maxN + 1, n_time).
    """
    max_n = math.ceil(n_time / max(base_period, 1))
    return _sine_predictors_n(n_time, base_period, max_n)


def _sine_predictors_n(n_time: int, base_period: int, max_n: int) -> np.ndarray:
    """Build sine/cosine + linear-trend predictor matrix with explicit harmonic count.

    Args:
        n_time: Number of time points.
        base_period: Base period in samples (longest period = max_n * base_period).
        max_n: Number of harmonics (sine + cosine pairs).

    Returns:
        Predictor matrix, shape (2*max_n + 1, n_time).
    """
    T = np.arange(n_time, dtype=np.float32)
    periods = (np.arange(1, max_n + 1, dtype=np.float32) * base_period).reshape(-1, 1)
    phase = np.float32(2 * np.pi) * T[np.newaxis, :] / periods
    linear = T / max(T[-1], np.float32(1.0)) - np.float32(0.5)
    return np.vstack([np.sin(phase), np.cos(phase), linear[np.newaxis, :]]).astype(np.float32)


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
    """Fit baseline via vectorized weighted bounded least squares.

    Ports extractTrial.m fitB (lines 315-341):
    For each pixel, regresses Y against predictors:
      [source activity (H*X), sine/cos drift, surround neuropil, constant]
    using weighted bounded least squares (lsqlin).

    Vectorized: accumulates normal equations (AtA, Atb) over time chunks,
    then batch-solves all pixels at once via np.linalg.solve.

    Args:
        Y: Observations, (n_pixels, n_time).
        X: Convolved activity, (n_sources, n_time).
        H: Spatial footprints, (n_pixels, n_sources).
        Finv: Inverse freshness, (n_pixels, n_time).
        B_prev: Previous baseline estimate, (n_pixels, n_time).
        sel_pix: 2D boolean mask.
        denoise_samps: Denoising window in samples.
        baseline_samps: Baseline window in samples.
        lam: Lambda (regularizer, typically 1.0 after normalization).
        n_concurrent: Number of subproblems running concurrently.

    Returns:
        Updated baseline estimate, (n_pixels, n_time).
    """
    n_pixels, n_time = Y.shape
    n_sources = H.shape[1]

    # Compute high-pass of residual
    _, HP = split_freq(Y - H @ X, 2 * denoise_samps,
                       max(1, math.ceil(baseline_samps / denoise_samps)))
    HP = np.nan_to_num(HP, nan=0.0).astype(np.float32)

    # Compute surround signals
    b_filter = build_baseline_filters(
        math.ceil(2.0 * 3)  # sel_radius; config.dXY default = 3
    )
    HP_surround = get_surround(HP, sel_pix, b_filter, n_concurrent=n_concurrent,
                               device=device)
    del HP
    n_filters = HP_surround.shape[2]

    # Per-pixel scale factor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scale = np.nanmean(Y, axis=1).astype(np.float32)  # (n_pixels,)
    scale = np.where((scale <= 0) | np.isnan(scale), np.float32(1.0), scale)

    # --- Downsample to page rate for regression ---
    # All baseline predictors are low-frequency (periods >= baseline_samps).
    # Fitting at page rate (one sample per denoise_samps frames) is equivalent
    # and avoids processing 168k+ timepoints with 200+ predictors.
    page_size = max(1, denoise_samps)
    n_pages = n_time // page_size

    if n_pages < 2:
        # Degenerate case: return constant baseline
        return np.full_like(Y, np.float32(max(lam / 10, float(np.nanmean(scale)))))

    # Page-average all time series (handles NaN via nanmean)
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

    # Build sine predictors at page rate
    bl_pages = max(1, baseline_samps // page_size)
    max_n_harmonics = math.ceil(n_pages / max(bl_pages, 1))
    sine_preds_pg = _sine_predictors(n_pages, bl_pages)
    sine_pg_T = np.ascontiguousarray(sine_preds_pg.T)  # (n_pages, n_sine)
    n_sine = sine_preds_pg.shape[0]
    del sine_preds_pg

    n_preds = n_sources + n_sine + n_filters + 1
    logger.debug(
        "fit_baseline: %d pixels, %d pages (from %d frames), %d preds",
        n_pixels, n_pages, n_time, n_preds,
    )

    # --- Build predictor matrix at page rate ---
    # (n_pixels, n_pages, n_preds) — fits in memory since n_pages is small
    preds = np.empty((n_pixels, n_pages, n_preds), dtype=np.float32)
    np.einsum('ps,sc->pcs', H, X_pg, out=preds[:, :, :n_sources])
    preds[:, :, n_sources:n_sources + n_sine] = sine_pg_T[np.newaxis, :, :]
    preds[:, :, n_sources + n_sine:n_sources + n_sine + n_filters] = surr_pg
    preds[:, :, -1] = scale[:, None]
    del X_pg, surr_pg

    # --- Compute weights ---
    denom = np.maximum(B_prev_pg + HX_pg, np.float32(1e-10))
    W = (np.sqrt(np.float32(1.0) / np.maximum(Finv_pg, np.float32(1e-10)))
         * B_prev_pg / denom)
    del denom, HX_pg, Finv_pg, B_prev_pg

    valid = np.isfinite(Y_pg) & np.isfinite(W)
    W = np.where(valid, W, np.float32(0.0))
    Y_pg = np.where(valid, Y_pg, np.float32(0.0))
    del valid

    # --- Weighted predictor and response matrices ---
    preds_w = preds * W[:, :, None]
    resp_w = Y_pg * W
    del preds, W, Y_pg

    # --- Ridge-regularized batch least squares ---
    # The normal equations (A^T A) are ill-conditioned (cond ~1e10-1e12).
    # Adding Tikhonov regularization alpha*diag(A^T A) reduces cond numbers
    # to ~1e3-1e4, making the batch solve stable, then we clip to bounds.
    AtA = np.einsum('ptk,ptj->pkj', preds_w, preds_w)  # (n_pixels, n_preds, n_preds)
    Atb = np.einsum('ptk,pt->pk', preds_w, resp_w)      # (n_pixels, n_preds)
    del preds_w, resp_w

    # Tikhonov regularization: add alpha * diag(AtA) to the diagonal
    alpha = 1e-4
    for k in range(n_preds):
        diag_vals = AtA[:, k, k].copy()
        # Use max of diagonal value and a small floor to avoid zero regularization
        diag_vals = np.maximum(diag_vals, np.float32(1e-6))
        AtA[:, k, k] += alpha * diag_vals

    # Batch solve: (n_pixels, n_preds, n_preds) \ (n_pixels, n_preds, 1) -> (n_pixels, n_preds)
    b_coeffs = np.linalg.solve(AtA, Atb[..., None]).squeeze(-1)
    del AtA, Atb

    # Clip to bounds (matches MATLAB lsqlin bounds)
    lb = -10.0 * scale[:, None] * np.ones((1, n_preds), dtype=np.float32)
    lb[:, :n_sources] = 0.0  # source coefficients >= 0
    ub = 10.0 * scale[:, None] * np.ones((1, n_preds), dtype=np.float32)
    b_coeffs = np.clip(b_coeffs, lb, ub)

    # --- Reconstruct baseline at full rate ---
    coeffs_sine = b_coeffs[:, n_sources:n_sources + n_sine].astype(np.float32)
    coeffs_surr = b_coeffs[:, n_sources + n_sine:n_sources + n_sine + n_filters].astype(np.float32)
    base_const = (scale * b_coeffs[:, -1].astype(np.float32))[:, None]
    del b_coeffs

    # Full-rate sine predictors for reconstruction (same harmonic count as page-rate)
    sine_preds_full = _sine_predictors_n(n_time, baseline_samps, max_n_harmonics)
    sine_full_T = np.ascontiguousarray(sine_preds_full.T)  # (n_time, n_sine)
    del sine_preds_full

    B = np.full_like(Y, np.float32(lam / 10))
    lam_floor = np.float32(lam / 10)

    # Reconstruct in chunks to bound memory
    recon_chunk = max(1000, min(50000, n_time))
    for t0 in range(0, n_time, recon_chunk):
        t1 = min(t0 + recon_chunk, n_time)

        # Sine contribution
        B_chunk = (sine_full_T[t0:t1, :] @ coeffs_sine.T).T

        # Surround contribution
        B_chunk += np.einsum('ptf,pf->pt', HP_surround[:, t0:t1, :], coeffs_surr)

        # Constant
        B_chunk += base_const

        B[:, t0:t1] = np.maximum(lam_floor, B_chunk)

    return B
