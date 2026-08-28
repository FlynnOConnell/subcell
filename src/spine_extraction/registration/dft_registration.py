"""Sub-pixel image registration via upsampled DFT cross-correlation.

Ports dftregistration_clipped.m (Guizar-Sicairos et al. 2008) with an added
clipping constraint that limits the maximum detectable shift.

The algorithm:
1. Compute cross-correlation via IFFT of buf1ft * conj(buf2ft)
2. Mask out shifts beyond the clip distance
3. Locate the peak to ~0.5 pixel accuracy
4. If usfac > 2, refine with matrix-multiply DFT around the peak

Provides both PyTorch (GPU/CPU) and pure-NumPy (CPU-optimized) implementations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RegistrationResult:
    """Result of DFT registration."""

    error: float
    diffphase: float
    row_shift: float
    col_shift: float


# ---------------------------------------------------------------------------
# Pure-NumPy implementation (avoids torch overhead on CPU for small images)
# ---------------------------------------------------------------------------

def _dftups_numpy(
    inp: np.ndarray,
    nor: int,
    noc: int,
    usfac: int,
    roff: float = 0.0,
    coff: float = 0.0,
) -> np.ndarray:
    """Upsampled DFT by matrix multiplies (NumPy version)."""
    nr, nc = inp.shape

    col_idx = np.fft.ifftshift(np.arange(nc, dtype=np.float64)) - nc // 2
    out_col = np.arange(noc, dtype=np.float64) - coff
    kernc = np.exp(
        (-1j * 2 * np.pi / (nc * usfac)) * col_idx[:, np.newaxis] * out_col[np.newaxis, :]
    )

    row_out = np.arange(nor, dtype=np.float64) - roff
    row_idx = np.fft.ifftshift(np.arange(nr, dtype=np.float64)) - nr // 2
    kernr = np.exp(
        (-1j * 2 * np.pi / (nr * usfac)) * row_out[:, np.newaxis] * row_idx[np.newaxis, :]
    )

    return kernr @ inp @ kernc


def dft_registration_clipped_numpy(
    buf1ft: np.ndarray,
    buf2ft: np.ndarray,
    usfac: int = 4,
    clip: int | tuple[int, int] = 10,
) -> RegistrationResult:
    """Pure-NumPy DFT registration (optimized for CPU, avoids torch overhead).

    Same algorithm as the torch version but ~3-5x faster on CPU for small images
    because it avoids tensor creation, dispatch, and .item() overhead.

    Args:
        buf1ft: FFT of reference image (complex ndarray).
        buf2ft: FFT of image to register (complex ndarray).
        usfac: Upsampling factor.
        clip: Maximum shift in pixels.

    Returns:
        RegistrationResult with error, diffphase, row_shift, col_shift.
    """
    if isinstance(clip, (int, float)):
        clip = (int(clip), int(clip))

    m, n = buf1ft.shape

    if usfac == 0:
        cc_max = np.sum(buf1ft * np.conj(buf2ft))
        rf_zero = np.sum(np.abs(buf1ft) ** 2)
        rg_zero = np.sum(np.abs(buf2ft) ** 2)
        error = 1.0 - (cc_max * np.conj(cc_max)).real / (rg_zero * rf_zero).real
        error = math.sqrt(abs(error))
        diffphase = math.atan2(cc_max.imag, cc_max.real)
        return RegistrationResult(error=error, diffphase=diffphase, row_shift=0.0, col_shift=0.0)

    # Partial-pixel shift (usfac >= 2)
    mlarge = m * 2
    nlarge = n * 2
    cc = np.zeros((mlarge, nlarge), dtype=buf1ft.dtype)

    r_start = m + 1 - m // 2 - 1
    r_end = r_start + m
    c_start = n + 1 - n // 2 - 1
    c_end = c_start + n
    cc[r_start:r_end, c_start:c_end] = (
        np.fft.fftshift(buf1ft) * np.conj(np.fft.fftshift(buf2ft))
    )

    cc = np.fft.ifft2(np.fft.ifftshift(cc))

    # Apply clipping mask
    # MATLAB: keep(fix(2*clip)+2 : end-fix(2*clip), :) = false
    # fix(2*clip)+2 is 1-indexed → subtract 1 for 0-indexed → r_clip + 1
    r_clip = 2 * clip[0]
    c_clip = 2 * clip[1]
    if r_clip + 1 < mlarge - r_clip:
        cc[r_clip + 1 : mlarge - r_clip, :] = 0
    if c_clip + 1 < nlarge - c_clip:
        cc[:, c_clip + 1 : nlarge - c_clip] = 0

    cc_real = cc.real
    flat_idx = np.argmax(cc_real)
    rloc = flat_idx // nlarge
    cloc = flat_idx % nlarge
    cc_max = cc[rloc, cloc]

    md2 = mlarge // 2
    nd2 = nlarge // 2
    # 0-indexed: use >= for the wrap-around boundary (MATLAB uses > with 1-indexed)
    row_shift = (rloc - mlarge if rloc >= md2 else rloc) / 2.0
    col_shift = (cloc - nlarge if cloc >= nd2 else cloc) / 2.0

    if usfac > 2:
        row_shift = round(row_shift * usfac) / usfac
        col_shift = round(col_shift * usfac) / usfac
        dft_shift = int(math.ceil(usfac * 1.5) // 2)
        nor = int(math.ceil(usfac * 1.5))
        noc = nor

        cc = np.conj(
            _dftups_numpy(
                buf2ft * np.conj(buf1ft),
                nor, noc, usfac,
                dft_shift - row_shift * usfac,
                dft_shift - col_shift * usfac,
            )
        ) / (md2 * nd2 * usfac**2)

        cc_real = cc.real
        flat_idx = np.argmax(cc_real)
        rloc = flat_idx // noc
        cloc = flat_idx % noc
        cc_max = cc[rloc, cloc]

        rg00 = _dftups_numpy(buf1ft * np.conj(buf1ft), 1, 1, usfac) / (
            md2 * nd2 * usfac**2
        )
        rf00 = _dftups_numpy(buf2ft * np.conj(buf2ft), 1, 1, usfac) / (
            md2 * nd2 * usfac**2
        )

        # 0-indexed argmax: no extra -1 (MATLAB has -1 to convert from 1-indexed)
        rloc = rloc - dft_shift
        cloc = cloc - dft_shift
        row_shift = row_shift + rloc / usfac
        col_shift = col_shift + cloc / usfac
    else:
        rg00 = np.sum(buf1ft * np.conj(buf1ft)) / (m * n)
        rf00 = np.sum(buf2ft * np.conj(buf2ft)) / (m * n)

    error_val = 1.0 - (cc_max * np.conj(cc_max)).real / (rg00 * rf00).real.squeeze()
    error = math.sqrt(abs(error_val))
    diffphase = math.atan2(cc_max.imag, cc_max.real)

    if m // 2 == 1:
        row_shift = 0.0
    if n // 2 == 1:
        col_shift = 0.0

    return RegistrationResult(
        error=error, diffphase=diffphase,
        row_shift=float(row_shift), col_shift=float(col_shift),
    )


# ---------------------------------------------------------------------------
# PyTorch implementation (for GPU acceleration)
# ---------------------------------------------------------------------------

def dft_registration_clipped(
    buf1ft: torch.Tensor,
    buf2ft: torch.Tensor,
    usfac: int = 4,
    clip: int | tuple[int, int] = 10,
) -> RegistrationResult:
    """Sub-pixel image registration via upsampled DFT cross-correlation with clipping.

    Args:
        buf1ft: FFT of reference image (complex). DC in (0,0), not fftshifted.
        buf2ft: FFT of image to register (complex). DC in (0,0), not fftshifted.
        usfac: Upsampling factor. Registration precision = 1/usfac pixel.
        clip: Maximum shift in pixels. Scalar or (row_clip, col_clip).

    Returns:
        RegistrationResult with error, diffphase, row_shift, col_shift.
    """
    if isinstance(clip, (int, float)):
        clip = (int(clip), int(clip))

    m, n = buf1ft.shape

    if usfac == 0:
        # No shift, just compute error
        cc_max = torch.sum(buf1ft * torch.conj(buf2ft))
        rf_zero = torch.sum(torch.abs(buf1ft) ** 2)
        rg_zero = torch.sum(torch.abs(buf2ft) ** 2)
        error = 1.0 - (cc_max * torch.conj(cc_max)).real / (rg_zero * rf_zero).real
        error = torch.sqrt(torch.abs(error)).item()
        diffphase = torch.atan2(cc_max.imag, cc_max.real).item()
        return RegistrationResult(error=error, diffphase=diffphase, row_shift=0.0, col_shift=0.0)

    if usfac == 1:
        # Whole-pixel shift
        cc = torch.fft.ifft2(buf1ft * torch.conj(buf2ft))
        # Apply clipping mask
        keep = torch.ones_like(cc, dtype=torch.bool)
        r_start = clip[0] // 2 + 2  # 1-indexed to 0-indexed adjustment
        r_end = m - clip[0] // 2
        c_start = clip[1] // 2 + 2
        c_end = n - clip[1] // 2
        if r_start < r_end:
            keep[r_start:r_end, :] = False
        if c_start < c_end:
            keep[:, c_start:c_end] = False
        cc[~keep] = 0

        cc_real = cc.real
        max_val, flat_idx = torch.max(cc_real.reshape(-1), 0)
        rloc = flat_idx.item() // n
        cloc = flat_idx.item() % n
        cc_max = cc[rloc, cloc]

        md2 = m // 2
        nd2 = n // 2
        row_shift = rloc - m if rloc > md2 else rloc
        col_shift = cloc - n if cloc > nd2 else cloc

        rf_zero = torch.sum(torch.abs(buf1ft) ** 2) / (m * n)
        rg_zero = torch.sum(torch.abs(buf2ft) ** 2) / (m * n)
        error = 1.0 - (cc_max * torch.conj(cc_max)).real / (rg_zero * rf_zero).real
        error = torch.sqrt(torch.abs(error)).item()
        diffphase = torch.atan2(cc_max.imag, cc_max.real).item()

        return RegistrationResult(
            error=error, diffphase=diffphase,
            row_shift=float(row_shift), col_shift=float(col_shift),
        )

    # Partial-pixel shift (usfac >= 2)
    # Step 1: 2x upsample via zero-padded FFT to get initial estimate
    mlarge = m * 2
    nlarge = n * 2
    cc = torch.zeros(mlarge, nlarge, dtype=buf1ft.dtype, device=buf1ft.device)

    # Embed in 2x array
    r_start = m + 1 - m // 2 - 1  # 0-indexed
    r_end = r_start + m
    c_start = n + 1 - n // 2 - 1
    c_end = c_start + n
    cc[r_start:r_end, c_start:c_end] = (
        torch.fft.fftshift(buf1ft) * torch.conj(torch.fft.fftshift(buf2ft))
    )

    cc = torch.fft.ifft2(torch.fft.ifftshift(cc))

    # Apply clipping mask on the 2x upsampled grid
    # MATLAB: keep(fix(2*clip)+2 : end-fix(2*clip), :) = false
    # fix(2*clip)+2 is 1-indexed → subtract 1 for 0-indexed → r_clip + 1
    keep = torch.ones(mlarge, nlarge, dtype=torch.bool, device=cc.device)
    r_clip = 2 * clip[0]
    c_clip = 2 * clip[1]
    if r_clip + 1 < mlarge - r_clip:
        keep[r_clip + 1 : mlarge - r_clip, :] = False
    if c_clip + 1 < nlarge - c_clip:
        keep[:, c_clip + 1 : nlarge - c_clip] = False
    cc[~keep] = 0

    cc_real = cc.real
    max_val, flat_idx = torch.max(cc_real.reshape(-1), 0)
    rloc = flat_idx.item() // nlarge
    cloc = flat_idx.item() % nlarge
    cc_max = cc[rloc, cloc]

    md2 = mlarge // 2
    nd2 = nlarge // 2
    # 0-indexed: use >= for the wrap-around boundary (MATLAB uses > with 1-indexed)
    row_shift = (rloc - mlarge if rloc >= md2 else rloc) / 2.0
    col_shift = (cloc - nlarge if cloc >= nd2 else cloc) / 2.0

    if usfac > 2:
        # Refine with matrix-multiply DFT
        row_shift = round(row_shift * usfac) / usfac
        col_shift = round(col_shift * usfac) / usfac
        dft_shift = int(np.ceil(usfac * 1.5) // 2)
        nor = int(np.ceil(usfac * 1.5))
        noc = nor

        cc = torch.conj(
            _dftups(
                buf2ft * torch.conj(buf1ft),
                nor, noc, usfac,
                dft_shift - row_shift * usfac,
                dft_shift - col_shift * usfac,
            )
        ) / (md2 * nd2 * usfac**2)

        cc_real = cc.real
        max_val, flat_idx = torch.max(cc_real.reshape(-1), 0)
        rloc = flat_idx.item() // noc
        cloc = flat_idx.item() % noc
        cc_max = cc[rloc, cloc]

        rg00 = _dftups(buf1ft * torch.conj(buf1ft), 1, 1, usfac) / (
            md2 * nd2 * usfac**2
        )
        rf00 = _dftups(buf2ft * torch.conj(buf2ft), 1, 1, usfac) / (
            md2 * nd2 * usfac**2
        )

        # 0-indexed argmax: no extra -1 (MATLAB has -1 to convert from 1-indexed)
        rloc = rloc - dft_shift
        cloc = cloc - dft_shift
        row_shift = row_shift + rloc / usfac
        col_shift = col_shift + cloc / usfac
    else:
        rg00 = torch.sum(buf1ft * torch.conj(buf1ft)) / (m * n)
        rf00 = torch.sum(buf2ft * torch.conj(buf2ft)) / (m * n)

    error = 1.0 - (cc_max * torch.conj(cc_max)).real / (rg00 * rf00).real.squeeze()
    error = torch.sqrt(torch.abs(error)).item()
    diffphase = torch.atan2(cc_max.imag, cc_max.real).item()

    # If only one row or column, zero that shift
    if m // 2 == 1:
        row_shift = 0.0
    if n // 2 == 1:
        col_shift = 0.0

    return RegistrationResult(
        error=error, diffphase=diffphase,
        row_shift=float(row_shift), col_shift=float(col_shift),
    )


def _dftups(
    inp: torch.Tensor,
    nor: int,
    noc: int,
    usfac: int,
    roff: float = 0.0,
    coff: float = 0.0,
) -> torch.Tensor:
    """Upsampled DFT by matrix multiplies in a small region.

    Ports the dftups inner function from dftregistration_clipped.m.
    Computes: kernr @ inp @ kernc

    Args:
        inp: Input 2D complex array.
        nor: Number of output rows.
        noc: Number of output columns.
        usfac: Upsampling factor.
        roff: Row offset in upsampled grid.
        coff: Column offset in upsampled grid.
    """
    nr, nc = inp.shape
    device = inp.device
    dtype = inp.dtype

    # Column kernel
    col_idx = torch.arange(nc, device=device, dtype=torch.float64)
    col_idx = torch.fft.ifftshift(col_idx) - nc // 2
    out_col = torch.arange(noc, device=device, dtype=torch.float64) - coff
    kernc = torch.exp(
        (-1j * 2 * np.pi / (nc * usfac)) * col_idx.unsqueeze(1) * out_col.unsqueeze(0)
    ).to(dtype)

    # Row kernel
    row_out = torch.arange(nor, device=device, dtype=torch.float64) - roff
    row_idx = torch.arange(nr, device=device, dtype=torch.float64)
    row_idx = torch.fft.ifftshift(row_idx) - nr // 2
    kernr = torch.exp(
        (-1j * 2 * np.pi / (nr * usfac)) * row_out.unsqueeze(1) * row_idx.unsqueeze(0)
    ).to(dtype)

    return kernr @ inp @ kernc
