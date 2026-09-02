"""Synthetic subcell data for the tests: a finished store, and a raw movie to run."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from subcell.config import ExtractionConfig
from subcell.io.zarr_store import AlignmentData, ExperimentStore

H, W = 40, 48
NUM_CHANNELS = 2
DS_FACTOR = 4
FULL_HZ = 400.0
SOURCE_ROWS = np.array([10.0, 20.0, 30.0])
SOURCE_COLS = np.array([12.0, 24.0, 36.0])
SEL_RADIUS = 3


def _disk_mask(rows, cols, radius, shape) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    mask = np.zeros(shape, bool)
    for r, c in zip(rows, cols):
        mask |= (yy - r) ** 2 + (xx - c) ** 2 <= radius**2
    return mask


def _events(n_time: int, rng, rate_hz: float, hz: float, tau_s: float) -> tuple[np.ndarray, np.ndarray]:
    spikes = (rng.random(n_time) < rate_hz / hz).astype(np.float32) * rng.uniform(0.5, 2.0, n_time).astype(np.float32)
    kernel = np.exp(-np.arange(int(6 * tau_s * hz) + 1) / (tau_s * hz)).astype(np.float32)
    denoised = np.convolve(spikes, kernel)[:n_time].astype(np.float32)
    return spikes, denoised


def make_store(root: Path, n_trials: int = 2, seed: int = 0) -> Path:
    """
    Write an ``experiment.zarr`` the way the pipeline would, with registration
    products for every trial and extraction for all of them. Trial 2 is shorter
    and has no full-rate movie, to exercise padding and the per-trial checks.
    """
    rng = np.random.default_rng(seed)
    store = ExperimentStore(root / "experiment.zarr")
    frame_time = 1.0 / FULL_HZ
    align_hz = FULL_HZ / DS_FACTOR
    sel_pix = _disk_mask(SOURCE_ROWS, SOURCE_COLS, SEL_RADIUS, (H, W))
    ys, xs = np.nonzero(sel_pix)
    n_sel = int(sel_pix.sum())
    n_src = len(SOURCE_ROWS)
    config = ExtractionConfig()

    mean_all = []
    act_all = []
    for pos in range(n_trials):
        trial = pos + 1
        n_ds = 30 - 5 * pos
        n_raw = n_ds * DS_FACTOR

        ds = rng.normal(100, 5, (H, W, n_ds * NUM_CHANNELS)).astype(np.float32)
        ds[:2, :, :] = np.nan  # motion border
        ds[:, -3:, :] = np.nan
        arr = store.create_registered_ds(trial, ds.shape, NUM_CHANNELS)
        arr[:] = ds
        if pos == 0:
            raw = rng.normal(100, 8, (H, W, n_raw * NUM_CHANNELS)).astype(np.float32)
            raw[:2, :, :] = np.nan
            arr = store.create_registered_raw(trial, raw.shape, NUM_CHANNELS)
            arr[:] = raw

        motion_ds = np.cumsum(rng.normal(0, 0.3, (2, n_ds)), axis=1)
        motion = np.repeat(motion_ds, DS_FACTOR, axis=1)[:, :n_raw]
        rec_err = rng.normal(-0.02, 0.005, n_ds)
        rec_err[8:11] = 0.1  # error rises toward zero in a motion burst; the censor flags it
        store.save_alignment_data(
            trial,
            AlignmentData(
                num_channels=NUM_CHANNELS,
                frame_time=frame_time,
                align_hz=align_hz,
                motion_r=motion[0],
                motion_c=motion[1],
                motion_ds_r=motion_ds[0],
                motion_ds_c=motion_ds[1],
                rec_neg_err=rec_err,
            ),
        )
        for ch in range(NUM_CHANNELS):
            store.save_avg_image(trial, ch + 1, rng.integers(0, 255, (H, W)).astype(np.uint8))
        mean = rng.normal(100, 3, (H, W, NUM_CHANNELS)).astype(np.float32)
        act = np.zeros((H, W), np.float32)
        act[sel_pix] = 5.0
        store.save_mean_image(trial, mean)
        store.save_activity_image(trial, act)
        mean_all.append(mean)
        act_all.append(act)

        footprints = np.zeros((n_sel, n_src), np.float32)
        for k, (r, c) in enumerate(zip(SOURCE_ROWS, SOURCE_COLS)):
            footprints[:, k] = np.exp(-((ys - r) ** 2 + (xs - c) ** 2) / (2 * 1.3**2)) * (
                (ys - r) ** 2 + (xs - c) ** 2 <= SEL_RADIUS**2
            )
        events = np.zeros((n_src, n_raw), np.float32)
        denoised = np.zeros((n_src, n_raw), np.float32)
        for k in range(n_src):
            events[k], denoised[k] = _events(n_raw, rng, 4.0, FULL_HZ, config.tau_s)
        ls = denoised + rng.normal(0, 0.05, denoised.shape).astype(np.float32)
        f0 = np.full((n_src, n_raw), 50.0, np.float32)
        snr = np.array([8.0, 3.5, 1.2], np.float32)
        discard = np.zeros(n_raw, bool)
        discard[32:44] = True
        store.save_extraction_results(
            trial,
            footprints=footprints,
            events=events,
            denoised=denoised,
            ls=ls,
            f0=f0,
            snr=snr,
            discard_frames=discard,
        )

    store.save_summary(
        mean_image=np.mean(mean_all, axis=0),
        activity_image=np.mean(act_all, axis=0),
        source_locations_r=SOURCE_ROWS,
        source_locations_c=SOURCE_COLS,
        sel_pix=sel_pix,
        valid_trials=np.arange(n_trials),
        trial_offsets=np.zeros((2, n_trials)),
        trial_corr_coeffs=np.full(n_trials, 0.97),
        params=config.model_dump(),
    )
    return root / "experiment.zarr"


def make_movie(
    n_trials: int = 2,
    n_frames: int = 480,
    size: int = 48,
    fs: float = 400.0,
    n_sources: int = 6,
    seed: int = 1,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    """
    A raw single-plane, single-channel movie with bright synaptic transients on
    a dim background and a slow rigid jitter, as ``(T, C, Z, Y, X)`` int16.

    Returns the movie, the frames per trial, and the ``(n_sources, 2)`` source
    centers as (row, col).
    """
    rng = np.random.default_rng(seed)
    tau_frames = 0.03 * fs
    kernel = np.exp(-np.arange(int(8 * tau_frames)) / tau_frames).astype(np.float32)
    yy, xx = np.mgrid[:size, :size]
    margin = 8
    centers = np.column_stack(
        [rng.integers(margin, size - margin, n_sources), rng.integers(margin, size - margin, n_sources)]
    )
    blobs = np.stack(
        [np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * 1.3**2)).astype(np.float32) for r, c in centers]
    )
    dendrite = 20.0 * np.exp(-((yy - size / 2) ** 2) / (2 * 3.0**2)).astype(np.float32)

    frames_per_trial = [n_frames] * n_trials
    total = n_frames * n_trials
    movie = np.empty((total, 1, 1, size, size), np.int16)
    for k in range(n_sources):
        spikes = (rng.random(total) < 3.0 / fs).astype(np.float32) * rng.uniform(60, 120, total)
        trace = np.convolve(spikes, kernel)[:total]
        if k == 0:
            signal = trace[:, None, None] * blobs[k][None]
        else:
            signal += trace[:, None, None] * blobs[k][None]
    jitter = np.round(np.cumsum(rng.normal(0, 0.35, (total, 2)), axis=0)).astype(int)
    jitter = np.clip(jitter, -2, 2)
    for t in range(total):
        frame = 30.0 + dendrite + signal[t] + rng.normal(0, 4.0, (size, size))
        frame = np.roll(frame, (int(jitter[t, 0]), int(jitter[t, 1])), axis=(0, 1))
        movie[t, 0, 0] = np.clip(frame, 0, 4000).astype(np.int16)
    return movie, frames_per_trial, centers
