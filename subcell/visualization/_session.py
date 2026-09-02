"""
A subcell ``experiment.zarr`` opened for viewing.

Registration and extraction each leave products in the store per trial. This
module reads the small ones (images, motion, traces) into memory and wraps the
big ones (registered movies) as lazy ``(trial, time, rows, cols)`` arrays, so
an ``NDWidget`` can put a trial slider next to the time slider and fetch one
frame at a time from disk. Trials whose registration is still in progress are
skipped, so the same loader serves a store that a pipeline run is filling in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from subcell.config import ExtractionConfig
from subcell.io.zarr_store import AlignmentData
from subcell.pipeline.runner import detect_motion_frames, upsample_discard_mask
from subcell.visualization._data_model import ExtractionData, TrialExtraction
from subcell.visualization._loaders import _prune_dead_sources

logger = logging.getLogger(__name__)

__all__ = [
    "InterleavedMovie",
    "RawTrialMovie",
    "SubcellSession",
    "TrialMovieStack",
    "TrialRecord",
    "empty_session",
    "load_session",
]


def _trial_index(key: str) -> int:
    return int(key.rsplit("_", 1)[-1])


class InterleavedMovie:
    """
    ``(T, H, W)`` view of one channel of a subcell registered array.

    The store keeps a registered trial as ``(H, W, T * C)`` with the channels
    interleaved along the last axis; this reads frame ``t`` of channel ``ch``
    at ``[:, :, t * C + ch]``. Frames come back float32 and, when ``pad_to``
    is larger than the stored canvas, NaN-padded at the bottom and right so
    every trial in a session shares one shape.
    """

    def __init__(self, arr, num_channels: int, channel: int, pad_to=None):
        self._arr = arr
        self._c = int(num_channels)
        self._ch = int(channel)
        h, w, n = (int(v) for v in arr.shape)
        self._t = n // self._c
        self._h, self._w = h, w
        self._H, self._W = (int(v) for v in (pad_to or (h, w)))
        if self._H < h or self._W < w:
            raise ValueError(f"pad_to {pad_to} is smaller than the stored canvas {(h, w)}")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self._t, self._H, self._W)

    @property
    def ndim(self) -> int:
        return 3

    @property
    def dtype(self):
        return np.dtype(np.float32)

    @property
    def n_frames(self) -> int:
        return self._t

    def __len__(self) -> int:
        return self._t

    def _pad(self, block: np.ndarray) -> np.ndarray:
        """``(..., h, w)`` -> ``(..., H, W)`` with NaN in the margin."""
        if (self._h, self._w) == (self._H, self._W):
            return block
        out = np.full(block.shape[:-2] + (self._H, self._W), np.nan, np.float32)
        out[..., : self._h, : self._w] = block
        return out

    def frames(self, start: int, stop: int, step: int = 1) -> np.ndarray:
        """``(n, H, W)`` float32 frames ``start:stop:step``."""
        start, stop, step = slice(start, stop, step).indices(self._t)
        n = len(range(start, stop, step))
        if n == 0:
            return np.zeros((0, self._H, self._W), np.float32)
        block = self._arr[:, :, start * self._c + self._ch : (stop - 1) * self._c + self._ch + 1 : step * self._c]
        block = np.asarray(block, dtype=np.float32)
        return self._pad(np.moveaxis(block, -1, 0))

    def frame(self, t: int) -> np.ndarray:
        """``(H, W)`` float32 frame ``t``; NaN past the end."""
        if not 0 <= t < self._t:
            return np.full((self._H, self._W), np.nan, np.float32)
        block = np.asarray(self._arr[:, :, t * self._c + self._ch], dtype=np.float32)
        return self._pad(block)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        t, rest = key[0], key[1:]
        if isinstance(t, (int, np.integer)):
            t = int(t)
            if t < 0:
                t += self._t
            out = self.frame(t)
        elif isinstance(t, slice):
            out = self.frames(*t.indices(self._t))
        else:
            idx = np.asarray(t, dtype=int).ravel()
            out = np.stack([self.frame(int(i)) for i in idx]) if idx.size else np.zeros((0, self._H, self._W), np.float32)
        return out[rest] if rest else out


class RawTrialMovie:
    """
    ``(T, H, W)`` view of one channel of one trial of an unregistered source.

    ``array`` is the 5D ``(T, C, Z, Y, X)`` movie the pipeline read, and
    ``t_slice`` the trial's span along T, as ``ArrayTrialSource`` records them.
    Shown beside the registered movies as the before picture.
    """

    def __init__(self, array, t_slice: slice, channel: int):
        self._arr = array
        self._t0 = int(t_slice.start or 0)
        self._t = int(t_slice.stop) - self._t0
        self._ch = int(channel)
        shape = tuple(int(s) for s in array.shape)
        self._h, self._w = shape[-2], shape[-1]

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self._t, self._h, self._w)

    @property
    def ndim(self) -> int:
        return 3

    @property
    def dtype(self):
        return np.dtype(np.float32)

    @property
    def n_frames(self) -> int:
        return self._t

    def __len__(self) -> int:
        return self._t

    def frame(self, t: int) -> np.ndarray:
        if not 0 <= t < self._t:
            return np.full((self._h, self._w), np.nan, np.float32)
        return np.asarray(self._arr[self._t0 + t, self._ch, 0, :, :], dtype=np.float32)

    def frames(self, start: int, stop: int, step: int = 1) -> np.ndarray:
        start, stop, step = slice(start, stop, step).indices(self._t)
        if stop <= start:
            return np.zeros((0, self._h, self._w), np.float32)
        block = self._arr[self._t0 + start : self._t0 + stop : step, self._ch, 0, :, :]
        return np.asarray(block, dtype=np.float32).reshape(-1, self._h, self._w)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        t, rest = key[0], key[1:]
        if isinstance(t, (int, np.integer)):
            out = self.frame(int(t) + (self._t if t < 0 else 0))
        else:
            out = self.frames(*t.indices(self._t))
        return out[rest] if rest else out


class TrialMovieStack:
    """
    ``(trial, time, H, W)`` lazy array over one movie per trial.

    ``NDWidget`` treats ``trial`` and ``time`` as slider dims, so one image
    graphic scrubs every registered trial of a session. Trials shorter than
    the longest return NaN frames past their end; a ``None`` entry (a trial
    without this movie) returns NaN frames throughout.
    """

    def __init__(self, movies: list, shape_hw: tuple[int, int], n_time: int | None = None):
        self._movies = list(movies)
        self._H, self._W = (int(v) for v in shape_hw)
        lengths = [m.n_frames for m in self._movies if m is not None]
        self._t = int(n_time if n_time is not None else (max(lengths) if lengths else 1))

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (len(self._movies), self._t, self._H, self._W)

    @property
    def ndim(self) -> int:
        return 4

    @property
    def dtype(self):
        return np.dtype(np.float32)

    @property
    def n_trials(self) -> int:
        return len(self._movies)

    def n_frames(self, trial: int) -> int:
        movie = self._movies[trial]
        return 0 if movie is None else movie.n_frames

    def has(self, trial: int) -> bool:
        return self._movies[trial] is not None

    def _blank(self, n: int | None = None) -> np.ndarray:
        shape = (self._H, self._W) if n is None else (n, self._H, self._W)
        return np.full(shape, np.nan, np.float32)

    def _read(self, trial: int, t) -> np.ndarray:
        movie = self._movies[trial]
        if isinstance(t, (int, np.integer)):
            t = int(t)
            if t < 0:
                t += self._t
            out = self._blank()
            if movie is None or not 0 <= t < movie.n_frames:
                return out
            frame = movie.frame(t)
            out[: frame.shape[0], : frame.shape[1]] = frame
            return out
        start, stop, step = t.indices(self._t)
        n = len(range(start, stop, step))
        out = self._blank(n)
        if movie is None or n == 0:
            return out
        have = movie.frames(start, min(stop, movie.n_frames), step)
        out[: len(have), : have.shape[1], : have.shape[2]] = have
        return out

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (4 - len(key))
        trial, t, rest = key[0], key[1], key[2:]
        if isinstance(trial, (int, np.integer)):
            trial = int(trial)
            if trial < 0:
                trial += len(self._movies)
            out = self._read(trial, t)
        else:
            trials = range(*trial.indices(len(self._movies))) if isinstance(trial, slice) else np.asarray(trial, int).ravel()
            frames = [self._read(int(k), t) for k in trials]
            out = np.stack(frames) if frames else self._blank(0)
        return out[(Ellipsis,) + tuple(rest)] if any(r != slice(None) for r in rest) else out


@dataclass
class TrialRecord:
    """Everything the store holds for one registered trial."""

    index: int
    num_channels: int
    align_hz: float
    full_hz: float
    n_ds_frames: int
    n_raw_frames: int
    shape: tuple[int, int]
    alignment: AlignmentData | None = None
    movies: dict[str, InterleavedMovie] = field(default_factory=dict)
    images: dict[str, np.ndarray] = field(default_factory=dict)
    extraction: TrialExtraction | None = None
    discard_ds: np.ndarray | None = None
    discard_full: np.ndarray | None = None

    @property
    def duration_s(self) -> float:
        return self.n_raw_frames / self.full_hz if self.full_hz > 0 else float(self.n_raw_frames)

    @property
    def ds_factor(self) -> int:
        if self.n_ds_frames <= 0:
            return 1
        return max(1, int(round(self.n_raw_frames / self.n_ds_frames)))

    @property
    def max_shift(self) -> float:
        if self.alignment is None or self.alignment.motion_r.size == 0:
            return 0.0
        return float(
            max(np.nanmax(np.abs(self.alignment.motion_r)), np.nanmax(np.abs(self.alignment.motion_c)))
        )

    @property
    def discard_fraction(self) -> float:
        if self.discard_full is None or self.discard_full.size == 0:
            return 0.0
        return float(self.discard_full.mean())


@dataclass
class SubcellSession:
    """A subcell store as the viewer sees it."""

    path: Path
    trials: list[TrialRecord]
    canvas: tuple[int, int]
    extraction: ExtractionData | None = None
    summary_images: dict[str, np.ndarray] = field(default_factory=dict)
    valid_trials: np.ndarray | None = None
    trial_offsets: np.ndarray | None = None
    trial_corr: np.ndarray | None = None
    params: dict = field(default_factory=dict)
    extraction_config: ExtractionConfig = field(default_factory=ExtractionConfig)

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def n_sources(self) -> int:
        return 0 if self.extraction is None else self.extraction.n_sources

    @property
    def num_channels(self) -> int:
        return max((t.num_channels for t in self.trials), default=1)

    @property
    def full_hz(self) -> float:
        rates = [t.full_hz for t in self.trials if t.full_hz > 0]
        return float(np.median(rates)) if rates else 1.0

    @property
    def max_duration_s(self) -> float:
        return max((t.duration_s for t in self.trials), default=1.0)

    @property
    def max_raw_frames(self) -> int:
        return max((t.n_raw_frames for t in self.trials), default=1)

    @property
    def trial_indices(self) -> list[int]:
        return [t.index for t in self.trials]

    def trial_position(self, trial_index: int) -> int | None:
        """Position in ``trials`` of the trial the store keys by ``trial_index``."""
        for pos, trial in enumerate(self.trials):
            if trial.index == trial_index:
                return pos
        return None

    def movie_names(self) -> list[str]:
        """Movie kinds any trial offers, in a stable order."""
        names: list[str] = []
        for trial in self.trials:
            for name in trial.movies:
                if name not in names:
                    names.append(name)
        return names

    def image_names(self) -> list[str]:
        names: list[str] = []
        for trial in self.trials:
            for name in trial.images:
                if name not in names:
                    names.append(name)
        return names

    def movie_stack(self, name: str) -> TrialMovieStack:
        """``(trial, time, H, W)`` lazy stack of one movie kind across trials."""
        movies = [trial.movies.get(name) for trial in self.trials]
        n_time = max((m.n_frames for m in movies if m is not None), default=1)
        return TrialMovieStack(movies, self.canvas, n_time=n_time)

    def image_stack(self, name: str) -> np.ndarray:
        """``(trial, H, W)`` float32 stack of one image kind, NaN where missing."""
        out = np.full((self.n_trials, *self.canvas), np.nan, np.float32)
        for k, trial in enumerate(self.trials):
            img = trial.images.get(name)
            if img is not None:
                out[k, : img.shape[0], : img.shape[1]] = img
        return out

    def movie_hz(self, name: str, trial: int) -> float:
        record = self.trials[trial]
        return record.align_hz if name.startswith("registered ds") else record.full_hz


def _read_alignment(grp) -> AlignmentData:
    return AlignmentData(
        num_channels=int(grp.attrs["num_channels"]),
        frame_time=float(grp.attrs["frame_time"]),
        align_hz=float(grp.attrs["align_hz"]),
        motion_r=np.asarray(grp["motion_r"]),
        motion_c=np.asarray(grp["motion_c"]),
        motion_ds_r=np.asarray(grp["motion_ds_r"]),
        motion_ds_c=np.asarray(grp["motion_ds_c"]),
        rec_neg_err=np.asarray(grp["rec_neg_err"]),
    )


def _read_extraction(grp) -> TrialExtraction:
    return TrialExtraction(
        footprints=np.asarray(grp["footprints"], dtype=np.float32),
        events=np.asarray(grp["events"], dtype=np.float32),
        denoised=np.asarray(grp["denoised"], dtype=np.float32),
        ls=np.asarray(grp["ls"], dtype=np.float32),
        f0=np.asarray(grp["F0"], dtype=np.float32),
        snr=np.asarray(grp["SNR"], dtype=np.float32),
        discard_frames=np.asarray(grp["discard_frames"]).astype(bool),
    )


def _read_trial(key: str, grp, config: ExtractionConfig) -> TrialRecord | None:
    """One trial's products, or None while its registration is still running."""
    if "alignment" not in grp or "registered_ds" not in grp:
        return None
    adata = _read_alignment(grp["alignment"])
    ds = grp["registered_ds"]
    num_channels = int(grp.attrs.get("num_channels", adata.num_channels)) or 1
    h, w = int(ds.shape[0]), int(ds.shape[1])
    n_ds = int(ds.shape[2]) // num_channels
    n_raw = int(adata.motion_r.shape[0])
    full_hz = 1.0 / adata.frame_time if adata.frame_time > 0 else adata.align_hz

    record = TrialRecord(
        index=_trial_index(key),
        num_channels=num_channels,
        align_hz=adata.align_hz,
        full_hz=full_hz,
        n_ds_frames=n_ds,
        n_raw_frames=n_raw,
        shape=(h, w),
        alignment=adata,
    )
    for ch in range(num_channels):
        record.movies[f"registered ds · ch{ch + 1}"] = InterleavedMovie(ds, num_channels, ch)
    if "registered_raw" in grp:
        raw = grp["registered_raw"]
        record.n_raw_frames = int(raw.shape[2]) // num_channels
        for ch in range(num_channels):
            record.movies[f"registered raw · ch{ch + 1}"] = InterleavedMovie(raw, num_channels, ch)

    if "mean_image" in grp:
        mean = np.asarray(grp["mean_image"], dtype=np.float32)
        if mean.ndim == 2:
            mean = mean[:, :, None]
        for ch in range(mean.shape[2]):
            record.images[f"mean · ch{ch + 1}"] = mean[:, :, ch]
    if "activity_image" in grp:
        record.images["activity"] = np.asarray(grp["activity_image"], dtype=np.float32)
    if "avg_images" in grp:
        avg = grp["avg_images"]
        for name in sorted(avg.keys()):
            ch = name.split("_")[0].replace("ch", "")
            record.images[f"avg 8bit · ch{ch}"] = np.asarray(avg[name], dtype=np.float32)

    if "extraction" in grp:
        try:
            record.extraction = _read_extraction(grp["extraction"])
        except KeyError:
            logger.debug("trial %s extraction incomplete", key)

    if adata.rec_neg_err.size:
        record.discard_ds = detect_motion_frames(adata, config)
        record.discard_full = upsample_discard_mask(record.discard_ds, record.n_raw_frames)
    return record


def _read_summary(root, trials: list[TrialRecord], session: SubcellSession) -> None:
    if "summary" not in root:
        return
    summ = root["summary"]
    session.params = dict(summ.attrs.get("params", {}) or {})
    if "mean_image" in summ:
        mean = np.asarray(summ["mean_image"], dtype=np.float32)
        if mean.ndim == 2:
            mean = mean[:, :, None]
        for ch in range(mean.shape[2]):
            session.summary_images[f"summary mean · ch{ch + 1}"] = mean[:, :, ch]
    if "activity_image" in summ:
        session.summary_images["summary activity"] = np.asarray(summ["activity_image"], dtype=np.float32)
    if "valid_trials" in summ:
        session.valid_trials = np.asarray(summ["valid_trials"]).astype(int)
    if "trial_alignment" in summ:
        ta = summ["trial_alignment"]
        if "offsets" in ta:
            session.trial_offsets = np.asarray(ta["offsets"], dtype=float)
        if "corr_coeffs" in ta:
            session.trial_corr = np.asarray(ta["corr_coeffs"], dtype=float)

    if "sources" not in summ or "mean_image" not in summ:
        return
    with_traces = {t.index: t.extraction for t in trials if t.extraction is not None}
    if not with_traces:
        return
    sources = summ["sources"]
    mean = np.asarray(summ["mean_image"], dtype=np.float32)
    if mean.ndim == 2:
        mean = mean[:, :, None]
    first = next(iter(with_traces.values()))
    analyze_hz = session.full_hz
    try:
        data = ExtractionData(
            mean_image=mean,
            sel_pix=np.asarray(sources["sel_pix"]).astype(bool),
            source_rows=np.asarray(sources["locations_r"], dtype=float),
            source_cols=np.asarray(sources["locations_c"], dtype=float),
            trials=with_traces,
            analyze_hz=analyze_hz,
            source_name="zarr",
            file_path=str(session.path),
        )
    except KeyError as error:
        logger.warning("summary/sources incomplete (%s); traces hidden", error)
        return
    if first.footprints.shape[0] != int(data.sel_pix.sum()):
        logger.warning(
            "footprints cover %d pixels but sel_pix has %d; traces hidden",
            first.footprints.shape[0],
            int(data.sel_pix.sum()),
        )
        return
    session.extraction = _prune_dead_sources(data)


def empty_session(canvas: tuple[int, int] = (64, 64)) -> SubcellSession:
    """A session with nothing in it, for a viewer opened before any data."""
    return SubcellSession(path=Path(""), trials=[], canvas=tuple(int(v) for v in canvas))


def load_session(
    path: str | Path,
    extraction_config: ExtractionConfig | None = None,
    min_canvas: tuple[int, int] | None = None,
) -> SubcellSession:
    """
    Open an ``experiment.zarr`` for viewing.

    Parameters
    ----------
    path : str or Path
        The store ``ExperimentStore`` wrote, or the folder holding it.
    extraction_config : ExtractionConfig, optional
        Thresholds for the motion-frame censor mask shown over the movies and
        traces. Defaults to the parameters recorded in the store's summary,
        else the package defaults.
    min_canvas : tuple of int, optional
        Grow the shared ``(H, W)`` canvas at least this large, so a raw
        movie shown beside the registered ones fits too.

    Returns
    -------
    SubcellSession
        Trials registered so far, in store order, with whatever extraction
        has been saved for them.
    """
    path = Path(path)
    if path.is_dir() and path.suffix != ".zarr" and (path / "experiment.zarr").exists():
        path = path / "experiment.zarr"
    root = zarr.open(str(path), mode="r")

    config = extraction_config
    if config is None:
        params = {}
        if "summary" in root:
            params = dict(root["summary"].attrs.get("params", {}) or {})
        try:
            config = ExtractionConfig(**params) if params else ExtractionConfig()
        except Exception:  # noqa: BLE001 - stale params must not block viewing
            config = ExtractionConfig()

    trials: list[TrialRecord] = []
    if "trials" in root:
        grp = root["trials"]
        for key in sorted(grp.keys(), key=_trial_index):
            try:
                record = _read_trial(key, grp[key], config)
            except Exception:  # noqa: BLE001 - a half-written trial is skipped, not fatal
                logger.exception("skipping trial %s", key)
                continue
            if record is not None:
                trials.append(record)

    canvas = (
        max((t.shape[0] for t in trials), default=1),
        max((t.shape[1] for t in trials), default=1),
    )
    session = SubcellSession(path=path, trials=trials, canvas=canvas, extraction_config=config)
    _read_summary(root, trials, session)
    if session.extraction is not None:
        h, w = session.extraction.image_shape
        session.canvas = (max(session.canvas[0], h), max(session.canvas[1], w))
    if min_canvas is not None:
        session.canvas = (max(session.canvas[0], int(min_canvas[0])), max(session.canvas[1], int(min_canvas[1])))
    logger.info(
        "loaded %s: %d trial(s), %d source(s), canvas %s",
        path.name,
        session.n_trials,
        session.n_sources,
        session.canvas,
    )
    return session


def footprints_of(session: SubcellSession, trial_pos: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None:
    """
    Per-source ``(ypix, xpix, lam)`` from one trial's NMF footprints.

    Footprint rows follow the True pixels of the union ``sel_pix`` mask in
    row-major order, which is how the extraction indexed them.
    """
    data = session.extraction
    if data is None:
        return None
    record = session.trials[trial_pos]
    trial = data.trials.get(record.index)
    if trial is None:
        return None
    ys, xs = np.nonzero(data.sel_pix)
    H = np.nan_to_num(trial.footprints, nan=0.0)
    out = []
    for k in range(H.shape[1]):
        lam = H[:, k]
        keep = lam > 0
        out.append((ys[keep].astype(np.int32), xs[keep].astype(np.int32), lam[keep].astype(np.float32)))
    return out


def sel_pix_footprints(session: SubcellSession, radius: int | None = None) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None:
    """Disk footprints around each source center, the shape the NMF started from."""
    data = session.extraction
    if data is None:
        return None
    r = int(radius if radius is not None else session.extraction_config.sel_radius)
    h, w = data.image_shape
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    disk = (yy**2 + xx**2) <= r**2
    dy, dx = yy[disk], xx[disk]
    out = []
    for row, col in zip(data.source_rows, data.source_cols):
        ys = np.round(row).astype(int) + dy
        xs = np.round(col).astype(int) + dx
        ok = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        inside = np.zeros_like(ok)
        inside[ok] = data.sel_pix[ys[ok], xs[ok]]
        out.append(
            (ys[inside].astype(np.int32), xs[inside].astype(np.int32), np.ones(int(inside.sum()), np.float32))
        )
    return out


def as_movie_like(obj: Any) -> bool:
    """True for anything that indexes like a movie stack."""
    return all(hasattr(obj, name) for name in ("shape", "dtype", "ndim", "__getitem__"))
