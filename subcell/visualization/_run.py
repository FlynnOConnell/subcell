"""
Run the subcell pipeline on a background thread, for a viewer to watch.

The viewer stays interactive while registration and extraction write into
the store; it re-opens the store as products land. The run captures the
pipeline's log lines so the panel can show them, and reports which stage and
trial it is on.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from subcell.config import ExtractionConfig, RegistrationConfig
from subcell.io.lazy_array import ArrayTrialSource, array_trial_source, trial_table_from_array
from subcell.io.trial_table import TrialEntry, TrialTable
from subcell.io.zarr_store import ExperimentStore

logger = logging.getLogger(__name__)

__all__ = ["PipelineRun", "STAGES", "resolve_source"]

STAGES = ("register", "extract")
STORE_NAME = "experiment.zarr"


class _DequeHandler(logging.Handler):
    """Keeps the last lines of the pipeline log for the panel."""

    def __init__(self, lines: deque):
        super().__init__(level=logging.INFO)
        self.lines = lines
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname).1s %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(self.format(record))
        except Exception:  # noqa: BLE001 - logging must never raise into the pipeline
            pass


class _ArrayWithMeta:
    """A plain array given the ``metadata`` attribute ``array_trial_source`` reads."""

    def __init__(self, array, metadata: dict):
        self.array = array
        self.metadata = dict(metadata)
        self.shape = tuple(int(s) for s in array.shape)
        self.ndim = len(self.shape)
        self.dtype = array.dtype

    def __getitem__(self, key):
        return self.array[key]


def _to_5d(array) -> Any:
    """``(T, Y, X)`` or ``(T, C, Y, X)`` arrays as ``(T, C, Z, Y, X)``; 5D passes through."""
    ndim = len(array.shape)
    if ndim == 5:
        return array
    arr = np.asarray(array)
    if ndim == 3:
        return arr[:, None, None, :, :]
    if ndim == 4:
        return arr[:, :, None, :, :]
    raise ValueError(f"expected a 3D, 4D or 5D movie, got shape {tuple(array.shape)}")


@dataclass
class ResolvedSource:
    """What a run reads: a trial table, and an array source when not TIFFs on disk."""

    trial_table: TrialTable
    array_source: ArrayTrialSource | None = None
    description: str = ""


def resolve_source(
    source: Any,
    fs: float | None = None,
    frames_per_trial: Sequence[int] | None = None,
) -> ResolvedSource:
    """
    Turn whatever the caller has into something ``register_bergamo`` reads.

    Parameters
    ----------
    source : str, Path, array, or LazyArray
        A directory of ScanImage TIFFs (one per trial), a single TIFF, an
        ``mbo_utilities`` LazyArray, or an in-memory movie: ``(T, Y, X)``,
        ``(T, C, Y, X)`` or ``(T, C, Z, Y, X)`` with one Z plane.
    fs : float, optional
        Frame rate for an array without one in its metadata.
    frames_per_trial : sequence of int, optional
        Trial lengths along T for an array; the whole array is one trial
        without it (and without ``frames_per_file`` metadata).
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir() and not path.suffix == ".zarr":
            tifs = sorted(p for p in path.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
            if tifs:
                table = TrialTable.from_directory(path)
                return ResolvedSource(table, None, f"{len(table.entries)} TIFF trial(s) in {path.name}")
        if path.is_file() and path.suffix.lower() in (".tif", ".tiff"):
            table = TrialTable(directory=str(path.parent), entries=[TrialEntry(path.name, 1, 1)])
            return ResolvedSource(table, None, path.name)
        try:
            from mbo_utilities import imread
        except ImportError as error:
            raise ValueError(
                f"{path} is not a TIFF or a folder of TIFFs, and mbo_utilities is not "
                "installed to read anything else"
            ) from error
        source = imread(str(path))
        description = path.name
    else:
        description = f"{type(source).__name__} {tuple(int(s) for s in source.shape)}"

    array = _to_5d(source)
    metadata = dict(getattr(array, "metadata", None) or {})
    if fs is not None:
        metadata["fs"] = float(fs)
    if frames_per_trial is not None:
        metadata["frames_per_file"] = [int(n) for n in frames_per_trial]
    if not hasattr(array, "metadata") or fs is not None or frames_per_trial is not None:
        array = _ArrayWithMeta(array, metadata)
    trial_source = array_trial_source(array)
    table = trial_table_from_array(array, directory=description)
    return ResolvedSource(table, trial_source, f"{table.n_trials} trial(s) from {description}")


class PipelineRun:
    """
    One background run of registration and/or extraction into a store.

    Parameters
    ----------
    source
        See :func:`resolve_source`.
    output : str or Path
        Folder the ``experiment.zarr`` goes in.
    registration, extraction : configs
        Stage parameters; defaults are the package defaults with
        ``save_full_resolution`` on, which extraction needs.
    stages : sequence of str
        Any of ``"register"`` and ``"extract"``, run in that order.
    device : str
        ``"auto"``, ``"cpu"`` or ``"cuda"``.

    Attributes
    ----------
    status : str
        ``"idle"``, ``"running"``, ``"done"``, ``"cancelled"`` or ``"error"``.
    stage : str
        The stage in progress, or the last one run.
    log : deque of str
        Recent pipeline log lines.
    """

    def __init__(
        self,
        source: Any,
        output: str | Path,
        registration: RegistrationConfig | None = None,
        extraction: ExtractionConfig | None = None,
        stages: Sequence[str] = STAGES,
        device: str = "auto",
        fs: float | None = None,
        frames_per_trial: Sequence[int] | None = None,
    ):
        self.source = source
        self.output = Path(output)
        self.registration = registration or RegistrationConfig(save_full_resolution=True)
        self.extraction = extraction or ExtractionConfig()
        self.stages = tuple(s for s in STAGES if s in stages)
        self.device = device
        self.fs = fs
        self.frames_per_trial = frames_per_trial

        self.status = "idle"
        self.stage = ""
        self.message = ""
        self.error: str | None = None
        self.log: deque = deque(maxlen=500)
        self.trials_done = 0
        self.n_trials = 0
        self.started = 0.0
        self.finished = 0.0
        self.products_changed = threading.Event()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def store_path(self) -> Path:
        return self.output / STORE_NAME

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def elapsed(self) -> float:
        if not self.started:
            return 0.0
        end = self.finished or time.perf_counter()
        return end - self.started

    def start(self) -> "PipelineRun":
        if self.running:
            raise RuntimeError("this run is already in progress")
        self._cancel.clear()
        self.status = "running"
        self.error = None
        self.started = time.perf_counter()
        self.finished = 0.0
        self._thread = threading.Thread(target=self._target, name="subcell-pipeline", daemon=True)
        self._thread.start()
        return self

    def cancel(self) -> None:
        """Stop after the trial in progress; a stage in flight is not interrupted."""
        self._cancel.set()
        self.message = "cancelling after the current trial"

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the run ends; True when it did within ``timeout``."""
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    # ------------------------------------------------------------------

    def _target(self) -> None:
        pipeline_logger = logging.getLogger("subcell")
        handler = _DequeHandler(self.log)
        previous_level = pipeline_logger.level
        if previous_level == logging.NOTSET or previous_level > logging.INFO:
            pipeline_logger.setLevel(logging.INFO)
        pipeline_logger.addHandler(handler)
        try:
            self._run()
            if self.status == "running":
                self.status = "cancelled" if self._cancel.is_set() else "done"
        except Exception as error:  # noqa: BLE001 - reported in the panel
            self.status = "error"
            self.error = f"{type(error).__name__}: {error}"
            self.log.append(traceback.format_exc())
            logger.exception("pipeline run failed")
        finally:
            self.finished = time.perf_counter()
            pipeline_logger.removeHandler(handler)
            pipeline_logger.setLevel(previous_level)
            self.products_changed.set()

    def _run(self) -> None:
        from subcell._utils.torch_helpers import get_device
        from subcell.pipeline.runner import run_extraction
        from subcell.registration.bergamo import register_bergamo

        self.message = "reading source"
        resolved = resolve_source(self.source, fs=self.fs, frames_per_trial=self.frames_per_trial)
        table = resolved.trial_table
        self.n_trials = table.n_trials
        self.output.mkdir(parents=True, exist_ok=True)
        store = ExperimentStore(self.store_path)
        device = get_device(self.device)
        logger.info("source: %s -> %s on %s", resolved.description, self.store_path, device)

        if "register" in self.stages:
            self.stage = "register"
            if resolved.array_source is None:
                # TIFF trials: one call, so the process pool can spread them
                self.message = f"registering {self.n_trials} trial(s)"
                register_bergamo(table, self.registration, store, device=device)
                self.trials_done = self.n_trials
                self.products_changed.set()
            else:
                src = resolved.array_source
                for pos, entry in enumerate(table.entries):
                    if self._cancel.is_set():
                        return
                    self.message = f"registering trial {entry.trial_index} of {self.n_trials}"
                    one = TrialTable(directory=table.directory, entries=[entry])
                    sub = ArrayTrialSource(
                        array=src.array,
                        trial_slices=[src.trial_slices[pos]],
                        filenames=[src.filenames[pos]],
                        metadata=src.metadata,
                    )
                    register_bergamo(one, self.registration, store, device=device, source=sub)
                    self.trials_done = pos + 1
                    self.products_changed.set()

        if "extract" in self.stages and not self._cancel.is_set():
            self.stage = "extract"
            self.message = "localizing, aligning and extracting"
            run_extraction(table, self.extraction, store, device=device)
            self.products_changed.set()
        self.message = "finished"
