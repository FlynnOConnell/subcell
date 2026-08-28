"""
Ingest an mbo_utilities LazyArray as a multi-trial Bergamo recording.

subcell's cross-trial alignment needs to know where one trial ends and the
next begins. A LazyArray flattens every source file into one T axis, so the
boundaries are recovered from the ``frames_per_file`` / ``file_paths``
metadata that multi-file readers deposit. A source without them is treated
as a single trial.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from subcell.io.tiff_reader import ScanImageMetadata
from subcell.io.trial_table import TrialEntry, TrialTable

logger = logging.getLogger(__name__)


def _get_meta(metadata: dict, name: str, default=None):
    """Read a canonical metadata key, resolving mbo_utilities aliases if present."""
    try:
        from mbo_utilities.metadata.params import get_param
    except ImportError:
        return metadata.get(name, default)
    return get_param(metadata, name, default=default)


def trial_boundaries(arr: Any) -> list[slice]:
    """
    Slices along T delimiting each trial.

    Parameters
    ----------
    arr : LazyArray
        Array whose metadata may carry ``frames_per_file``.

    Returns
    -------
    list of slice
        One slice per trial, covering T contiguously. A single full-extent
        slice when no per-file counts are available.
    """
    metadata = getattr(arr, "metadata", None) or {}
    n_time = int(arr.shape[0])
    counts = _get_meta(metadata, "frames_per_file")

    if not counts:
        logger.info(
            "No frames_per_file metadata; treating all %d timepoints as one trial. "
            "Cross-trial alignment has nothing to align.",
            n_time,
        )
        return [slice(0, n_time)]

    counts = [int(c) for c in counts]
    total = sum(counts)
    if total != n_time:
        logger.warning(
            "frames_per_file sums to %d but T is %d; falling back to one trial",
            total,
            n_time,
        )
        return [slice(0, n_time)]

    bounds = []
    start = 0
    for count in counts:
        bounds.append(slice(start, start + count))
        start += count
    return bounds


def _filenames(arr: Any, n_trials: int) -> list[str]:
    """Per-trial source names, synthesised when the array does not carry them."""
    metadata = getattr(arr, "metadata", None) or {}
    names = getattr(arr, "filenames", None) or _get_meta(metadata, "file_paths") or []
    names = [Path(str(n)).name for n in names]
    if len(names) == n_trials:
        return names
    return [f"trial_{i + 1:03d}" for i in range(n_trials)]


@dataclass
class ArrayTrialSource:
    """
    A LazyArray presented to subcell as a sequence of Bergamo trials.

    Attributes
    ----------
    array : LazyArray
        5D source, ordered TCZYX.
    trial_slices : list of slice
        T-range backing each trial.
    filenames : list of str
        Display name per trial.
    metadata : ScanImageMetadata
        Channel count and frame time, as the registration expects them.
    """

    array: Any
    trial_slices: list[slice] = field(default_factory=list)
    filenames: list[str] = field(default_factory=list)
    metadata: ScanImageMetadata = field(default_factory=ScanImageMetadata)

    @property
    def n_trials(self) -> int:
        """Number of trials in this source."""
        return len(self.trial_slices)

    def read_trial(self, trial_index: int) -> np.ndarray:
        """
        Materialize one trial.

        Parameters
        ----------
        trial_index : int
            Zero-based index into ``trial_slices``.

        Returns
        -------
        np.ndarray
            (rows, cols, channels, frames), matching what the registration's
            ``reshape_interleaved`` would have produced from a TIFF.
        """
        t_slice = self.trial_slices[trial_index]
        block = np.asarray(self.array[t_slice, :, 0, :, :])
        return np.transpose(block, (2, 3, 1, 0))


def array_trial_source(arr: Any) -> ArrayTrialSource:
    """
    Build an :class:`ArrayTrialSource` from an mbo_utilities LazyArray.

    Parameters
    ----------
    arr : LazyArray
        5D TCZYX source. Bergamo data is single-plane, so Z must be 1.

    Returns
    -------
    ArrayTrialSource

    Raises
    ------
    ValueError
        If the array is not 5D or carries more than one Z plane.
    """
    shape = tuple(int(s) for s in arr.shape)
    if len(shape) != 5:
        raise ValueError(f"expected a 5D TCZYX array, got shape {shape}")
    if shape[2] != 1:
        raise ValueError(
            f"subcell is single-plane (Bergamo); array has {shape[2]} Z planes. "
            "Select one plane before running."
        )

    metadata = getattr(arr, "metadata", None) or {}
    fs = _get_meta(metadata, "fs", default=0.0) or 0.0
    slices = trial_boundaries(arr)

    return ArrayTrialSource(
        array=arr,
        trial_slices=slices,
        filenames=_filenames(arr, len(slices)),
        metadata=ScanImageMetadata(
            num_channels=shape[1],
            frame_rate=float(fs),
            frame_time=1.0 / float(fs) if fs else 0.0,
            num_frames=shape[0],
            num_rows=shape[3],
            num_cols=shape[4],
        ),
    )


def trial_table_from_array(arr: Any, directory: str | Path = "") -> TrialTable:
    """
    Build a :class:`TrialTable` from a LazyArray's trial boundaries.

    Parameters
    ----------
    arr : LazyArray
        5D TCZYX source.
    directory : str or Path
        Recorded on the table for provenance; the data is read from ``arr``,
        not from this path.

    Returns
    -------
    TrialTable
        One entry per trial, indexed from 1 as the MATLAB port expects.
    """
    source = array_trial_source(arr)
    return TrialTable(
        directory=str(directory),
        entries=[
            TrialEntry(filename=name, trial_index=i + 1, epoch=1)
            for i, name in enumerate(source.filenames)
        ],
    )
