"""Unified data model for NMF extraction results.

Decouples the viewer from the storage format (Zarr vs MATLAB .mat).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrialExtraction:
    """Extraction results for a single trial."""

    footprints: np.ndarray  # (n_sel_pix, n_sources) float32
    events: np.ndarray  # (n_sources, n_time) float32
    denoised: np.ndarray  # (n_sources, n_time) float32
    ls: np.ndarray  # (n_sources, n_time) float32
    f0: np.ndarray  # (n_sources, n_time) float32
    snr: np.ndarray  # (n_sources,) float32
    discard_frames: np.ndarray  # (n_time,) bool

    @property
    def n_time(self) -> int:
        return self.events.shape[1]

    @property
    def n_sources(self) -> int:
        return self.events.shape[0]


@dataclass
class ExtractionData:
    """Unified data model produced by both Zarr and MATLAB loaders."""

    mean_image: np.ndarray  # (H, W, C) float32
    sel_pix: np.ndarray  # (H, W) bool - union pixel selection mask
    source_rows: np.ndarray  # (n_sources,) float64, peak row coords
    source_cols: np.ndarray  # (n_sources,) float64, peak col coords

    trials: dict[int, TrialExtraction]  # trial_index -> TrialExtraction

    analyze_hz: float  # framerate for time axis
    source_name: str  # "zarr" or "mat"
    file_path: str  # path that was loaded

    initial_footprints: np.ndarray | None = None  # (H, W, n_sources) bool

    @property
    def n_sources(self) -> int:
        return len(self.source_rows)

    @property
    def n_channels(self) -> int:
        return self.mean_image.shape[2]

    @property
    def image_shape(self) -> tuple[int, int]:
        return self.mean_image.shape[0], self.mean_image.shape[1]

    @property
    def trial_keys(self) -> list[int]:
        return sorted(self.trials.keys())

    def pixel_index_map(self) -> np.ndarray:
        """
        Precompute map from (row, col) to index within sel_pix True pixels.

        Returns (H, W) int32 array. -1 for pixels not in sel_pix.
        """
        h, w = self.image_shape
        pix_map = np.full((h, w), -1, dtype=np.int32)
        pix_map[self.sel_pix] = np.arange(int(np.sum(self.sel_pix)))
        return pix_map
