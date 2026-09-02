"""Zarr output store for experiment results.

Manages the Zarr directory hierarchy for registered data, alignment metadata,
extraction results, and experiment summaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numcodecs
import numpy as np
import zarr

logger = logging.getLogger(__name__)

_COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=3)


@dataclass
class AlignmentData:
    """Alignment metadata for a single trial."""

    num_channels: int
    frame_time: float
    align_hz: float
    motion_r: np.ndarray  # (raw_frames,) row motion at full framerate
    motion_c: np.ndarray  # (raw_frames,) col motion at full framerate
    motion_ds_r: np.ndarray  # (ds_frames,) downsampled row motion
    motion_ds_c: np.ndarray  # (ds_frames,) downsampled col motion
    rec_neg_err: np.ndarray  # (ds_frames,) reconstruction error


class ExperimentStore:
    """Manages the Zarr directory store for an experiment's outputs."""

    def __init__(self, root_path: Path):
        self.root_path = Path(root_path)
        self.root = zarr.open(str(root_path), mode="a", zarr_format=2)
        logger.info("Opened Zarr store at %s", root_path)

    def _trial_group(self, trial_idx: int) -> zarr.Group:
        """Get or create the group for a trial."""
        key = f"trials/trial_{trial_idx:03d}"
        return self.root.require_group(key)

    def create_registered_ds(
        self,
        trial_idx: int,
        shape: tuple,
        num_channels: int,
        chunks: tuple | None = None,
    ) -> zarr.Array:
        """
        Create an empty zarr array for downsampled registered data.

        Returns the zarr Array for streaming chunk-by-chunk writes.
        """
        grp = self._trial_group(trial_idx)
        if chunks is None:
            chunks = (shape[0], shape[1], min(500, shape[2]))
        arr = grp.create_array(
            "registered_ds",
            shape=shape,
            dtype=np.float32,
            chunks=chunks,
            compressor=_COMPRESSOR,
            fill_value=np.nan,
            overwrite=True,
        )
        grp.attrs["num_channels"] = num_channels
        logger.debug("Created registered_ds for trial %d: shape %s", trial_idx, shape)
        return arr

    def create_registered_raw(
        self,
        trial_idx: int,
        shape: tuple,
        num_channels: int,
        chunks: tuple | None = None,
    ) -> zarr.Array:
        """
        Create an empty zarr array for full-resolution registered data.

        Returns the zarr Array for streaming chunk-by-chunk writes.
        """
        grp = self._trial_group(trial_idx)
        if chunks is None:
            chunks = (shape[0], shape[1], min(500, shape[2]))
        arr = grp.create_array(
            "registered_raw",
            shape=shape,
            dtype=np.float32,
            chunks=chunks,
            compressor=_COMPRESSOR,
            fill_value=np.nan,
            overwrite=True,
        )
        grp.attrs["num_channels"] = num_channels
        logger.debug("Created registered_raw for trial %d: shape %s", trial_idx, shape)
        return arr

    def save_alignment_data(self, trial_idx: int, adata: AlignmentData) -> None:
        """Save alignment metadata for a trial."""
        grp = self._trial_group(trial_idx).require_group("alignment")
        for name in (
            "motion_r",
            "motion_c",
            "motion_ds_r",
            "motion_ds_c",
            "rec_neg_err",
        ):
            arr = getattr(adata, name)
            grp.create_array(name, data=arr.astype(np.float64), overwrite=True)
        grp.attrs.update(
            {
                "num_channels": adata.num_channels,
                "frame_time": adata.frame_time,
                "align_hz": adata.align_hz,
            }
        )
        logger.debug("Saved alignment data for trial %d", trial_idx)

    def has_alignment_data(self, trial_idx: int) -> bool:
        """Whether alignment metadata exists for a trial."""
        return f"trials/trial_{trial_idx:03d}/alignment" in self.root

    def load_alignment_data(self, trial_idx: int) -> AlignmentData:
        """Load alignment metadata for a trial."""
        grp = self.root[f"trials/trial_{trial_idx:03d}/alignment"]
        return AlignmentData(
            num_channels=int(grp.attrs["num_channels"]),
            frame_time=float(grp.attrs["frame_time"]),
            align_hz=float(grp.attrs["align_hz"]),
            motion_r=np.array(grp["motion_r"]),
            motion_c=np.array(grp["motion_c"]),
            motion_ds_r=np.array(grp["motion_ds_r"]),
            motion_ds_c=np.array(grp["motion_ds_c"]),
            rec_neg_err=np.array(grp["rec_neg_err"]),
        )

    def save_avg_image(self, trial_idx: int, channel: int, image: np.ndarray) -> None:
        """Save an 8-bit average image for a channel."""
        grp = self._trial_group(trial_idx).require_group("avg_images")
        grp.create_array(
            f"ch{channel}_8bit", data=image.astype(np.uint8), overwrite=True
        )

    def save_mean_image(self, trial_idx: int, image: np.ndarray) -> None:
        """Save the mean image for a trial (all channels)."""
        grp = self._trial_group(trial_idx)
        grp.create_array("mean_image", data=image.astype(np.float32), overwrite=True)

    def save_activity_image(self, trial_idx: int, image: np.ndarray) -> None:
        """Save the activity image for a trial."""
        grp = self._trial_group(trial_idx)
        grp.create_array(
            "activity_image", data=image.astype(np.float32), overwrite=True
        )

    def load_registered_ds(self, trial_idx: int) -> np.ndarray:
        """Load downsampled registered data for a trial."""
        return np.array(self.root[f"trials/trial_{trial_idx:03d}/registered_ds"])

    def load_registered_raw(self, trial_idx: int) -> np.ndarray:
        """Load full-resolution registered data for a trial into memory."""
        return np.array(self.root[f"trials/trial_{trial_idx:03d}/registered_raw"])

    def has_registered_raw(self, trial_idx: int) -> bool:
        """Whether full-resolution registered data exists for a trial."""
        return f"trials/trial_{trial_idx:03d}/registered_raw" in self.root

    def load_registered_raw_pixels(
        self,
        trial_idx: int,
        pixel_mask: np.ndarray,
        num_channels: int,
        channel: int,
        block_size: int = 600,
    ) -> np.ndarray:
        """
        Read one channel at the masked pixels, streaming blocks of frames.

        Avoids materializing the full-resolution movie, which for a long trial
        is far larger than the selected pixels actually needed.

        Parameters
        ----------
        pixel_mask : np.ndarray
            Boolean mask over the field of view, (h, w).
        num_channels : int
            Channels interleaved along the stored time axis.
        channel : int
            Zero-indexed channel to read.
        block_size : int
            Frames per read.

        Returns
        -------
        np.ndarray
            Traces for the masked pixels, (pixel_mask.sum(), n_frames), float32.
        """
        arr = self.root[f"trials/trial_{trial_idx:03d}/registered_raw"]
        n_frames = arr.shape[2] // num_channels
        pixel_mask = np.asarray(pixel_mask, dtype=bool)
        n_pixels = int(pixel_mask.sum())
        h, w = int(arr.shape[0]), int(arr.shape[1])

        # The mask is drawn on the canvas shared by every trial, which can be
        # larger than this trial's own registered canvas (or, if extraction is
        # re-run after registering more trials, smaller). Read the pixels the
        # two have in common and leave the rest NaN: never observed here.
        mh, mw = pixel_mask.shape
        ch_, cw_ = min(h, mh), min(w, mw)
        local_mask = np.zeros((h, w), dtype=bool)
        local_mask[:ch_, :cw_] = pixel_mask[:ch_, :cw_]
        rows_of = np.full(pixel_mask.shape, -1, dtype=np.int64)
        rows_of[pixel_mask] = np.arange(n_pixels)
        target_rows = rows_of[:ch_, :cw_][local_mask[:ch_, :cw_]]
        if local_mask.sum() == n_pixels:
            out = np.empty((n_pixels, n_frames), dtype=np.float32)
        else:
            logger.info(
                "Trial %d canvas %dx%d vs mask %dx%d: %d of %d pixels lie outside it",
                trial_idx, h, w, mh, mw, n_pixels - int(local_mask.sum()), n_pixels,
            )
            out = np.full((n_pixels, n_frames), np.nan, dtype=np.float32)

        for start in range(0, n_frames, block_size):
            stop = min(start + block_size, n_frames)
            block = arr[:, :, start * num_channels : stop * num_channels]
            block = block.reshape(
                block.shape[0], block.shape[1], stop - start, num_channels
            )
            out[target_rows, start:stop] = block[:, :, :, channel][local_mask, :]

        return out

    def save_extraction_results(
        self,
        trial_idx: int,
        footprints: np.ndarray,
        events: np.ndarray,
        denoised: np.ndarray,
        ls: np.ndarray,
        f0: np.ndarray,
        snr: np.ndarray,
        discard_frames: np.ndarray,
        global_f: np.ndarray | None = None,
    ) -> None:
        """Save signal extraction results for a trial."""
        grp = self._trial_group(trial_idx).require_group("extraction")
        datasets = {
            "footprints": (footprints, np.float32),
            "events": (events, np.float32),
            "denoised": (denoised, np.float32),
            "ls": (ls, np.float32),
            "F0": (f0, np.float32),
            "SNR": (snr, np.float32),
            "discard_frames": (discard_frames, bool),
        }
        for name, (data, dtype) in datasets.items():
            grp.create_array(name, data=np.asarray(data, dtype=dtype), overwrite=True)
        if global_f is not None:
            grp.create_array(
                "global_F", data=global_f.astype(np.float32), overwrite=True
            )
        logger.debug("Saved extraction results for trial %d", trial_idx)

    def save_summary(
        self,
        mean_image: np.ndarray,
        activity_image: np.ndarray,
        source_locations_r: np.ndarray,
        source_locations_c: np.ndarray,
        sel_pix: np.ndarray,
        valid_trials: np.ndarray,
        trial_offsets: np.ndarray,
        trial_corr_coeffs: np.ndarray,
        params: dict,
    ) -> None:
        """Save experiment-level summary data."""
        summ = self.root.require_group("summary")
        for name, data in [
            ("mean_image", mean_image.astype(np.float32)),
            ("activity_image", activity_image.astype(np.float32)),
            ("valid_trials", valid_trials.astype(np.int32)),
        ]:
            summ.create_array(name, data=data, overwrite=True)

        sources = summ.require_group("sources")
        for name, data in [
            ("locations_r", source_locations_r.astype(np.float64)),
            ("locations_c", source_locations_c.astype(np.float64)),
            ("sel_pix", sel_pix.astype(bool)),
        ]:
            sources.create_array(name, data=data, overwrite=True)

        align = summ.require_group("trial_alignment")
        for name, data in [
            ("offsets", trial_offsets.astype(np.float64)),
            ("corr_coeffs", trial_corr_coeffs.astype(np.float64)),
        ]:
            align.create_array(name, data=data, overwrite=True)

        summ.attrs.update(
            {
                "num_sources": len(source_locations_r),
                "params": params,
            }
        )
        logger.info("Saved experiment summary with %d sources", len(source_locations_r))
