"""Configuration models mirroring the MATLAB setParams() defaults."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class RegistrationConfig(BaseModel):
    """Motion correction parameters, porting setParams('stripRegBergamo')."""

    maxshift: int = Field(
        50, description="Maximum frame offset in pixels for motion search"
    )
    clip_shift: int = Field(10, description="Maximum allowable shift per frame")
    n_workers: int = Field(
        4, description="Number of parallel workers for trial-level processing"
    )
    remove_lines: int = Field(
        4, description="Flyback lines to remove from top of each image"
    )
    ds_time: int = Field(
        3, description="Temporal downsampling exponent: factor = 2^ds_time"
    )
    frame_rate: float = Field(
        0.0, description="Frame rate in Hz; 0 = auto-detect from metadata"
    )
    overwrite_existing: bool = Field(
        False, description="Re-register trials that already have outputs"
    )
    init_frames: int = Field(
        1000, description="Number of frames used for initial template"
    )
    min_cluster_size: int = Field(
        100, description="Minimum cluster size for template selection"
    )
    template_min_count: int = Field(
        100, description="Pixels with fewer measurements stay NaN in template"
    )
    save_full_resolution: bool = Field(
        False,
        description=(
            "Save full-resolution registered data, which signal extraction "
            "requires. Grows the store by 2^ds_time."
        ),
    )

    @property
    def ds_factor(self) -> int:
        """Temporal downsampling factor, 2^ds_time."""
        return 2**self.ds_time


class ExtractionConfig(BaseModel):
    """Signal extraction parameters, porting setParams('summarize_LoCo')."""

    microscope: Literal["bergamo"] = Field("bergamo", description="Microscope type")
    sigma_px: float = Field(1.33, description="PSF Gaussian sigma in pixels")
    nmf_iter: int = Field(2, description="NMF refinement iterations")
    dXY: int = Field(3, description="Source radius in pixels")
    lambda_param: float | None = Field(
        None, description="Single-photon amplitude; None = auto-estimate"
    )
    denoise_window_s: float = Field(
        0.2, description="Temporal denoising window in seconds"
    )
    baseline_window_glu_s: float = Field(
        4.0, description="Glutamate baseline (F0) window in seconds"
    )
    baseline_window_ca_s: float = Field(
        4.0, description="Calcium baseline window in seconds"
    )
    activity_channel: int = Field(
        1, description="Channel containing glutamate signal (1-indexed)"
    )
    tau_s: float = Field(0.03, description="Indicator decay time constant in seconds")
    max_synapse_density: float = Field(
        0.01, description="Maximum synapses per valid pixel"
    )
    n_parallel_workers: int = Field(
        12, description="Upper bound on subproblems solved concurrently"
    )
    motion_thresh: float = Field(
        2.5, description="Motion detection threshold for censoring"
    )
    nan_thresh: float = Field(
        0.33, description="Max NaN fraction to consider a pixel valid"
    )
    discard_initial_s: float = Field(
        0.0, description="Seconds to discard from trial start"
    )
    block_size: int = Field(
        600, description="Frames per block when streaming full-res data"
    )
    sel_radius_factor: float = Field(
        2.0, description="Selection radius = ceil(factor * dXY)"
    )
    analyze_hz: float = Field(
        0.0, description="Analysis framerate; 0 = use full framerate"
    )
    device: str = Field(
        "auto", description="'cpu', 'cuda', or 'auto' for GPU-accelerated FFT"
    )
    cross_trial_maxshift: int = Field(
        5, description="Max shift for cross-trial alignment in pixels"
    )
    valid_trial_corr_min: float = Field(
        0.90, description="Min correlation for a trial to be valid"
    )

    @property
    def sel_radius(self) -> int:
        """Radius in pixels of the pixel set selected around each source."""
        return math.ceil(self.sel_radius_factor * self.dXY)

    def compute_derived(self, framerate: float) -> dict:
        """
        Parameters that depend on the acquisition framerate.

        Parameters
        ----------
        framerate : float
            Framerate of the data being analyzed, used when ``analyze_hz`` is 0.

        Returns
        -------
        dict
            Framerate, decay constant in samples, baseline and denoise window
            lengths in samples, and the decay kernel half-length.
        """
        analyze_hz = self.analyze_hz if self.analyze_hz > 0 else framerate
        tau_full = self.tau_s * analyze_hz
        return {
            "analyze_hz": analyze_hz,
            "tau_full": tau_full,
            "baseline_window_samps": int(self.baseline_window_glu_s * analyze_hz),
            "denoise_window_samps": int(self.denoise_window_s * analyze_hz),
            "kernel_half_len": math.ceil(6 * tau_full),
        }


class PipelineConfig(BaseModel):
    """Top-level configuration combining all stages."""

    data_directory: Path
    output_directory: Path | None = Field(
        None, description="Output dir; default = data_directory / subcell_output"
    )
    device: str = Field("auto", description="'cpu', 'cuda', or 'auto'")
    registration: RegistrationConfig = RegistrationConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    log_level: str = Field("INFO", description="Logging level")

    def get_output_directory(self) -> Path:
        """Configured output directory, or a default beside the data."""
        if self.output_directory is not None:
            return self.output_directory
        return self.data_directory / "subcell_output"

    @classmethod
    def from_yaml(cls, path: Path, **overrides) -> PipelineConfig:
        """
        Load configuration from YAML.

        Parameters
        ----------
        **overrides
            Dotted keys such as ``registration.n_workers``. None values are ignored.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        for key, value in overrides.items():
            if value is not None:
                _set_nested(data, key, value)
        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to a YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(mode="json"),
                f,
                default_flow_style=False,
                sort_keys=False,
            )


def _set_nested(d: dict, key: str, value) -> None:
    """Set a dotted key such as 'registration.n_workers' in a nested dict."""
    parts = key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value
