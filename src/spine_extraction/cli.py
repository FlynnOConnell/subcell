"""Command-line interface for spine-extract."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from spine_extraction import __version__
from spine_extraction.config import PipelineConfig, RegistrationConfig, ExtractionConfig


@click.group()
@click.version_option(version=__version__)
def cli():
    """spine-extract: Synaptic signal extraction from two-photon imaging data."""


@cli.command()
@click.option("--output", "-o", type=click.Path(), default="config.yaml",
              help="Output path for the generated config file.")
def init_config(output):
    """Generate a default configuration YAML file."""
    cfg = PipelineConfig(data_directory=Path("."))
    cfg.to_yaml(Path(output))
    click.echo(f"Default config written to {output}")


@cli.command("build-trial-table")
@click.argument("data_dir", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output path for trial table JSON.")
def build_trial_table(data_dir, output):
    """Build trial table from a directory of TIFF files."""
    from spine_extraction.io.trial_table import TrialTable

    data_path = Path(data_dir)
    tt = TrialTable.from_directory(data_path)
    out_path = Path(output) if output else data_path / "trial_table.json"
    tt.save(out_path)
    click.echo(f"Trial table with {len(tt.entries)} trials saved to {out_path}")


@cli.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.option("--config", "-c", type=click.Path(exists=True), default=None,
              help="Path to config YAML file.")
@click.option("--trial-table", type=click.Path(exists=True), default=None,
              help="Path to trial_table.json.")
@click.option("--device", type=click.Choice(["cpu", "cuda", "auto"]), default=None)
@click.option("--workers", "-j", type=int, default=None)
def register(data_dir, config, trial_table, device, workers):
    """Register (motion-correct) all trials."""
    from spine_extraction.io.trial_table import TrialTable
    from spine_extraction.io.zarr_store import ExperimentStore
    from spine_extraction.registration.bergamo import register_bergamo
    from spine_extraction._utils.torch_helpers import get_device
    from spine_extraction._utils.logging import setup_logging

    cfg = _load_config(config, data_dir, device=device,
                       registration_workers=workers)
    setup_logging(cfg.log_level)
    logger = logging.getLogger(__name__)

    tt_path = Path(trial_table) if trial_table else Path(data_dir) / "trial_table.json"
    if not tt_path.exists():
        logger.info("No trial table found, building one automatically...")
        tt = TrialTable.from_directory(Path(data_dir))
        tt.save(tt_path)
    else:
        tt = TrialTable.load(tt_path)

    out_dir = cfg.get_output_directory()
    out_dir.mkdir(parents=True, exist_ok=True)
    store = ExperimentStore(out_dir / "experiment.zarr")
    torch_device = get_device(cfg.device)

    tt = register_bergamo(tt, cfg.registration, store, device=torch_device)
    tt.save(tt_path)
    click.echo(f"Registration complete. Updated trial table at {tt_path}")


@cli.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.option("--config", "-c", type=click.Path(exists=True), default=None)
@click.option("--trial-table", type=click.Path(exists=True), default=None)
@click.option("--device", type=click.Choice(["cpu", "cuda", "auto"]), default=None)
@click.option("--workers", "-j", type=int, default=None)
def extract(data_dir, config, trial_table, device, workers):
    """Extract synaptic signals from registered data."""
    from spine_extraction.io.trial_table import TrialTable
    from spine_extraction.io.zarr_store import ExperimentStore
    from spine_extraction.pipeline.runner import run_extraction
    from spine_extraction._utils.torch_helpers import get_device
    from spine_extraction._utils.logging import setup_logging

    cfg = _load_config(config, data_dir, device=device,
                       extraction_workers=workers)
    setup_logging(cfg.log_level)

    tt_path = Path(trial_table) if trial_table else Path(data_dir) / "trial_table.json"
    tt = TrialTable.load(tt_path)

    out_dir = cfg.get_output_directory()
    store = ExperimentStore(out_dir / "experiment.zarr")
    torch_device = get_device(cfg.device)

    run_extraction(tt, cfg.extraction, store, device=torch_device)
    click.echo("Extraction complete.")


@cli.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.option("--config", "-c", type=click.Path(exists=True), default=None)
@click.option("--device", type=click.Choice(["cpu", "cuda", "auto"]), default=None)
@click.option("--workers", "-j", type=int, default=None)
def run(data_dir, config, device, workers):
    """Run full pipeline: build-trial-table -> register -> extract."""
    from spine_extraction.pipeline.runner import run_full_pipeline
    from spine_extraction._utils.torch_helpers import get_device
    from spine_extraction._utils.logging import setup_logging

    cfg = _load_config(config, data_dir, device=device,
                       registration_workers=workers, extraction_workers=workers)
    setup_logging(cfg.log_level)
    torch_device = get_device(cfg.device)

    run_full_pipeline(cfg, device=torch_device)
    click.echo("Full pipeline complete.")


def _load_config(
    config_path: str | None,
    data_dir: str,
    device: str | None = None,
    registration_workers: int | None = None,
    extraction_workers: int | None = None,
) -> PipelineConfig:
    """Load config from YAML or create default, applying CLI overrides."""
    if config_path:
        overrides = {}
        if device:
            overrides["device"] = device
        if registration_workers:
            overrides["registration.n_workers"] = registration_workers
        if extraction_workers:
            overrides["extraction.n_parallel_workers"] = extraction_workers
        cfg = PipelineConfig.from_yaml(Path(config_path), **overrides)
    else:
        reg = RegistrationConfig()
        ext = ExtractionConfig()
        if registration_workers:
            reg.n_workers = registration_workers
        if extraction_workers:
            ext.n_parallel_workers = extraction_workers
        cfg = PipelineConfig(
            data_directory=Path(data_dir),
            device=device or "auto",
            registration=reg,
            extraction=ext,
        )
    # Ensure data_directory is set from CLI argument
    cfg.data_directory = Path(data_dir)
    return cfg
