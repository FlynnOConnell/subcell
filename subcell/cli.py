"""Command-line interface for subcell."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from subcell import __version__
from subcell._utils.logging import setup_logging
from subcell._utils.torch_helpers import get_device
from subcell.config import ExtractionConfig, PipelineConfig, RegistrationConfig
from subcell.io.trial_table import TrialTable
from subcell.io.zarr_store import ExperimentStore
from subcell.pipeline.runner import run_extraction, run_full_pipeline
from subcell.registration.bergamo import register_bergamo

logger = logging.getLogger(__name__)


def _load_config(
    config_path: str | None,
    data_dir: str,
    device: str | None = None,
    registration_workers: int | None = None,
    extraction_workers: int | None = None,
) -> PipelineConfig:
    """
    Load configuration from YAML or defaults, applying CLI overrides.

    Parameters
    ----------
    config_path : str, optional
        YAML matching the ``PipelineConfig`` schema. Defaults are used if None.
    data_dir : str
        Always overrides ``data_directory``.

    Returns
    -------
    PipelineConfig
        Config whose ``device`` is propagated to the extraction stage.
    """
    if config_path:
        overrides = {}
        if device is not None:
            overrides["device"] = device
        if registration_workers is not None:
            overrides["registration.n_workers"] = registration_workers
        if extraction_workers is not None:
            overrides["extraction.n_parallel_workers"] = extraction_workers
        cfg = PipelineConfig.from_yaml(Path(config_path), **overrides)
    else:
        reg = RegistrationConfig()
        ext = ExtractionConfig()
        if registration_workers is not None:
            reg.n_workers = registration_workers
        if extraction_workers is not None:
            ext.n_parallel_workers = extraction_workers
        cfg = PipelineConfig(
            data_directory=Path(data_dir),
            device=device or "auto",
            registration=reg,
            extraction=ext,
        )

    cfg.data_directory = Path(data_dir)
    cfg.extraction.device = cfg.device
    return cfg


@click.group()
@click.version_option(version=__version__)
def cli():
    """subcell: synaptic signal extraction from two-photon imaging data."""


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="config.yaml",
    help="Output path for the generated config file.",
)
def init_config(output):
    """Generate a default configuration YAML file."""
    cfg = PipelineConfig(data_directory=Path("."))
    cfg.to_yaml(Path(output))
    click.echo(f"Default config written to {output}")


@cli.command("build-trial-table")
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output path for trial table JSON.",
)
def build_trial_table(data_dir, output):
    """Build trial table from a directory of TIFF files."""
    data_path = Path(data_dir)
    tt = TrialTable.from_directory(data_path)
    out_path = Path(output) if output else data_path / "trial_table.json"
    tt.save(out_path)
    click.echo(f"Trial table with {len(tt.entries)} trials saved to {out_path}")


@cli.command()
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default=None,
    help="Path to config YAML file.",
)
@click.option(
    "--trial-table",
    type=click.Path(exists=True),
    default=None,
    help="Path to trial_table.json.",
)
@click.option("--device", type=click.Choice(["cpu", "cuda", "auto"]), default=None)
@click.option("--workers", "-j", type=int, default=None)
def register(data_dir, config, trial_table, device, workers):
    """Register (motion-correct) all trials."""
    cfg = _load_config(config, data_dir, device=device, registration_workers=workers)
    setup_logging(cfg.log_level)

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

    tt = register_bergamo(tt, cfg.registration, store, device=get_device(cfg.device))
    tt.save(tt_path)
    click.echo(f"Registration complete. Updated trial table at {tt_path}")


@cli.command()
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--config", "-c", type=click.Path(exists=True), default=None)
@click.option("--trial-table", type=click.Path(exists=True), default=None)
@click.option("--device", type=click.Choice(["cpu", "cuda", "auto"]), default=None)
@click.option("--workers", "-j", type=int, default=None)
def extract(data_dir, config, trial_table, device, workers):
    """Extract synaptic signals from registered data."""
    cfg = _load_config(config, data_dir, device=device, extraction_workers=workers)
    setup_logging(cfg.log_level)

    tt_path = Path(trial_table) if trial_table else Path(data_dir) / "trial_table.json"
    tt = TrialTable.load(tt_path)

    store = ExperimentStore(cfg.get_output_directory() / "experiment.zarr")
    run_extraction(tt, cfg.extraction, store, device=get_device(cfg.device))
    click.echo("Extraction complete.")


@cli.command()
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--config", "-c", type=click.Path(exists=True), default=None)
@click.option("--device", type=click.Choice(["cpu", "cuda", "auto"]), default=None)
@click.option("--workers", "-j", type=int, default=None)
def run(data_dir, config, device, workers):
    """Run the full pipeline: build-trial-table, register, extract."""
    cfg = _load_config(
        config,
        data_dir,
        device=device,
        registration_workers=workers,
        extraction_workers=workers,
    )
    # RegistrationConfig defaults to standalone registration, which does not
    # keep the full-rate movie that extraction reads. With no config file to
    # say otherwise the full pipeline needs it, so turn it on rather than fail
    # the check in run_full_pipeline. An explicit False in YAML still stands.
    if config is None:
        cfg.registration.save_full_resolution = True
    setup_logging(cfg.log_level)
    run_full_pipeline(cfg, device=get_device(cfg.device))
    click.echo("Full pipeline complete.")


@cli.command()
@click.argument("store", required=False, type=click.Path())
@click.option("--raw", default=None, help="Unregistered movie to show beside the registered one.")
@click.option("--labels", default=None, help="Comma-separated class names, e.g. spine,shaft,junk.")
@click.option("--size", default="1700x950", show_default=True, help="Figure size WxH in pixels.")
def vis(store, raw, labels, size):
    """Open SubcellVis on a store, or empty to run the pipeline from the window."""
    argv = [store] if store else []
    if raw:
        argv += ["--raw", raw]
    if labels:
        argv += ["--labels", labels]
    argv += ["--size", size]
    from subcell.visualization.subcell_vis import main

    main(argv)
