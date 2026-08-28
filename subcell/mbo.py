"""
mbo_utilities plugin: exposes subcell in the mbo studio pipelines dropdown.

Discovered through the ``mbo_utilities.pipelines`` entry-point group, so
mbo_utilities needs no knowledge of this package. Importing this module does
not require mbo_utilities to be installed; the widget class is only built
when the GUI asks for it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from imgui_bundle import imgui
from mbo_utilities.gui._selection_ui import draw_selection_table, resolve_dim_labels
from mbo_utilities.gui.widgets.pipelines._base import PipelineWidget
from mbo_utilities.pipeline_registry import PipelineInfo

logger = logging.getLogger(__name__)

TASK_TYPE = "subcell"

INFO = PipelineInfo(
    name="Subcell",
    description="Synaptic signal extraction from two-photon Bergamo imaging data",
    input_patterns=["**/*.tif", "**/*.tiff"],
    output_patterns=["**/experiment.zarr"],
    input_extensions=["tif", "tiff"],
    output_extensions=["zarr"],
    marker_files=["experiment.zarr"],
    category="segmentation",
)


def task_subcell(args: dict, logger) -> None:
    """
    Worker entry point: register and extract one session.

    Runs in the process the GUI spawns, so every import stays local to keep
    the GUI's own import cost down.

    Parameters
    ----------
    args : dict
        ``input_path``, ``output_dir``, ``activity_channel``, ``settings``
        and ``reader_kwargs`` as assembled by :class:`SubcellPipelineWidget`.
    logger : logging.Logger
        Worker logger; progress goes here, not to stdout.
    """
    from mbo_utilities import imread

    from subcell._utils.torch_helpers import get_device
    from subcell.config import ExtractionConfig, RegistrationConfig
    from subcell.io.lazy_array import array_trial_source, trial_table_from_array
    from subcell.io.zarr_store import ExperimentStore
    from subcell.pipeline.runner import run_extraction
    from subcell.registration.bergamo import register_bergamo

    input_path = args["input_path"]
    output_dir = Path(args["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    arr = imread(input_path, **(args.get("reader_kwargs") or {}))
    source = array_trial_source(arr)
    trial_table = trial_table_from_array(arr, directory=input_path)
    logger.info("subcell: %d trial(s) from %s", trial_table.n_trials, input_path)

    settings = args.get("settings") or {}
    reg_config = RegistrationConfig(**settings.get("registration", {}))
    ext_config = ExtractionConfig(
        **{
            **settings.get("extraction", {}),
            "activity_channel": int(args.get("activity_channel", 1)),
        }
    )

    device = get_device(args.get("device", "auto"))
    store = ExperimentStore(output_dir / "experiment.zarr")

    logger.info("subcell: registering...")
    trial_table = register_bergamo(
        trial_table, reg_config, store, device=device, source=source
    )

    logger.info("subcell: extracting...")
    run_extraction(trial_table, ext_config, store, device=device)
    logger.info("subcell: done -> %s", output_dir / "experiment.zarr")


class SubcellPipelineWidget(PipelineWidget):
    """Run-tab widget for the subcell pipeline."""

    name = "Subcell"
    install_command = "uv pip install subcell"
    info = INFO

    task_type = TASK_TYPE
    task_func = staticmethod(task_subcell)

    # Cross-trial alignment consumes every timepoint of every trial jointly,
    # Bergamo data is single-plane, and exactly one channel is extracted
    # (all channels still contribute to the mean image).
    axes_consumed: ClassVar[dict[str, str]] = {
        "T": "all",
        "Z": "none",
        "C": "select-one",
    }

    @property
    def is_available(self) -> bool:
        """Whether subcell itself is importable."""
        try:
            import subcell  # noqa: F401
        except ImportError:
            return False
        return True

    def __init__(self, parent):
        super().__init__(parent)
        self._outdir = ""
        self._status = ""

    @classmethod
    def applies_to(cls, arr) -> bool:
        """True for single-plane arrays, which is what Bergamo produces."""
        if arr is None:
            return False
        try:
            return int(arr.shape[2]) == 1
        except (AttributeError, IndexError, TypeError, ValueError):
            return False

    def _dims(self) -> tuple[int, int]:
        """(num_timepoints, num_channels) of the loaded array."""
        arr = getattr(self.parent, "image_widget", None)
        data = getattr(arr, "data", None) if arr is not None else None
        if not data:
            return 0, 1
        shape = data[0].shape
        return int(shape[0]), int(shape[1])

    def _activity_channel(self) -> int:
        """1-based channel the user picked for extraction."""
        return int(getattr(self.parent, "_subcell_c_start", 1))

    def draw_config(self) -> None:
        """Draw the output row, the axis selection table, and the run button."""
        n_time, n_channels = self._dims()

        imgui.text("Output")
        imgui.same_line()
        imgui.set_next_item_width(320)
        _, self._outdir = imgui.input_text("##subcell_out", self._outdir)

        imgui.separator()
        tp_label, z_label, c_label = resolve_dim_labels(self.parent)
        draw_selection_table(
            self.parent,
            n_time,
            1,
            tp_attr="_subcell_tp",
            z_attr="_subcell_z",
            c_attr="_subcell_c",
            id_suffix="_subcell",
            num_channels=n_channels,
            tp_label=tp_label,
            z_label=z_label,
            c_label=c_label,
            axes=self.axes_consumed,
        )

        imgui.text_disabled(
            "All trials are aligned jointly; timepoints cannot be subset."
        )
        imgui.separator()

        ready = bool(self._outdir) and bool(getattr(self.parent, "fpath", None))
        if not ready:
            imgui.begin_disabled()
        if imgui.button("Run Subcell", imgui.ImVec2(180, 0)):
            self._submit()
        if not ready:
            imgui.end_disabled()

        if self._status:
            imgui.text(self._status)

    def _submit(self) -> None:
        """Spawn the worker for the loaded session."""
        from mbo_utilities.gui.widgets.process_manager import get_process_manager

        pm = get_process_manager()
        pid = pm.spawn(
            task_type=TASK_TYPE,
            args={
                "input_path": str(self.parent.fpath),
                "output_dir": self._outdir,
                "activity_channel": self._activity_channel(),
                "device": "auto",
                "settings": {},
                "reader_kwargs": {},
            },
            description=f"Subcell ch{self._activity_channel()}",
            output_path=self._outdir,
        )
        self._status = f"Started (PID {pid})" if pid else "Failed to start worker."
