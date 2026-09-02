"""
``SubcellVis``: the subcell spine-extraction pipeline in one NDWidget window.

One ``fastplotlib.NDWidget`` shows the field of view with two slider dims,
``trial`` and ``time``, so a single image graphic scrubs every trial of a
session and any of its layers: the raw movie, the registered movies at the
alignment and the full frame rate, the mean, activity and average images,
and the cross-trial summary images. The NMF footprints, the pixel set the
NMF started from and the source centers draw over it.

imgui panels sit on the figure's edges: sources to label, sort and filter on
the left; traces, motion, a trial table and the pipeline log on the right;
view, labels and pipeline controls on top. The pipeline card runs
registration and extraction on a background thread into a store, and the
window re-reads the store as products land, so the same object drives the
pipeline and shows what it did.

Built the way masknmf's ``ClassificationVis`` and ``SignalSelectionVis``
are: construct with the data, ``show()`` to put it on screen, ``close()``
to take it down. In a notebook construct and ``show()`` in one cell; in a
script call ``fastplotlib.loop.run()`` after ``show()``.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

try:
    import fastplotlib as fpl
    from imgui_bundle import icons_fontawesome_6 as fa
    from imgui_bundle import imgui, implot
    from imgui_bundle import portable_file_dialogs as pfd
except ImportError as error:  # pragma: no cover - import guard
    raise ImportError(
        "SubcellVis needs fastplotlib and imgui-bundle: pip install 'subcell[vis]'"
    ) from error

try:
    from masknmf.visualization.imgui import (
        UNLABEL_ALL,
        UNLABELED,
        LabelSet,
        RoiOrder,
        RowAction,
        StrokeDrawer,
        close_figure,
        draw_keybinds_popup,
        draw_label_buttons,
        draw_label_editor,
        draw_label_filter,
        draw_progress,
        draw_range_filter,
        draw_roi_table,
    )
    from masknmf.visualization.imgui.files import NATIVE_DIALOGS, PathPrompt, draw_path_prompt
    from masknmf.visualization.imgui.theme import (
        THEME,
        card,
        em,
        popup,
        set_tooltip,
        to_vec4,
        toggle_button,
    )
    from masknmf.visualization.rois import FootprintSet, build_pick_map, feathered_rgba
except ImportError as error:  # pragma: no cover - import guard
    raise ImportError(
        "SubcellVis shares its imgui panels with masknmf: pip install masknmf "
        "(or 'subcell[vis]')"
    ) from error

from subcell.config import ExtractionConfig, RegistrationConfig
from subcell.visualization._run import STAGES, PipelineRun, resolve_source
from subcell.visualization._session import (
    RawTrialMovie,
    SubcellSession,
    TrialMovieStack,
    empty_session,
    footprints_of,
    load_session,
    sel_pix_footprints,
)
from subcell.visualization._vis_utils import (
    auto_contrast,
    decimate_minmax,
    draw_model_fields,
    format_seconds,
    spans_from_mask,
    visible_slice,
)

logger = logging.getLogger(__name__)

__all__ = ["SubcellVis", "main"]

DEFAULT_LABEL_NAMES = ("spine", "shaft", "junk")
SIGNALS = ("denoised", "events", "ls", "F0")
CMAPS = ("gray", "viridis", "magma", "inferno", "turbo")
DEVICES = ("auto", "cpu", "cuda")

_CONTROLS_HEIGHT = 200
_SOURCES_WIDTH = 380
_RIGHT_WIDTH = 600
_SOURCE_COLUMNS = ("id", "label", "snr", "area", "peak")
_PLOT_POINTS = 4000
_RELOAD_EVERY_S = 2.5
_CURSOR_COLOR = imgui.ImVec4(1.0, 0.85, 0.3, 0.9)
_DISCARD_COLOR = imgui.ImVec4(1.0, 0.35, 0.35, 0.22)
_PICK_RADIUS = 4.0

_KEYBINDS = (
    ("up / down", "previous / next source in the table"),
    ("f", "center the view on the selection"),
    ("1-9", "label the selection, then advance"),
    ("0", "clear its label"),
    ("u", "next unlabeled source"),
    ("b", "toggle the footprint overlay"),
    ("m", "toggle the pixel-set overlay"),
    ("left / right", "step one frame"),
    ("space", "play / pause"),
    ("[ / ]", "previous / next trial"),
    ("esc", "empty the group"),
    ("click", "select the source under the cursor"),
    ("ctrl+click", "add the source to the group"),
)


class _TimeToFrame:
    """Reference seconds to an array index at a fixed frame rate."""

    def __init__(self, hz: float):
        self.hz = float(hz)

    def __call__(self, seconds) -> int:
        return int(float(seconds) * self.hz + 1e-6)


def _find_store(store) -> Optional[Path]:
    if store is None:
        return None
    path = Path(store)
    if path.suffix == ".zarr":
        return path
    if (path / "experiment.zarr").exists():
        return path / "experiment.zarr"
    return path


def _lighten(rgb, amount: float) -> tuple:
    return tuple(c + (1.0 - c) * amount for c in rgb[:3])


def _line_color(rgb, alpha: float = 1.0) -> imgui.ImVec4:
    r, g, b = rgb[:3]
    return imgui.ImVec4(r, g, b, alpha)


def _line_colormap(rgb) -> int:
    """
    A registered single-color colormap for one plotted line.

    This implot build has no per-line color argument, so a line takes its
    color from the pushed colormap; looked up by name so a recreated context
    re-registers rather than duplicating. Same device masknmf's viewers use.
    """
    key = tuple(int(round(float(v) * 255)) for v in rgb[:3])
    name = "subcell_line_{}_{}_{}".format(*key)
    index = implot.get_colormap_index(name)
    if index < 0:
        color = (key[0] / 255.0, key[1] / 255.0, key[2] / 255.0, 1.0)
        index = implot.add_colormap(name, np.array([color, color], np.float32))
    return int(index)


class SubcellVis:
    """
    Look at, label and run the subcell spine-extraction pipeline.

    Parameters
    ----------
    store : str or Path, optional
        An ``experiment.zarr`` the pipeline wrote, or the folder holding one.
        None opens an empty window; load a store from the pipeline card or
        :meth:`open`, or start a run with :meth:`run`.
    raw : array, LazyArray or path, optional
        The unregistered movie the store came from, shown as the ``raw``
        layers. Its trials are matched to the store's by position.
    label_names : sequence of str
        Classes the sources can be labeled with; 1-9 are hotkeys.
    extraction_config : ExtractionConfig, optional
        Censor thresholds for the motion-frame mask; defaults to what the
        store recorded.
    size : tuple of int
        Figure size in pixels. The panels are sized in pixels, so a figure
        much under 1400 wide leaves the field of view a narrow strip.
    figure_kwargs
        Passed to the figure, e.g. ``canvas="jupyter"``.
    """

    def __init__(
        self,
        store: str | Path | None = None,
        raw: Any = None,
        label_names: Sequence[str] = DEFAULT_LABEL_NAMES,
        extraction_config: ExtractionConfig | None = None,
        size: tuple[int, int] = (1700, 950),
        **figure_kwargs,
    ):
        self._store_path = _find_store(store)
        self._extraction_config = extraction_config
        self._raw_source = None
        self._raw_shape: tuple[int, int] | None = None
        if raw is not None:
            self._set_raw(raw)

        self._session = self._load(self._store_path) if self._store_path else empty_session()

        # layers
        self._layers: dict[str, Any] = {}
        self._layer_kind: dict[str, str] = {}
        self._layer_names: list[str] = []
        self._layer: Optional[str] = None
        self._contrast: dict[str, tuple[float, float]] = {}
        self._cmap: dict[str, str] = {}

        # overlays and sources
        self._fp_overlay = None
        self._mask_overlay = None
        self._markers = None
        self._sel_marker = None
        self._footprint_sets: dict[int, Optional[FootprintSet]] = {}
        self._disks: Optional[FootprintSet] = None
        self._trial = 0
        self._selected = -1
        self._group: list[int] = []
        self._show_fp = True
        self._fp_opacity = 0.55
        self._show_disks = False
        self._show_mask = False
        self._show_centers = True
        self._follow = False
        self._scroll_to_selection = False
        self._label_names = tuple(label_names)
        self._labels = LabelSet(0, self._label_names)
        self._order = RoiOrder({"snr": np.zeros(0, np.int64)}, self._labels.labels, 0)
        self._new_label = ""
        self._columns: dict[str, np.ndarray] = {}
        self._snr_float = np.zeros(0, np.float32)

        # traces
        self._signals = {"denoised": True, "events": True, "ls": False, "F0": False}
        self._per_f0 = False
        self._autofit = True
        self._trace_fit = True
        self._force_fit = False
        self._plot_key = None
        self._spans: dict[int, np.ndarray] = {}
        self._tab = "traces"

        # pipeline
        self._run: Optional[PipelineRun] = None
        self._run_input = ""
        self._run_output = str(self._store_path.parent) if self._store_path else ""
        self._run_stages = {stage: True for stage in STAGES}
        self._device_idx = 0
        self._reg_config = RegistrationConfig(save_full_resolution=True)
        self._ext_config = (
            extraction_config.model_copy() if extraction_config is not None else ExtractionConfig()
        )
        self._last_reload = 0.0
        self._settings_open = False
        self._keybinds_open = False
        self._open_prompt = PathPrompt(
            "Open experiment.zarr", action="open", hint="a subcell experiment.zarr or its folder, read by this process"
        )
        self._file_dialog = None
        self._status = ""
        self._error: Optional[str] = None

        self._build_figure(size, figure_kwargs)
        self._apply_session(self._session)
        self._drawer = StrokeDrawer(self._fov, lambda stroke: None, self._pick)

        figure = self._ndw.figure
        figure.add_imgui_window(self._draw_controls, location="top", size=_CONTROLS_HEIGHT, title="Subcell")
        figure.add_imgui_window(self._draw_sources_panel, location="left", size=_SOURCES_WIDTH, title="Sources")
        figure.add_imgui_window(self._draw_right_panel, location="right", size=_RIGHT_WIDTH, title="Traces")
        for subplot in figure:
            subplot.tooltip.enabled = False
            subplot.toolbar = False
        self._status = self._describe()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> SubcellSession:
        return load_session(path, extraction_config=self._extraction_config, min_canvas=self._raw_shape)

    def _set_raw(self, raw) -> None:
        resolved = resolve_source(raw)
        if resolved.array_source is None:
            raise ValueError("raw must be an array or a path an array reader opens, not TIFFs")
        self._raw_source = resolved.array_source
        shape = self._raw_source.array.shape
        self._raw_shape = (int(shape[-2]), int(shape[-1]))

    def _build_figure(self, size, figure_kwargs) -> None:
        session = self._session
        ranges = {
            "trial": (0, max(1, session.n_trials), 1),
            "time": (0.0, max(session.max_duration_s, 1e-3), 1.0 / max(session.full_hz, 1e-6)),
        }
        self._ndw = fpl.NDWidget(ranges, shape=(1, 1), names=["fov"], size=size, **figure_kwargs)
        self._fov = self._ndw.figure["fov"]
        self._ndw.indices.add_event_handler(self._on_indices)

    def _layer_specs(self) -> dict[str, tuple[str, Any, tuple, Optional[float]]]:
        """name -> (kind, data, dims, hz) for every layer the session offers."""
        session = self._session
        specs: dict[str, tuple] = {}
        if self._raw_source is not None and session.n_trials:
            src = self._raw_source
            n_ch = int(src.array.shape[1])
            for ch in range(n_ch):
                movies = []
                for pos in range(session.n_trials):
                    movies.append(RawTrialMovie(src.array, src.trial_slices[pos], ch) if pos < src.n_trials else None)
                stack = TrialMovieStack(movies, session.canvas, n_time=session.max_raw_frames)
                specs[f"raw · ch{ch + 1}"] = ("movie", stack, ("trial", "time", "m", "n"), session.full_hz)
        for name in session.movie_names():
            hz = session.movie_hz(name, 0) if session.n_trials else session.full_hz
            rates = [session.movie_hz(name, k) for k in range(session.n_trials) if name in session.trials[k].movies]
            hz = float(np.median(rates)) if rates else hz
            specs[name] = ("movie", session.movie_stack(name), ("trial", "time", "m", "n"), hz)
        for name in session.image_names():
            specs[name] = ("image", session.image_stack(name), ("trial", "m", "n"), None)
        for name, img in session.summary_images.items():
            canvas = np.full(session.canvas, np.nan, np.float32)
            canvas[: img.shape[0], : img.shape[1]] = img
            specs[name] = ("summary", canvas, ("m", "n"), None)
        if not specs:
            specs["empty"] = ("summary", np.zeros(session.canvas, np.float32), ("m", "n"), None)
        return specs

    def _remove_layer(self, name: str) -> None:
        nd = self._layers.pop(name)
        self._layer_kind.pop(name, None)
        nd_subplot = self._ndw["fov"]
        if nd.graphic is not None:
            try:
                self._fov.delete_graphic(nd.graphic)
            except Exception:  # noqa: BLE001 - already gone
                pass
        try:
            nd_subplot._nd_graphics.remove(nd)
        except ValueError:
            pass

    def _sync_layers(self) -> None:
        """Create, replace or drop the NDImages so they match the session."""
        specs = self._layer_specs()
        for name in [n for n in self._layers if n not in specs]:
            self._remove_layer(name)
        nd_subplot = self._ndw["fov"]
        for name, (kind, data, dims, hz) in specs.items():
            transforms = {"time": _TimeToFrame(hz)} if hz else None
            nd = self._layers.get(name)
            if nd is not None and tuple(nd.dims) == tuple(dims):
                nd.pause = False
                nd.data = data
                if transforms:
                    nd.slider_dim_transforms = dict(transforms)
            else:
                if nd is not None:
                    self._remove_layer(name)
                nd = nd_subplot.add_nd_image(
                    data,
                    list(dims),
                    ["m", "n"],
                    slider_dim_transforms=dict(transforms) if transforms else None,
                    compute_histogram=False,
                    name=name,
                )
                self._layers[name] = nd
            self._layer_kind[name] = kind
            nd.graphic.visible = False
            nd.pause = True
        self._layer_names = list(specs)
        self._contrast = {k: v for k, v in self._contrast.items() if k in specs}
        current = self._layer if self._layer in specs else self._default_layer()
        self._layer = None
        self.set_layer(current)

    def _default_layer(self) -> str:
        names = self._layer_names
        preferred = f"registered ds · ch{self._session.extraction_config.activity_channel}"
        if preferred in names:
            return preferred
        for name in names:
            if self._layer_kind[name] == "movie" and not name.startswith("raw"):
                return name
        return names[0]

    def _build_overlays(self) -> None:
        for graphic in (self._fp_overlay, self._mask_overlay, self._markers, self._sel_marker):
            if graphic is not None:
                try:
                    self._fov.delete_graphic(graphic)
                except Exception:  # noqa: BLE001 - already gone
                    pass
        h, w = self._session.canvas
        blank = np.zeros((h, w, 4), np.uint8)
        self._mask_overlay = self._fov.add_image(blank, name="pixel_set", alpha_mode="blend", offset=(0, 0, 1.0))
        self._fp_overlay = self._fov.add_image(blank.copy(), name="footprints", alpha_mode="blend", offset=(0, 0, 1.5))
        for overlay in (self._mask_overlay, self._fp_overlay):
            overlay.vmin, overlay.vmax = 0, 255
            for tile in overlay.world_object.children:
                tile.material.pick_write = False
        self._markers = None
        self._sel_marker = None
        data = self._session.extraction
        if data is not None and data.n_sources:
            pts = np.column_stack([data.source_cols, data.source_rows, np.full(data.n_sources, 2.0)]).astype(np.float32)
            # a (n, 4) array with n <= 4 reads as one color to fastplotlib, so
            # start uniform in vertex mode and paint per-source colors after
            self._markers = self._fov.add_scatter(pts, colors="w", color_mode="vertex", sizes=5, name="centers")
            self._sel_marker = self._fov.add_scatter(
                np.array([[0.0, 0.0, 2.5]], np.float32), colors="yellow", sizes=18, markers="ring", name="selected"
            )
            self._sel_marker.visible = False
            for graphic in (self._markers, self._sel_marker):
                graphic.world_object.material.pick_write = False

    def _apply_session(self, session: SubcellSession) -> None:
        """Point every panel at a (re)loaded session."""
        self._session = session
        self._footprint_sets.clear()
        self._spans.clear()
        self._disks = None
        ranges = self._ndw.ranges
        ranges["trial"].stop = max(1, session.n_trials)
        ranges["time"].stop = max(session.max_duration_s, 1e-3)
        self._trial = min(self._trial, max(session.n_trials - 1, 0))
        self._sync_layers()
        self._build_overlays()
        self._rebuild_sources()
        self._ndw.indices.set({"trial": self._trial, "time": min(self.current_time(), ranges["time"].stop - ranges["time"].step)})
        self._refresh_overlays()
        self._trace_fit = True

    def _describe(self) -> str:
        s = self._session
        if not s.n_trials:
            return "no store loaded: open one, or run the pipeline"
        raw = sum(1 for t in s.trials if any(n.startswith("registered raw") for n in t.movies))
        parts = [f"{s.path.name}: {s.n_trials} trial(s)", f"{raw} with full-rate movies"]
        parts.append(f"{s.n_sources} source(s)" if s.extraction is not None else "no extraction yet")
        return " · ".join(parts)

    # ------------------------------------------------------------------
    # public state
    # ------------------------------------------------------------------

    @property
    def session(self) -> SubcellSession:
        return self._session

    @property
    def store_path(self) -> Optional[Path]:
        return self._store_path

    @property
    def fov_widget(self) -> fpl.NDWidget:
        return self._ndw

    @property
    def figure(self):
        return self._ndw.figure

    @property
    def layer(self) -> Optional[str]:
        """Name of the layer the field of view shows."""
        return self._layer

    @property
    def layer_names(self) -> list[str]:
        return list(self._layer_names)

    @property
    def selected(self) -> int:
        """Selected source index, -1 for none."""
        return self._selected

    @property
    def group(self) -> list[int]:
        return list(self._group)

    @property
    def labels(self) -> np.ndarray:
        return self._labels.labels

    @property
    def label_names(self) -> tuple:
        return self._labels.names

    @property
    def pipeline(self) -> Optional[PipelineRun]:
        """The run in progress or last finished, if any."""
        return self._run

    @property
    def n_sources(self) -> int:
        return self._session.n_sources

    def current_trial(self) -> int:
        return self._trial

    def set_trial(self, position: int) -> None:
        n = max(self._session.n_trials, 1)
        self._ndw.indices.set_dim_index("trial", int(np.clip(position, 0, n - 1)))

    def current_time(self) -> float:
        return float(self._ndw.indices["time"])

    def set_time(self, seconds: float) -> None:
        self._ndw.indices.set_dim_index("time", float(seconds))

    def current_frame(self) -> int:
        """Frame at the full frame rate of the current trial."""
        hz = self._trial_hz()
        return int(self.current_time() * hz + 1e-6)

    def set_frame(self, frame: int) -> None:
        self.set_time(int(frame) / self._trial_hz())

    def set_layer(self, name: str) -> None:
        """Show one layer and pause the others, so only it fetches frames."""
        if name == self._layer or name not in self._layers:
            return
        if self._layer is not None and self._layer in self._layers:
            old = self._layers[self._layer]
            old.graphic.visible = False
            old.pause = True
        new = self._layers[name]
        new.pause = False
        new.graphic.visible = True
        self._layer = name
        self._fov.title = name
        self._apply_contrast(name)
        self._ndw.indices.set({dim: value for dim, value in self._ndw.indices})

    def show(self, **kwargs):
        return self._ndw.figure.show(**kwargs)

    def close(self) -> None:
        try:
            self._drawer.close()
        except Exception:  # noqa: BLE001 - the renderer may be gone already
            pass
        close_figure(self._ndw.figure)

    # ------------------------------------------------------------------
    # loading and running
    # ------------------------------------------------------------------

    def open(self, store: str | Path) -> None:
        """Load a store into the window."""
        path = _find_store(store)
        session = self._load(path)
        self._store_path = path
        self._run_output = str(path.parent)
        self._selected = -1
        self._group.clear()
        self._apply_session(session)
        self._status = self._describe()
        self._error = None

    def reload(self) -> None:
        """Re-read the store, keeping the selection where it can."""
        if self._store_path is None or not self._store_path.exists():
            return
        try:
            session = self._load(self._store_path)
        except Exception as error:  # noqa: BLE001 - a half-written store must not kill the window
            self._error = f"reload failed: {type(error).__name__}: {error}"
            return
        keep = self._selected
        self._apply_session(session)
        self.select(keep if 0 <= keep < self.n_sources else -1)
        self._status = self._describe()
        self._last_reload = time.perf_counter()

    def run(
        self,
        source: Any = None,
        output: str | Path | None = None,
        registration: RegistrationConfig | None = None,
        extraction: ExtractionConfig | None = None,
        stages: Sequence[str] = STAGES,
        device: str | None = None,
        fs: float | None = None,
        frames_per_trial: Sequence[int] | None = None,
    ) -> PipelineRun:
        """
        Register and extract on a background thread, watching the store fill.

        Parameters
        ----------
        source
            TIFF folder, TIFF, LazyArray, or movie array; the ``raw`` the
            window was built with when omitted.
        output : str or Path, optional
            Folder for ``experiment.zarr``; the loaded store's folder, else
            ``./subcell_output``.
        registration, extraction : configs, optional
            Stage parameters; the pipeline card's settings when omitted.
        stages : sequence of str
            ``"register"``, ``"extract"``, or both.
        """
        if self._run is not None and self._run.running:
            raise RuntimeError("a run is already in progress")
        if source is None:
            source = self._raw_source.array if self._raw_source is not None else (self._run_input or None)
        if source is None:
            raise ValueError("nothing to run on: pass a source or build the window with raw=")
        if output is None:
            output = self._run_output or (self._store_path.parent if self._store_path else Path.cwd() / "subcell_output")
        registration = registration or self._reg_config
        extraction = extraction or self._ext_config
        if "extract" in stages and not registration.save_full_resolution:
            registration = registration.model_copy(update={"save_full_resolution": True})
        run = PipelineRun(
            source,
            output,
            registration=registration,
            extraction=extraction,
            stages=stages,
            device=device or DEVICES[self._device_idx],
            fs=fs,
            frames_per_trial=frames_per_trial,
        )
        self._run = run
        self._run_output = str(Path(output))
        self._store_path = run.store_path
        self._error = None
        run.start()
        self._status = f"running: {run.message}"
        if isinstance(source, (str, Path)):
            self._run_input = str(source)
        return run

    def _poll_run(self) -> None:
        run = self._run
        if run is None:
            return
        if run.products_changed.is_set() and time.perf_counter() - self._last_reload > _RELOAD_EVERY_S:
            run.products_changed.clear()
            self.reload()
        if run.running:
            self._status = f"{run.stage}: {run.message} ({format_seconds(run.elapsed)})"
        elif run.status in ("done", "cancelled", "error") and not getattr(run, "_reported", False):
            run._reported = True
            self.reload()
            self._status = f"{run.status} in {format_seconds(run.elapsed)}: {self._describe()}"
            self._error = run.error

    # ------------------------------------------------------------------
    # sources
    # ------------------------------------------------------------------

    def _labels_path(self) -> Optional[Path]:
        if self._store_path is None:
            return None
        return self._store_path.parent / f"{self._store_path.stem}.labels.npz"

    def _rebuild_sources(self) -> None:
        n = self.n_sources
        labels = None
        names = self._labels.names if len(self._labels.names) else self._label_names
        path = self._labels_path()
        if path is not None and path.exists():
            try:
                saved = np.load(path, allow_pickle=False)
                if saved["labels"].shape[0] == n:
                    labels = saved["labels"].astype(np.int64)
                    names = tuple(str(v) for v in saved["label_names"]) or names
            except Exception:  # noqa: BLE001 - a stale sidecar is ignored
                logger.exception("could not read %s", path)
        if labels is None and self._labels.labels.shape[0] == n:
            labels = self._labels.labels
        self._labels = LabelSet(n, names, labels)
        self._refresh_columns()
        self._order = RoiOrder(self._columns, self._labels.labels, n)
        self._order.set_range_column("area")
        self._order.rebuild()
        if not 0 <= self._selected < n:
            self._selected = -1
        self._group = [k for k in self._group if 0 <= k < n]

    def _refresh_columns(self) -> None:
        """Per-source columns for the current trial."""
        n = self.n_sources
        snr = np.full(n, np.nan, np.float32)
        area = np.zeros(n, np.int64)
        peak = np.zeros(n, np.float32)
        fs = self._footprint_set(self._trial)
        trial = self._trial_extraction()
        if trial is not None:
            snr = np.asarray(trial.snr, np.float32)
            with np.errstate(all="ignore"):
                peak = np.nan_to_num(np.nanmax(trial.events, axis=1), nan=0.0).astype(np.float32)
        if fs is not None:
            area = np.array([fs.area(k) for k in range(len(fs))], np.int64)
        self._snr_float = snr
        self._columns = {
            "snr": np.nan_to_num(np.round(snr), nan=0.0).astype(np.int64),
            "area": area,
            "peak": np.nan_to_num(peak, nan=0.0),
        }
        if hasattr(self, "_order") and self._order.n_items == n:
            self._order.columns = self._columns
            self._order.refresh_range()
            self._order.rebuild()

    def _trial_extraction(self):
        if not self._session.n_trials or self._session.extraction is None:
            return None
        record = self._session.trials[self._trial]
        return self._session.extraction.trials.get(record.index)

    def _trial_hz(self) -> float:
        if not self._session.n_trials:
            return max(self._session.full_hz, 1e-6)
        return max(self._session.trials[self._trial].full_hz, 1e-6)

    def _footprint_set(self, trial: int) -> Optional[FootprintSet]:
        if trial in self._footprint_sets:
            fs = self._footprint_sets[trial]
        else:
            fps = footprints_of(self._session, trial) if self._session.n_trials else None
            fs = None
            if fps is not None:
                fs = FootprintSet(f"trial {trial}", fps, build_pick_map(fps, self._session.canvas))
            self._footprint_sets[trial] = fs
        if fs is not None:
            fs.classes = {k: int(v) for k, v in enumerate(self._labels.labels) if v >= 0}
        return fs

    def _disk_set(self) -> Optional[FootprintSet]:
        if self._disks is None:
            fps = sel_pix_footprints(self._session)
            if fps is not None:
                self._disks = FootprintSet("initial disks", fps, build_pick_map(fps, self._session.canvas))
        return self._disks

    def _formatters(self) -> dict:
        return {
            "snr": lambda k: "-" if not np.isfinite(self._snr_float[k]) else f"{self._snr_float[k]:.1f}",
            "area": lambda k: f"{self._columns['area'][k]}",
            "peak": lambda k: f"{self._columns['peak'][k]:.3g}",
        }

    # ------------------------------------------------------------------
    # selection
    # ------------------------------------------------------------------

    def select(self, index: int, center: bool = False) -> None:
        """Select one source; anything out of range clears the selection."""
        self._selected = int(index) if 0 <= int(index) < self.n_sources else -1
        if self._selected >= 0:
            cleared = self._order.reveal(self._selected)
            if cleared:
                self._status = f"selected source {self._selected}; cleared filter {', '.join(cleared)}"
            self._scroll_to_selection = True
        self._trace_fit = True
        self._refresh_overlays()
        if center or self._follow:
            self.center_selection()

    def toggle_group(self, index: int) -> None:
        if not 0 <= index < self.n_sources:
            return
        if index in self._group:
            self._group.remove(index)
        else:
            self._group.append(index)
        self._trace_fit = True
        self._refresh_overlays()

    def clear_group(self) -> None:
        if self._group:
            self._group.clear()
            self._trace_fit = True
            self._refresh_overlays()

    def plotted_sources(self) -> list[int]:
        """Sources the trace panel draws: the selection, then the group."""
        out = [self._selected] if self._selected >= 0 else []
        out += [k for k in self._group if k != self._selected]
        return out

    def step(self, delta: int) -> None:
        if self._order.step(delta) and self._order.current is not None:
            self.select(self._order.current)

    def next_unlabeled(self) -> None:
        if self._order.next_unlabeled() and self._order.current is not None:
            self.select(self._order.current)

    def center_selection(self) -> None:
        data = self._session.extraction
        if data is None or self._selected < 0:
            return
        fs = self._footprint_set(self._trial)
        if fs is not None and fs.area(self._selected):
            ypix, xpix, _lam = fs.footprints[self._selected]
            y0, y1 = float(ypix.min()), float(ypix.max())
            x0, x1 = float(xpix.min()), float(xpix.max())
        else:
            y0 = y1 = float(data.source_rows[self._selected])
            x0 = x1 = float(data.source_cols[self._selected])
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        half = max(max(y1 - y0, x1 - x0, 1.0) * 3.0, 20.0)
        self._fov.camera.show_rect(cx - half, cx + half, cy - half, cy + half)

    def _pick(self, row: int, col: int, mods: frozenset = frozenset()) -> None:
        data = self._session.extraction
        if data is None:
            return
        hit = -1
        fs = self._footprint_set(self._trial)
        h, w = self._session.canvas
        if fs is not None and self._show_fp and 0 <= row < h and 0 <= col < w:
            hit = int(fs.pick_map[row, col])
        if hit < 0:
            d = np.hypot(data.source_rows - row, data.source_cols - col)
            k = int(np.argmin(d))
            if d[k] <= _PICK_RADIUS:
                hit = k
        if "Ctrl" in mods:
            if hit >= 0:
                self.toggle_group(hit)
            return
        self.clear_group()
        self.select(hit)

    # ------------------------------------------------------------------
    # labels
    # ------------------------------------------------------------------

    def assign_class(self, class_index: int, advance: bool = True) -> None:
        """Label the selection and the group; UNLABELED clears them."""
        targets = self.plotted_sources()
        if not targets:
            return
        self._labels.assign(targets, class_index)
        self._after_labels_changed()
        if advance and class_index != UNLABELED and len(targets) == 1:
            self.step(1)

    def add_label(self, name: str) -> bool:
        added = self._labels.add(name)
        if added:
            self._after_labels_changed()
        return added

    def _after_labels_changed(self) -> None:
        self._order.labels = self._labels.labels
        self._order.rebuild()
        self._footprint_set(self._trial)
        self._refresh_overlays()
        self.save_labels()

    def save_labels(self, path: str | Path | None = None) -> Optional[Path]:
        """Write labels and names beside the store; None when there is no store."""
        path = Path(path) if path is not None else self._labels_path()
        if path is None or not self.n_sources:
            return None
        try:
            np.savez(
                path,
                labels=self._labels.labels,
                label_names=np.array(self._labels.names, dtype=str),
                source_rows=self._session.extraction.source_rows,
                source_cols=self._session.extraction.source_cols,
            )
        except OSError as error:
            self._error = f"label save failed: {error}"
            return None
        return path

    # ------------------------------------------------------------------
    # overlays and contrast
    # ------------------------------------------------------------------

    def _refresh_overlays(self) -> None:
        if self._fp_overlay is None:
            return
        h, w = self._session.canvas
        fs = self._footprint_set(self._trial)
        if fs is not None and self._show_fp:
            rgba = fs.rgba((h, w), self._fp_opacity, self._selected if self._selected >= 0 else None, set(self._group))
        else:
            rgba = np.zeros((h, w, 4), np.uint8)
        self._fp_overlay.data = rgba

        mask = np.zeros((h, w, 4), np.uint8)
        data = self._session.extraction
        if data is not None and self._show_mask:
            hh, ww = data.sel_pix.shape
            mask[:hh, :ww][data.sel_pix] = (255, 255, 255, 55)
        if data is not None and self._show_disks:
            disks = self._disk_set()
            if disks is not None:
                comps = [(y, x, lam, (0.4, 0.9, 1.0), 0.3) for y, x, lam in disks.footprints]
                disk_rgba = feathered_rgba((h, w), comps)
                win = disk_rgba[..., 3] > mask[..., 3]
                mask[win] = disk_rgba[win]
        self._mask_overlay.data = mask

        if self._markers is not None:
            n = self.n_sources
            colors = np.ones((n, 4), np.float32)
            if fs is not None:
                for k in range(n):
                    colors[k, :3] = fs.color(k)
            colors[:, 3] = 0.9 if self._show_centers else 0.0
            self._markers.colors[:] = colors
        if self._sel_marker is not None:
            if self._selected >= 0 and data is not None:
                self._sel_marker.data[0, :2] = (data.source_cols[self._selected], data.source_rows[self._selected])
                self._sel_marker.visible = True
            else:
                self._sel_marker.visible = False

    def _sample_layer(self, name: str) -> np.ndarray:
        nd = self._layers[name]
        data = nd.data
        kind = self._layer_kind[name]
        if kind == "movie":
            n = data.shape[1]
            idx = np.unique(np.linspace(0, max(n - 1, 0), 6).astype(int))
            return np.stack([data[self._trial, int(t)] for t in idx])
        if kind == "image":
            return np.asarray(data[self._trial])
        return np.asarray(data)

    def _apply_contrast(self, name: str) -> None:
        nd = self._layers[name]
        if name not in self._contrast:
            self._contrast[name] = auto_contrast(self._sample_layer(name))
        vmin, vmax = self._contrast[name]
        nd.graphic.vmin, nd.graphic.vmax = vmin, vmax
        nd.graphic.cmap = self._cmap.get(name, "viridis" if "activity" in name else "gray")

    def auto_contrast(self, name: str | None = None) -> None:
        name = name or self._layer
        if name in self._layers:
            self._contrast.pop(name, None)
            self._apply_contrast(name)

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------

    def _on_indices(self, indices: dict) -> None:
        trial = int(indices.get("trial", 0))
        if trial != self._trial and self._session.n_trials:
            self._trial = int(np.clip(trial, 0, self._session.n_trials - 1))
            self._refresh_columns()
            self._refresh_overlays()
            self._trace_fit = True

    def _handle_keys(self) -> None:
        io = imgui.get_io()
        if io.want_text_input:
            return
        if imgui.is_key_pressed(imgui.Key.escape, False):
            self.clear_group()
        if imgui.is_key_pressed(imgui.Key.down_arrow, True):
            self.step(1)
        if imgui.is_key_pressed(imgui.Key.up_arrow, True):
            self.step(-1)
        if imgui.is_key_pressed(imgui.Key.right_arrow, True):
            self.set_frame(self.current_frame() + 1)
        if imgui.is_key_pressed(imgui.Key.left_arrow, True):
            self.set_frame(self.current_frame() - 1)
        if imgui.is_key_pressed(imgui.Key.left_bracket, False):
            self.set_trial(self._trial - 1)
        if imgui.is_key_pressed(imgui.Key.right_bracket, False):
            self.set_trial(self._trial + 1)
        if io.key_ctrl:
            return
        if imgui.is_key_pressed(imgui.Key.space, False):
            self.toggle_play()
        if imgui.is_key_pressed(imgui.Key.f, False):
            self._follow = not self._follow
            if self._follow:
                self.center_selection()
        if imgui.is_key_pressed(imgui.Key.u, False):
            self.next_unlabeled()
        if imgui.is_key_pressed(imgui.Key.b, False):
            self._show_fp = not self._show_fp
            self._refresh_overlays()
        if imgui.is_key_pressed(imgui.Key.m, False):
            self._show_mask = not self._show_mask
            self._refresh_overlays()
        picked = self._labels.hotkey_pressed()
        if picked is not None:
            self.assign_class(picked)

    def toggle_play(self) -> None:
        ui = getattr(self._ndw, "_sliders_ui", None)
        playing = getattr(ui, "_playing", None)
        if isinstance(playing, dict) and "time" in playing:
            playing["time"] = not playing["time"]
            if playing["time"]:
                ui._last_frame_time["time"] = 0.0

    # ------------------------------------------------------------------
    # imgui: top controls
    # ------------------------------------------------------------------

    def _draw_controls(self) -> None:
        self._poll_run()
        self._poll_file_dialog()
        self._handle_keys()
        gap = em(0.6)
        avail = imgui.get_content_region_avail().x
        height = max(imgui.get_content_region_avail().y - em(1.6), em(6))
        w_view = max(avail * 0.36, em(20))
        w_labels = max(avail * 0.27, em(16))
        w_run = max(avail - w_view - w_labels - 2 * gap, em(18))
        self._draw_view_card(height, w_view)
        imgui.same_line(0, gap)
        self._draw_labels_card(height, w_labels)
        imgui.same_line(0, gap)
        self._draw_pipeline_card(height, w_run)
        self._draw_status()
        self._keybinds_open = draw_keybinds_popup(_KEYBINDS, self._keybinds_open)
        self._draw_settings_popup()
        self._draw_open_prompt()

    def _draw_view_card(self, height: float, width: float) -> None:
        with card("##view", "VIEW", height, width):
            imgui.set_next_item_width(em(13))
            idx = self._layer_names.index(self._layer) if self._layer in self._layer_names else 0
            changed, idx = imgui.combo("layer", idx, self._layer_names)
            if changed:
                self.set_layer(self._layer_names[idx])
            set_tooltip("what the field of view shows; movies fetch one frame per slider move")
            imgui.same_line(0, em(0.6))
            cmaps = list(CMAPS)
            current = self._cmap.get(self._layer, "viridis" if self._layer and "activity" in self._layer else "gray")
            imgui.set_next_item_width(em(6))
            changed, cidx = imgui.combo("##cmap", cmaps.index(current) if current in cmaps else 0, cmaps)
            if changed and self._layer:
                self._cmap[self._layer] = cmaps[cidx]
                self._apply_contrast(self._layer)

            if self._layer in self._contrast:
                vmin, vmax = self._contrast[self._layer]
                span = max(abs(vmax - vmin), 1e-6)
                imgui.set_next_item_width(em(13))
                changed, lo, hi = imgui.drag_float_range2(
                    "##contrast", vmin, vmax, span / 200.0, 0.0, 0.0, "min %.3g", "max %.3g"
                )
                if changed and hi > lo:
                    self._contrast[self._layer] = (lo, hi)
                    self._apply_contrast(self._layer)
                imgui.same_line(0, em(0.4))
                if imgui.button("auto"):
                    self.auto_contrast()
                set_tooltip("1st to 99.5th percentile of a few frames of this trial")

            dirty = False
            changed, self._show_fp = imgui.checkbox("footprints", self._show_fp)
            dirty |= changed
            imgui.same_line(0, em(0.3))
            imgui.text_disabled("(b)")
            imgui.same_line(0, em(0.6))
            imgui.set_next_item_width(em(6))
            changed, self._fp_opacity = imgui.slider_float("##fp_opacity", self._fp_opacity, 0.05, 1.0, "%.2f")
            dirty |= changed
            changed, self._show_disks = imgui.checkbox("initial disks", self._show_disks)
            dirty |= changed
            set_tooltip("the pixel disk around each source the NMF started from")
            imgui.same_line(0, em(0.6))
            changed, self._show_mask = imgui.checkbox("pixel set", self._show_mask)
            dirty |= changed
            imgui.same_line(0, em(0.3))
            imgui.text_disabled("(m)")
            imgui.same_line(0, em(0.6))
            changed, self._show_centers = imgui.checkbox("centers", self._show_centers)
            dirty |= changed
            if dirty:
                self._refresh_overlays()

            if self._session.n_trials:
                record = self._session.trials[self._trial]
                imgui.text_disabled(
                    f"trial {self._trial + 1}/{self._session.n_trials} (store #{record.index}) · "
                    f"{record.n_raw_frames} frames @ {record.full_hz:.0f} Hz · "
                    f"t = {self.current_time():.3f} s, frame {self.current_frame()}"
                )
                if record.discard_full is not None and record.discard_full.size:
                    frame = min(self.current_frame(), record.discard_full.size - 1)
                    if record.discard_full[frame]:
                        imgui.same_line(0, em(0.6))
                        imgui.text_colored(to_vec4(THEME.err), f"{fa.ICON_FA_TRIANGLE_EXCLAMATION} censored frame")

    def _draw_labels_card(self, height: float, width: float) -> None:
        with card("##labels", "SOURCES", height, width):
            if not self.n_sources:
                imgui.text_disabled("no sources: extraction has not run")
                return
            if draw_progress(self._labels, "_src"):
                self.next_unlabeled()
            picked = draw_label_buttons(self._labels, "_src")
            if picked == UNLABEL_ALL:
                self._labels.clear()
                self._after_labels_changed()
            elif picked is not None:
                self.assign_class(picked)
            self._new_label, changed = draw_label_editor(self._labels, self._new_label, "_src")
            if changed:
                self._after_labels_changed()
            k = self._selected
            if k >= 0:
                snr = self._snr_float[k]
                imgui.text(
                    f"source {k} · snr {snr:.1f} · {self._columns['area'][k]} px · "
                    f"{self._labels.name_of(k)}"
                )
            else:
                imgui.text_disabled("click a source, or pick one in the table")
            if self._group:
                imgui.same_line(0, em(0.8))
                imgui.text_colored(to_vec4(THEME.warn), f"group of {len(self._group)}")
            with toggle_button(self._follow):
                if imgui.button("follow (f)"):
                    self._follow = not self._follow
                    if self._follow:
                        self.center_selection()
            set_tooltip("keep the camera on the selection as it changes")
            imgui.same_line(0, em(0.5))
            if imgui.button("keybinds"):
                self._keybinds_open = True
            imgui.same_line(0, em(0.5))
            if imgui.button("save labels"):
                path = self.save_labels()
                self._status = f"labels saved to {path}" if path else "no store to save beside"

    def _draw_pipeline_card(self, height: float, width: float) -> None:
        run = self._run
        busy = run is not None and run.running
        with card("##pipeline", "PIPELINE", height, width):
            imgui.set_next_item_width(-em(8))
            _, self._run_input = imgui.input_text_with_hint(
                "##input", "TIFF folder, TIFF, or array path", self._run_input
            )
            set_tooltip(
                "what to register: a folder of per-trial ScanImage TIFFs, one TIFF, or a path "
                "mbo_utilities reads. Leave empty to use the raw array the window was built with."
            )
            imgui.same_line(0, em(0.4))
            imgui.text_disabled("input")
            imgui.set_next_item_width(-em(8))
            _, self._run_output = imgui.input_text_with_hint("##output", "output folder", self._run_output)
            imgui.same_line(0, em(0.4))
            imgui.text_disabled("output")
            for stage in STAGES:
                _, self._run_stages[stage] = imgui.checkbox(stage, self._run_stages[stage])
                imgui.same_line(0, em(0.8))
            imgui.set_next_item_width(em(5))
            _, self._device_idx = imgui.combo("##device", self._device_idx, list(DEVICES))
            imgui.same_line(0, em(0.8))
            if imgui.button("settings"):
                self._settings_open = True
            set_tooltip("registration and extraction parameters, from the pydantic configs")

            stages = [s for s in STAGES if self._run_stages[s]]
            can_run = bool(stages) and not busy and bool(self._run_input or self._raw_source is not None)
            if busy:
                if imgui.button("cancel"):
                    run.cancel()
            else:
                if not can_run:
                    imgui.begin_disabled()
                if imgui.button(f"{fa.ICON_FA_PLAY} run"):
                    try:
                        self.run(stages=stages)
                    except Exception as error:  # noqa: BLE001 - shown in the status row
                        self._error = f"{type(error).__name__}: {error}"
                if not can_run:
                    imgui.end_disabled()
            imgui.same_line(0, em(0.6))
            if imgui.button("open store"):
                self._open_prompt.start(self._store_path.parent if self._store_path else os.getcwd())
            imgui.same_line(0, em(0.6))
            if imgui.button("reload"):
                self.reload()
            if run is not None:
                color = THEME.warn if busy else (THEME.err if run.status == "error" else THEME.ok)
                imgui.text_colored(to_vec4(color), f"{run.status}: {run.stage} · {run.message}")
                if busy and run.stage == "register" and run.n_trials:
                    imgui.progress_bar(run.trials_done / run.n_trials, imgui.ImVec2(-1, em(0.8)), f"{run.trials_done}/{run.n_trials} trials")

    def _draw_status(self) -> None:
        if self._error:
            imgui.text_colored(to_vec4(THEME.err), f"{fa.ICON_FA_TRIANGLE_EXCLAMATION} {self._error}")
            imgui.same_line(0, em(1))
        imgui.text_colored(to_vec4(THEME.text_dim), self._status)

    def _draw_settings_popup(self) -> None:
        if not self._settings_open:
            return
        opened, self._settings_open = popup("Pipeline settings", self._settings_open)
        if opened:
            imgui.text_disabled("edits apply to the next run")
            if imgui.begin_table("##settings", 2, imgui.TableFlags_.borders_inner_v):
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_colored(to_vec4(THEME.accent), "registration")
                draw_model_fields(self._reg_config, "_reg")
                imgui.table_next_column()
                imgui.text_colored(to_vec4(THEME.accent), "extraction")
                if draw_model_fields(self._ext_config, "_ext", skip=("microscope", "device")):
                    self._session.extraction_config = self._ext_config
                imgui.end_table()
            if imgui.button("close", imgui.ImVec2(em(6), 0)):
                self._settings_open = False
        imgui.end()

    def _draw_open_prompt(self) -> None:
        submitted, browse = draw_path_prompt(self._open_prompt)
        if browse and NATIVE_DIALOGS:
            self._file_dialog = pfd.select_folder("Open experiment.zarr", self._open_prompt.path or os.getcwd())
        if submitted:
            try:
                self.open(submitted)
                self._open_prompt.open = False
            except Exception as error:  # noqa: BLE001 - shown inside the prompt
                self._open_prompt.status = f"{type(error).__name__}: {error}"

    def _poll_file_dialog(self) -> None:
        if self._file_dialog is None or not self._file_dialog.ready():
            return
        result = self._file_dialog.result()
        self._file_dialog = None
        if result:
            self._open_prompt.path = str(result)

    # ------------------------------------------------------------------
    # imgui: sources panel
    # ------------------------------------------------------------------

    def _table_select(self, k: int) -> None:
        self.clear_group()
        self.select(k)

    def _table_ctrl(self, k: int) -> None:
        self.toggle_group(k)

    def _draw_sources_panel(self) -> None:
        if not self.n_sources:
            imgui.text_disabled("no sources yet")
            imgui.text_wrapped(
                "Sources appear once extraction has run: localize on each trial, align the "
                "trials, pick peaks in the averaged activity image, then fit the NMF."
            )
            return
        changed = draw_label_filter(self._order, self._labels, "_src")
        set_tooltip("filter by class label")
        changed |= draw_range_filter(self._order, "_src")
        set_tooltip("footprint area in pixels, in this trial")
        imgui.text_disabled(f"{len(self._order.order)}/{self._order.n_items} in view")
        if changed:
            self._order.rebuild()
        footer = imgui.get_frame_height_with_spacing() + em(0.6)
        if imgui.begin_child("##source_table", imgui.ImVec2(0, -footer)):
            pos = self._order.pos
            actions = (
                RowAction(fa.ICON_FA_CROSSHAIRS, "center the view on this source", lambda k: self.select(k, center=True)),
                RowAction(fa.ICON_FA_CHART_LINE, "add to the plotted group (ctrl+click does too)", self.toggle_group),
            )
            self._scroll_to_selection = draw_roi_table(
                self._order,
                self._labels,
                _SOURCE_COLUMNS,
                self._formatters(),
                self._scroll_to_selection,
                table_id="subcell_sources",
                on_select=self._table_select,
                actions=actions,
                is_grouped=lambda k: k in self._group,
                on_ctrl_select=self._table_ctrl,
            )
            if self._order.pos != pos and self._order.current is not None:
                self.select(self._order.current)
        imgui.end_child()
        imgui.separator()
        if self._group:
            if imgui.button(f"clear group ({len(self._group)})"):
                self.clear_group()
            imgui.same_line(0, em(0.5))
        if imgui.button("plot all in view"):
            self._group = [int(k) for k in self._order.order[:50]]
            self._trace_fit = True
            self._refresh_overlays()
        set_tooltip("group the first 50 sources the table lists")

    # ------------------------------------------------------------------
    # imgui: right panel
    # ------------------------------------------------------------------

    def select_tab(self, name: str) -> None:
        """Bring one right-panel tab to the front on the next frame."""
        self._select_tab = name

    def _draw_right_panel(self) -> None:
        if not imgui.begin_tab_bar("##right_tabs"):
            return
        wanted = getattr(self, "_select_tab", None)
        for name, draw in (
            ("traces", self._draw_traces_tab),
            ("motion", self._draw_motion_tab),
            ("trials", self._draw_trials_tab),
            ("log", self._draw_log_tab),
        ):
            flags = imgui.TabItemFlags_.set_selected if name == wanted else 0
            selected, _ = imgui.begin_tab_item(name, None, flags)
            if selected:
                self._tab = name
                draw()
                imgui.end_tab_item()
        if wanted is not None:
            self._select_tab = None
        imgui.end_tab_bar()

    # traces ------------------------------------------------------------

    def _x_seconds(self, n: int, hz: float) -> np.ndarray:
        return np.arange(n, dtype=np.float32) / np.float32(hz)

    def _trace(self, k: int, signal: str, trial) -> np.ndarray:
        y = np.asarray(getattr(trial, signal.lower() if signal != "F0" else "f0")[k], np.float32)
        if self._per_f0 and signal != "F0":
            f0 = np.asarray(trial.f0[k], np.float32)
            with np.errstate(all="ignore"):
                y = np.where(f0 > 0, y / f0 * 100.0, np.nan).astype(np.float32)
        return y

    def _plot_lines(self) -> list[tuple[str, int, str]]:
        sources = self.plotted_sources()
        return [(f"{k} {sig}", k, sig) for k in sources for sig in SIGNALS if self._signals[sig]]

    def _draw_trace_options(self) -> None:
        for sig in SIGNALS:
            _, self._signals[sig] = imgui.checkbox(sig, self._signals[sig])
            imgui.same_line(0, em(0.6))
        changed, self._per_f0 = imgui.checkbox("% of F0", self._per_f0)
        set_tooltip("divide by the source's fitted baseline F0, in percent")
        if changed:
            self._trace_fit = True
        imgui.same_line(0, em(0.8))
        changed, self._autofit = imgui.checkbox("autofit", self._autofit)
        if changed and self._autofit:
            self._force_fit = True
        imgui.same_line(0, em(0.4))
        if imgui.button("fit"):
            self._force_fit = True

    def _draw_traces_tab(self) -> None:
        trial = self._trial_extraction()
        lines = self._plot_lines() if trial is not None else []
        self._draw_trace_options()
        if trial is None:
            imgui.text_disabled("no traces: this trial has no extraction")
            return
        if not lines:
            imgui.text_disabled("select a source in the table or on the image; ctrl+click groups")
            return
        record = self._session.trials[self._trial]
        hz = record.full_hz
        imgui.text_disabled(
            f"trial {self._trial + 1} · {len(self.plotted_sources())} source(s) · {trial.n_time} samples @ {hz:.0f} Hz"
            f" · censored {100 * record.discard_fraction:.1f}%"
        )
        height = max(imgui.get_content_region_avail().y * 0.7, em(8))
        self._draw_signal_plot("##traces", lines, trial, hz, height)
        self._draw_trace_stats(trial)

    def _draw_signal_plot(self, plot_id: str, lines, trial, hz: float, height: float) -> None:
        if implot.get_current_context() is None:
            implot.create_context()
        key = tuple(label for label, _k, _s in lines) + (self._trial, self._per_f0)
        if key != self._plot_key:
            self._plot_key = key
            self._trace_fit = True
        fit = (self._trace_fit and self._autofit) or self._force_fit
        self._trace_fit = False
        self._force_fit = False
        if fit:
            implot.set_next_axes_to_fit()
        flags = implot.Flags_.no_title
        if not implot.begin_plot(plot_id, imgui.ImVec2(-1, height), flags):
            return
        try:
            implot.setup_axes("time (s)", "% F0" if self._per_f0 else "signal")
            implot.setup_legend(implot.Location_.north_west, implot.LegendFlags_.outside)
            fs = self._footprint_set(self._trial)
            x_full = self._x_seconds(trial.n_time, hz)
            limits = implot.get_plot_limits()
            window = visible_slice(x_full, limits.x.min, limits.x.max) if not fit else slice(0, x_full.size)
            for label, k, sig in lines:
                y = self._trace(k, sig, trial)
                xs, ys = decimate_minmax(x_full[window], y[window], _PLOT_POINTS)
                rgb = fs.color(k) if fs is not None else (1.0, 1.0, 1.0)
                if sig == "ls":
                    rgb = _lighten(rgb, 0.45)
                elif sig == "F0":
                    rgb = _lighten(rgb, 0.7)
                implot.push_colormap(_line_colormap(rgb))
                if sig == "events":
                    implot.plot_stems(label, xs, ys)
                else:
                    implot.plot_line(label, xs, ys)
                implot.pop_colormap()
            self._draw_discard_bands(self._discard_spans(self._trial))
            self._draw_playhead(x_full)
        finally:
            implot.end_plot()

    def _discard_spans(self, trial: int) -> np.ndarray:
        if trial not in self._spans:
            record = self._session.trials[trial]
            mask = record.discard_full if record.discard_full is not None else np.zeros(0, bool)
            self._spans[trial] = spans_from_mask(mask, record.full_hz)
        return self._spans[trial]

    def _draw_discard_bands(self, spans: np.ndarray) -> None:
        if spans.shape[0] == 0 or spans.shape[0] > 5000:
            return
        limits = implot.get_plot_limits()
        x0, x1 = limits.x.min, limits.x.max
        visible = spans[(spans[:, 1] >= x0) & (spans[:, 0] <= x1)]
        if visible.shape[0] == 0:
            return
        draw_list = implot.get_plot_draw_list()
        color = imgui.get_color_u32(_DISCARD_COLOR)
        implot.push_plot_clip_rect()
        for start, stop in visible:
            p0 = implot.plot_to_pixels(float(start), limits.y.max)
            p1 = implot.plot_to_pixels(float(stop), limits.y.min)
            draw_list.add_rect_filled(imgui.ImVec2(p0.x, p0.y), imgui.ImVec2(max(p1.x, p0.x + 1.0), p1.y), color)
        implot.pop_plot_clip_rect()

    def _draw_playhead(self, xs: np.ndarray) -> None:
        t = self.current_time()
        moved, at = implot.drag_line_x(0, float(t), _CURSOR_COLOR, 1.5)[:2]
        if moved:
            self.set_time(float(np.clip(at, 0.0, xs[-1] if xs.size else 0.0)))

    def _draw_trace_stats(self, trial) -> None:
        sources = self.plotted_sources()
        flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h | imgui.TableFlags_.scroll_y
        if not imgui.begin_table("##trace_stats", 6, flags, imgui.ImVec2(0, 0)):
            return
        for name in ("source", "label", "snr", "events", "peak", "mean F0"):
            imgui.table_setup_column(name)
        imgui.table_headers_row()
        for k in sources:
            events = trial.events[k]
            with np.errstate(all="ignore"):
                n_events = int(np.nansum(events > 0))
                peak = float(np.nanmax(events)) if np.isfinite(events).any() else 0.0
                f0 = float(np.nanmean(trial.f0[k])) if np.isfinite(trial.f0[k]).any() else 0.0
            cells = (
                f"{k}",
                self._labels.name_of(k),
                f"{trial.snr[k]:.1f}",
                f"{n_events}",
                f"{peak:.3g}",
                f"{f0:.3g}",
            )
            imgui.table_next_row()
            for cell in cells:
                imgui.table_next_column()
                imgui.text(cell)
        imgui.end_table()

    # motion ------------------------------------------------------------

    def _draw_motion_tab(self) -> None:
        if not self._session.n_trials:
            imgui.text_disabled("no trials registered")
            return
        record = self._session.trials[self._trial]
        adata = record.alignment
        if adata is None:
            imgui.text_disabled("this trial has no alignment data")
            return
        offset = corr = None
        if self._session.trial_offsets is not None and self._trial < self._session.trial_offsets.shape[1]:
            offset = self._session.trial_offsets[:, self._trial]
        if self._session.trial_corr is not None and self._trial < self._session.trial_corr.shape[0]:
            corr = self._session.trial_corr[self._trial]
        imgui.text_disabled(
            f"trial {self._trial + 1} · max shift {record.max_shift:.1f} px · censored {100 * record.discard_fraction:.1f}%"
            + (f" · cross-trial offset ({offset[0]:.1f}, {offset[1]:.1f}) px" if offset is not None else "")
            + (f" · corr {corr:.3f}" if corr is not None and np.isfinite(corr) else "")
        )
        if implot.get_current_context() is None:
            implot.create_context()
        avail = imgui.get_content_region_avail().y
        h1 = max(avail * 0.5 - em(1), em(6))
        h2 = max(avail * 0.5 - em(1), em(6))
        x_full = self._x_seconds(adata.motion_r.size, record.full_hz)
        x_ds = self._x_seconds(adata.rec_neg_err.size, record.align_hz)
        spans = self._discard_spans(self._trial)

        if implot.begin_plot("##shifts", imgui.ImVec2(-1, h1), implot.Flags_.no_title):
            try:
                implot.setup_axes("time (s)", "shift (px)")
                implot.setup_legend(implot.Location_.north_west, implot.LegendFlags_.outside)
                limits = implot.get_plot_limits()
                window = visible_slice(x_full, limits.x.min, limits.x.max)
                for label, values, rgb in (
                    ("rows", adata.motion_r, (0.4, 0.7, 1.0)),
                    ("cols", adata.motion_c, (1.0, 0.6, 0.3)),
                ):
                    xs, ys = decimate_minmax(x_full[window], np.asarray(values, np.float32)[window], _PLOT_POINTS)
                    implot.push_colormap(_line_colormap(rgb))
                    implot.plot_line(label, xs, ys)
                    implot.pop_colormap()
                    self._draw_discard_bands(spans)
                self._draw_playhead(x_full)
            finally:
                implot.end_plot()
        if implot.begin_plot("##recerr", imgui.ImVec2(-1, h2), implot.Flags_.no_title):
            try:
                implot.setup_axes("time (s)", "reconstruction error")
                limits = implot.get_plot_limits()
                window = visible_slice(x_ds, limits.x.min, limits.x.max)
                xs, ys = decimate_minmax(x_ds[window], np.asarray(adata.rec_neg_err, np.float32)[window], _PLOT_POINTS)
                implot.push_colormap(_line_colormap((0.9, 0.4, 0.4)))
                implot.plot_line("rec err", xs, ys)
                implot.pop_colormap()
                self._draw_discard_bands(spans)
                self._draw_playhead(x_ds)
            finally:
                implot.end_plot()

    # trials ------------------------------------------------------------

    def _draw_trials_tab(self) -> None:
        session = self._session
        if not session.n_trials:
            imgui.text_disabled("no trials registered")
            return
        valid = set(int(v) for v in session.valid_trials) if session.valid_trials is not None else None
        columns = ("#", "store", "frames", "Hz", "dur", "shift", "censored", "corr", "offset", "raw", "traces")
        flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h | imgui.TableFlags_.scroll_y | imgui.TableFlags_.resizable
        if not imgui.begin_table("##trials", len(columns), flags, imgui.ImVec2(0, 0)):
            return
        imgui.table_setup_scroll_freeze(0, 1)
        for name in columns:
            imgui.table_setup_column(name)
        imgui.table_headers_row()
        for pos, record in enumerate(session.trials):
            corr = session.trial_corr[pos] if session.trial_corr is not None and pos < len(session.trial_corr) else np.nan
            offset = session.trial_offsets[:, pos] if session.trial_offsets is not None and pos < session.trial_offsets.shape[1] else None
            has_raw = any(n.startswith("registered raw") for n in record.movies)
            has_traces = session.extraction is not None and record.index in session.extraction.trials
            imgui.table_next_row()
            imgui.table_next_column()
            clicked, _ = imgui.selectable(f"{pos + 1}##trial{pos}", pos == self._trial, imgui.SelectableFlags_.span_all_columns)
            if clicked:
                self.set_trial(pos)
            cells = (
                f"{record.index}",
                f"{record.n_raw_frames}",
                f"{record.full_hz:.0f}",
                f"{record.duration_s:.1f} s",
                f"{record.max_shift:.1f}",
                f"{100 * record.discard_fraction:.1f}%",
                "-" if not np.isfinite(corr) else f"{corr:.3f}",
                "-" if offset is None or not np.all(np.isfinite(offset)) else f"{offset[0]:.1f}, {offset[1]:.1f}",
                fa.ICON_FA_CHECK if has_raw else "",
                fa.ICON_FA_CHECK if has_traces else "",
            )
            for i, cell in enumerate(cells):
                imgui.table_next_column()
                if i == 5 and valid is not None and pos not in valid:
                    imgui.text_colored(to_vec4(THEME.err), cell + " ✕")
                else:
                    imgui.text(cell)
        imgui.end_table()

    # log ---------------------------------------------------------------

    def _draw_log_tab(self) -> None:
        run = self._run
        if run is None:
            imgui.text_disabled("no pipeline run in this window yet")
            return
        imgui.text_disabled(f"{run.status} · {run.stage} · {format_seconds(run.elapsed)} · {run.store_path}")
        if imgui.begin_child("##log", imgui.ImVec2(0, 0), child_flags=imgui.ChildFlags_.borders):
            for line in list(run.log):
                imgui.text_unformatted(line)
            if run.running and imgui.get_scroll_y() >= imgui.get_scroll_max_y() - em(2):
                imgui.set_scroll_here_y(1.0)
        imgui.end_child()


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Subcell spine extraction: view a store, or run and watch")
    parser.add_argument("store", nargs="?", help="experiment.zarr or the folder holding it; omit to open empty")
    parser.add_argument("--raw", default=None, help="the unregistered movie (TIFF or a path mbo_utilities reads)")
    parser.add_argument("--labels", default=None, help="comma-separated class names, e.g. spine,shaft,junk")
    parser.add_argument("--size", default="1700x950", help="figure size WxH in pixels")
    args = parser.parse_args(argv)

    width, height = (int(v) for v in args.size.lower().split("x"))
    label_names = tuple(args.labels.split(",")) if args.labels else DEFAULT_LABEL_NAMES
    vis = SubcellVis(args.store, raw=args.raw, label_names=label_names, size=(width, height))
    vis.show()
    fpl.loop.run()


if __name__ == "__main__":
    main()
