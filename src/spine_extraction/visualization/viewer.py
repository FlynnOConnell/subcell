"""Top-level interactive NMF extraction viewer.

Wires together the image panel, ROI list, and trace panel into a
Panel layout that works in Jupyter notebooks.

Usage::

    from spine_extraction.visualization import ExtractionViewer

    viewer = ExtractionViewer()
    viewer.show()                    # empty, use Browse button
    # or
    viewer.load("path/to/registered.zarr")
    viewer.show()
"""

from __future__ import annotations

import logging
from pathlib import Path

import panel as pn

from ._data_model import ExtractionData
from ._loaders import load_auto
from ._image_panel import ImagePanel
from ._roi_list import ROIListPanel
from ._trace_panel import TracePanel

logger = logging.getLogger(__name__)

_SIDEBAR_WIDTH = 80


def _browse_file() -> str | None:
    """Open a chooser that lets the user pick a .mat file or .zarr folder."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        result = [None]

        chooser = tk.Tk()
        chooser.title("Open extraction results")
        chooser.attributes("-topmost", True)
        chooser.resizable(False, False)

        # Center on screen
        w, h = 300, 100
        x = (chooser.winfo_screenwidth() - w) // 2
        y = (chooser.winfo_screenheight() - h) // 2
        chooser.geometry(f"{w}x{h}+{x}+{y}")

        tk.Label(chooser, text="Select data source:").pack(pady=(10, 5))

        btn_frame = tk.Frame(chooser)
        btn_frame.pack(pady=5)

        def pick_mat():
            chooser.withdraw()
            path = filedialog.askopenfilename(
                title="Select .mat file",
                filetypes=[("MATLAB file", "*.mat"), ("All files", "*.*")],
            )
            if path:
                result[0] = path
            chooser.destroy()

        def pick_zarr():
            chooser.withdraw()
            path = filedialog.askdirectory(title="Select .zarr folder")
            if path:
                result[0] = path
            chooser.destroy()

        tk.Button(btn_frame, text="  .mat File  ", command=pick_mat).pack(side=tk.LEFT, padx=8)
        tk.Button(btn_frame, text="  .zarr Folder  ", command=pick_zarr).pack(side=tk.LEFT, padx=8)

        chooser.protocol("WM_DELETE_WINDOW", chooser.destroy)
        chooser.wait_window()

        return result[0]
    except Exception:
        return None


class ExtractionViewer:
    """Interactive NMF extraction results viewer for Jupyter notebooks."""

    def __init__(self, dark: bool = True):
        self._data: ExtractionData | None = None
        self._image_panel: ImagePanel | None = None
        self._roi_list: ROIListPanel | None = None
        self._trace_panel: TracePanel | None = None
        self._selected: set[int] = set()
        self._dark = dark
        self._loading = False  # suppress callbacks during load

        # Control widgets — all fixed to sidebar width
        sw = _SIDEBAR_WIDTH
        self._browse_button = pn.widgets.Button(
            name="Browse...", button_type="primary", width=sw,
        )
        self._browse_button.on_click(self._on_browse_click)

        self._channel_select = pn.widgets.Select(
            name="Channel", options=[], width=sw,
        )
        self._channel_select.param.watch(self._on_channel_change, "value")

        self._trial_select = pn.widgets.Select(
            name="Trial", options=[], width=sw,
        )
        self._trial_select.param.watch(self._on_trial_change, "value")

        self._signal_select = pn.widgets.Select(
            name="Signal",
            options=["denoised", "events", "ls"],
            value="denoised",
            width=sw,
        )
        self._signal_select.param.watch(self._on_signal_change, "value")

        self._initial_toggle = pn.widgets.Toggle(
            name="Initial footprints",
            value=False,
            width=sw,
            button_type="default",
        )
        self._initial_toggle.param.watch(self._on_initial_toggle, "value")

        self._status = pn.pane.Markdown(
            "*Browse for a .zarr or .mat file to begin.*",
            sizing_mode="stretch_width",
        )

        # Layout container
        self._layout = pn.Column(sizing_mode="stretch_width")
        self._rebuild_layout()

    def load(self, path: str | Path) -> None:
        """Load extraction results from a file path."""
        path = Path(path)
        self._do_load(path)

    def show(self) -> pn.Column:
        """Display the viewer in a Jupyter notebook."""
        pn.extension(sizing_mode="stretch_width")
        return self._layout

    def _do_load(self, path: Path) -> None:
        """Internal: load data and rebuild UI."""
        try:
            self._status.object = f"*Loading {path.name}...*"
            data = load_auto(path)

            # Null out old panels BEFORE updating widgets to prevent
            # callbacks from firing on stale panels during reload.
            self._loading = True
            self._image_panel = None
            self._roi_list = None
            self._trace_panel = None
            self._data = data

            # Update control widgets (callbacks are suppressed by _loading flag)
            ch_options = {f"Ch {i+1}": i for i in range(data.n_channels)}
            self._channel_select.options = ch_options
            self._channel_select.value = 0

            trial_options = {f"Trial {k}": k for k in data.trial_keys}
            self._trial_select.options = trial_options
            self._trial_select.value = data.trial_keys[0]

            # Show/hide initial footprint toggle based on data availability
            self._initial_toggle.visible = data.initial_footprints is not None
            self._initial_toggle.value = False

            # Create panels
            self._selected = set()
            self._image_panel = ImagePanel(
                data, on_source_tapped=self._on_source_tapped,
                dark=self._dark,
            )
            self._roi_list = ROIListPanel(
                data, on_selection_change=self._on_list_selection_changed,
            )
            self._trace_panel = TracePanel(data, dark=self._dark)
            self._loading = False

            self._rebuild_layout()

            trial = data.trials[data.trial_keys[0]]
            self._status.object = (
                f"**Loaded**: {path.name} — "
                f"{data.n_sources} sources, "
                f"{trial.n_time} frames @ {data.analyze_hz:.1f} Hz, "
                f"{data.n_channels} channels "
                f"({data.source_name})"
            )

        except Exception as e:
            self._loading = False
            logger.exception("Failed to load %s", path)
            self._status.object = f"**Error**: {e}"

    def _rebuild_layout(self) -> None:
        """Rebuild the Panel layout.

        Layout:
        +------------------+--------------------------------------+
        | Browse...        |                                      |
        | Channel ▼        |                                      |
        | Trial ▼          |          Image Panel                 |
        | Signal ▼         |         (stretch both)               |
        | [Init footprints]|                                      |
        | Min SNR ═══      |                                      |
        | ┌──────────────┐ |                                      |
        | │ Sources list │ |                                      |
        | └──────────────┘ |                                      |
        +------------------+--------------------------------------+
        | Status: Loaded: ...                                     |
        +---------------------------------------------------------+
        | Trace Panel (full width, stretch)                       |
        +---------------------------------------------------------+
        """
        sidebar_items = [
            self._browse_button,
            self._channel_select,
            self._trial_select,
            self._signal_select,
            self._initial_toggle,
        ]

        if self._data is not None and self._image_panel is not None:
            # Add ROI list widgets to sidebar
            sidebar_items.append(self._roi_list.snr_slider)
            self._roi_list.select_widget.width = _SIDEBAR_WIDTH
            sidebar_items.append(self._roi_list.select_widget)

            # Compute Sources list height to fill sidebar alongside image+traces.
            # Image height ≈ 600*H/W (from _image_panel init_w=600), traces=280.
            # Sidebar fixed widgets ≈ 280px. Each list row ≈ 18px.
            H, W = self._data.image_shape
            img_h = int(600 * H / W)
            total_h = img_h + 280  # image + traces
            sidebar_fixed = 280  # approx height of controls above Sources
            remaining = max(total_h - sidebar_fixed, 100)
            self._roi_list.select_widget.size = max(remaining // 18, 6)

            sidebar = pn.Column(
                *sidebar_items,
                width=_SIDEBAR_WIDTH + 20,
                sizing_mode="fixed",
            )

            image_pane = pn.pane.Bokeh(
                self._image_panel.figure,
                sizing_mode="stretch_width",
            )

            trace_pane = pn.pane.Bokeh(
                self._trace_panel.figure,
                sizing_mode="stretch_width",
                height=280,
            )

            # Right column: image on top, traces below
            right_col = pn.Column(
                image_pane,
                trace_pane,
                sizing_mode="stretch_width",
            )

            main = pn.Row(
                sidebar,
                right_col,
                sizing_mode="stretch_width",
            )

            self._layout.objects = [
                self._status,
                main,
            ]
        else:
            sidebar = pn.Column(
                *sidebar_items,
                width=_SIDEBAR_WIDTH + 20,
                sizing_mode="fixed",
            )

            self._layout.objects = [
                self._status,
                pn.Row(
                    sidebar,
                    pn.Spacer(sizing_mode="stretch_both", min_height=300),
                    sizing_mode="stretch_width",
                ),
                pn.Spacer(height=100),
            ]

    # ---- Event handlers ----

    def _on_browse_click(self, event) -> None:
        path = _browse_file()
        if path:
            self._do_load(Path(path))

    def _on_channel_change(self, event) -> None:
        if self._loading or event.new is None:
            return
        if self._image_panel:
            self._image_panel.set_channel(int(event.new))

    def _on_trial_change(self, event) -> None:
        if self._loading or event.new is None:
            return
        trial_key = int(event.new)
        if self._image_panel:
            self._image_panel.set_trial(trial_key)
        if self._roi_list:
            self._roi_list.set_trial(trial_key)
        if self._trace_panel:
            self._trace_panel.set_trial(trial_key)

    def _on_signal_change(self, event) -> None:
        if self._loading or not event.new:
            return
        if self._trace_panel:
            self._trace_panel.set_signal_type(event.new)

    def _on_initial_toggle(self, event) -> None:
        if self._loading:
            return
        if self._image_panel:
            self._image_panel.set_show_initial(bool(event.new))

    def _on_source_tapped(self, source_idx: int, ctrl: bool = False) -> None:
        """Handle click on a source in the image panel.

        Regular click: replace selection with this source (or deselect if
        it is the only one selected).
        Ctrl+click: toggle this source in/out of the current selection.
        """
        if ctrl:
            if source_idx in self._selected:
                self._selected.discard(source_idx)
            else:
                self._selected.add(source_idx)
        else:
            if self._selected == {source_idx}:
                self._selected = set()
            else:
                self._selected = {source_idx}
        self._sync_selection()

    def _on_list_selection_changed(self, indices: list[int]) -> None:
        """Handle selection change from the ROI list."""
        self._selected = set(indices)
        self._sync_selection_from_list()

    def _sync_selection(self) -> None:
        """Push current selection to all panels."""
        if self._image_panel:
            self._image_panel.update_selection(self._selected)
        if self._roi_list:
            self._roi_list.set_selection(self._selected)
        if self._trace_panel:
            self._trace_panel.update_traces(sorted(self._selected))

    def _sync_selection_from_list(self) -> None:
        """Push selection from list to image and traces (no list update needed)."""
        if self._image_panel:
            self._image_panel.update_selection(self._selected)
        if self._trace_panel:
            self._trace_panel.update_traces(sorted(self._selected))
