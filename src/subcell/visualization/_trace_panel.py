"""Stacked time-series trace panel using Bokeh.

Displays selected ROI signals as vertically offset traces with
min-max downsampling for performance at 770K+ points per trace.
"""

from __future__ import annotations

import numpy as np
from bokeh.models import ColumnDataSource, CustomJSTickFormatter, FixedTicker, Range1d
from bokeh.plotting import figure

from ._data_model import ExtractionData
from ._image_panel import _COLORS


def minmax_downsample(
    x: np.ndarray, y: np.ndarray, n_out: int = 3000
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample preserving local minima and maxima.

    For each bucket, emits (t_min, y_min) and (t_max, y_max) in temporal order.
    This guarantees no peaks are clipped in the display.

    Parameters
    ----------
    x : np.ndarray
        (N,) time values.
    y : np.ndarray
        (N,) signal values (NaN-safe).
    n_out : int
        target number of output points (actual output is ~n_out).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (x_ds, y_ds) downsampled arrays.
    """
    n = len(x)
    if n <= n_out:
        return x.copy(), y.copy()

    n_buckets = n_out // 2
    bucket_size = n / n_buckets

    x_out = np.empty(2 * n_buckets + 2, dtype=x.dtype)
    y_out = np.empty(2 * n_buckets + 2, dtype=y.dtype)

    x_out[0] = x[0]
    y_out[0] = y[0]
    pos = 1

    for i in range(n_buckets):
        start = int(i * bucket_size)
        end = min(int((i + 1) * bucket_size), n)
        if start >= end:
            continue
        segment = y[start:end]

        if np.all(np.isnan(segment)):
            x_out[pos] = x[start]
            y_out[pos] = np.nan
            pos += 1
            continue

        idx_min = start + np.nanargmin(segment)
        idx_max = start + np.nanargmax(segment)

        if idx_min <= idx_max:
            x_out[pos] = x[idx_min]
            y_out[pos] = y[idx_min]
            x_out[pos + 1] = x[idx_max]
            y_out[pos + 1] = y[idx_max]
        else:
            x_out[pos] = x[idx_max]
            y_out[pos] = y[idx_max]
            x_out[pos + 1] = x[idx_min]
            y_out[pos + 1] = y[idx_min]
        pos += 2

    x_out[pos] = x[-1]
    y_out[pos] = y[-1]
    pos += 1

    return x_out[:pos], y_out[:pos]


MAX_TRACES = 10


class TracePanel:
    """Stacked time-series traces for selected NMF sources."""

    def __init__(
        self, data: ExtractionData, n_display_points: int = 4000, dark: bool = True
    ):
        self._data = data
        self._n_display_points = n_display_points
        self._current_trial: int = data.trial_keys[0]
        self._signal_type: str = "denoised"
        self._selected: list[int] = []
        self._spacing = 1.0  # vertical spacing between traces
        self._dark = dark

        trial = data.trials[self._current_trial]
        self._t_full = np.arange(trial.n_time) / data.analyze_hz

        self._sources: dict[int, ColumnDataSource] = {}
        self._fig = self._build_figure()

    def _build_figure(self):
        trial = self._data.trials[self._current_trial]
        t_max = trial.n_time / self._data.analyze_hz

        fig = figure(
            height=300,
            x_range=Range1d(0, t_max),
            y_range=Range1d(-1, 1),
            tools="xpan,pan,xwheel_zoom,box_zoom,ybox_zoom,reset,save",
            active_drag="xpan",
            active_scroll="xwheel_zoom",
            output_backend="webgl",
            sizing_mode="stretch_width",
        )

        if self._dark:
            bg = "#1e1e1e"
            fg = "#cccccc"
            grid_color = "#444444"
        else:
            bg = "#ffffff"
            fg = "#333333"
            grid_color = "#dddddd"

        fig.min_border_left = 50
        fig.background_fill_color = bg
        fig.border_fill_color = bg
        fig.outline_line_color = None
        fig.toolbar.logo = None

        fig.xaxis.axis_label = "Time (s)"
        fig.xaxis.axis_label_text_color = fg
        fig.xaxis.axis_line_color = fg
        fig.xaxis.major_tick_line_color = fg
        fig.xaxis.minor_tick_line_color = fg
        fig.xaxis.major_label_text_color = fg

        fig.yaxis.axis_label = ""
        fig.yaxis.visible = True
        fig.yaxis.axis_line_color = fg
        fig.yaxis.major_tick_line_color = fg
        fig.yaxis.minor_tick_line_color = fg
        fig.yaxis.major_label_text_color = fg

        fig.grid.grid_line_color = grid_color
        fig.grid.grid_line_alpha = 0.4

        fig.x_range.on_change("start", self._on_range_change)
        fig.x_range.on_change("end", self._on_range_change)

        return fig

    @property
    def figure(self):
        return self._fig

    def set_trial(self, trial_key: int) -> None:
        """Switch the displayed trial."""
        self._current_trial = trial_key
        trial = self._data.trials[trial_key]
        self._t_full = np.arange(trial.n_time) / self._data.analyze_hz
        self._fig.x_range.end = self._t_full[-1]
        self.update_traces(self._selected)

    def set_signal_type(self, signal_type: str) -> None:
        """Switch displayed signal: 'denoised', 'events', or 'ls'."""
        self._signal_type = signal_type
        self.update_traces(self._selected)

    def update_traces(self, selected: list[int]) -> None:
        """Redraw traces for the selected source indices."""
        selected = sorted(selected)[:MAX_TRACES]
        selection_changed = list(selected) != self._selected
        self._selected = list(selected)
        trial = self._data.trials[self._current_trial]

        if not selected:
            self._fig.renderers = [
                r for r in self._fig.renderers if not hasattr(r, "_trace_tag")
            ]
            self._sources.clear()
            self._fig.y_range.start = -1
            self._fig.y_range.end = 1
            self._fig.yaxis.ticker = FixedTicker(ticks=[])
            self._fig.yaxis.major_label_overrides = {}
            return

        signal_data = getattr(trial, self._signal_type)

        if selection_changed:
            amplitudes = []
            for src_idx in selected:
                y = signal_data[src_idx, :]
                valid = y[~np.isnan(y)]
                if len(valid) > 0:
                    amplitudes.append(
                        np.percentile(valid, 99) - np.percentile(valid, 1)
                    )
                else:
                    amplitudes.append(1.0)
            self._spacing = max(np.median(amplitudes) * 1.5, 1e-6)

        colors = _COLORS
        tick_positions = []
        tick_labels = {}
        global_y_min = np.inf
        global_y_max = -np.inf
        trace_data = []  # list of (src_idx, t_ds, y_ds, offset)

        for i, src_idx in enumerate(selected):
            offset = -i * self._spacing
            y_full = signal_data[src_idx, :].copy()

            t_ds, y_ds = minmax_downsample(self._t_full, y_full, self._n_display_points)
            y_ds = y_ds + offset

            valid = y_ds[~np.isnan(y_ds)]
            if len(valid) > 0:
                global_y_min = min(global_y_min, np.min(valid))
                global_y_max = max(global_y_max, np.max(valid))

            trace_data.append((src_idx, t_ds, y_ds))
            tick_positions.append(offset)
            tick_labels[offset] = src_idx

        margin = (
            (global_y_max - global_y_min) * 0.05 if global_y_max > global_y_min else 0.5
        )
        y_lo = global_y_min - margin
        y_hi = global_y_max + margin

        doc = self._fig.document
        if doc:
            doc.hold("combine")

        try:
            self._fig.y_range.start = y_lo
            self._fig.y_range.end = y_hi
            self._fig.y_range.reset_start = y_lo
            self._fig.y_range.reset_end = y_hi

            if selection_changed:
                self._fig.renderers = [
                    r for r in self._fig.renderers if not hasattr(r, "_trace_tag")
                ]
                self._sources.clear()

            for src_idx, t_ds, y_ds in trace_data:
                if selection_changed:
                    cds = ColumnDataSource(data={"t": t_ds, "y": y_ds})
                    self._sources[src_idx] = cds

                    r, g, b = colors[src_idx % len(colors)]
                    color = f"#{r:02x}{g:02x}{b:02x}"
                    r = self._fig.line(
                        "t",
                        "y",
                        source=cds,
                        line_color=color,
                        line_width=1.2,
                        line_alpha=0.85,
                    )
                    r._trace_tag = True
                else:
                    self._sources[src_idx].data = {"t": t_ds, "y": y_ds}

            self._fig.yaxis.ticker = FixedTicker(ticks=tick_positions)

            pairs = ", ".join(
                f"[{offset}, {src_idx}]" for offset, src_idx in tick_labels.items()
            )
            self._fig.yaxis.formatter = CustomJSTickFormatter(
                code=f"""
                var pairs = [{pairs}];
                var best = '';
                var best_dist = 1e9;
                for (var i = 0; i < pairs.length; i++) {{
                    var dist = Math.abs(tick - pairs[i][0]);
                    if (dist < best_dist) {{
                        best_dist = dist;
                        best = '' + pairs[i][1];
                    }}
                }}
                return best_dist < 0.01 ? best : '';
            """
            )
        finally:
            if doc:
                doc.unhold()

    def _on_range_change(self, attr, old, new):
        """Placeholder — no dynamic re-downsample needed."""
