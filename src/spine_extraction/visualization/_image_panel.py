"""Image panel with NMF footprint overlays using Bokeh.

Displays the mean image with semi-transparent colored footprint overlays.
Supports channel switching and click-to-select via TapTool.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from bokeh.events import Tap
from bokeh.models import ColumnDataSource, Range1d
from bokeh.plotting import figure

from ._data_model import ExtractionData

# Qualitative colormap: 20 distinct colors as (R, G, B) uint8
_COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
    (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
    (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
]


class ImagePanel:
    """Bokeh image display with footprint RGBA overlays."""

    # Max distance (pixels) from a marker to register a click
    _CLICK_RADIUS = 10

    def __init__(
        self,
        data: ExtractionData,
        on_source_tapped: Callable[[int, bool], None] | None = None,
        dark: bool = True,
    ):
        self._data = data
        self._on_source_tapped = on_source_tapped
        self._channel: int = 0
        self._current_trial: int = data.trial_keys[0]
        self._selected: set[int] = set()
        self._show_initial: bool = False

        # Precompute sel_pix coordinate arrays for peak lookups
        self._sel_rows, self._sel_cols = np.where(data.sel_pix)

        # Compute NMF footprint peak locations for markers
        self._peak_rows, self._peak_cols, self._active = self._compute_peak_locations()

        # Assign colors to sources
        self._colors = [_COLORS[i % len(_COLORS)] for i in range(data.n_sources)]

        H, W = data.image_shape

        # Set initial pixel dimensions that respect the image aspect ratio;
        # stretch_width will scale proportionally from this ratio.
        init_w = 600
        init_h = max(int(init_w * H / W), 100)
        self._fig = figure(
            width=init_w,
            height=init_h,
            x_range=Range1d(0, W),
            y_range=Range1d(H, 0),  # flipped for image coords
            tools="tap,pan,wheel_zoom,reset",
            sizing_mode="stretch_width",
        )
        self._fig.toolbar.active_drag = None  # pan available but not default
        self._fig.min_border_left = 50  # match trace panel to prevent shift
        self._fig.axis.visible = False
        self._fig.grid.visible = False
        self._fig.toolbar.logo = None

        # Transparent background to match notebook theme
        bg = "#1e1e1e" if dark else "#ffffff"
        self._fig.background_fill_color = bg
        self._fig.border_fill_color = bg
        self._fig.outline_line_color = None

        # Mean image layer (bottom)
        mean_rgba = self._build_mean_rgba(self._channel)
        self._mean_source = ColumnDataSource(data=dict(
            image=[self._rgba_to_uint32(mean_rgba)],
            x=[0], y=[0], dw=[W], dh=[H],
        ))
        self._fig.image_rgba(
            "image", source=self._mean_source,
            x="x", y="y", dw="dw", dh="dh",
        )

        # Overlay layer (top)
        overlay_rgba = self._build_overlay_rgba()
        self._overlay_source = ColumnDataSource(data=dict(
            image=[self._rgba_to_uint32(overlay_rgba)],
            x=[0], y=[0], dw=[W], dh=[H],
        ))
        self._fig.image_rgba(
            "image", source=self._overlay_source,
            x="x", y="y", dw="dw", dh="dh",
        )

        # NMF footprint peak markers (only sources active in current trial)
        if data.n_sources > 0:
            self._markers_source = ColumnDataSource(data=dict(
                x=self._peak_cols[self._active].tolist(),
                y=self._peak_rows[self._active].tolist(),
            ))
            self._fig.scatter(
                "x", "y", source=self._markers_source,
                size=8, fill_alpha=0, line_color="red", line_width=1.0,
                marker="cross", level="overlay",
            )

        # Tap callback
        self._fig.on_event(Tap, self._on_tap)

    @property
    def figure(self):
        return self._fig

    def set_channel(self, channel: int) -> None:
        """Switch the background mean image channel."""
        self._channel = channel
        rgba = self._build_mean_rgba(channel)
        H, W = self._data.image_shape
        self._mean_source.data = dict(
            image=[self._rgba_to_uint32(rgba)],
            x=[0], y=[0], dw=[W], dh=[H],
        )

    def set_trial(self, trial_key: int) -> None:
        """Switch displayed trial (affects footprint weights and peak markers)."""
        self._current_trial = trial_key
        self._peak_rows, self._peak_cols, self._active = self._compute_peak_locations()
        if hasattr(self, "_markers_source"):
            self._markers_source.data = dict(
                x=self._peak_cols[self._active].tolist(),
                y=self._peak_rows[self._active].tolist(),
            )
        self._update_overlay()

    def set_show_initial(self, show: bool) -> None:
        """Toggle display of initial footprint estimates."""
        self._show_initial = show
        self._update_overlay()

    def update_selection(self, selected: set[int]) -> None:
        """Update which sources are highlighted."""
        self._selected = set(selected)
        self._update_overlay()

    def _update_overlay(self) -> None:
        overlay_rgba = self._build_overlay_rgba()
        H, W = self._data.image_shape
        self._overlay_source.data = dict(
            image=[self._rgba_to_uint32(overlay_rgba)],
            x=[0], y=[0], dw=[W], dh=[H],
        )

    def _build_mean_rgba(self, channel: int) -> np.ndarray:
        """Convert a single channel of mean_image to RGBA uint8."""
        H, W = self._data.image_shape
        im = self._data.mean_image[:, :, channel].copy()
        im = np.nan_to_num(im, nan=0.0)

        vmin = np.percentile(im[im > 0], 1) if np.any(im > 0) else 0
        vmax = np.percentile(im, 99.5)
        if vmax <= vmin:
            vmax = vmin + 1
        im = np.clip((im - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[:, :, 0] = im
        rgba[:, :, 1] = im
        rgba[:, :, 2] = im
        rgba[:, :, 3] = 255
        return rgba

    def _build_overlay_rgba(self) -> np.ndarray:
        """Build RGBA overlay from NMF footprints with enhanced contrast."""
        H, W = self._data.image_shape
        rgba = np.zeros((H, W, 4), dtype=np.uint8)

        if self._current_trial not in self._data.trials:
            return rgba

        trial = self._data.trials[self._current_trial]
        sel_pix = self._data.sel_pix

        # Only render selected sources (not all)
        if not self._selected:
            if self._show_initial and self._data.initial_footprints is not None:
                self._draw_initial_outlines(rgba)
            return rgba

        for src_idx in self._selected:
            col = trial.footprints[:, src_idx]
            weight = np.zeros((H, W), dtype=np.float32)
            weight[sel_pix] = np.nan_to_num(col, nan=0.0)

            w_max = np.max(weight)
            if w_max <= 0:
                continue

            # Normalize and apply power-law contrast enhancement
            weight = np.clip(weight / w_max, 0.0, 1.0)
            weight = np.power(weight, 0.4)  # gamma < 1 boosts mid-range values

            r, g, b = self._colors[src_idx]
            base_alpha = 0.85
            alpha = (weight * base_alpha * 255).astype(np.uint8)

            mask = weight > 0.05
            if not np.any(mask):
                continue

            # Additive blending
            for ch, color_val in enumerate([r, g, b]):
                rgba[mask, ch] = np.clip(
                    rgba[mask, ch].astype(np.int16)
                    + (color_val * alpha[mask].astype(np.float32) / 255).astype(np.int16),
                    0, 255,
                ).astype(np.uint8)
            rgba[mask, 3] = np.clip(
                rgba[mask, 3].astype(np.int16) + alpha[mask].astype(np.int16),
                0, 255,
            ).astype(np.uint8)

        # Draw initial footprint outlines if toggled on
        if self._show_initial and self._data.initial_footprints is not None:
            self._draw_initial_outlines(rgba)

        return rgba

    def _draw_initial_outlines(self, rgba: np.ndarray) -> None:
        """Draw thin outlines of initial footprint masks onto the overlay."""
        from scipy.ndimage import binary_erosion

        init_fp = self._data.initial_footprints  # (H, W, n_sources) bool

        for src_idx in range(self._data.n_sources):
            mask = init_fp[:, :, src_idx]
            if not np.any(mask):
                continue
            eroded = binary_erosion(mask, iterations=1)
            outline = mask & ~eroded

            r, g, b = self._colors[src_idx]
            is_selected = src_idx in self._selected
            outline_alpha = 220 if is_selected else 140

            rgba[outline, 0] = np.clip(r, 0, 255)
            rgba[outline, 1] = np.clip(g, 0, 255)
            rgba[outline, 2] = np.clip(b, 0, 255)
            rgba[outline, 3] = outline_alpha

    def _compute_peak_locations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute (row, col) of peak NMF footprint weight for each source.

        Returns (peak_rows, peak_cols, active) where active is a bool mask
        of sources with non-zero footprints in the current trial.
        """
        trial = self._data.trials.get(self._current_trial)
        n = self._data.n_sources
        active = np.zeros(n, dtype=bool)
        peak_rows = np.zeros(n, dtype=np.float64)
        peak_cols = np.zeros(n, dtype=np.float64)

        if trial is None:
            return peak_rows, peak_cols, active

        for src_idx in range(n):
            col = trial.footprints[:, src_idx]
            col_clean = np.nan_to_num(col, nan=0.0)
            if np.max(col_clean) <= 0:
                continue
            active[src_idx] = True
            best_pix = int(np.argmax(col_clean))
            peak_rows[src_idx] = self._sel_rows[best_pix]
            peak_cols[src_idx] = self._sel_cols[best_pix]

        return peak_rows, peak_cols, active

    def _on_tap(self, event):
        """Handle click on image — select source with closest peak marker."""
        if self._data.n_sources == 0 or not np.any(self._active):
            return

        x, y = event.x, event.y

        # Only consider active sources (visible markers)
        dist = np.full(self._data.n_sources, np.inf)
        dist[self._active] = np.sqrt(
            (self._peak_cols[self._active] - x) ** 2
            + (self._peak_rows[self._active] - y) ** 2
        )
        closest = int(np.argmin(dist))

        if dist[closest] > self._CLICK_RADIUS:
            return

        # Detect Ctrl modifier
        ctrl = False
        try:
            ctrl = bool(event.modifiers.ctrl)
        except (AttributeError, TypeError):
            pass

        if self._on_source_tapped:
            self._on_source_tapped(closest, ctrl)

    @staticmethod
    def _rgba_to_uint32(rgba: np.ndarray) -> np.ndarray:
        """Convert (H, W, 4) uint8 RGBA to (H, W) uint32 for Bokeh image_rgba."""
        return rgba.view(dtype=np.uint32).reshape(rgba.shape[0], rgba.shape[1])
