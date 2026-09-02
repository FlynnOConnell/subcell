"""Small pure helpers behind ``SubcellVis``: decimation, spans, config panels."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
    "auto_contrast",
    "decimate_minmax",
    "draw_model_fields",
    "spans_from_mask",
    "visible_slice",
]


def spans_from_mask(mask: np.ndarray, hz: float) -> np.ndarray:
    """
    ``(n, 2)`` start and end times in seconds of each run of True in ``mask``.

    ``hz`` converts frame indices to seconds; the end is exclusive, so a run
    of one frame spans ``1 / hz``.
    """
    mask = np.asarray(mask, bool)
    if mask.size == 0 or not mask.any():
        return np.zeros((0, 2), np.float64)
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return np.column_stack([starts, stops]).astype(np.float64) / float(hz)


def visible_slice(x: np.ndarray, x0: float, x1: float, pad: int = 2) -> slice:
    """Indices of a sorted ``x`` covering ``[x0, x1]`` with ``pad`` samples of margin."""
    if x.size == 0:
        return slice(0, 0)
    i0 = int(np.searchsorted(x, x0, side="left")) - pad
    i1 = int(np.searchsorted(x, x1, side="right")) + pad
    return slice(max(i0, 0), min(i1, x.size))


def decimate_minmax(x: np.ndarray, y: np.ndarray, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Min-max decimation that keeps every peak and trough visible.

    Each of ``n_out // 2`` buckets contributes its minimum and its maximum, in
    time order, so a plot of the result outlines the full-resolution trace.
    NaN samples are ignored inside a bucket; an all-NaN bucket yields NaN, so
    a censored span still shows as a gap.
    """
    n = x.size
    if n <= n_out or n_out < 4:
        return x, y
    n_buckets = n_out // 2
    edges = np.linspace(0, n, n_buckets + 1).astype(int)
    xs = np.empty(2 * n_buckets, x.dtype)
    ys = np.empty(2 * n_buckets, y.dtype)
    for b in range(n_buckets):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            hi = lo + 1
        seg = y[lo:hi]
        if np.all(np.isnan(seg)):
            xs[2 * b] = x[lo]
            xs[2 * b + 1] = x[hi - 1]
            ys[2 * b] = np.nan
            ys[2 * b + 1] = np.nan
            continue
        i_min = int(np.nanargmin(seg))
        i_max = int(np.nanargmax(seg))
        first, second = sorted((i_min, i_max))
        xs[2 * b] = x[lo + first]
        ys[2 * b] = seg[first]
        xs[2 * b + 1] = x[lo + second]
        ys[2 * b + 1] = seg[second]
    return xs, ys


def auto_contrast(sample: np.ndarray, lo: float = 1.0, hi: float = 99.5) -> tuple[float, float]:
    """Percentile contrast limits of a NaN-laden sample; (0, 1) when nothing is finite."""
    values = np.asarray(sample, np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    vmin, vmax = (float(v) for v in np.percentile(values, (lo, hi)))
    if not vmax > vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def draw_model_fields(model: Any, id_suffix: str = "", skip: tuple[str, ...] = ()) -> bool:
    """
    imgui editors for every scalar field of a pydantic model, in place.

    bool draws a checkbox, int an integer input, float a float input; a field
    holding None or a non-scalar shows read-only. The field description is the
    tooltip. Returns True when a value changed.
    """
    from imgui_bundle import imgui

    changed = False
    fields = getattr(type(model), "model_fields", {})
    for name, info in fields.items():
        if name in skip:
            continue
        value = getattr(model, name)
        label = f"{name}##{id_suffix}"
        imgui.set_next_item_width(imgui.get_font_size() * 8)
        if isinstance(value, bool):
            edited, new = imgui.checkbox(label, value)
        elif isinstance(value, int):
            edited, new = imgui.input_int(label, value)
        elif isinstance(value, float):
            edited, new = imgui.input_float(label, value, 0.0, 0.0, "%.4g")
        else:
            imgui.text_disabled(f"{name}: {value if value is not None else 'auto'}")
            edited, new = False, value
        if edited and new != value:
            try:
                setattr(model, name, new)
                changed = True
            except (ValueError, TypeError):
                pass
        description = getattr(info, "description", None)
        if description and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(description)
    return changed


def format_seconds(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "-"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, sec = divmod(seconds, 60)
    return f"{int(minutes)} min {sec:.0f} s"
