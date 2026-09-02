"""
Interactive visualization of subcell results.

``SubcellVis`` is the NDWidget + imgui viewer that also drives the pipeline;
``ExtractionViewer`` is the older Panel/Bokeh notebook viewer. Each is
imported on first use, so importing this package pulls in neither GUI stack.
"""

from __future__ import annotations

from importlib import import_module

from ._loaders import load_auto, load_mat, load_zarr

__all__ = [
    "ExtractionViewer",
    "SubcellVis",
    "load_auto",
    "load_mat",
    "load_session",
    "load_zarr",
]

_LAZY = {
    "SubcellVis": ("subcell.visualization.subcell_vis", "SubcellVis"),
    "ExtractionViewer": ("subcell.visualization.viewer", "ExtractionViewer"),
    "load_session": ("subcell.visualization._session", "load_session"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = target
    value = getattr(import_module(module), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
