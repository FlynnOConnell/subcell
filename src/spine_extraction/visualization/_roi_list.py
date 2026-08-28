"""ROI list panel using Panel MultiSelect widget.

Scrollable list of source names with SNR, filterable by SNR threshold.
Supports bidirectional selection sync with the image panel.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import panel as pn

from ._data_model import ExtractionData


class ROIListPanel:
    """Scrollable multi-select list of NMF source ROIs."""

    def __init__(
        self,
        data: ExtractionData,
        on_selection_change: Callable[[list[int]], None] | None = None,
    ):
        self._data = data
        self._callback = on_selection_change
        self._updating = False  # guard for bidirectional sync
        self._current_trial: int = data.trial_keys[0]

        # Build option list (always sorted by index)
        self._options = self._build_options()

        self._widget = pn.widgets.MultiSelect(
            name="Sources",
            options=self._options,
            size=min(12, max(6, data.n_sources)),
            width=80,
        )
        self._widget.param.watch(self._on_change, "value")

        # SNR threshold slider
        trial = data.trials[self._current_trial]
        snr_max = float(np.nanmax(trial.snr)) if len(trial.snr) > 0 else 5.0
        self._snr_slider = pn.widgets.FloatSlider(
            name="Min SNR",
            start=0.0,
            end=max(snr_max, 0.1),
            value=0.0,
            step=0.05,
        )
        self._snr_slider.param.watch(self._on_filter_change, "value")

    def _build_options(self, min_snr: float = 0.0) -> dict:
        """Build {label: index} option dict for the MultiSelect."""
        trial = self._data.trials[self._current_trial]

        # Only include sources with non-zero footprints
        fp_max = np.nanmax(trial.footprints, axis=0)  # (n_sources,)
        indices = [i for i in range(self._data.n_sources) if fp_max[i] > 0]

        # Filter by SNR
        if min_snr > 0:
            indices = [i for i in indices if trial.snr[i] >= min_snr]

        options = {}
        for i in indices:
            snr = trial.snr[i]
            snr_str = f"{snr:.2f}" if not np.isnan(snr) else "?"
            options[f"{i} ({snr_str})"] = i
        return options

    @property
    def snr_slider(self):
        """Expose the SNR slider widget for external layout."""
        return self._snr_slider

    @property
    def select_widget(self):
        """Expose the MultiSelect widget for external layout."""
        return self._widget

    def set_trial(self, trial_key: int) -> None:
        """Update when trial changes."""
        self._current_trial = trial_key
        trial = self._data.trials[trial_key]
        snr_max = float(np.nanmax(trial.snr)) if len(trial.snr) > 0 else 5.0
        self._snr_slider.end = max(snr_max, 0.1)
        self._refresh_options()

    def set_selection(self, indices: set[int]) -> None:
        """Programmatically set selected sources (from image panel click)."""
        self._updating = True
        try:
            # Filter to only indices that exist in current options
            available = set(self._widget.options.values())
            valid = sorted(indices & available)
            self._widget.value = valid
        finally:
            self._updating = False

    def get_selection(self) -> list[int]:
        """Get currently selected source indices."""
        return list(self._widget.value)

    def _on_change(self, event):
        """Handle user selection change in the list."""
        if self._updating:
            return
        if self._callback:
            self._callback(list(event.new))

    def _on_filter_change(self, event):
        self._refresh_options()

    def _refresh_options(self):
        """Rebuild options with current filter, preserving selection."""
        old_selection = set(self._widget.value)
        self._options = self._build_options(
            min_snr=self._snr_slider.value,
        )
        self._updating = True
        try:
            self._widget.options = self._options
            # Restore selection for items that still exist
            available = set(self._options.values())
            self._widget.value = sorted(old_selection & available)
        finally:
            self._updating = False
